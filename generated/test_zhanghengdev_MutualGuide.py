import sys
_module = sys.modules[__name__]
del sys
data = _module
coco = _module
data_augment = _module
data_prefetcher = _module
voc0712 = _module
voc_eval = _module
xml_dataset = _module
distil = _module
models = _module
backbone = _module
cspdarknet_backbone = _module
efficientnet_backbone = _module
efficientnetv2_backbone = _module
gpunet_backbone = _module
regnet_backbone = _module
repvgg_backbone = _module
resnet_backbone = _module
shufflenet_backbone = _module
swin_backbone = _module
vgg_backbone = _module
base_blocks = _module
detector = _module
neck = _module
fpn_neck = _module
pafpn_neck = _module
ssd_neck = _module
test = _module
train = _module
utils = _module
box = _module
box_utils = _module
detection = _module
prior_box = _module
seq_matcher = _module
ema = _module
flops_counter = _module
loss = _module
balanced_l1_loss = _module
focal_loss = _module
gfocal_loss = _module
giou_loss = _module
hint_loss = _module
multibox_loss = _module
siou_loss = _module
lr_scheduler = _module
timer = _module

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


import torch.utils.data as data


import numpy as np


import random


import math


import torch


import torch.nn as nn


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


import re


import collections


from functools import partial


from torch import nn


from torch.nn import functional as F


from torch.utils import model_zoo


import copy


from typing import Any


from typing import Callable


from typing import Dict


from typing import Optional


from typing import List


from torch import Tensor


import torch.utils.model_zoo as model_zoo


from typing import Tuple


from collections import OrderedDict


import torch.nn.init as init


import torchvision


from math import sqrt as sqrt


from itertools import product as product


from copy import deepcopy


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

    def switch_to_deploy(self) ->None:
        fusedconv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, groups=self.conv.groups, bias=True).requires_grad_(False)
        w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
        w_bn = torch.diag(self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
        b_conv = torch.zeros(self.conv.weight.size(0), device=self.conv.weight.device) if self.conv.bias is None else self.conv.bias
        b_bn = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(torch.sqrt(self.bn.running_var + self.bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        self.conv = fusedconv
        delattr(self, 'bn')
        self.forward = self.fuseforward


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super().__init__()
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


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class CSPDarkNetBackbone(nn.Module):

    def __init__(self, version='cspdarknet-0.5', pretrained=True):
        super(CSPDarkNetBackbone, self).__init__()
        if version == 'cspdarknet-0.5':
            dep_mul, wid_mul = 0.33, 0.5
            self.out_channels = 256, 512
        elif version == 'cspdarknet-0.75':
            dep_mul, wid_mul = 0.67, 0.75
            self.out_channels = 384, 768
        elif version == 'cspdarknet-1.0':
            dep_mul, wid_mul = 1.0, 1.0
            self.out_channels = 512, 1024
        else:
            raise ValueError
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.stem = Focus(3, base_channels, ksize=3)
        self.dark2 = nn.Sequential(BaseConv(base_channels, base_channels * 2, 3, 2), CSPLayer(base_channels * 2, base_channels * 2, n=base_depth))
        self.dark3 = nn.Sequential(BaseConv(base_channels * 2, base_channels * 4, 3, 2), CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3))
        self.dark4 = nn.Sequential(BaseConv(base_channels * 4, base_channels * 8, 3, 2), CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3))
        self.dark5 = nn.Sequential(BaseConv(base_channels * 8, base_channels * 16, 3, 2), SPPBottleneck(base_channels * 16, base_channels * 16), CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False))
        if pretrained:
            None
            pretrained_dict = torch.load('weights/TorchPretrained/yolox_{}.pth'.format(version))['model']
            keys = list(pretrained_dict.keys())
            for k in keys:
                if k.startswith('backbone.backbone.'):
                    pretrained_dict[k[18:]] = pretrained_dict[k]
                    pretrained_dict.pop(k)
                else:
                    pretrained_dict.pop(k)
            self.load_state_dict(pretrained_dict, strict=True)
            for module in [self.stem, self.dark2]:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        out1 = self.dark4(x)
        out2 = self.dark5(out1)
        return out1, out2


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


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size"""

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
    """2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
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


def drop_connect(inputs, p, training):
    """Drop connect."""
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
    Static padding is necessary for ONNX exporting of models."""
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

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = self._block_args.se_ratio is not None and 0 < self._block_args.se_ratio <= 1
        self.id_skip = block_args.id_skip
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(in_channels=oup, out_channels=oup, groups=oup, kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)
        final_oup = self._block_args.output_filters
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


class Conv2dNormActivation(torch.nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: Optional[int]=None, groups: int=1, norm_layer: Optional[Callable[..., torch.nn.Module]]=torch.nn.BatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]]=torch.nn.ReLU, dilation: int=1, inplace: Optional[bool]=True, bias: Optional[bool]=None, conv_layer: Callable[..., torch.nn.Module]=torch.nn.Conv2d) ->None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        layers = [conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params = {} if inplace is None else {'inplace': inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


def _make_divisible(v: float, divisor: int, min_value: Optional[int]=None) ->int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.
    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(self, input_channels: int, squeeze_channels: int, activation: Callable[..., torch.nn.Module]=torch.nn.ReLU, scale_activation: Callable[..., torch.nn.Module]=torch.nn.Sigmoid) ->None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) ->Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) ->Tensor:
        scale = self._scale(input)
        return scale * input


class EfficientNetBackbone(nn.Module):

    def __init__(self) ->None:
        super().__init__()
        inverted_residual_setting = [FusedMBConvConfig(1, 3, 1, 24, 24, 2), FusedMBConvConfig(4, 3, 2, 24, 48, 4), FusedMBConvConfig(4, 3, 2, 48, 64, 4), MBConvConfig(4, 3, 2, 64, 128, 6), MBConvConfig(6, 3, 1, 128, 160, 9), MBConvConfig(6, 3, 2, 160, 256, 15)]
        norm_layer = partial(nn.BatchNorm2d, eps=0.001)
        layers: List[nn.Module] = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU))
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                stage.append(block_cnf.block(block_cnf, norm_layer))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
        self.features = nn.Sequential(*layers)
        None
        url = 'https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth'
        pretrained_dict = model_zoo.load_url(url)
        self.load_state_dict(pretrained_dict, strict=False)
        self.features1 = self.features[:-1]
        self.features2 = self.features[-1]
        self.features = None
        del self.features

    def forward(self, x: Tensor) ->Tensor:
        out1 = self.features1(x)
        out2 = self.features2(out1)
        return out1, out2


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family
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
        self.act1 = act_layer
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = gate_layer

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def get_padding(kernel_size: int, stride: int=1, dilation: int=1, **_) ->int:
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


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


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.
    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        raise NotImplementedError
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = in_channels if depthwise else kwargs.pop('groups', 1)
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            raise NotImplementedError
        else:
            m = create_conv2d_pad(in_channels, out_channels, kernel_size, groups=groups, **kwargs)
    return m


def drop_path(x, drop_prob: float=0.0, training: bool=False):
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
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


def get_act(actType: str=''):
    if actType == 'swish':
        return nn.SiLU
    elif actType == 'relu':
        return nn.ReLU
    else:
        raise NotImplementedError


class ConvBnAct(nn.Module):
    """Conv + Norm Layer + Activation w/ optional skip connection"""

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, dilation=1, pad_type='', skip=False, act_layer='relu', norm_layer=nn.BatchNorm2d, drop_path_rate=0.0):
        super(ConvBnAct, self).__init__()
        self.has_residual = skip and stride == 1 and in_chs == out_chs
        self.drop_path_rate = drop_path_rate
        self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, eps=0.001)
        self.act1 = get_act(act_layer)(inplace=True)
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.kernel_size = kernel_size
        self.stride = stride
        self.act_layer = act_layer

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def __repr__(self):
        name = 'conv_k{}_i{}_o{}_s{}_{}'.format(self.kernel_size, self.in_channels, self.out_channels, self.stride, self.act_layer)
        return name

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    """DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, pad_type='', noskip=False, pw_kernel_size=1, pw_act=False, act_layer='relu', norm_layer=nn.BatchNorm2d, se_layer=None, drop_path_rate=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act
        self.drop_path_rate = drop_path_rate
        self.conv_dw = create_conv2d(in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, eps=0.001)
        self.act1 = get_act(act_layer)(inplace=True)
        self.se = se_layer(in_chs, act_layer=get_act(act_layer)) if se_layer else nn.Identity()
        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, eps=0.001)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)
        return info

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(inp), nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()
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


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class EdgeResidual(nn.Module):
    """Residual block with expansion convolution followed by pointwise-linear w/ stride
    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html
    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, stride=1, dilation=1, pad_type='', force_in_chs=0, noskip=False, exp_ratio=1.0, pw_kernel_size=1, act_layer='relu', norm_layer=nn.BatchNorm2d, use_se=False, se_ratio=0.25, drop_path_rate=0.0):
        super(EdgeResidual, self).__init__()
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_path_rate = drop_path_rate
        self.conv_exp = create_conv2d(in_chs, mid_chs, exp_kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(mid_chs, eps=0.001)
        self.act1 = get_act(act_layer)(inplace=True)
        self.use_se = use_se
        if use_se:
            rd_ratio = se_ratio / exp_ratio
            self.se = SqueezeExcite(mid_chs, act_layer=get_act(act_layer), rd_ratio=rd_ratio)
        else:
            self.se = nn.Identity()
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, eps=0.001)
        self.kernel_size = exp_kernel_size
        self.expansion = exp_ratio
        self.in_channels = in_chs
        self.out_channels = out_chs
        self.stride = stride
        self.act_layer = act_layer

    def feature_info(self, location):
        if location == 'expansion':
            info = dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:
            info = dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)
        return info

    def __repr__(self):
        name = 'er_k{}_e{}_i{}_o{}_s{}_{}_se_{}'.format(self.kernel_size, self.expansion, self.in_channels, self.out_channels, self.stride, self.act_layer, self.use_se)
        return name

    def forward(self, x):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_residual:
            if self.drop_path_rate > 0.0:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class Fused_IRB(nn.Module):

    def __init__(self, num_in_channels: int=1, num_out_channels: int=1, kernel_size: int=3, stride: int=1, expansion: int=1):
        super().__init__()
        self.drop_connect_rate = 0.0
        self.in_channels = num_in_channels
        self.out_channels = num_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion
        self.body = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels * self.expansion, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(self.in_channels * self.expansion, eps=0.001), nn.ReLU(), nn.Conv2d(in_channels=self.in_channels * self.expansion, out_channels=self.out_channels, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(self.out_channels, eps=0.001))
        if self.stride == 1 and self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def drop_connect(self, inputs, training=False, drop_connect_rate=0.0):
        """Apply drop connect."""
        if not training:
            return inputs
        keep_prob = 1 - drop_connect_rate
        random_tensor = keep_prob + torch.rand((inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
        random_tensor.floor_()
        output = inputs.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        res = self.body(x)
        if self.shortcut is not None:
            if self.drop_connect_rate > 0 and self.training:
                res = self.drop_connect(res, self.training, self.drop_connect_rate)
            res = res + self.shortcut(x)
            return res
        else:
            return res

    def __repr__(self):
        name = 'k{}_e{}_i{}_o{}_s{}'.format(self.kernel_size, self.expansion, self.in_channels, self.out_channels, self.stride)
        return name


class Inverted_Residual_Block(nn.Module):

    def __init__(self, num_in_channels, num_out_channels, kernel_size, stride, expansion):
        super().__init__()
        self.drop_connect_rate = 0.0
        self.in_channels = num_in_channels
        self.out_channels = num_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion = expansion
        self.body = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels * self.expansion, 1, bias=False), nn.BatchNorm2d(self.in_channels * self.expansion), nn.ReLU(), nn.Conv2d(self.in_channels * self.expansion, self.in_channels * self.expansion, kernel_size, padding=kernel_size // 2, stride=stride, groups=self.in_channels * self.expansion, bias=False), nn.BatchNorm2d(self.in_channels * self.expansion), nn.ReLU(), nn.Conv2d(self.in_channels * self.expansion, self.out_channels, 1, bias=False), nn.BatchNorm2d(self.out_channels))
        if self.stride == 1 and self.in_channels == self.out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def drop_connect(self, inputs, training=False, drop_connect_rate=0.0):
        """Apply drop connect."""
        if not training:
            return inputs
        keep_prob = 1 - drop_connect_rate
        random_tensor = keep_prob + torch.rand((inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
        random_tensor.floor_()
        output = inputs.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        res = self.body(x)
        if self.shortcut is not None:
            if self.drop_connect_rate > 0 and self.training:
                res = self.drop_connect(res, self.training, self.drop_connect_rate)
            res = res + self.shortcut(x)
            return res
        else:
            return res

    def __repr__(self):
        name = 'k{}_e{}_i{}_o{}_s{}'.format(self.kernel_size, self.expansion, self.in_channels, self.out_channels, self.stride)
        return name


class Prologue(nn.Module):

    def __init__(self, num_in_channels, num_out_channels, act_layer='relu'):
        super().__init__()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.net = nn.Sequential(nn.Conv2d(self.num_in_channels, self.num_out_channels, 3, padding=1, stride=2, bias=False), nn.BatchNorm2d(self.num_out_channels, eps=0.001), get_act(act_layer)(inplace=True))
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.act_layer = act_layer

    def __repr__(self):
        name = 'prologue_i{}_o{}_s{}_{}'.format(self.num_in_channels, self.num_out_channels, 2, self.act_layer)
        return name

    def forward(self, x):
        return self.net(x)


class GPUNet(nn.Module):

    def __init__(self, config, ckpt):
        super(GPUNet, self).__init__()
        layerCounter = 1
        for layerConfig in config:
            layerCounter = layerCounter + 1
            assert 'layer_type' in layerConfig.keys()
            layerType = layerConfig['layer_type']
            if layerType == 'head':
                name = 'head: ' + str(layerCounter)
                layer = Prologue(num_in_channels=layerConfig['num_in_channels'], num_out_channels=layerConfig['num_out_channels'], act_layer=layerConfig.get('act', 'swish'))
                self.add_module(name, layer)
            elif layerType == 'conv':
                name = 'stage: ' + str(layerConfig['stage']) + ' layer'
                name += str(layerCounter)
                layer = ConvBnAct(in_chs=layerConfig['num_in_channels'], out_chs=layerConfig['num_out_channels'], kernel_size=layerConfig['kernel_size'], stride=layerConfig['stride'], act_layer=layerConfig['act'])
                self.add_module(name, layer)
            elif layerType == 'irb':
                name = 'stage: ' + str(layerConfig['stage']) + ' layer'
                name += str(layerCounter)
                layer = InvertedResidual(in_chs=layerConfig['num_in_channels'], out_chs=layerConfig['num_out_channels'], dw_kernel_size=layerConfig['kernel_size'], stride=layerConfig['stride'], exp_ratio=layerConfig['expansion'], use_se=layerConfig['use_se'], act_layer=layerConfig['act'])
                self.add_module(name, layer)
            elif layerType == 'fused_irb':
                name = 'stage: ' + str(layerConfig['stage']) + ' layer'
                name += str(layerCounter)
                layer = EdgeResidual(in_chs=layerConfig['num_in_channels'], out_chs=layerConfig['num_out_channels'], exp_kernel_size=layerConfig['kernel_size'], stride=layerConfig['stride'], dilation=1, pad_type='same', exp_ratio=layerConfig['expansion'], use_se=layerConfig['use_se'], act_layer=layerConfig['act'])
                self.add_module(name, layer)
            else:
                raise NotImplementedError
        ckpt = torch.load(ckpt, map_location=torch.device('cpu'))
        ckpt = {k[8:]: v for k, v in ckpt['state_dict'].items()}
        self.load_state_dict(ckpt, strict=False)

    def forward(self, x):
        out = list()
        for n, m in self.named_children():
            x = m(x)
            if n == '' or n == '':
                out.append(x)
            None
            None
        return x


class ConvNormActivation(torch.nn.Sequential):
    """
    Configurable block used for Convolution-Normalzation-Activation blocks.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalzation-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in wich case it will calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolutiuon layer. If ``None`` this layer wont be used. Default: ``torch.nn.BatchNorm2d``
        activation_layer (Callable[..., torch.nn.Module], optinal): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: Optional[int]=None, groups: int=1, norm_layer: Optional[Callable[..., torch.nn.Module]]=torch.nn.BatchNorm2d, activation_layer: Optional[Callable[..., torch.nn.Module]]=torch.nn.ReLU, dilation: int=1, inplace: bool=True) ->None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


class SimpleStemIN(ConvNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, width_in: int, width_out: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module]) ->None:
        super().__init__(width_in, width_out, kernel_size=3, stride=1, norm_layer=norm_layer, activation_layer=activation_layer)


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float]) ->None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width
        layers['a'] = ConvNormActivation(width_in, w_b, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=activation_layer)
        layers['b'] = ConvNormActivation(w_b, w_b, kernel_size=3, stride=stride, groups=g, norm_layer=norm_layer, activation_layer=activation_layer)
        if se_ratio:
            width_se_out = int(round(se_ratio * width_in))
            layers['se'] = SqueezeExcitation(input_channels=w_b, squeeze_channels=width_se_out, activation=activation_layer)
        layers['c'] = ConvNormActivation(w_b, width_out, kernel_size=1, stride=1, norm_layer=norm_layer, activation_layer=None)
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, width_in: int, width_out: int, stride: int, norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int=1, bottleneck_multiplier: float=1.0, se_ratio: Optional[float]=None) ->None:
        super().__init__()
        self.proj = None
        should_proj = width_in != width_out or stride != 1
        if should_proj:
            self.proj = ConvNormActivation(width_in, width_out, kernel_size=1, stride=stride, norm_layer=norm_layer, activation_layer=None)
        self.f = BottleneckTransform(width_in, width_out, stride, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio)
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) ->Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, width_in: int, width_out: int, stride: int, depth: int, block_constructor: Callable[..., nn.Module], norm_layer: Callable[..., nn.Module], activation_layer: Callable[..., nn.Module], group_width: int, bottleneck_multiplier: float, se_ratio: Optional[float]=None, stage_index: int=0) ->None:
        super().__init__()
        for i in range(depth):
            block = block_constructor(width_in if i == 0 else width_out, width_out, stride if i == 0 else 1, norm_layer, activation_layer, group_width, bottleneck_multiplier, se_ratio)
            self.add_module(f'block{stage_index}-{i}', block)


class BlockParams:

    def __init__(self, depths: List[int], widths: List[int], group_widths: List[int], bottleneck_multipliers: List[float], strides: List[int], se_ratio: Optional[float]=None) ->None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(cls, depth: int, w_0: int, w_a: float, w_m: float, group_width: int, bottleneck_multiplier: float=1.0, se_ratio: Optional[float]=None, **kwargs: Any) ->'BlockParams':
        """
        Programatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """
        QUANT = 8
        STRIDE = 2
        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError('Invalid RegNet settings')
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT)) * QUANT).int().tolist()
        num_stages = len(set(block_widths))
        split_helper = zip(block_widths + [0], [0] + block_widths, block_widths + [0], [0] + block_widths)
        splits = [(w != wp or r != rp) for w, wp, r, rp in split_helper]
        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = torch.diff(torch.tensor([d for d, t in enumerate(splits) if t])).int().tolist()
        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(stage_widths, bottleneck_multipliers, group_widths)
        return cls(depths=stage_depths, widths=stage_widths, group_widths=group_widths, bottleneck_multipliers=bottleneck_multipliers, strides=strides, se_ratio=se_ratio)

    def _get_expanded_params(self):
        return zip(self.widths, self.strides, self.depths, self.group_widths, self.bottleneck_multipliers)

    @staticmethod
    def _adjust_widths_groups_compatibilty(stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]) ->Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """

        def _make_divisible(v: float, divisor: int, min_value: Optional[int]=None) ->int:
            """
            This function is taken from the original tf repo.
            It ensures that all layers have a channel number that is divisible by 8
            It can be seen here:
            https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
            """
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]
        ws_bot = [_make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


model_urls = {'regnet_y_400mf': 'https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth', 'regnet_y_800mf': 'https://download.pytorch.org/models/regnet_y_800mf-1b27b58c.pth', 'regnet_y_1_6gf': 'https://download.pytorch.org/models/regnet_y_1_6gf-b11a554e.pth', 'regnet_y_3_2gf': 'https://download.pytorch.org/models/regnet_y_3_2gf-b5a9779c.pth', 'regnet_y_8gf': 'https://download.pytorch.org/models/regnet_y_8gf-d0d0e4a8.pth', 'regnet_y_16gf': 'https://download.pytorch.org/models/regnet_y_16gf-9e6ed7dd.pth', 'regnet_y_32gf': 'https://download.pytorch.org/models/regnet_y_32gf-4dee3f7a.pth', 'regnet_x_400mf': 'https://download.pytorch.org/models/regnet_x_400mf-adf1edd5.pth', 'regnet_x_800mf': 'https://download.pytorch.org/models/regnet_x_800mf-ad17e45c.pth', 'regnet_x_1_6gf': 'https://download.pytorch.org/models/regnet_x_1_6gf-e3633e7f.pth', 'regnet_x_3_2gf': 'https://download.pytorch.org/models/regnet_x_3_2gf-f342aeae.pth', 'regnet_x_8gf': 'https://download.pytorch.org/models/regnet_x_8gf-03ceed89.pth', 'regnet_x_16gf': 'https://download.pytorch.org/models/regnet_x_16gf-2007eb11.pth', 'regnet_x_32gf': 'https://download.pytorch.org/models/regnet_x_32gf-9d47f8d0.pth'}


class RegNetBackbone(nn.Module):

    def __init__(self, mf: int=400, pretrained: bool=True, stem_width: int=32, stem_type: Optional[Callable[..., nn.Module]]=None, block_type: Optional[Callable[..., nn.Module]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None, activation: Optional[Callable[..., nn.Module]]=None) ->None:
        super().__init__()
        self.mf = mf
        if self.mf == 400:
            block_params = BlockParams.from_init_params(depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25)
        elif self.mf == 800:
            block_params = BlockParams.from_init_params(depth=14, w_0=56, w_a=38.84, w_m=2.4, group_width=16, se_ratio=0.25)
        norm_layer = nn.BatchNorm2d
        block_type = ResBottleneckBlock
        activation = nn.ReLU
        self.stem = SimpleStemIN(3, stem_width, norm_layer, activation)
        current_width = stem_width
        for i, (width_out, stride, depth, group_width, bottleneck_multiplier) in enumerate(block_params._get_expanded_params()):
            setattr(self, f'block{i + 1}', AnyStage(current_width, width_out, stride, depth, block_type, norm_layer, activation, group_width, bottleneck_multiplier, block_params.se_ratio, stage_index=i + 1))
            current_width = width_out
        if pretrained:
            self.load_pre_trained_weights()

    def forward(self, x: Tensor) ->Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        out1 = self.block3(x)
        out2 = self.block4(out1)
        return out1, out2

    def load_pre_trained_weights(self) ->None:
        arch = 'regnet_y_{}mf'.format(self.mf)
        if arch not in model_urls:
            raise ValueError(f'No checkpoint is available for model type {arch}')
        None
        state_dict = model_zoo.load_url(model_urls[arch])
        for key in list(state_dict.keys()):
            if key.startswith('trunk_output.'):
                state_dict[key[13:]] = state_dict[key]
                state_dict.pop(key)
            if key.startswith('fc.'):
                state_dict.pop(key)
        self.load_state_dict(state_dict, strict=True)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels, kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride, padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias


class REPVGGBackbone(nn.Module):

    def __init__(self, version='repvgg-A2', override_groups_map=None, deploy=False, pretrained=True):
        super(REPVGGBackbone, self).__init__()
        self.version = version
        if self.version == 'repvgg-A0':
            num_blocks = [2, 4, 14, 1]
            width_multiplier = [0.75, 0.75, 0.75, 2.5]
            self.out_channels = 192, 1280
        elif self.version == 'repvgg-A1':
            num_blocks = [2, 4, 14, 1]
            width_multiplier = [1, 1, 1, 2.5]
            self.out_channels = 256, 1280
        elif self.version == 'repvgg-A2':
            num_blocks = [2, 4, 14, 1]
            width_multiplier = [1.5, 1.5, 1.5, 2.75]
            self.out_channels = 384, 1280
        else:
            raise ValueError
        assert len(width_multiplier) == 4
        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        if pretrained:
            self.load_pre_trained_weights()
            self.stage0.eval()
            for param in self.stage0.parameters():
                param.requires_grad = False
            self.stage0.switch_to_deploy()
            self.stage1.eval()
            for param in self.stage1.parameters():
                param.requires_grad = False
            self.stage1[0].switch_to_deploy()
            self.stage1[1].switch_to_deploy()

    def load_pre_trained_weights(self):
        None
        pretrained_dict = {'repvgg-A0': 'weights/TorchPretrained/RepVGG-A0-train.pth', 'repvgg-A1': 'weights/TorchPretrained/RepVGG-A1-train.pth', 'repvgg-A2': 'weights/TorchPretrained/RepVGG-A2-train.pth', 'repvgg-B1': 'weights/TorchPretrained/RepVGG-B1-train.pth', 'repvgg-B2': 'weights/TorchPretrained/RepVGG-B2-train.pth'}
        pretrained_dict = torch.load(pretrained_dict[self.version])
        pretrained_dict.pop('linear.weight')
        pretrained_dict.pop('linear.bias')
        self.load_state_dict(pretrained_dict, strict=True)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        out1 = self.stage3(x)
        out2 = self.stage4(out1)
        return out1, out2


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(BasicBlock, self).__init__()
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


class ResNetBackbone(nn.Module):

    def __init__(self, version=18, pretrained=True):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.version = version
        if self.version == 'resnet18':
            block, layers = BasicBlock, [2, 2, 2, 2]
            self.out_channels = 256, 512
        elif self.version == 'resnet34':
            block, layers = BasicBlock, [3, 4, 6, 3]
            self.out_channels = 256, 512
        elif self.version == 'resnet50':
            block, layers = Bottleneck, [3, 4, 6, 3]
            self.out_channels = 1024, 2048
        elif self.version == 'resnet101':
            block, layers = Bottleneck, [3, 4, 23, 3]
            self.out_channels = 1024, 2048
        else:
            raise ValueError
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if pretrained:
            self.load_pre_trained_weights()
            for module in [self.conv1, self.bn1, self.layer1]:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def load_pre_trained_weights(self):
        None
        pretrained_dict = {'resnet18': 'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'}
        pretrained_dict = model_zoo.load_url(pretrained_dict[self.version])
        pretrained_dict.pop('fc.weight')
        pretrained_dict.pop('fc.bias')
        self.load_state_dict(pretrained_dict, strict=True)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out1 = self.layer3(x)
        out2 = self.layer4(out1)
        return out1, out2


class ShuffleNetBackbone(nn.Module):

    def __init__(self, version='shufflenet-1.0', pretrained=True):
        super(ShuffleNetBackbone, self).__init__()
        self.version = version
        if self.version == 'shufflenet-0.5':
            self._stage_out_channels = [24, 48, 96, 192, 1024]
            self.out_channels = 96, 192
        elif self.version == 'shufflenet-1.0':
            self._stage_out_channels = [24, 116, 232, 464, 1024]
            self.out_channels = 232, 464
        elif self.version == 'shufflenet-1.5':
            self._stage_out_channels = [24, 176, 352, 704, 1024]
            self.out_channels = 352, 704
        elif self.version == 'shufflenet-2.0':
            self._stage_out_channels = [24, 244, 488, 976, 2048]
            self.out_channels = 488, 976
        else:
            raise ValueError
        input_channels = 3
        stages_repeats = [4, 8, 4]
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
        if pretrained:
            self.load_pre_trained_weights()
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

    def load_pre_trained_weights(self):
        None
        if self.version == 'shufflenet-0.5':
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth')
        elif self.version == 'shufflenet-1.0':
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth')
        elif self.version == 'shufflenet-1.5':
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth')
        elif self.version == 'shufflenet-2.0':
            state_dict = model_zoo.load_url('https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth')
        else:
            raise ValueError
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        out1 = self.stage3(x)
        out2 = self.stage4(out1)
        return out1, out2


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, drop_prob: float=0.0, training: bool=False):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """Multilayer perceptron."""

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


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
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
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, 'input feature has wrong size'
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    """Patch Merging Layer"""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B, H, W, C)
        pad_input = H % 2 == 1 or W % 2 == 1
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.blocks = nn.ModuleList([SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if i % 2 == 0 else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer) for i in range(depth)])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        w_slices = slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None)
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x.size()
        if W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SwinTransformerBackbone(nn.Module):
    """Swin Transformer backbone."""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, norm_layer=nn.LayerNorm, patch_norm=True, out_indices=(1, 2, 3), pretrained=True):
        super().__init__()
        self.version = 'T'
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if i_layer < self.num_layers - 1 else None)
            self.layers.append(layer)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        if pretrained:
            self.load_pre_trained_weights()

    def load_pre_trained_weights(self):
        None
        pretrained_dict = {'T': 'weights/TorchPretrained/swin_tiny_patch4_window7_224.pth'}
        pretrained_dict = torch.load(pretrained_dict[self.version])
        self.load_state_dict(pretrained_dict['model'], strict=False)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return tuple(outs)


class VGGBackbone(nn.Module):

    def __init__(self, version='vgg11', pretrained=True):
        super(VGGBackbone, self).__init__()
        if version == 'vgg11':
            cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
            dict_url = 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth'
            break_layer = 21
        elif version == 'vgg16':
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
            dict_url = 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
            break_layer = 33
        else:
            raise ValueError
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        self.layer1 = nn.Sequential(*layers[:break_layer])
        self.layer2 = nn.Sequential(*layers[break_layer:])
        if pretrained:
            None
            pretrained_dict = model_zoo.load_url(dict_url)
            pretrained_dict = {k.replace('features.', '', 1): v for k, v in pretrained_dict.items() if 'features' in k}
            self.layer1.load_state_dict({k: v for k, v in pretrained_dict.items() if int(k.split('.')[0]) < break_layer})
            self.layer2.load_state_dict({self._rename(k, break_layer): v for k, v in pretrained_dict.items() if int(k.split('.')[0]) >= break_layer})

    def _rename(self, k, num):
        a = int(k.split('.')[0])
        return k.replace('{}.'.format(a), '{}.'.format(a - num), 1)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return out1, out2


class BasicConv(nn.Module):
    """Basic Convolution Module"""

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, groups: int=1, bias: bool=False, act: bool=True, bn: bool=True) ->None:
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.act = nn.SiLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

    def switch_to_deploy(self) ->None:
        if self.bn is None:
            return
        fusedconv = nn.Conv2d(self.conv.in_channels, self.conv.out_channels, kernel_size=self.conv.kernel_size, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True).requires_grad_(False)
        w_conv = self.conv.weight.clone().view(self.conv.out_channels, -1)
        w_bn = torch.diag(self.bn.weight.div(torch.sqrt(self.bn.eps + self.bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
        b_conv = torch.zeros(self.conv.weight.size(0), device=self.conv.weight.device) if self.conv.bias is None else self.conv.bias
        b_bn = self.bn.bias - self.bn.weight.mul(self.bn.running_mean).div(torch.sqrt(self.bn.running_var + self.bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        self.conv = fusedconv
        self.bn = None


class DepthwiseConv(nn.Module):
    """Depthwise Convolution Module"""

    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int=1, padding: int=0, dilation: int=1, bias: bool=False, act: bool=True, bn: bool=True) ->None:
        super(DepthwiseConv, self).__init__()
        if kernel_size == 1:
            self.dconv = None
        else:
            kernel_size, padding = 5, 2
            self.dconv = BasicConv(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_planes, bias=bias, act=False, bn=True)
        self.pconv = BasicConv(in_planes, out_planes, kernel_size=1, act=act, bn=bn)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        if self.dconv is not None:
            x = self.dconv(x)
        x = self.pconv(x)
        return x


class CEM(nn.Module):
    """Context Enhancement Module"""

    def __init__(self, channels: list, fea_channel: int, conv_block: nn.Module) ->None:
        super(CEM, self).__init__()
        for i, c in enumerate(channels):
            layer_name = f'conv{i + 1}'
            if i == 0:
                layer = conv_block(c, fea_channel, kernel_size=1, act=False)
            else:
                layer = nn.Sequential(conv_block(c, fea_channel, kernel_size=1, act=False), nn.Upsample(scale_factor=2 ** i, mode='bilinear'))
            self.add_module(layer_name, layer)
        layer_name = f'conv{i + 2}'
        layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv_block(channels[-1], fea_channel, kernel_size=1, act=False))
        self.add_module(layer_name, layer)
        self.act = nn.SiLU(inplace=True)

    def forward(self, inputs: list) ->torch.Tensor:
        out = None
        for i, x in enumerate(inputs):
            layer = getattr(self, f'conv{i + 1}')
            x = layer(x)
            out = x if out is None else x + out
        layer = getattr(self, f'conv{i + 2}')
        context = layer(inputs[-1])
        return self.act(out + context)


def fpn_extractor(fpn_level: int, fea_channel: int, conv_block: nn.Module) ->nn.ModuleList:
    layers = []
    for _ in range(fpn_level - 1):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=3, stride=2, padding=1))
    return nn.ModuleList(layers)


class SSDNeck(nn.Module):

    def __init__(self, fpn_level: int, channels: list, fea_channel: int, conv_block: nn.Module) ->None:
        super(SSDNeck, self).__init__()
        self.fpn_level = fpn_level
        self.ft_module = CEM(channels, fea_channel, conv_block)
        self.pyramid_ext = fpn_extractor(self.fpn_level, fea_channel, conv_block)

    def forward(self, x: list) ->list:
        x = self.ft_module(x)
        fpn_fea = [x]
        for v in self.pyramid_ext:
            x = v(x)
            fpn_fea.append(x)
        return fpn_fea


def fpn_convs(fpn_level: int, fea_channel: int, conv_block: nn.Module) ->nn.ModuleList:
    layers = []
    for _ in range(fpn_level):
        layers.append(conv_block(fea_channel, fea_channel, kernel_size=3, stride=1, padding=1))
    return nn.ModuleList(layers)


class FPNNeck(SSDNeck):

    def __init__(self, fpn_level: int, channels: list, fea_channel: int, conv_block: nn.Module) ->None:
        SSDNeck.__init__(self, fpn_level, channels, fea_channel, conv_block)
        self.lateral_convs = fpn_convs(self.fpn_level, fea_channel, conv_block)
        self.fpn_convs = fpn_convs(self.fpn_level, fea_channel, conv_block)

    def forward(self, x: list) ->list:
        fpn_fea = super().forward(x)
        fpn_fea = [lateral_conv(x) for x, lateral_conv in zip(fpn_fea, self.lateral_convs)]
        for i in range(self.fpn_level - 1, 0, -1):
            fpn_fea[i - 1] = fpn_fea[i - 1] + F.interpolate(fpn_fea[i], scale_factor=2.0, mode='bilinear')
        fpn_fea = [fpn_conv(x) for x, fpn_conv in zip(fpn_fea, self.fpn_convs)]
        return fpn_fea


class PAFPNNeck(FPNNeck):

    def __init__(self, fpn_level: int, channels: list, fea_channel: int, conv_block: nn.Module) ->None:
        FPNNeck.__init__(self, fpn_level, channels, fea_channel, conv_block)
        self.downsample_convs = fpn_extractor(self.fpn_level, fea_channel, conv_block)
        self.pafpn_convs = fpn_convs(self.fpn_level, fea_channel, conv_block)

    def forward(self, x: list) ->list:
        fpn_fea = super().forward(x)
        for i in range(0, self.fpn_level - 1):
            fpn_fea[i + 1] = fpn_fea[i + 1] + self.downsample_convs[i](fpn_fea[i])
        fpn_fea = [pafpn_conv(x) for x, pafpn_conv in zip(fpn_fea, self.pafpn_convs)]
        return fpn_fea


def multibox(fpn_level: int, num_anchors: int, num_classes: int, fea_channel: int, dis_channel: int, conv_block: nn.Module) ->tuple:
    loc_layers, conf_layers, dist_layers = list(), list(), list()
    for _ in range(fpn_level):
        loc_layers.append(nn.Sequential(conv_block(fea_channel, fea_channel, 3, padding=1), nn.Conv2d(fea_channel, num_anchors * 4, 1)))
        conf_layers.append(nn.Sequential(conv_block(fea_channel, fea_channel, 3, padding=1), nn.Conv2d(fea_channel, num_anchors * num_classes, 1)))
        dist_layers.append(nn.Sequential(conv_block(fea_channel, fea_channel, 3, padding=1), nn.Conv2d(fea_channel, dis_channel, 1)))
    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers), nn.ModuleList(dist_layers)


class Detector(nn.Module):
    """Student Detector Model"""

    def __init__(self, base_size: int, num_classes: int, backbone: str, neck: str, mode: str) ->None:
        super(Detector, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = 6
        self.mode = mode
        self.fpn_level = 3 if base_size <= 640 else 4
        if 'vgg' in backbone:
            self.backbone = VGGBackbone(version=backbone)
            self.conv_block = BasicConv
        elif 'resnet' in backbone:
            self.backbone = ResNetBackbone(version=backbone)
            self.conv_block = BasicConv
        elif 'repvgg' in backbone:
            self.backbone = REPVGGBackbone(version=backbone)
            self.conv_block = BasicConv
        elif 'cspdarknet' in backbone:
            self.backbone = CSPDarkNetBackbone(version=backbone)
            self.conv_block = BasicConv
        elif 'shufflenet' in backbone:
            self.backbone = ShuffleNetBackbone(version=backbone)
            self.conv_block = DepthwiseConv
        elif 'efficientnet' in backbone:
            self.backbone = EfficientNetBackbone.from_name(backbone)
            self.conv_block = DepthwiseConv
        else:
            raise ValueError('Error: Sorry backbone {} is not supported!'.format(backbone))
        if self.conv_block is BasicConv:
            self.fea_channel = self.dis_channel = 256
        elif self.conv_block is DepthwiseConv:
            self.fea_channel = self.dis_channel = 128
        else:
            raise ValueError('Error: Sorry conv_block {} is not supported!'.format(self.conv_block))
        if neck == 'ssd':
            neck_func = SSDNeck
        elif neck == 'fpn':
            neck_func = FPNNeck
        elif neck == 'pafpn':
            neck_func = PAFPNNeck
        else:
            raise ValueError('Error: Sorry neck {} is not supported!'.format(neck))
        self.neck = neck_func(self.fpn_level, self.backbone.out_channels, self.fea_channel, self.conv_block)
        self.loc, self.conf, self.dist = multibox(self.fpn_level, self.num_anchors, self.num_classes, self.fea_channel, self.dis_channel, self.conv_block)
        bias_value = 0
        for modules in self.loc:
            torch.nn.init.normal_(modules[-1].weight, std=0.01)
            torch.nn.init.constant_(modules[-1].bias, bias_value)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for modules in self.conf:
            torch.nn.init.normal_(modules[-1].weight, std=0.01)
            torch.nn.init.constant_(modules[-1].bias, bias_value)

    def deploy(self) ->None:
        for module in self.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self.eval()

    def _forward_func_tea(self, fp: list) ->dict:
        fea = list()
        loc = list()
        conf = list()
        for x, l, c in zip(fp, self.loc, self.conf):
            fea.append(x.permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        fea = torch.cat([o.view(o.size(0), -1) for o in fea], 1)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return {'loc': loc.view(loc.size(0), -1, 4), 'conf': conf.view(conf.size(0), -1, self.num_classes), 'feature': fea.view(conf.size(0), -1, self.dis_channel)}

    def _forward_func_stu(self, fp: list) ->dict:
        fea = list()
        loc = list()
        conf = list()
        for x, l, c, d in zip(fp, self.loc, self.conf, self.dist):
            fea.append(d(x).permute(0, 2, 3, 1).contiguous())
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        fea = torch.cat([o.view(o.size(0), -1) for o in fea], 1)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return {'loc': loc.view(loc.size(0), -1, 4), 'conf': conf.view(conf.size(0), -1, self.num_classes), 'feature': fea.view(conf.size(0), -1, self.dis_channel)}

    def _forward_func_nor(self, fp: list) ->dict:
        loc = list()
        conf = list()
        for x, l, c in zip(fp, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        return {'loc': loc.view(loc.size(0), -1, 4), 'conf': conf.view(conf.size(0), -1, self.num_classes)}

    def forward(self, x: torch.Tensor) ->dict:
        x = self.backbone(x)
        fp = self.neck(x)
        if self.mode == 'teacher':
            return self._forward_func_tea(fp)
        elif self.mode == 'student':
            return self._forward_func_stu(fp)
        elif self.mode == 'normal':
            return self._forward_func_nor(fp)
        raise NotImplementedError


class BalancedL1Loss(nn.Module):

    def __init__(self, alpha: float=0.5, gamma: float=1.5, beta: float=0.11, loss_weight: float=1.0) ->None:
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor=None) ->torch.Tensor:
        diff = torch.abs(pred - target)
        b = np.e ** (self.gamma / self.alpha) - 1
        loss = torch.where(diff < self.beta, self.alpha / b * (b * diff + 1) * torch.log(b * diff / self.beta + 1) - self.alpha * diff, self.gamma * diff + self.gamma / b - self.alpha * self.beta)
        if weights is None:
            return loss.mean() * self.loss_weight
        else:
            return (loss * weights).sum() * self.loss_weight


class FocalLoss(nn.Module):

    def __init__(self, alpha: float=0.25, gamma: float=2.0, loss_weight: float=2.0) ->None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None) ->torch.Tensor:
        if mask is not None:
            pred, target = pred[mask], target[mask]
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        loss = loss.sum() / (target > 0).float().sum()
        return loss * self.loss_weight


class GFocalLoss(nn.Module):

    def __init__(self, alpha: float=0.75, gamma: float=2.0, loss_weight: float=2.0, epsilon: float=1.0) ->None:
        super(GFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor=None) ->torch.Tensor:
        if mask is not None:
            pred, target = pred[mask], target[mask]
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        focal_weight = target * (target > 0.0).float() + self.alpha * pred_sigmoid.pow(self.gamma) * (target == 0.0).float()
        pt = pred_sigmoid * (target > 0.0).float() + (1 - pred_sigmoid) * (target == 0.0).float()
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight + self.epsilon * (1 - pt).pow(self.gamma + 1)
        loss = loss.sum() / (target > 0.0).float().sum()
        return loss * self.loss_weight


class GIOULoss(nn.Module):

    def __init__(self, loss_weight: float=2.0) ->None:
        super(GIOULoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, weights: torch.Tensor=None) ->torch.Tensor:
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        pred_x2 = torch.max(pred_x1, pred_x2)
        pred_y2 = torch.max(pred_y1, pred_y2)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_x1 = gt_boxes[:, 0]
        target_y1 = gt_boxes[:, 1]
        target_x2 = gt_boxes[:, 2]
        target_y2 = gt_boxes[:, 3]
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        x1_intersect = torch.max(pred_x1, target_x1)
        y1_intersect = torch.max(pred_y1, target_y1)
        x2_intersect = torch.min(pred_x2, target_x2)
        y2_intersect = torch.min(pred_y2, target_y2)
        area_intersect = torch.zeros(pred_x1.size())
        mask = (y2_intersect > y1_intersect) * (x2_intersect > x1_intersect)
        area_intersect[mask] = (x2_intersect[mask] - x1_intersect[mask]) * (y2_intersect[mask] - y1_intersect[mask])
        x1_enclosing = torch.min(pred_x1, target_x1)
        y1_enclosing = torch.min(pred_y1, target_y1)
        x2_enclosing = torch.max(pred_x2, target_x2)
        y2_enclosing = torch.max(pred_y2, target_y2)
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing) + 1e-07
        area_union = pred_area + target_area - area_intersect + 1e-07
        ious = area_intersect / area_union
        gious = ious - (area_enclosing - area_union) / area_enclosing
        loss = 1 - gious
        if weights is None:
            return loss.mean() * self.loss_weight
        else:
            return (loss * weights).sum() * self.loss_weight


class HintLoss(nn.Module):

    def __init__(self, mode: str='pdf', loss_weight: float=5.0) ->None:
        super(HintLoss, self).__init__()
        self.mode = mode
        self.loss_weight = loss_weight
        None

    def forward(self, pred_t: torch.Tensor, pred_s: torch.Tensor) ->torch.Tensor:
        conf_t, fea_t = pred_t['conf'].detach(), pred_t['feature'].detach()
        conf_s, fea_s = pred_s['conf'].detach(), pred_s['feature']
        if self.mode == 'mse':
            return ((fea_s - fea_t) ** 2).mean() * self.loss_weight
        if self.mode == 'pdf':
            with torch.no_grad():
                disagree = (conf_t.sigmoid() - conf_s.sigmoid()) ** 2
                weight = disagree.mean(-1).unsqueeze(1)
                weight = F.avg_pool1d(weight, kernel_size=6, stride=6, padding=0)
                weight = weight.squeeze() / weight.sum()
            return (weight * ((fea_s - fea_t) ** 2).mean(-1)).sum() * self.loss_weight
        raise NotImplementedError


class SIOULoss(nn.Module):

    def __init__(self, loss_weight: float=2.0) ->None:
        super(SIOULoss, self).__init__()
        self.iou_type = 'siou'
        self.loss_weight = loss_weight
        self.eps = 1e-07

    def __call__(self, box1, box2, weights):
        """calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4]."""
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = 4 / math.pi ** 2 * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou
        if weights is None:
            return loss.mean() * self.loss_weight
        else:
            return (loss * weights).sum() * self.loss_weight


def decode(loc: torch.Tensor, priors: torch.Tensor, variances: list=[0.1, 0.2]) ->torch.Tensor:
    """Decode locations from predictions using priors"""
    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def encode(matched: torch.Tensor, priors: torch.Tensor, variances: list=[0.1, 0.2]) ->torch.Tensor:
    """Encode from the priorbox layers to ground truth boxes"""
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= variances[0] * priors[:, 2:]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    targets = torch.cat([g_cxcy, g_wh], 1)
    return targets


def jaccard(box_a: torch.Tensor, box_b: torch.Tensor) ->torch.Tensor:
    """Compute the jaccard overlap of two sets of boxes"""
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def point_form(boxes: torch.Tensor) ->torch.Tensor:
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)"""
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)


@torch.no_grad()
def match(truths: torch.Tensor, labels: torch.Tensor, priors: torch.Tensor, loc_t: torch.Tensor, conf_t: torch.Tensor, overlap_t: torch.Tensor, idx: int) ->None:
    """Match each prior box with the ground truth box"""
    overlaps = jaccard(truths, point_form(priors))
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    best_prior_overlap, best_prior_idx = overlaps.max(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 1)
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    overlap_t[idx] = best_truth_overlap
    conf_t[idx] = labels[best_truth_idx]
    loc_t[idx] = truths[best_truth_idx]


@torch.no_grad()
def mutual_match(truths: torch.Tensor, labels: torch.Tensor, regress: torch.Tensor, classif: torch.Tensor, priors: torch.Tensor, loc_t: torch.Tensor, conf_t: torch.Tensor, overlap_t: torch.Tensor, pred_t: torch.Tensor, idx: int, topk: int=15, sigma: float=2.0) ->None:
    """Classify to regress and regress to classify, Mutual Match for label assignement"""
    qualities = jaccard(truths, decode(regress, priors))
    qualities[qualities != qualities.max(dim=0, keepdim=True)[0]] = 0.0
    for quality in qualities:
        num_pos = max(1, torch.topk(quality, topk, largest=True)[0].sum().int())
        num_pos = min(num_pos, (quality > 0).sum())
        pos_mask = torch.topk(quality, num_pos, largest=True)[1]
        quality[pos_mask] += 3.0
    best_truth_overlap, best_truth_idx = qualities.max(dim=0)
    overlap_t[idx] = best_truth_overlap
    conf_t[idx] = labels[best_truth_idx]
    qualities = (jaccard(truths, point_form(priors)) * torch.exp(classif.sigmoid().t()[labels, :] / sigma)).clamp_(max=1)
    qualities[qualities != qualities.max(dim=0, keepdim=True)[0]] = 0.0
    for quality in qualities:
        num_pos = max(1, torch.topk(quality, topk, largest=True)[0].sum().int())
        num_pos = min(num_pos, (quality > 0).sum())
        pos_mask = torch.topk(quality, num_pos, largest=True)[1]
        quality[pos_mask] += 3.0
    best_truth_overlap, best_truth_idx = qualities.max(dim=0)
    pred_t[idx] = best_truth_overlap
    loc_t[idx] = truths[best_truth_idx]


class MultiBoxLoss(nn.Module):
    """Object Detection Loss"""

    def __init__(self, mutual_guide: bool=True) ->None:
        super(MultiBoxLoss, self).__init__()
        self.mutual_guide = mutual_guide
        self.focal_loss = FocalLoss()
        self.gfocal_loss = GFocalLoss()
        self.l1_loss = BalancedL1Loss()
        self.iou_loss = SIOULoss()

    def forward(self, predictions: dict, priors: torch.Tensor, targets: list) ->tuple:
        loc_p, cls_p = predictions['loc'], predictions['conf']
        num, num_priors, num_classes = cls_p.size()
        if self.mutual_guide:
            loc_t = torch.zeros(num, num_priors, 4)
            cls_t = torch.zeros(num, num_priors).long()
            cls_w = torch.zeros(num, num_priors)
            loc_w = torch.zeros(num, num_priors)
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                regress = loc_p[idx, :, :]
                classif = cls_p[idx, :, :]
                mutual_match(truths, labels, regress, classif, priors, loc_t, cls_t, cls_w, loc_w, idx)
            pos = loc_w >= 3.0
            priors = priors.unsqueeze(0).expand_as(loc_p)
            mask = pos.unsqueeze(-1).expand_as(loc_p)
            weights = (loc_w - 3.0).relu().unsqueeze(-1).expand_as(loc_p)
            weights = weights[mask].view(-1, 4)
            weights = weights / weights.sum()
            loc_p = loc_p[mask].view(-1, 4)
            priors = priors[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            loss_l = self.l1_loss(loc_p, encode(loc_t, priors), weights=weights) + self.iou_loss(decode(loc_p, priors), loc_t, weights=weights.sum(-1))
            cls_t = cls_t + 1
            neg = cls_w <= 1.0
            cls_t[neg] = 0
            cls_t = torch.zeros(num * num_priors, num_classes + 1).scatter_(1, cls_t.view(-1, 1), 1)
            cls_t = cls_t[:, 1:].view(num, num_priors, num_classes)
            cls_w = (cls_w - 3.0).relu().unsqueeze(-1).expand_as(cls_t)
            loss_c = self.gfocal_loss(cls_p, cls_t * cls_w)
            return loss_l + loss_c
        else:
            cls_w = torch.zeros(num, num_priors)
            loc_t = torch.zeros(num, num_priors, 4)
            cls_t = torch.zeros(num, num_priors).long()
            for idx in range(num):
                truths = targets[idx][:, :-1]
                labels = targets[idx][:, -1].long()
                match(truths, labels, priors, loc_t, cls_t, cls_w, idx)
            pos = cls_w >= 0.5
            ign = (cls_w < 0.5) * (cls_w >= 0.4)
            neg = cls_w < 0.4
            priors = priors.unsqueeze(0).expand_as(loc_p)
            mask = pos.unsqueeze(-1).expand_as(loc_p)
            loc_p = loc_p[mask].view(-1, 4)
            loc_t = loc_t[mask].view(-1, 4)
            priors = priors[mask].view(-1, 4)
            loss_l = self.l1_loss(loc_p, encode(loc_t, priors))
            cls_t[neg] = 0
            batch_label = torch.zeros(num * num_priors, num_classes + 1).scatter_(1, cls_t.view(-1, 1), 1)
            batch_label = batch_label[:, 1:].view(num, num_priors, num_classes)
            ign = ign.unsqueeze(-1).expand_as(batch_label)
            batch_label[ign] *= -1
            mask = batch_label >= 0
            loss_c = self.focal_loss(cls_p, batch_label, mask)
            return loss_l + loss_c


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AnyStage,
     lambda: ([], {'width_in': 4, 'width_out': 4, 'stride': 1, 'depth': 1, 'block_constructor': _mock_layer, 'norm_layer': 1, 'activation_layer': 1, 'group_width': 4, 'bottleneck_multiplier': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BalancedL1Loss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BaseConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckTransform,
     lambda: ([], {'width_in': 4, 'width_out': 4, 'stride': 1, 'norm_layer': _mock_layer, 'activation_layer': _mock_layer, 'group_width': 4, 'bottleneck_multiplier': 4, 'se_ratio': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dDynamicSamePadding,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv2dNormActivation,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBnAct,
     lambda: ([], {'in_chs': 4, 'out_chs': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvNormActivation,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthwiseSeparableConv,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EdgeResidual,
     lambda: ([], {'in_chs': 4, 'out_chs': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Focus,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Fused_IRB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (GFocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GIOULoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InvertedResidual,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PatchEmbed,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Prologue,
     lambda: ([], {'num_in_channels': 4, 'num_out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RegNetBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResBottleneckBlock,
     lambda: ([], {'width_in': 4, 'width_out': 4, 'stride': 1, 'norm_layer': _mock_layer, 'activation_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleNetBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (SimpleStemIN,
     lambda: ([], {'width_in': 4, 'width_out': 4, 'norm_layer': _mock_layer, 'activation_layer': _mock_layer}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SqueezeExcitation,
     lambda: ([], {'input_channels': 4, 'squeeze_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGGBackbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_zhanghengdev_MutualGuide(_paritybench_base):
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

