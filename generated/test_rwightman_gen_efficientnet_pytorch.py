import sys
_module = sys.modules[__name__]
del sys
caffe2_benchmark = _module
caffe2_validate = _module
data = _module
dataset = _module
loader = _module
tf_preprocessing = _module
transforms = _module
geffnet = _module
activations = _module
activations = _module
activations_autofn = _module
activations_jit = _module
config = _module
conv2d_layers = _module
efficientnet_builder = _module
gen_efficientnet = _module
helpers = _module
mobilenetv3 = _module
model_factory = _module
version = _module
hubconf = _module
onnx_export = _module
onnx_optimize = _module
onnx_to_caffe = _module
setup = _module
utils = _module
validate = _module

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


from torch import nn as nn


from torch.nn import functional as F


import torch


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


import numpy as np


import math


import re


from copy import deepcopy


import time


import torch.nn.parallel


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


def mish(x, inplace: bool=False):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    """
    return x.mul(F.softplus(x).tanh())


class Mish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return mish(x, self.inplace)


class Sigmoid(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.sigmoid_() if self.inplace else x.sigmoid()


class Tanh(nn.Module):

    def __init__(self, inplace: bool=False):
        super(Tanh, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.tanh_() if self.inplace else x.tanh()


def hard_swish(x, inplace: bool=False):
    inner = F.relu6(x + 3.0).div_(6.0)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):

    def __init__(self, inplace: bool=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)


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


class SwishAutoFn(torch.autograd.Function):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    Memory efficient variant from:
     https://medium.com/the-artificial-impostor/more-memory-efficient-swish-activation-function-e07c22c12a76
    """

    @staticmethod
    def forward(ctx, x):
        result = x.mul(torch.sigmoid(x))
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        return grad_output.mul(x_sigmoid * (1 + x * (1 - x_sigmoid)))


class SwishAuto(nn.Module):

    def __init__(self, inplace: bool=False):
        super(SwishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishAutoFn.apply(x)


class MishAutoFn(torch.autograd.Function):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    Experimental memory-efficient variant
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.mul(torch.tanh(F.softplus(x)))
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_sigmoid = torch.sigmoid(x)
        x_tanh_sp = F.softplus(x).tanh()
        return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp *
            x_tanh_sp))


class MishAuto(nn.Module):

    def __init__(self, inplace: bool=False):
        super(MishAuto, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishAutoFn.apply(x)


@torch.jit.script
def swish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    return grad_output * (x_sigmoid * (1 + x * (1 - x_sigmoid)))


@torch.jit.script
def swish_jit_fwd(x):
    return x.mul(torch.sigmoid(x))


class SwishJitAutoFn(torch.autograd.Function):
    """ torch.jit.script optimised Swish
    Inspired by conversation btw Jeremy Howard & Adam Pazske
    https://twitter.com/jeremyphoward/status/1188251041835315200
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return swish_jit_bwd(x, grad_output)


class SwishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(SwishJit, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return SwishJitAutoFn.apply(x)


@torch.jit.script
def mish_jit_bwd(x, grad_output):
    x_sigmoid = torch.sigmoid(x)
    x_tanh_sp = F.softplus(x).tanh()
    return grad_output.mul(x_tanh_sp + x * x_sigmoid * (1 - x_tanh_sp *
        x_tanh_sp))


@torch.jit.script
def mish_jit_fwd(x):
    return x.mul(torch.tanh(F.softplus(x)))


class MishJitAutoFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return mish_jit_fwd(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return mish_jit_bwd(x, grad_output)


class MishJit(nn.Module):

    def __init__(self, inplace: bool=False):
        super(MishJit, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return MishJitAutoFn.apply(x)


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


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels,
            kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def _same_pad_arg(input_size, kernel_size, stride, dilation):
    ih, iw = input_size
    kh, kw = kernel_size
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    return [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]


class Conv2dSameExport(nn.Conv2d):
    """ ONNX export friendly Tensorflow like 'SAME' convolution wrapper for 2D convolutions

    NOTE: This does not currently work with torch.jit.script
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSameExport, self).__init__(in_channels, out_channels,
            kernel_size, stride, 0, dilation, groups, bias)
        self.pad = None
        self.pad_input_size = 0, 0

    def forward(self, x):
        input_size = x.size()[-2:]
        if self.pad is None:
            pad_arg = _same_pad_arg(input_size, self.weight.size()[-2:],
                self.stride, self.dilation)
            self.pad = nn.ZeroPad2d(pad_arg)
            self.pad_input_size = input_size
        else:
            assert self.pad_input_size == input_size
        x = self.pad(x)
        return F.conv2d(x, self.weight, self.bias, self.stride, self.
            padding, self.dilation, self.groups)


def _split_channels(num_chan, num_groups):
    split = [(num_chan // num_groups) for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = (stride - 1 + dilation * (kernel_size - 1)) // 2
    return padding


def _is_static_pad(kernel_size, stride=1, dilation=1, **_):
    return stride == 1 and dilation * (kernel_size - 1) % 2 == 0


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


_EXPORTABLE = False


def is_exportable():
    return _EXPORTABLE


_SCRIPTABLE = False


def is_scriptable():
    return _SCRIPTABLE


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        if is_exportable():
            assert not is_scriptable()
            return Conv2dSameExport(in_chs, out_chs, kernel_size, **kwargs)
        else:
            return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **
            kwargs)


class MixedConv2d(nn.ModuleDict):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding='', dilation=1, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [
            kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits,
            out_splits)):
            conv_groups = out_ch if depthwise else 1
            self.add_module(str(idx), create_conv2d_pad(in_ch, out_ch, k,
                stride=stride, padding=padding, dilation=dilation, groups=
                conv_groups, **kwargs))
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [conv(x_split[i]) for i, conv in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


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
        init_weight = get_condconv_initializer(partial(nn.init.
            kaiming_uniform_, a=math.sqrt(5)), self.num_experts, self.
            weight_shape)
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(partial(nn.init.uniform_,
                a=-bound, b=bound), self.num_experts, self.bias_shape)
            init_bias(self.bias)

    def forward(self, x, routing_weights):
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


def make_divisible(v: int, divisor: int=8, min_value: int=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def sigmoid(x, inplace: bool=False):
    return x.sigmoid_() if inplace else x.sigmoid()


class SqueezeExcite(nn.Module):

    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
        act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) *
            se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    assert 'groups' not in kwargs
    if isinstance(kernel_size, list):
        assert 'num_experts' not in kwargs
        m = MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else 1
        if 'num_experts' in kwargs and kwargs['num_experts'] > 0:
            m = CondConv2d(in_chs, out_chs, kernel_size, groups=groups, **
                kwargs)
        else:
            m = create_conv2d_pad(in_chs, out_chs, kernel_size, groups=
                groups, **kwargs)
    return m


class ConvBnAct(nn.Module):

    def __init__(self, in_chs, out_chs, kernel_size, stride=1, pad_type='',
        act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        assert stride in [1, 2]
        norm_kwargs = norm_kwargs or {}
        self.conv = select_conv2d(in_chs, out_chs, kernel_size, stride=
            stride, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


def drop_connect(inputs, training: bool=False, drop_connect_rate: float=0.0):
    """Apply drop connect."""
    if not training:
        return inputs
    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand((inputs.size()[0], 1, 1, 1),
        dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()
    output = inputs.div(keep_prob) * random_tensor
    return output


_SE_ARGS_DEFAULT = dict(gate_fn=sigmoid, act_layer=None, reduce_mid=False,
    divisor=1)


def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with optional first pw conv.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1,
        pad_type='', act_layer=nn.ReLU, noskip=False, pw_kernel_size=1,
        pw_act=False, se_ratio=0.0, se_kwargs=None, norm_layer=nn.
        BatchNorm2d, norm_kwargs=None, drop_connect_rate=0.0):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        norm_kwargs = norm_kwargs or {}
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.drop_connect_rate = drop_connect_rate
        self.conv_dw = select_conv2d(in_chs, in_chs, dw_kernel_size, stride
            =stride, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        if se_ratio is not None and se_ratio > 0.0:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = nn.Identity()
        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size,
            padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True) if pw_act else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1,
        pad_type='', act_layer=nn.ReLU, noskip=False, exp_ratio=1.0,
        exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.0, se_kwargs=None,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, conv_kwargs=None,
        drop_connect_rate=0.0):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs: int = make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_connect_rate = drop_connect_rate
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size,
            padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self.conv_dw = select_conv2d(mid_chs, mid_chs, dw_kernel_size,
            stride=stride, padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        if se_ratio is not None and se_ratio > 0.0:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = nn.Identity()
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size,
            padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def forward(self, x):
        residual = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class EdgeResidual(nn.Module):
    """ EdgeTPU Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, exp_ratio=1.0,
        fake_in_chs=0, stride=1, pad_type='', act_layer=nn.ReLU, noskip=
        False, pw_kernel_size=1, se_ratio=0.0, se_kwargs=None, norm_layer=
        nn.BatchNorm2d, norm_kwargs=None, drop_connect_rate=0.0):
        super(EdgeResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        mid_chs = make_divisible(fake_in_chs * exp_ratio
            ) if fake_in_chs > 0 else make_divisible(in_chs * exp_ratio)
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_connect_rate = drop_connect_rate
        self.conv_exp = select_conv2d(in_chs, mid_chs, exp_kernel_size,
            padding=pad_type)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        if se_ratio is not None and se_ratio > 0.0:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)
        else:
            self.se = nn.Identity()
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size,
            stride=stride, padding=pad_type)
        self.bn2 = nn.BatchNorm2d(out_chs, **norm_kwargs)

    def forward(self, x):
        residual = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3, stride=1,
        pad_type='', act_layer=nn.ReLU, noskip=False, exp_ratio=1.0,
        exp_kernel_size=1, pw_kernel_size=1, se_ratio=0.0, se_kwargs=None,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, num_experts=0,
        drop_connect_rate=0.0):
        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)
        super(CondConvResidual, self).__init__(in_chs, out_chs,
            dw_kernel_size=dw_kernel_size, stride=stride, pad_type=pad_type,
            act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio,
            exp_kernel_size=exp_kernel_size, pw_kernel_size=pw_kernel_size,
            se_ratio=se_ratio, se_kwargs=se_kwargs, norm_layer=norm_layer,
            norm_kwargs=norm_kwargs, conv_kwargs=conv_kwargs,
            drop_connect_rate=drop_connect_rate)
        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        residual = x
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)
        if self.has_residual:
            if self.drop_connect_rate > 0.0:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)


class EfficientNetBuilder:
    """ Build Trunk Blocks for Efficient/Mobile Networks

    This ended up being somewhat of a cross between
    https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mnasnet_models.py
    and
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/backbone/fbnet_builder.py

    """

    def __init__(self, channel_multiplier=1.0, channel_divisor=8,
        channel_min=None, pad_type='', act_layer=None, se_kwargs=None,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_connect_rate=0.0):
        self.channel_multiplier = channel_multiplier
        self.channel_divisor = channel_divisor
        self.channel_min = channel_min
        self.pad_type = pad_type
        self.act_layer = act_layer
        self.se_kwargs = se_kwargs
        self.norm_layer = norm_layer
        self.norm_kwargs = norm_kwargs
        self.drop_connect_rate = drop_connect_rate
        self.in_chs = None
        self.block_idx = 0
        self.block_count = 0

    def _round_channels(self, chs):
        return round_channels(chs, self.channel_multiplier, self.
            channel_divisor, self.channel_min)

    def _make_block(self, ba):
        bt = ba.pop('block_type')
        ba['in_chs'] = self.in_chs
        ba['out_chs'] = self._round_channels(ba['out_chs'])
        if 'fake_in_chs' in ba and ba['fake_in_chs']:
            ba['fake_in_chs'] = self._round_channels(ba['fake_in_chs'])
        ba['norm_layer'] = self.norm_layer
        ba['norm_kwargs'] = self.norm_kwargs
        ba['pad_type'] = self.pad_type
        ba['act_layer'] = ba['act_layer'] if ba['act_layer'
            ] is not None else self.act_layer
        assert ba['act_layer'] is not None
        if bt == 'ir':
            ba['drop_connect_rate'
                ] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_kwargs'] = self.se_kwargs
            if ba.get('num_experts', 0) > 0:
                block = CondConvResidual(**ba)
            else:
                block = InvertedResidual(**ba)
        elif bt == 'ds' or bt == 'dsa':
            ba['drop_connect_rate'
                ] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_kwargs'] = self.se_kwargs
            block = DepthwiseSeparableConv(**ba)
        elif bt == 'er':
            ba['drop_connect_rate'
                ] = self.drop_connect_rate * self.block_idx / self.block_count
            ba['se_kwargs'] = self.se_kwargs
            block = EdgeResidual(**ba)
        elif bt == 'cn':
            block = ConvBnAct(**ba)
        else:
            assert False, 'Uknkown block type (%s) while building model.' % bt
        self.in_chs = ba['out_chs']
        return block

    def _make_stack(self, stack_args):
        blocks = []
        for i, ba in enumerate(stack_args):
            if i >= 1:
                ba['stride'] = 1
            block = self._make_block(ba)
            blocks.append(block)
            self.block_idx += 1
        return nn.Sequential(*blocks)

    def __call__(self, in_chs, block_args):
        """ Build the blocks
        Args:
            in_chs: Number of input-channels passed to first block
            block_args: A list of lists, outer list defines stages, inner
                list contains strings defining block configuration(s)
        Return:
             List of block stacks (each stack wrapped in nn.Sequential)
        """
        self.in_chs = in_chs
        self.block_count = sum([len(x) for x in block_args])
        self.block_idx = 0
        blocks = []
        for stack_idx, stack in enumerate(block_args):
            assert isinstance(stack, list)
            stack = self._make_stack(stack)
            blocks.append(stack)
        return blocks


def initialize_weight_default(m, n=''):
    if isinstance(m, CondConv2d):
        init_fn = get_condconv_initializer(partial(nn.init.kaiming_normal_,
            mode='fan_out', nonlinearity='relu'), m.num_experts, m.weight_shape
            )
        init_fn(m.weight)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='linear'
            )


def initialize_weight_goog(m, n='', fix_group_fanout=True):
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(lambda w: w.data.normal_(
            0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()


class GenEfficientNet(nn.Module):
    """ Generic EfficientNets

    An implementation of mobile optimized networks that covers:
      * EfficientNet (B0-B8, L2, CondConv, EdgeTPU)
      * MixNet (Small, Medium, and Large, XL)
      * MNASNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3,
        num_features=1280, stem_size=32, fix_stem=False, channel_multiplier
        =1.0, channel_divisor=8, channel_min=None, pad_type='', act_layer=
        nn.ReLU, drop_rate=0.0, drop_connect_rate=0.0, se_kwargs=None,
        norm_layer=nn.BatchNorm2d, norm_kwargs=None, weight_init='goog'):
        super(GenEfficientNet, self).__init__()
        self.drop_rate = drop_rate
        if not fix_stem:
            stem_size = round_channels(stem_size, channel_multiplier,
                channel_divisor, channel_min)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2,
            padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        in_chs = stem_size
        builder = EfficientNetBuilder(channel_multiplier, channel_divisor,
            channel_min, pad_type, act_layer, se_kwargs, norm_layer,
            norm_kwargs, drop_connect_rate)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs
        self.conv_head = select_conv2d(in_chs, num_features, 1, padding=
            pad_type)
        self.bn2 = norm_layer(num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(num_features, num_classes)
        for n, m in self.named_modules():
            if weight_init == 'goog':
                initialize_weight_goog(m, n)
            else:
                initialize_weight_default(m, n)

    def features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.
            global_pool, nn.Flatten(), nn.Dropout(self.drop_rate), self.
            classifier])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


class MobileNetV3(nn.Module):
    """ MobileNet-V3

    A this model utilizes the MobileNet-v3 specific 'efficient head', where global pooling is done before the
    head convolution without a final batch-norm layer before the classifier.

    Paper: https://arxiv.org/abs/1905.02244
    """

    def __init__(self, block_args, num_classes=1000, in_chans=3, stem_size=
        16, num_features=1280, head_bias=True, channel_multiplier=1.0,
        pad_type='', act_layer=HardSwish, drop_rate=0.0, drop_connect_rate=
        0.0, se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
        weight_init='goog'):
        super(MobileNetV3, self).__init__()
        self.drop_rate = drop_rate
        stem_size = round_channels(stem_size, channel_multiplier)
        self.conv_stem = select_conv2d(in_chans, stem_size, 3, stride=2,
            padding=pad_type)
        self.bn1 = nn.BatchNorm2d(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        in_chs = stem_size
        builder = EfficientNetBuilder(channel_multiplier, pad_type=pad_type,
            act_layer=act_layer, se_kwargs=se_kwargs, norm_layer=norm_layer,
            norm_kwargs=norm_kwargs, drop_connect_rate=drop_connect_rate)
        self.blocks = nn.Sequential(*builder(in_chs, block_args))
        in_chs = builder.in_chs
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = select_conv2d(in_chs, num_features, 1, padding=
            pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.classifier = nn.Linear(num_features, num_classes)
        for m in self.modules():
            if weight_init == 'goog':
                initialize_weight_goog(m)
            else:
                initialize_weight_default(m)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2, nn.
            Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rwightman_gen_efficientnet_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(CondConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {})

    def test_001(self):
        self._check(Conv2dSame(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Conv2dSameExport(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ConvBnAct(*[], **{'in_chs': 4, 'out_chs': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(DepthwiseSeparableConv(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(EdgeResidual(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(HardSigmoid(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(HardSwish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(InvertedResidual(*[], **{'in_chs': 4, 'out_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_009(self):
        self._check(Mish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(MishAuto(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_011(self):
        self._check(MishJit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(MixedConv2d(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_013(self):
        self._check(Sigmoid(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_014(self):
        self._check(SqueezeExcite(*[], **{'in_chs': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_015(self):
        self._check(Swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_016(self):
        self._check(SwishAuto(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_017(self):
        self._check(SwishJit(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_018(self):
        self._check(Tanh(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

