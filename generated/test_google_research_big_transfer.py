import sys
_module = sys.modules[__name__]
del sys
master = _module
bit_common = _module
bit_hyperrule = _module
bit_jax = _module
models = _module
tf2jax = _module
train = _module
bit_pytorch = _module
fewshot = _module
lbtoolbox = _module
models = _module
train = _module
bit_tf2 = _module
normalization = _module
input_pipeline_tf2_or_jax = _module

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


from collections import OrderedDict


import torch


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


def standardize(x, axis, eps):
    x = x - jnp.mean(x, axis=axis, keepdims=True)
    x = x / jnp.sqrt(jnp.mean(jnp.square(x), axis=axis, keepdims=True) + eps)
    return x


class GroupNorm(nn.Module):
    """Group normalization (arxiv.org/abs/1803.08494)."""

    def apply(self, x, num_groups=32):
        input_shape = x.shape
        group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)
        x = x.reshape(group_shape)
        x = standardize(x, axis=[1, 2, 4], eps=1e-05)
        x = x.reshape(input_shape)
        bias_scale_shape = tuple([1, 1, 1] + [input_shape[-1]])
        x = x * self.param('scale', bias_scale_shape, nn.initializers.ones)
        x = x + self.param('bias', bias_scale_shape, nn.initializers.zeros)
        return x


def fixed_padding(x, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    x = jax.lax.pad(x, 0.0, ((0, 0, 0), (pad_beg, pad_end, 0), (pad_beg,
        pad_end, 0), (0, 0, 0)))
    return x


class ResidualUnit(nn.Module):
    """Bottleneck ResNet block."""

    def apply(self, x, nout, strides=(1, 1)):
        x_shortcut = x
        needs_projection = x.shape[-1] != nout * 4 or strides != (1, 1)
        group_norm = GroupNorm
        conv = StdConv.partial(bias=False)
        x = group_norm(x, name='gn1')
        x = nn.relu(x)
        if needs_projection:
            x_shortcut = conv(x, nout * 4, (1, 1), strides, name='conv_proj')
        x = conv(x, nout, (1, 1), name='conv1')
        x = group_norm(x, name='gn2')
        x = nn.relu(x)
        x = fixed_padding(x, 3)
        x = conv(x, nout, (3, 3), strides, name='conv2', padding='VALID')
        x = group_norm(x, name='gn3')
        x = nn.relu(x)
        x = conv(x, nout * 4, (1, 1), name='conv3')
        return x + x_shortcut


class ResidualBlock(nn.Module):

    def apply(self, x, block_size, nout, first_stride):
        x = ResidualUnit(x, nout, strides=first_stride, name='unit01')
        for i in range(1, block_size):
            x = ResidualUnit(x, nout, strides=(1, 1), name=f'unit{i + 1:02d}')
        return x


_block_sizes = {(50): [3, 4, 6, 3], (101): [3, 4, 23, 3], (152): [3, 8, 36, 3]}


class ResNet(nn.Module):
    """ResNetV2."""

    def apply(self, x, num_classes=1000, width_factor=1, num_layers=50):
        block_sizes = _block_sizes[num_layers]
        width = 64 * width_factor
        root_block = RootBlock.partial(width=width)
        x = root_block(x, name='root_block')
        for i, block_size in enumerate(block_sizes):
            x = ResidualBlock(x, block_size, width * 2 ** i, first_stride=(
                1, 1) if i == 0 else (2, 2), name=f'block{i + 1}')
        x = GroupNorm(x, name='norm-pre-head')
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(x, num_classes, name='conv_head', kernel_init=nn.
            initializers.zeros)
        return x.astype(jnp.float32)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.
            dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1,
        bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0,
        bias=bias)


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW."""
    if conv_weights.ndim == 4:
        conv_weights = conv_weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

  Follows the implementation of "Identity Mappings in Deep Residual Networks":
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

  Except it puts the stride on 3x3 conv when available.
  """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        out = self.relu(self.gn1(x))
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))
        return out + residual

    def load_from(self, weights, prefix=''):
        convname = 'standardized_conv2d'
        with torch.no_grad():
            self.conv1.weight.copy_(tf2th(weights[
                f'{prefix}a/{convname}/kernel']))
            self.conv2.weight.copy_(tf2th(weights[
                f'{prefix}b/{convname}/kernel']))
            self.conv3.weight.copy_(tf2th(weights[
                f'{prefix}c/{convname}/kernel']))
            self.gn1.weight.copy_(tf2th(weights[f'{prefix}a/group_norm/gamma'])
                )
            self.gn2.weight.copy_(tf2th(weights[f'{prefix}b/group_norm/gamma'])
                )
            self.gn3.weight.copy_(tf2th(weights[f'{prefix}c/group_norm/gamma'])
                )
            self.gn1.bias.copy_(tf2th(weights[f'{prefix}a/group_norm/beta']))
            self.gn2.bias.copy_(tf2th(weights[f'{prefix}b/group_norm/beta']))
            self.gn3.bias.copy_(tf2th(weights[f'{prefix}c/group_norm/beta']))
            if hasattr(self, 'downsample'):
                w = weights[f'{prefix}a/proj/{convname}/kernel']
                self.downsample.weight.copy_(tf2th(w))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, head_size=21843,
        zero_head=False):
        super().__init__()
        wf = width_factor
        self.root = nn.Sequential(OrderedDict([('conv', StdConv2d(3, 64 *
            wf, kernel_size=7, stride=2, padding=3, bias=False)), ('pad',
            nn.ConstantPad2d(1, 0)), ('pool', nn.MaxPool2d(kernel_size=3,
            stride=2, padding=0))]))
        self.body = nn.Sequential(OrderedDict([('block1', nn.Sequential(
            OrderedDict([('unit01', PreActBottleneck(cin=64 * wf, cout=256 *
            wf, cmid=64 * wf))] + [(f'unit{i:02d}', PreActBottleneck(cin=
            256 * wf, cout=256 * wf, cmid=64 * wf)) for i in range(2, 
            block_units[0] + 1)]))), ('block2', nn.Sequential(OrderedDict([
            ('unit01', PreActBottleneck(cin=256 * wf, cout=512 * wf, cmid=
            128 * wf, stride=2))] + [(f'unit{i:02d}', PreActBottleneck(cin=
            512 * wf, cout=512 * wf, cmid=128 * wf)) for i in range(2, 
            block_units[1] + 1)]))), ('block3', nn.Sequential(OrderedDict([
            ('unit01', PreActBottleneck(cin=512 * wf, cout=1024 * wf, cmid=
            256 * wf, stride=2))] + [(f'unit{i:02d}', PreActBottleneck(cin=
            1024 * wf, cout=1024 * wf, cmid=256 * wf)) for i in range(2, 
            block_units[2] + 1)]))), ('block4', nn.Sequential(OrderedDict([
            ('unit01', PreActBottleneck(cin=1024 * wf, cout=2048 * wf, cmid
            =512 * wf, stride=2))] + [(f'unit{i:02d}', PreActBottleneck(cin
            =2048 * wf, cout=2048 * wf, cmid=512 * wf)) for i in range(2, 
            block_units[3] + 1)])))]))
        self.zero_head = zero_head
        self.head = nn.Sequential(OrderedDict([('gn', nn.GroupNorm(32, 2048 *
            wf)), ('relu', nn.ReLU(inplace=True)), ('avg', nn.
            AdaptiveAvgPool2d(output_size=1)), ('conv', nn.Conv2d(2048 * wf,
            head_size, kernel_size=1, bias=True))]))

    def forward(self, x):
        x = self.head(self.body(self.root(x)))
        assert x.shape[-2:] == (1, 1)
        return x[..., 0, 0]

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(tf2th(weights[
                f'{prefix}root_block/standardized_conv2d/kernel']))
            self.head.gn.weight.copy_(tf2th(weights[
                f'{prefix}group_norm/gamma']))
            self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(tf2th(weights[
                    f'{prefix}head/conv2d/kernel']))
                self.head.conv.bias.copy_(tf2th(weights[
                    f'{prefix}head/conv2d/bias']))
            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_google_research_big_transfer(_paritybench_base):
    pass

    def test_000(self):
        self._check(StdConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})
