import sys
_module = sys.modules[__name__]
del sys
profiler = _module
tests = _module
conftest = _module
custom_attention = _module
dense_net = _module
ldc = _module
models = _module
u_net = _module
test_exception = _module
test_meta_tensor = _module
test_ordered_set = _module
test_torchview = _module
test_torchview_text = _module
test_torchview_vision = _module
test_transformers = _module
torchview = _module
computation_graph = _module
computation_node = _module
base_node = _module
compute_node = _module
recorder_tensor = _module
torchview = _module
utils = _module

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


from torch import nn


import torch.nn.functional as F


import torch


import math


import torch.nn as nn


import torch.utils.checkpoint as cp


from typing import Any


from torch.nn import functional as F


from typing import Callable


import torchtext


import torchvision


from torch import __version__ as torch_version


from torchtext import __version__ as torchtext_version


from torchvision import __version__ as torchvision_version


from typing import Union


from collections import Counter


from torch.nn.modules import Identity


from typing import Tuple


from collections.abc import Callable


from typing import Iterable


from typing import Mapping


from typing import TypeVar


import warnings


from typing import Sequence


from typing import Optional


from typing import Iterator


from typing import List


from torch.jit import ScriptModule


from collections.abc import Iterable


from typing import MutableSet


from typing import Generator


from torch.nn.parameter import Parameter


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(c_fc=nn.Linear(config.n_embd, 4 * config.n_embd), c_proj=nn.Linear(4 * config.n_embd, config.n_embd), act=nn.ReLU(inplace=True), dropout=nn.Dropout(config.resid_pdrop)))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


def _bn_function_factory(norm, relu, conv):

    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


class BottleneckUnit(nn.Module):

    def __init__(self, in_channel, growth_rate, expansion, p_dropout, activation, efficient=False):
        super(BottleneckUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channel, expansion * growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * growth_rate)
        self.conv2 = nn.Conv2d(expansion * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.p_dropout = p_dropout
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.bn1, self.activation, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.activation(self.bn2(bottleneck_output)))
        if self.p_dropout > 0:
            new_features = F.dropout(new_features, p=self.p_dropout, training=self.training)
        return new_features


class _InitBlock(nn.Module):

    def __init__(self, in_channel, out_channel, small_inputs, activation):
        super(_InitBlock, self).__init__()
        self.activation = activation
        self.small_inputs = small_inputs
        if small_inputs:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn = nn.BatchNorm2d(out_channel)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

    def forward(self, x):
        out = self.conv(x)
        if not self.small_inputs:
            out = self.pool(self.activation(self.bn(out)))
        return out


class _TransitionBlock(nn.Module):

    def __init__(self, in_channel, out_channel, activation):
        super(_TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.activation = activation
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.activation(self.bn(x))
        out = self.pool(self.conv(out))
        return out


class _DenseLayer(nn.Sequential):

    def __init__(self, input_features: int, out_features: int) ->None:
        super().__init__()
        self.add_module('conv1', nn.Conv2d(input_features, out_features, kernel_size=3, stride=1, padding=2, bias=True))
        self.add_module('norm1', nn.BatchNorm2d(out_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, bias=True))
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) ->tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = inputs
        new_features = super().forward(F.relu(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers: int, input_features: int, out_features: int) ->None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module(f'denselayer{i + 1}', layer)
            input_features = out_features


class CfgNode:
    """ a lightweight configuration class inspired by yacs """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append('%s:\n' % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append('%s: %s\n' % (k, v))
        parts = [(' ' * (indent * 4) + p) for p in parts]
        return ''.join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return {k: (v.to_dict() if isinstance(v, CfgNode) else v) for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        n_channel (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        n_class (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32.
        Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing.
        Much more memory efficient, but slower.
    """

    @staticmethod
    def get_activation(name: str):
        activation_map = {'relu': F.relu, 'elu': F.elu, 'gelu': F.gelu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}
        return activation_map[name]

    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.model_type = 'densenet'
        C.growth_rate = 24
        C.n_channel = C.growth_rate * 2
        C.n_channel = 32
        C.img_channel = 3
        C.activation = 'relu'
        C.p_dropout = 0.16
        C.compression = 0.5
        C.expansion = 4
        C.fc_pdrop = 0.16
        C.small_inputs = True
        C.efficient = True
        C.n_class = 10
        C.n_blocks = [5, 5, 5]
        return C

    def __init__(self, config: CfgNode, unit_module=BottleneckUnit):
        self.activation = F.relu if config.activation is None else self.get_activation(config.activation)
        super(DenseNet, self).__init__()
        assert 0 < config.compression <= 1, 'compression of densenet should be between 0 and 1'
        self.features = nn.Sequential()
        self.features.add_module('init_block', _InitBlock(config.img_channel, config.n_channel, config.small_inputs, self.activation))
        n_feature = config.n_channel
        for i, n_unit in enumerate(config.n_blocks):
            block = _DenseBlock(dense_unit=unit_module, n_unit=n_unit, in_channel=n_feature, expansion=config.expansion, growth_rate=config.growth_rate, p_dropout=config.p_dropout, efficient=config.efficient, activation=self.activation)
            self.features.add_module('dense_block_%d' % (i + 1), block)
            n_feature = n_feature + n_unit * config.growth_rate
            if i != len(config.n_blocks) - 1:
                trans = _TransitionBlock(in_channel=n_feature, out_channel=int(n_feature * config.compression), activation=self.activation)
                self.features.add_module('transition_%d' % (i + 1), trans)
                n_feature = int(n_feature * config.compression)
        self.bn_final = nn.BatchNorm2d(n_feature)
        self.fc = nn.Linear(n_feature, config.n_class)
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2.0 / n))
            elif 'bn' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'bn' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'fc' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        out = self.features(x)
        out = self.activation(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class CoFusion(nn.Module):

    def __init__(self, in_ch: int, out_ch: int) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32)

    def forward(self, x: torch.Tensor) ->Any:
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return (x * attn).sum(1).unsqueeze(1)


class UpConvBlock(nn.Module):

    def __init__(self, in_features: int, up_scale: int) ->None:
        super().__init__()
        self.up_factor = 2
        self.constant_features = 16
        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features: int, up_scale: int) ->list[nn.Module]:
        layers: list[nn.Module] = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx: int, up_scale: int) ->int:
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x: torch.Tensor) ->Any:
        return self.features(x)


class SingleConvBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, stride: int, use_bs: bool=True) ->None:
        super().__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class LDC(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self) ->None:
        super().__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(2, 32, 64)
        self.dblock_4 = _DenseBlock(3, 64, 96)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.side_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_3 = SingleConvBlock(32, 64, 1)
        self.pre_dense_4 = SingleConvBlock(64, 96, 1)
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(64, 2)
        self.up_block_4 = UpConvBlock(96, 3)
        self.block_cat = CoFusion(4, 4)

    def forward(self, x: torch.Tensor) ->list[torch.Tensor]:
        assert x.ndim == 4, x.shape
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down + block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        results: list[torch.Tensor] = [out_1, out_2, out_3, out_4]
        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)
        results.append(block_cat)
        return results


class IdentityModel(nn.Module):
    """Identity Model."""

    def __init__(self) ->None:
        super().__init__()
        self.identity = nn.Identity()

    def forward(self, x: Any) ->Any:
        return self.identity(x)


class SingleInputNet(nn.Module):
    """Simple CNN model."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class MultipleInputNetDifferentDtypes(nn.Module):
    """Model with multiple inputs containing different dtypes."""

    def __init__(self) ->None:
        super().__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)
        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) ->torch.Tensor:
        x1 = F.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = x2.type(torch.float)
        x2 = F.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = torch.cat((x1, x2), 0)
        return F.log_softmax(x, dim=1)


class ScalarNet(nn.Module):
    """Model that takes a scalar as a parameter."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.identity = IdentityModel()

    def forward(self, x: torch.Tensor, scalar: float) ->torch.Tensor:
        out = x
        scalar = self.identity(scalar)
        if scalar == 5:
            out = self.conv1(out)
        else:
            out = self.conv2(out)
        return out


class EdgeCaseModel(nn.Module):
    """Model that throws an exception when used."""

    def __init__(self, throw_error: bool=False, return_str: bool=False, return_class: bool=False) ->None:
        super().__init__()
        self.throw_error = throw_error
        self.return_str = return_str
        self.return_class = return_class
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.model = IdentityModel()

    def forward(self, x: torch.Tensor) ->Any:
        x = self.conv1(x)
        x = self.model('string output' if self.return_str else x)
        if self.throw_error:
            x = self.conv1(x)
        if self.return_class:
            x = self.model(EdgeCaseModel)
        return x


class MLP(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool=True) ->None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace), nn.Linear(64, 32), nn.ReLU(inplace), nn.Linear(32, 16), nn.ReLU(inplace), nn.Linear(16, 8), nn.ReLU(inplace), nn.Linear(8, 4), nn.ReLU(inplace), nn.Linear(4, 2))

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.layers(x)
        return x


class LSTMNet(nn.Module):
    """Batch-first LSTM model."""

    def __init__(self, vocab_size: int=20, embed_dim: int=300, hidden_dim: int=512, num_layers: int=2) ->None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor]:
        embed = self.embedding(x)
        out, hidden = self.encoder(embed)
        out = self.decoder(out)
        out = out.view(-1, out.size(2))
        return out, hidden


class RecursiveNet(nn.Module):
    """Model that uses a layer recursively in computation."""

    def __init__(self, inplace: bool=True) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.activation = nn.ReLU(inplace)

    def forward(self, x: torch.Tensor, args1: Any=None, args2: Any=None) ->torch.Tensor:
        del args1, args2
        out = x
        for _ in range(3):
            out = self.activation(self.conv1(out))
            out = self.conv1(out)
        return out


class SimpleRNN(nn.Module):
    """Simple RNN"""

    def __init__(self, inplace: bool=True) ->None:
        super().__init__()
        self.hid_dim = 2
        self.input_dim = 3
        self.max_length = 4
        self.lstm = nn.LSTMCell(self.input_dim, self.hid_dim)
        self.activation = nn.LeakyReLU(inplace=inplace)
        self.projection = nn.Linear(self.hid_dim, self.input_dim)

    def forward(self, token_embedding: torch.Tensor) ->torch.Tensor:
        b_size = token_embedding.size()[0]
        hx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        cx = torch.randn(b_size, self.hid_dim, device=token_embedding.device)
        for _ in range(self.max_length):
            hx, cx = self.lstm(token_embedding, (hx, cx))
            hx = self.activation(hx)
        return hx


class SiameseNets(nn.Module):
    """Model with MaxPool and ReLU layers."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 10)
        self.conv2 = nn.Conv2d(64, 128, 7)
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.pooling = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) ->torch.Tensor:
        x1 = self.pooling(F.relu(self.conv1(x1)))
        x1 = self.pooling(F.relu(self.conv2(x1)))
        x1 = self.pooling(F.relu(self.conv3(x1)))
        x1 = self.pooling(F.relu(self.conv4(x1)))
        x2 = self.pooling(F.relu(self.conv1(x2)))
        x2 = self.pooling(F.relu(self.conv2(x2)))
        x2 = self.pooling(F.relu(self.conv3(x2)))
        x2 = self.pooling(F.relu(self.conv4(x2)))
        batch_size = x1.size(0)
        x1 = x1.view(batch_size, -1)
        x2 = x2.view(batch_size, -1)
        x1 = self.fc1(x1)
        x2 = self.fc1(x2)
        metric = torch.abs(x1 - x2)
        similarity = torch.sigmoid(self.fc2(self.dropout(metric)))
        return similarity


class FunctionalNet(nn.Module):
    """Model that uses many functional torch layers."""

    def __init__(self) ->None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 1600)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class RecursiveRelu(nn.Module):
    """Model with many recursive layers"""

    def __init__(self, seq_len: int=8) ->None:
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        for _ in range(self.seq_len):
            x = self.activation(x)
        return x


class Tower(nn.Module):
    """Tower Model"""

    def __init__(self, length: int=1) ->None:
        super().__init__()
        self.layers = []
        for i in range(length):
            lazy_layer = nn.LazyLinear(out_features=10)
            self.add_module(f'tower{i}', lazy_layer)
            self.layers.append(lazy_layer)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        for l_layer in self.layers:
            x = l_layer(x)
        return x


class TowerBranches(nn.Module):
    """Model with different length of tower used for expand_nested"""

    def __init__(self) ->None:
        super().__init__()
        self.tower1 = Tower(2)
        self.tower2 = Tower(3)
        self.tower3 = Tower(4)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return torch.add(self.tower1(x) + self.tower2(x), self.tower3(x))


class OutputReused(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool=True) ->None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace), nn.Linear(64, 32), nn.ReLU(inplace), nn.Linear(32, 16), nn.ReLU(inplace), nn.Linear(16, 8), nn.ReLU(inplace), nn.Linear(8, 4), nn.ReLU(inplace), nn.Linear(4, 2))
        self.empty = nn.Identity(())
        self.act = nn.ReLU(inplace)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.layers(x1)
        x = x + x2
        y = self.empty(self.act(x2)) + x3
        return x, y, x3, x4


class InputNotUsed(nn.Module):
    """Multi Layer Perceptron with inplace option.
    Make sure inplace=true and false has the same visual graph"""

    def __init__(self, inplace: bool=True) ->None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace), nn.Linear(64, 32), nn.ReLU(inplace), nn.Linear(32, 16), nn.ReLU(inplace), nn.Linear(16, 8), nn.ReLU(inplace), nn.Linear(8, 4), nn.ReLU(inplace), nn.Linear(4, 2))
        self.empty = nn.Identity(())
        self.act = nn.ReLU(inplace)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor]:
        x = self.layers(x1)
        x = x + x2
        y = self.empty(self.act(x2)) + x3
        return x, y


class CreateTensorsInside(nn.Module):
    """Module that creates tensor during forward prop"""

    def __init__(self) ->None:
        super().__init__()
        self.layer1 = nn.Linear(10, 30)
        self.layer2 = nn.Linear(30, 50)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        x = self.layer1(x)
        x += torch.abs(torch.ones(1, 1))
        x = self.layer2(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, activation, mid_channels=None, use_bn=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_bn:
            self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), nn.BatchNorm2d(mid_channels), activation, nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), activation)
        else:
            self.double_conv = nn.Sequential(activation, activation, nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), activation, nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), activation)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation, use_bn=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, activation, use_bn=use_bn))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, activation, bilinear=False, use_bn=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, activation, in_channels // 2, use_bn=use_bn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, activation, use_bn=use_bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False, residual=False, activation_type='relu', use_bn=True):
        super(UNet2, self).__init__()
        if activation_type == 'leaky_relu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_type == 'relu':
            activation = nn.ReLU(inplace=True)
        else:
            raise TypeError
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 96, activation, use_bn=use_bn)
        self.down1 = Down(96, 96 * 2, activation, use_bn=use_bn)
        self.down2 = Down(96 * 2, 96 * 4, activation, use_bn=use_bn)
        self.up1 = Up(96 * 4, 96 * 2, activation, use_bn=use_bn)
        self.up2 = Up(96 * 2, 96 * 1, activation, use_bn=use_bn)
        self.outc = OutConv(96, n_classes)
        self.residual = residual

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.outc(x)
        if self.residual:
            x += input
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'config': _mock_config(n_embd=4, n_head=4, attn_pdrop=0.5, resid_pdrop=0.5, block_size=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CausalSelfAttention,
     lambda: ([], {'config': _mock_config(n_embd=4, n_head=4, attn_pdrop=0.5, resid_pdrop=0.5, block_size=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CoFusion,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EdgeCaseModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (FunctionalNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 32, 32])], {}),
     True),
    (IdentityModel,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (OutConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RecursiveNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (RecursiveRelu,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScalarNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64]), 0], {}),
     False),
    (SingleConvBlock,
     lambda: ([], {'in_features': 4, 'out_features': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Tower,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TowerBranches,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (UNet2,
     lambda: ([], {'n_channels': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpConvBlock,
     lambda: ([], {'in_features': 4, 'up_scale': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_DenseBlock,
     lambda: ([], {'num_layers': 1, 'input_features': 4, 'out_features': 4}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (_DenseLayer,
     lambda: ([], {'input_features': 4, 'out_features': 4}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (_InitBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'small_inputs': 4, 'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (_TransitionBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mert_kurttutan_torchview(_paritybench_base):
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

