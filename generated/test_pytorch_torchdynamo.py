import sys
_module = sys.modules[__name__]
del sys
check_coverage = _module
check_csv = _module
setup = _module
test = _module
dynamo = _module
mock_modules = _module
mock_module1 = _module
mock_module2 = _module
mock_module3 = _module
test_aot_autograd = _module
test_aot_cudagraphs = _module
test_distributed = _module
test_dynamic_shapes = _module
test_export = _module
test_functions = _module
test_global = _module
test_global_declaration = _module
test_misc = _module
test_model_output = _module
test_modules = _module
test_no_fake_tensors = _module
test_nops = _module
test_optimizations = _module
test_optimizers = _module
test_python_autograd = _module
test_repros = _module
test_skip_non_tensor = _module
test_subgraphs = _module
test_unspec = _module
test_verify_correctness = _module
inductor = _module
test_torchinductor = _module
torchdynamo = _module
torchinductor = _module

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


import functools


from torch.testing._internal.common_utils import TEST_WITH_ROCM


import torch.distributed as dist


from torch import nn


import torch.utils._pytree as pytree


from torch.fx.experimental.proxy_tensor import make_fx


import collections


import inspect


import itertools


from typing import Any


from torch import sub


from torch.nn import functional as F


import copy


import enum


import logging


import math


import typing


import numpy as np


import torch.onnx.operators


from torch.testing._internal.jit_utils import JitTestCase


from copy import deepcopy


from torch.nn.modules.lazy import LazyModuleMixin


from torch.nn.parameter import Parameter


from torch.nn.parameter import UninitializedParameter


from typing import Callable


from typing import Dict


from typing import List


from typing import NamedTuple


from typing import Optional


import random


from abc import ABC


from collections import namedtuple


from torch.testing._internal.common_utils import TEST_WITH_ASAN


from torch.testing._internal.common_utils import TestCase as TorchTestCase


from torch.utils._python_dispatch import TorchDispatchMode


from torch.utils._pytree import tree_flatten


from torch.utils._pytree import tree_unflatten


import torch._dynamo


import torch._inductor


class ToyModel(nn.Module):

    def __init__(self, in_feat=10, hidden_feat=5000, num_hidden=2, out_feat=5):
        super().__init__()
        self.net = nn.Sequential(*([nn.Linear(in_feat, hidden_feat), nn.ReLU()] + [nn.Linear(5000, 5000), nn.ReLU()] * num_hidden + [nn.Linear(5000, 5), nn.ReLU()]))

    def forward(self, inputs):
        return self.net(inputs)


class CustomFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, foo):
        return foo + foo

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Module1(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, foo):
        return CustomFunc().apply(foo)


class Module2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fn = CustomFunc.apply

    def forward(self, foo):
        return self.fn(foo)


class BasicModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class FnMember(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = F.relu

    def forward(self, x):
        x = self.linear1(x)
        if self.activation:
            x = self.activation(x)
        return x


class FnMemberCmp(torch.nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.activation is None:
            x = torch.sigmoid(x)
        return x


class SubmoduleExample(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x * self.scale


class IsTrainingCheck(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.train(True)

    def forward(self, x):
        if self.training:
            mod = self.linear1
        else:
            mod = self.linear2
        return F.relu(mod(x))


class IsEvalCheck(IsTrainingCheck):

    def __init__(self):
        super().__init__()
        self.train(False)


class ModuleMethodCall(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        return x * self.scale

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        x2 = self.call_and_scale(self.layer2, x)
        return x1 + x2


class UnsupportedMethodCall(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def call_and_scale(self, mod, x):
        x = mod(x)
        x = x * self.scale
        return unsupported(x, x)

    def forward(self, x):
        x1 = self.call_and_scale(self.layer1, x)
        return x + x1


class UnsupportedModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        x = self.layer1(x) * self.scale
        return unsupported(x, x)


class UnsupportedModuleCall(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mod = UnsupportedModule()

    def forward(self, x):
        return 1 + self.mod(x * 1.5)


class ModuleStaticMethodCall(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @staticmethod
    def call_and_scale(scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleClassMethodCall(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = BasicModule()
        self.scale = torch.randn(1, 10)

    @classmethod
    def call_and_scale(cls, scale, mod, x):
        x = mod(x)
        return x * scale

    def forward(self, x):
        x1 = self.call_and_scale(self.scale, self.layer1, x)
        x2 = self.call_and_scale(self.scale, self.layer2, x)
        return x1 + x2


class ModuleProperty(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = torch.randn(1, 10)

    @property
    def scale_alias(self):
        return self.scale

    def forward(self, x):
        return x * self.scale_alias


class ConstLoop(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.count = 3

    def forward(self, x):
        for i in range(self.count):
            x = torch.sigmoid(self.linear1(x))
        return x


class ViaModuleCall(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        return test_functions.constant3(torch.sigmoid(self.linear1(x)), x)


class IsNoneLayer(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)
        self.layer2 = None
        self.train(True)

    def forward(self, x):
        if self.layer1 is not None:
            x = self.layer1(x)
        if self.layer2 is not None:
            x = self.layer2(x)
        return x


class LayerList(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = [torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 10)]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModuleList(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 10), torch.nn.ReLU()])

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        for layer in self.layers:
            x = layer(x)
        for layer, val in zip(self.layers, (x, x, x, x)):
            x = layer(x) + val
        for layer, val in zip(self.layers, (1, 2, 3, 4)):
            x = layer(x) + val
        for idx, layer in enumerate(self.layers):
            x = layer(x) * idx
        for idx, layer in enumerate(self.layers[::-1]):
            x = layer(x) * idx
        return x


class ModuleDict(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleDict({'0': torch.nn.Linear(10, 10)})

    def forward(self, x):
        x = self.layers['0'](x)
        return x


class TensorList(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.randn((1, 10)), torch.randn((10, 1)), torch.randn((1, 10)), torch.randn((10, 1))

    def forward(self, x):
        for layer in self.layers:
            x = x * layer
        return x


class Children(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(10, 10)
        self.l2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(10, 10)
        self.l4 = torch.nn.ReLU()

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class IntArg(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(10, 10)

    def forward(self, x, offset=1):
        x = F.relu(self.layer1(x)) + offset
        return x


class Seq(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 10), torch.nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class Cfg:

    def __init__(self):
        self.val = 0.5
        self.count = 3


class CfgModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.cfg = Cfg()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        for i in range(self.cfg.count):
            x = self.layer(x + self.cfg.val)
        return x


class StringMember(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.mode = 'some_string'

    def forward(self, x):
        if self.mode == 'some_string':
            return F.relu(self.linear1(x))


class _Block(torch.nn.Module):

    def forward(self, x):
        return 1.5 * torch.cat(x, 1)


class _DenseBlock(torch.nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers: int=3) ->None:
        super().__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1), _Block())

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNetBlocks(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = _DenseBlock()

    def forward(self, x):
        return self.layers(x)


class MaterializedModule(torch.nn.Module):
    """Once the below lazy module is initialized with its first input,
    it is transformed into this module."""
    param: Parameter

    def __init__(self):
        super().__init__()
        self.register_parameter('param', None)

    def forward(self, x):
        return x


class LazyModule(LazyModuleMixin, MaterializedModule):
    param: UninitializedParameter
    cls_to_become = MaterializedModule

    def __init__(self):
        super().__init__()
        self.param = UninitializedParameter()

    def initialize_parameters(self, x):
        self.param.materialize(x.shape)


def requires_grad1(module: torch.nn.Module, recurse: bool=False) ->bool:
    requires_grad = any([p.requires_grad for p in module.parameters(recurse)])
    return requires_grad


class ParametersModule1(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        if not requires_grad1(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


def requires_grad2(module: torch.nn.Module, recurse: bool=False) ->bool:
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


class ParametersModule2(ParametersModule1):

    def forward(self, x):
        if not requires_grad2(self):
            return F.relu(self.linear1(x)) * self.scale
        else:
            return x + 1


class ParametersModule3(ParametersModule1):

    def forward(self, x):
        ones = torch.ones(10, dtype=next(self.parameters()).dtype)
        return F.relu(self.linear1(x)) * self.scale + ones


class SuperModule(BasicModule):

    def forward(self, x):
        x = super().forward(x)
        return x + 10.0


class ComplicatedSuperParent(torch.nn.Module):

    @classmethod
    def custom_add(cls, x):
        x = x + x
        return x


class SuperChildCallsClassMethod(ComplicatedSuperParent):

    @classmethod
    def child_func(cls, x):
        x = super().custom_add(x)
        return x

    def forward(self, x):
        x = self.child_func(x)
        return x


class HasAttrModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.randn(1, 10))

    def forward(self, x):
        x = F.relu(x)
        if hasattr(self, 'scale'):
            x *= self.scale
        if hasattr(self, 'scale2'):
            x *= self.scale2
        return x


class EnumValues(torch.nn.ModuleDict):

    def __init__(self, num_layers: int=3) ->None:
        super().__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1), _Block())

    def forward(self, init_features):
        features = [init_features]
        for idx, layer in enumerate(self.values()):
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class CallForwardDirectly(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = BasicModule()
        self.layer2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        return x


class ModuleNameString(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)

    def forward(self, x):
        if self.__class__.__name__ == 'ABC':
            return 10
        if self.linear1.__class__.__name__ == 'Linear':
            return F.relu(self.linear1(x) + 10)
        return 11


class SelfMutatingModule(torch.nn.Module):

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.counter = 0

    def forward(self, x):
        result = self.layer(x) + self.counter
        self.counter += 1
        return F.relu(result)


class ModuleAttributePrecedenceBase(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def linear(self, x):
        return x * 2.0


class ModuleAttributePrecedence(ModuleAttributePrecedenceBase):

    def __init__(self):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.linear = torch.nn.Linear(10, 10)
        self.initializer = torch.ones([10, 10])
        self.scale = 0.5

    def activation(self, x):
        return x * 1.2

    def initializer(self):
        return torch.zeros([10, 10])

    def scale(self):
        return 2.0

    def forward(self, x):
        return self.activation(self.linear(self.initializer + x)) * self.scale


class Conv_Bn_Relu(torch.nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv_Bn_Relu, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


ReformerEncoderOutput = namedtuple('ReformerEncoderOutput', ['hidden_states', 'all_hidden_states', 'all_attentions', 'past_buckets_states'])


ReformerBackwardOutput = namedtuple('ReformerBackwardOutput', ['attn_output', 'hidden_states', 'grad_attn_output', 'grad_hidden_states'])


class _ReversibleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden_states, layers, attention_mask, head_mask, num_hashes, all_hidden_states, all_attentions, past_buckets_states, use_cache, orig_sequence_length, output_hidden_states, output_attentions):
        all_buckets = ()
        hidden_states, attn_output = torch.chunk(hidden_states, 2, dim=-1)
        for layer_id, (layer, layer_head_mask) in enumerate(zip(layers, head_mask)):
            if output_hidden_states is True:
                all_hidden_states.append(hidden_states)
            attn_output = layer(attn_output)
        if output_hidden_states is True:
            all_hidden_states.append(hidden_states)
        ctx.save_for_backward(attn_output.detach(), hidden_states.detach())
        ctx.layers = layers
        ctx.all_buckets = all_buckets
        ctx.head_mask = head_mask
        ctx.attention_mask = attention_mask
        return torch.cat([attn_output, hidden_states], dim=-1)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 2, dim=-1)
        attn_output, hidden_states = ctx.saved_tensors
        output = ReformerBackwardOutput(attn_output=attn_output, hidden_states=hidden_states, grad_attn_output=grad_attn_output, grad_hidden_states=grad_hidden_states)
        del grad_attn_output, grad_hidden_states, attn_output, hidden_states
        layers = ctx.layers
        all_buckets = ctx.all_buckets
        head_mask = ctx.head_mask
        attention_mask = ctx.attention_mask
        for idx, layer in enumerate(layers[::-1]):
            buckets = all_buckets[-1]
            all_buckets = all_buckets[:-1]
            output = layer.backward_pass(next_attn_output=output.attn_output, hidden_states=output.hidden_states, grad_attn_output=output.grad_attn_output, grad_hidden_states=output.grad_hidden_states, head_mask=head_mask[len(layers) - idx - 1], attention_mask=attention_mask, buckets=buckets)
        assert all_buckets == (), 'buckets have to be empty after backpropagation'
        grad_hidden_states = torch.cat([output.grad_attn_output, output.grad_hidden_states], dim=-1)
        return grad_hidden_states, None, None, None, None, None, None, None, None, None, None, None


class ReformerEncoder(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dropout = 0.5
        self.layer_norm = torch.nn.LayerNorm(512, eps=1e-12)
        self.layers = [torch.nn.Linear(256, 256)]

    def forward(self, hidden_states, attention_mask=None, head_mask=[None] * 6, num_hashes=None, use_cache=False, orig_sequence_length=64, output_hidden_states=False, output_attentions=False):
        all_hidden_states = []
        all_attentions = []
        past_buckets_states = [(None, None) for i in range(len(self.layers))]
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(hidden_states, self.layers, attention_mask, head_mask, num_hashes, all_hidden_states, all_attentions, past_buckets_states, use_cache, orig_sequence_length, output_hidden_states, output_attentions)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return ReformerEncoderOutput(hidden_states=hidden_states, all_hidden_states=all_hidden_states, all_attentions=all_attentions, past_buckets_states=past_buckets_states)


class PartialT5(torch.nn.Module):

    def __init__(self):
        super(PartialT5, self).__init__()
        self.q = torch.nn.Linear(512, 512)
        self.k = torch.nn.Linear(512, 512)
        self.v = torch.nn.Linear(512, 512)

    def forward(self, hidden_states, key_value_states=None, past_key_value=None, query_length=None):
        batch_size, seq_length = hidden_states.shape[:2]
        real_seq_length = seq_length
        if past_key_value is not None:
            assert len(past_key_value) == 2, f'past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states'
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, 8, 64).transpose(1, 2)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                hidden_states = shape(proj_layer(key_value_states))
            if past_key_value is not None:
                if key_value_states is None:
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    hidden_states = past_key_value
            return hidden_states
        query_states = shape(self.q(hidden_states))
        key_states = project(hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None)
        value_states = project(hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        return scores, value_states


def apply_chunking_to_forward(forward_fn, *input_tensors):
    assert len(input_tensors) > 0
    tensor_shape = input_tensors[0].shape[1]
    assert all(input_tensor.shape[1] == tensor_shape for input_tensor in input_tensors)
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError()
    return forward_fn(*input_tensors)


class ChunkReformerFeedForward(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(256, eps=1e-12)
        self.dense = torch.nn.Linear(256, 256)
        self.output = torch.nn.Linear(256, 256)

    def forward(self, attention_output):
        return apply_chunking_to_forward(self.forward_chunk, attention_output + 1)

    def forward_chunk(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        return self.output(hidden_states)


class FakeMamlInner(torch.nn.Module):

    def __init__(self):
        super(FakeMamlInner, self).__init__()
        self.linear = torch.nn.Linear(784, 5)

    def forward(self, x, ignored=None, bn_training=False):
        return self.linear(x.view(x.shape[0], -1))


class PartialMaml(torch.nn.Module):

    def __init__(self):
        super(PartialMaml, self).__init__()
        self.net = FakeMamlInner()
        self.update_step_test = 10
        self.update_lr = 0.4

    def forward(self, x_spt, y_spt, x_qry, y_qry):
        querysz = x_qry.size(0)
        corrects = [(0) for _ in range(self.update_step_test + 1)]
        net = deepcopy(self.net)
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct
        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct
        del net
        accs = torch.tensor(corrects) / querysz
        return accs


class SequentialAppendList(torch.nn.Sequential):
    """from timm/models/vovnet.py"""

    def __init__(self, *args):
        super(SequentialAppendList, self).__init__(*args)

    def forward(self, x: torch.Tensor, concat_list: List[torch.Tensor]) ->torch.Tensor:
        for i, module in enumerate(self):
            if i == 0:
                concat_list.append(module(x))
            else:
                concat_list.append(module(concat_list[-1]))
        x = torch.cat(concat_list, dim=1)
        return x, concat_list


class BatchNormAct2d(torch.nn.BatchNorm2d):
    """Taken from timm"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, act_layer=torch.nn.ReLU, inplace=True):
        super(BatchNormAct2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.act = act_layer(inplace=inplace)

    @torch.jit.ignore
    def _forward_python(self, x):
        return super().forward(x)

    def forward(self, x):
        if torch.jit.is_scripting():
            x = self._forward_jit(x)
        else:
            x = self._forward_python(x)
        x = self.act(x)
        return x


class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward, activation, dropout) ->None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.ReLU(), layer_norm_eps=1e-05):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.ff_block = FeedForwardLayer(d_model, dim_feedforward, activation, dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def _ff_block(self, x):
        return self.ff_block(x)


class TestModule(torch.nn.Module):

    def inner_fn(self, left, right):
        return tuple(left) == tuple(right)

    def fn(self, tensor):
        if type(tensor) is int:
            return False
        torch.add(tensor, tensor)
        return self.inner_fn(tensor.shape, (1, 2, 3))


class ToTuple(torch.nn.Module):

    def forward(self, x):
        return x,


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNormAct2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ChunkReformerFeedForward,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 256, 256])], {}),
     False),
    (Conv_Bn_Relu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DenseNetBlocks,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (EnumValues,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FeedForwardLayer,
     lambda: ([], {'d_model': 4, 'dim_feedforward': 4, 'activation': _mock_layer(), 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HasAttrModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 10])], {}),
     True),
    (LazyModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaterializedModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Module1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Module2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ModuleAttributePrecedence,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 10, 10])], {}),
     False),
    (ModuleProperty,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 10])], {}),
     True),
    (ParametersModule1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ParametersModule2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SelfMutatingModule,
     lambda: ([], {'layer': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SuperChildCallsClassMethod,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TensorList,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 10, 10])], {}),
     True),
    (ToTuple,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerEncoderLayer,
     lambda: ([], {'d_model': 4, 'nhead': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (_DenseBlock,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_pytorch_torchdynamo(_paritybench_base):
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

