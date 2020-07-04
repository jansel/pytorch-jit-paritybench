import sys
_module = sys.modules[__name__]
del sys
api_reference = _module
index = _module
model = _module
train = _module
utils = _module
train = _module
model = _module
train = _module
model = _module
setup = _module
torchmeta = _module
datasets = _module
cifar100 = _module
base = _module
cifar_fs = _module
fc100 = _module
cub = _module
doublemnist = _module
helpers = _module
miniimagenet = _module
omniglot = _module
pascal5i = _module
tcga = _module
tieredimagenet = _module
triplemnist = _module
modules = _module
batchnorm = _module
container = _module
conv = _module
linear = _module
module = _module
normalization = _module
parallel = _module
tests = _module
test_datasets_helpers = _module
test_container = _module
test_conv = _module
test_dataparallel = _module
test_linear = _module
toy = _module
test_toy = _module
transforms = _module
test_splitters = _module
test_dataloaders = _module
test_gradient_based = _module
test_prototype = _module
harmonic = _module
sinusoid = _module
sinusoid_line = _module
augmentations = _module
categorical = _module
splitters = _module
target_transforms = _module
data = _module
dataloader = _module
dataset = _module
sampler = _module
task = _module
gradient_based = _module
metrics = _module
prototype = _module
version = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import inspect


import re


import logging


import torch.nn as nn


import torch


import torch.nn.functional as F


from collections import OrderedDict


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn import DataParallel as DataParallel_


from torch.nn.parallel import parallel_apply


from torch.nn.parallel.scatter_gather import scatter_kwargs


from torch.nn.parallel.replicate import _broadcast_coalesced_reshape


import numpy as np


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3,
        padding=1, **kwargs), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.
        MaxPool2d(2))


class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.features = nn.Sequential(conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size), conv3x3(hidden_size,
            hidden_size), conv3x3(hidden_size, hidden_size))
        self.classifier = nn.Linear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs)
        features = features.view((features.size(0), -1))
        logits = self.classifier(features)
        return logits


class PrototypicalNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size=64):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size), conv3x3(hidden_size,
            hidden_size), conv3x3(hidden_size, out_channels))

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if key is None or key == '':
        return dictionary
    key_re = re.compile('^{0}\\.(.+)'.format(re.escape(key)))
    if not any(filter(key_re.match, dictionary.keys())):
        key_re = re.compile('^module\\.{0}\\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub('\\1', k), value) for k, value in
        dictionary.items() if key_re.match(k) is not None)


class MetaModule(nn.Module):
    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.

    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(lambda module: module._parameters.items() if
            isinstance(module, MetaModule) else [], prefix=prefix, recurse=
            recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param


class MetaLayerNorm(nn.LayerNorm, MetaModule):
    __doc__ = nn.LayerNorm.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weight = params.get('weight', None)
        bias = params.get('bias', None)
        return F.layer_norm(input, self.normalized_shape, weight, bias,
            self.eps)


class DataParallel(DataParallel_, MetaModule):
    __doc__ = DataParallel_.__doc__

    def scatter(self, inputs, kwargs, device_ids):
        try:
            params = kwargs.pop('params')
        except KeyError:
            return super(DataParallel, self).scatter(inputs, kwargs, device_ids
                )
        inputs_, kwargs_ = scatter_kwargs(inputs, kwargs, device_ids, dim=
            self.dim)
        replicas = self._replicate_params(params, inputs_, device_ids,
            detach=not torch.is_grad_enabled())
        kwargs_ = tuple(dict(params=replica, **kwarg) for kwarg, replica in
            zip(kwargs_, replicas))
        return inputs_, kwargs_

    def _replicate_params(self, params, inputs, device_ids, detach=False):
        if params is None:
            return tuple(None for _ in inputs)
        replicas = _broadcast_coalesced_reshape(list(params.values()),
            device_ids[:len(inputs)], detach)
        replicas = tuple(OrderedDict(zip(params.keys(), replica)) for
            replica in replicas)
        return replicas


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_tristandeleu_pytorch_meta(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DataParallel(*[], **{'module': _mock_layer()}), [], {'input': torch.rand([4, 4])})

    @_fails_compile()
    def test_001(self):
        self._check(MetaLayerNorm(*[], **{'normalized_shape': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(PrototypicalNetwork(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 64, 64])], {})

