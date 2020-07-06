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
utils = _module
model = _module
train = _module
utils = _module
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
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import inspect


import re


import logging


import torch.nn as nn


import torch


import torch.nn.functional as F


from collections import OrderedDict


import numpy as np


import copy


from torch.nn.modules.batchnorm import _BatchNorm


from torch.nn.modules.utils import _single


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


from torch.nn import DataParallel as DataParallel_


from torch.nn.parallel import parallel_apply


from torch.nn.parallel.scatter_gather import scatter_kwargs


from torch.nn.parallel.replicate import _broadcast_coalesced_reshape


from torch.utils.data import DataLoader


from collections import defaultdict


from torch.utils.data.dataloader import default_collate


from torch.utils.data.dataset import Dataset as TorchDataset


import random


from itertools import combinations


from torch.utils.data.sampler import SequentialSampler


from torch.utils.data.sampler import RandomSampler


from torch.utils.data import ConcatDataset


from torch.utils.data import Subset


from torch.utils.data import Dataset as Dataset_


from torchvision.transforms import Compose


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
        gen = self._named_members(lambda module: module._parameters.items() if isinstance(module, MetaModule) else [], prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param


class MetaLinear(nn.Linear, MetaModule):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.linear(input, params['weight'], bias)


def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if key is None or key == '':
        return dictionary
    key_re = re.compile('^{0}\\.(.+)'.format(re.escape(key)))
    if not any(filter(key_re.match, dictionary.keys())):
        key_re = re.compile('^module\\.{0}\\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub('\\1', k), value) for k, value in dictionary.items() if key_re.match(k) is not None)


class MetaSequential(nn.Sequential, MetaModule):
    __doc__ = nn.Sequential.__doc__

    def forward(self, input, params=None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError('The module must be either a torch module (inheriting from `nn.Module`), or a `MetaModule`. Got type: `{0}`'.format(type(module)))
        return input


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs), nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(2))


class ConvolutionalNeuralNetwork(MetaModule):

    def __init__(self, in_channels, out_features, hidden_size=64):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.features = MetaSequential(conv3x3(in_channels, hidden_size), conv3x3(hidden_size, hidden_size), conv3x3(hidden_size, hidden_size), conv3x3(hidden_size, hidden_size))
        self.classifier = MetaLinear(hidden_size, out_features)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=get_subdict(params, 'classifier'))
        return logits


class PrototypicalNetwork(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_size=64):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(conv3x3(in_channels, hidden_size), conv3x3(hidden_size, hidden_size), conv3x3(hidden_size, hidden_size), conv3x3(hidden_size, out_channels))

    def forward(self, inputs):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)


class _MetaBatchNorm(_BatchNorm, MetaModule):

    def forward(self, input, params=None):
        self._check_input_dim(input)
        if params is None:
            params = OrderedDict(self.named_parameters())
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        weight = params.get('weight', None)
        bias = params.get('bias', None)
        return F.batch_norm(input, self.running_mean, self.running_var, weight, bias, self.training or not self.track_running_stats, exponential_average_factor, self.eps)


class MetaBatchNorm1d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm1d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class MetaBatchNorm2d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class MetaBatchNorm3d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm3d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))


class MetaConv1d(nn.Conv1d, MetaModule):
    __doc__ = nn.Conv1d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        if self.padding_mode == 'circular':
            expanded_padding = (self.padding[0] + 1) // 2, self.padding[0] // 2
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'), params['weight'], bias, self.stride, _single(0), self.dilation, self.groups)
        return F.conv1d(input, params['weight'], bias, self.stride, self.padding, self.dilation, self.groups)


class MetaConv2d(nn.Conv2d, MetaModule):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        if self.padding_mode == 'circular':
            expanded_padding = (self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'), params['weight'], bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, params['weight'], bias, self.stride, self.padding, self.dilation, self.groups)


class MetaConv3d(nn.Conv3d, MetaModule):
    __doc__ = nn.Conv3d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        if self.padding_mode == 'circular':
            expanded_padding = (self.padding[2] + 1) // 2, self.padding[2] // 2, (self.padding[1] + 1) // 2, self.padding[1] // 2, (self.padding[0] + 1) // 2, self.padding[0] // 2
            return F.conv3d(F.pad(input, expanded_padding, mode='circular'), params['weight'], bias, self.stride, _triple(0), self.dilation, self.groups)
        return F.conv3d(input, params['weight'], bias, self.stride, self.padding, self.dilation, self.groups)


class MetaBilinear(nn.Bilinear, MetaModule):
    __doc__ = nn.Bilinear.__doc__

    def forward(self, input1, input2, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.bilinear(input1, input2, params['weight'], bias)


class MetaLayerNorm(nn.LayerNorm, MetaModule):
    __doc__ = nn.LayerNorm.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weight = params.get('weight', None)
        bias = params.get('bias', None)
        return F.layer_norm(input, self.normalized_shape, weight, bias, self.eps)


class DataParallel(DataParallel_, MetaModule):
    __doc__ = DataParallel_.__doc__

    def scatter(self, inputs, kwargs, device_ids):
        try:
            params = kwargs.pop('params')
        except KeyError:
            return super(DataParallel, self).scatter(inputs, kwargs, device_ids)
        inputs_, kwargs_ = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        replicas = self._replicate_params(params, inputs_, device_ids, detach=not torch.is_grad_enabled())
        kwargs_ = tuple(dict(params=replica, **kwarg) for kwarg, replica in zip(kwargs_, replicas))
        return inputs_, kwargs_

    def _replicate_params(self, params, inputs, device_ids, detach=False):
        if params is None:
            return tuple(None for _ in inputs)
        replicas = _broadcast_coalesced_reshape(list(params.values()), device_ids[:len(inputs)], detach)
        replicas = tuple(OrderedDict(zip(params.keys(), replica)) for replica in replicas)
        return replicas


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvolutionalNeuralNetwork,
     lambda: ([], {'in_channels': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 16, 16])], {}),
     False),
    (DataParallel,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (MetaBatchNorm1d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MetaBatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MetaBatchNorm3d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4])], {}),
     False),
    (MetaBilinear,
     lambda: ([], {'in1_features': 4, 'in2_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MetaConv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (MetaConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MetaConv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     False),
    (MetaLayerNorm,
     lambda: ([], {'normalized_shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MetaLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MetaSequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PrototypicalNetwork,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 64, 64])], {}),
     False),
]

class Test_tristandeleu_pytorch_meta(_paritybench_base):
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

