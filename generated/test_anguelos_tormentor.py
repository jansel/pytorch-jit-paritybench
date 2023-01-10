import sys
_module = sys.modules[__name__]
del sys
diamond_square = _module
diamond_square = _module
render_augmentation_examples = _module
conf = _module
tweak = _module
ds_compare = _module
synth = _module
bunet = _module
dibco = _module
rr_ds = _module
setup = _module
benchmark_augmentations = _module
_test_augmentation_cross_device_deterninism = _module
test_augmentation = _module
test_augmented_pointclouds = _module
test_augmeting_labels = _module
test_base_augmentation = _module
test_cocodataset = _module
test_dataset = _module
test_factory = _module
test_resiszing = _module
util = _module
tormentor = _module
ablation = _module
augmentation_cascade = _module
augmentation_choice = _module
augmented_dataloader = _module
augmented_dataset = _module
base_augmentation = _module
color_augmentations = _module
deterministic_image_augmentation = _module
factory = _module
random = _module
random_network = _module
resizing_augmentation = _module
sampling_fileds = _module
spatial_augmentations = _module
spatial_image_augmentation = _module
static_image_augmentation = _module
util = _module
version = _module
wrap = _module

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


import math


import torch


import numpy as np


import matplotlib.pyplot as plt


from matplotlib import pyplot as plt


from collections import defaultdict


from torch import nn


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torchvision


import re


import string


import time


from typing import Tuple


from typing import Union


from typing import List


from typing import Type


import itertools


from typing import Any


from torch import Tensor


from torch import FloatTensor


from torch import LongTensor


from itertools import count


import types


def _diamond_square_seed(replicates, width, height, random, device):
    assert width == 3 or height == 3
    if height == 3:
        transpose = True
        width, height = height, width
    else:
        transpose = False
    assert height % 2 == 1 and height > 2
    res = random([replicates, 1, width, height])
    res[:, :, ::2, ::2] = random([replicates, 1, 2, (height + 1) // 2])
    res[:, :, 1, 1::2] = (res[:, :, ::2, :-2:2] + res[:, :, ::2, 2::2]).sum(dim=2) / 4.0
    if width > 3:
        res[:, :, 1, 2:-3:2] = (res[:, :, 0, 2:-3:2] + res[:, :, 2, 2:-3:2] + res[:, :, 1, 0:-4:2] + res[:, :, 1, 2:-3:2]) / 4.0
    res[:, :, 1, 0] = (res[:, :, 0, 0] + res[:, :, 1, 1] + res[:, :, 2, 0]) / 3.0
    res[:, :, 1, -1] = (res[:, :, -1, -1] + res[:, :, 1, -2] + res[:, :, 2, 0]) / 3.0
    res[:, :, 0, 1::2] = (res[:, :, 0, 0:-2:2] + res[:, :, 0, 2::2] + res[:, :, 1, 1::2]) / 3.0
    res[:, :, 2, 1::2] = (res[:, :, 2, 0:-2:2] + res[:, :, 2, 2::2] + res[:, :, 1, 1::2]) / 3.0
    if device is not None:
        res = res
    if transpose:
        return res.transpose(2, 3)
    else:
        return res


class DiamondSquare(torch.nn.Module):

    def get_current_device(self):
        return self.initial_rnd_scale.data.device

    def __init__(self, recursion_steps=1, rnd_scale=1.0, rand=torch.rand):
        self.recursion_steps = recursion_steps
        self.rand = rand
        self.initial_rnd_range = torch.nn.Parameter(torch.tensor([rnd_scale], requires_grad=True))
        self.diamond_kernel = [[0.25, 0.0, 0.25], [0.0, 0.0, 0.0], [0.25, 0.0, 0.25]]
        self.diamond_kernel = torch.nn.Parameter(torch.tensor(self.diamond_kernel, requires_grad=True).unsqueeze(dim=0).unsqueeze(dim=0))
        self.square_kernel = [[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]
        self.square_kernel = torch.nn.Parameter(torch.tensor(self.square_kernel, requires_grad=True).unsqueeze(dim=0).unsqueeze(dim=0))
        self.initial_rnd_scale = torch.nn.Parameter(torch.tensor(rnd_scale, requires_grad=False))

    def forward(self, input_img=None, seed_size=None):
        if input is None:
            img = _diamond_square_seed(seed_size, device=self.get_current_device())
        else:
            img = input_img
        rnd_scale = self.initial_rnd_range.clone()
        for _ in range(self.recursion_steps):
            img = one_diamond_one_square(img, random=self.rand, diamond_kernel=self.diamond_kernel, square_kernel=self.square_kernel, rnd_scale=rnd_scale)
        return img


def tensor2pil(tensor):
    if len(tensor.size()) == 4:
        assert tensor.size(0) == 1
        tensor = tensor[0, :, :, :]
    if tensor.size(0) in (1, 2):
        tensor = tensor[0, :, :]
    else:
        assert len(tensor.size()) == 2 or tensor.size(0) in (3, 4)
    array = np.uint8(tensor.cpu().numpy().astype('float') * 255)
    return Image.fromarray(array)


class BUNet(nn.Module):

    @staticmethod
    def resume(fname, **kwargs):
        try:
            if 'device' in kwargs.keys():
                device = kwargs['device']
            else:
                device = ''
            state_dict = torch.load(fname, map_location='cpu')
            constructor_params = state_dict['constructor_params']
            del state_dict['constructor_params']
            validation_epochs = state_dict['validation_epochs']
            del state_dict['validation_epochs']
            train_epochs = state_dict['train_epochs']
            del state_dict['train_epochs']
            args_history = state_dict['args_history']
            del state_dict['args_history']
            net = BUNet(**constructor_params)
            net.load_state_dict(state_dict)
            net.validation_epochs = validation_epochs
            net.train_epochs = train_epochs
            net.args_history = args_history
            net = net
            return net
        except FileNotFoundError:
            return BUNet(**kwargs)

    def __init__(self, input_channels=3, target_channels=2, channels=(64, 128, 256, 384), stack_size=2, device='cuda') ->None:
        super().__init__()
        self.input2iunet = nn.Conv2d(in_channels=input_channels, out_channels=channels[0], kernel_size=3, padding=1)
        self.iunet2output = nn.Conv2d(in_channels=channels[0], out_channels=target_channels, kernel_size=3, padding=1)
        self.iunet = iunets.iUNet(in_channels=channels[0], channels=channels[1:], architecture=(stack_size,) * (len(channels) - 1), dim=2)
        self.train_epochs = []
        self.validation_epochs = {}
        self.constructor_params = {'input_channels': input_channels, 'target_channels': target_channels, 'channels': channels, 'stack_size': stack_size, 'device': device}
        self.args_history = {}
        self

    def forward(self, x):
        x = self.input2iunet(x)
        x = self.iunet(x)
        x = self.iunet2output(x)
        return x

    def binarize(self, input, to_pil=False, threshold=False):
        if isinstance(input, torch.Tensor):
            with torch.no_grad():
                if len(input.size()) == 3:
                    input = torch.unsqueeze(input, 0)
                assert len(input.size()) == 4
                input = input
                output = self.forward(input)
                output = F.softmax(output, dim=1)[0, 1, :, :]
                if threshold:
                    output = (output > 0.5).float()
                if to_pil:
                    return tensor2pil(output)
                else:
                    return output
        elif isinstance(input, DataLoader):
            results = []
            for sample_data in input:
                sample_input_image = sample_data[0]
                results.append(tuple([tensor2pil(s) for s in (self.binarize(sample_input_image),) + tuple(sample_data)]))
            return results
        else:
            raise ValueError('Expect dataloader or dataset')

    def save(self, fname, args=None):
        state_dict = self.state_dict()
        state_dict['constructor_params'] = self.constructor_params
        state_dict['validation_epochs'] = self.validation_epochs
        state_dict['train_epochs'] = self.train_epochs
        state_dict['args_history'] = self.args_history
        torch.save(state_dict, fname)


TensorSize = Union[torch.Size, int, Tuple[int], Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int, int]]


class Distribution(torch.nn.Module):

    def __init__(self, do_rsample):
        super().__init__()
        self.do_rsample = do_rsample

    def forward(self, size: TensorSize=1, device='cpu'):
        """Samples the probabillity distribution

        Args:
            size: the size to be sampled
            device: the device on witch to sample

        Returns:
            a tuple with the sampled probabillity and path probabillities.

        """
        raise NotImplementedError()

    def copy(self, do_rsample=None):
        raise NotImplementedError()

    def get_distribution_parameters(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError
        return type(self) is type(other) and self.get_distribution_parameters() == other.get_distribution_parameters()

    @property
    def device(self):
        raise NotImplementedError()

    def to(self, device):
        if device != self.device:
            super()

    def __hash__(self):
        return hash(tuple(sorted(self.get_distribution_parameters().items())))


TupleRange = Tuple[float, float]


class Uniform(Distribution):

    def __init__(self, value_range: TupleRange=(0.0, 1.0), do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        self.min = torch.nn.Parameter(torch.Tensor([value_range[0]]), requires_grad=True)
        self.max = torch.nn.Parameter(torch.Tensor([value_range[1]]), requires_grad=True)
        self.distribution = torch.distributions.Uniform(low=self.min, high=self.max)
        self.do_rsample = do_rsample

    def __repr__(self):
        range_str = f'({self.distribution.low.item()}, {self.distribution.high.item()})'
        param_str = f' do_rsample={self.do_rsample}'
        return f'{self.__class__.__qualname__}(value_range={range_str}, {param_str})'

    def __str__(self):
        range_str = f'({self.distribution.low.item():.3}, {self.distribution.high.item():.3})'
        return f'{self.__class__.__qualname__}(value_range={range_str})'

    def forward(self, size: TensorSize=1, device='cpu') ->torch.Tensor:
        self
        if not hasattr(size, '__getitem__'):
            size = [size]
        if self.do_rsample:
            return self.distribution.rsample(size).view(size)
        else:
            return self.distribution.sample(size).view(size)

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Uniform(value_range=(self.min.item(), self.max.item()), do_rsample=do_rsample)

    def get_distribution_parameters(self):
        return {'min': self.min, 'max': self.max}

    @property
    def device(self):
        return self.min.device


class Constant(Distribution):

    def __init__(self, value: float, do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        self.value = torch.nn.Parameter(torch.Tensor([value]))

    def __repr__(self):
        value_str = tuple(self.value.detach().cpu().numpy())
        return f'{self.__class__.__qualname__}(value={value_str})'

    def forward(self, size: TensorSize=1, device='cpu') ->torch.Tensor:
        self
        if not hasattr(size, '__getitem__'):
            size = size,
        if self.do_rsample:
            return self.value.repeat(size)
        else:
            return self.value.repeat(size)

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Constant(value=self.value.item(), do_rsample=do_rsample)

    def get_distribution_parameters(self):
        return {'value': self.value}

    @property
    def device(self):
        return self.value.device


class Bernoulli(Distribution):

    def __init__(self, prob: float=0.5, do_rsample: object=False) ->object:
        super().__init__(do_rsample)
        self.prob = torch.nn.Parameter(torch.tensor([prob]))
        self.distribution = torch.distributions.Bernoulli(probs=self.prob)

    def forward(self, size: TensorSize=1, device='cpu'):
        self
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, '__getitem__'):
                size = [size]
            torch.ones(size, device=device)
            res = self.distribution.sample(size).view(size)
            return res

    def __repr__(self) ->str:
        name = self.__class__.__qualname__
        prob = self.prob.cpu().tolist()
        return f'{name}(prob={repr(prob)}, do_rsample={self.do_rsample})'

    def __str__(self) ->str:
        name = self.__class__.__qualname__
        prob = float(self.prob.cpu()[0])
        return f'{name}(prob={repr(prob):.3})'

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Bernoulli(prob=self.prob.item(), do_rsample=do_rsample)

    def get_distribution_parameters(self):
        return {'prob': self.prob}

    @property
    def device(self):
        return self.prob.device


class Categorical(Distribution):

    def __init__(self, n_categories: int=0, probs=(), do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        if n_categories == 0:
            assert probs != ()
        else:
            assert probs == ()
            probs = tuple([(1 / n_categories) for _ in range(n_categories)])
        self.probs = torch.autograd.Variable(torch.Tensor([probs]))
        self.distribution = torch.distributions.Categorical(probs=self.probs)

    def forward(self, size: TensorSize=1, device='cpu'):
        self
        if self.do_rsample:
            raise NotImplemented
        else:
            if not hasattr(size, '__getitem__'):
                size = [size]
            result = self.distribution.sample(size).view(size)
            return result

    def __repr__(self) ->str:
        name = self.__class__.__qualname__
        probs = tuple(self.probs.tolist())
        return f'{name}(prob={probs}, do_rsample={self.do_rsample})'

    def __str__(self) ->str:
        name = self.__class__.__qualname__
        probs = '(' + ', '.join([f'{p:.3}' for p in self.probs.tolist()]) + ')'
        return f'{name}(prob={probs})'

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Categorical(probs=tuple(self.probs.tolist()), do_rsample=do_rsample)

    def get_distribution_parameters(self) ->dict:
        return {'probs': self.probs}

    @property
    def device(self):
        return self.probs.device


class Normal(Distribution):

    def __init__(self, mean=0.0, deviation=1.0, do_rsample=False):
        super().__init__(do_rsample=do_rsample)
        self.mean = torch.nn.Parameter(torch.Tensor([mean]))
        self.deviation = torch.nn.Parameter(torch.Tensor([deviation]))
        self.distribution = torch.distributions.Normal(loc=self.mean, scale=self.deviation)

    def forward(self, size: TensorSize=1, device='cpu'):
        self
        if not hasattr(size, '__getitem__'):
            size = [size]
        if self.do_rsample:
            return self.distribution.rsample(size).view(size)
        else:
            return self.distribution.sample(size).view(size)

    def __repr__(self) ->str:
        name = self.__class__.__qualname__
        params = f'mean={self.mean.item()}, deviation={self.deviation.item()}'
        return f'{name}({params}, do_rsample={self.do_rsample})'

    def copy(self, do_rsample=None):
        if do_rsample is None:
            do_rsample = self.do_rsample
        return Normal(mean=self.mean, deviation=self.deviation, do_rsample=do_rsample)

    def get_distribution_parameters(self) ->dict:
        return {'mean': self.mean, 'deviation': self.deviation}

    @property
    def device(self):
        return self.mean.device


class DistributionNetwork(Distribution):

    def __init__(self):
        super().__init__()

    def forward(self, size: TensorSize=1, device='cpu'):
        raise NotImplementedError()

    def copy(self, do_rsample=None):
        raise NotImplementedError()

    def get_distribution_parameters(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError
        return type(self) is type(other) and self.get_distribution_parameters() == other.get_distribution_parameters()

    @property
    def device(self):
        raise NotImplementedError()

    def to(self, device):
        if device != self.device:
            super()

    def __hash__(self):
        return hash(tuple(sorted(self.get_distribution_parameters().items())))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Bernoulli,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (Constant,
     lambda: ([], {'value': 4}),
     lambda: ([], {}),
     False),
    (Normal,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (Uniform,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
]

class Test_anguelos_tormentor(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

