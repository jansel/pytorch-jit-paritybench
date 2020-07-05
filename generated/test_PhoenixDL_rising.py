import sys
_module = sys.modules[__name__]
del sys
conf = _module
rising = _module
_version = _module
interface = _module
loading = _module
collate = _module
dataset = _module
loader = _module
ops = _module
tensor = _module
random = _module
abstract = _module
continuous = _module
discrete = _module
transforms = _module
abstract = _module
affine = _module
channel = _module
compose = _module
crop = _module
format = _module
functional = _module
affine = _module
intensity = _module
spatial = _module
utility = _module
kernel = _module
spatial = _module
utils = _module
checktype = _module
shape = _module
setup = _module
tests = _module
_utils = _module
test_collate = _module
test_dataset = _module
test_loader = _module
test_tensor = _module
rand = _module
test_abstract = _module
test_continuous = _module
test_discrete = _module
test_interface = _module
test_affine = _module
test_channel = _module
test_crop = _module
test_device = _module
test_intensity = _module
test_spatial = _module
test_utility = _module
test_abstract_transform = _module
test_compose = _module
test_format_transforms = _module
test_intensity_transforms = _module
test_kernel_transforms = _module
test_spatial_transforms = _module
test_utility_transforms = _module
test_checktype = _module
versioneer = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from abc import abstractmethod


from typing import Union


from typing import Sequence


from typing import Optional


from typing import Callable


from typing import Any


from typing import Tuple


from random import shuffle


from typing import Mapping


import warnings


from torch import Tensor


import math


from itertools import combinations


from torch.multiprocessing import Value


import random


def reshape_list(flat_list: list, size: Union[torch.Size, tuple]) ->list:
    """
    Reshape a (nested) list to a given shape

    Args:
        flat_list: (nested) list to reshape
        size: shape to reshape to

    Returns:
        list: reshape list
    """
    if len(size) == 1:
        return [flat_list.pop(0) for _ in range(size[0])]
    else:
        return [reshape_list(flat_list, size[1:]) for _ in range(size[0])]


def reshape(value: Union[list, torch.Tensor], size: Union[Sequence, torch.Size]) ->Union[torch.Tensor, list]:
    """
    Reshape sequence (list or tensor) to given size

    Args:
        value: sequence to reshape
        size: size to reshape to

    Returns:
        Union[torch.Tensor, list]: reshaped sequence
    """
    if isinstance(value, torch.Tensor):
        return value.view(size)
    else:
        return reshape_list(value, size)


class AbstractParameter(torch.nn.Module):
    """
    Abstract Parameter class to inject randomness to transforms
    """

    @staticmethod
    def _get_n_samples(size: Union[Sequence, torch.Size]=(1,)):
        """
        Calculates the number of elements in the given size

        Args:
            size: Sequence or torch.Size

        Returns:
            int: the number of elements
        """
        if not isinstance(size, torch.Size):
            size = torch.Size(size)
        return size.numel()

    @abstractmethod
    def sample(self, n_samples: int) ->Union[torch.Tensor, list]:
        """
        Abstract sampling function

        Args:
            n_samples : the number of samples to return

        Returns:
            torch.Tensor or list: the sampled values
        """
        raise NotImplementedError

    def forward(self, size: Optional[Union[Sequence, torch.Size]]=None, device: Union[torch.device, str]=None, dtype: Union[torch.dtype, str]=None, tensor_like: torch.Tensor=None) ->Union[None, list, torch.Tensor]:
        """
        Forward function (will also be called if the module is called).
        Calculates the number of samples from the given shape, performs the
        sampling and converts it back to the correct shape.

        Args:
            size: the size of the sampled values. If None, it samples one value
                without reshaping
            device : the device the result value should be set to, if it is a tensor
            dtype : the dtype, the result value should be casted to, if it is a tensor
            tensor_like: the tensor, having the correct dtype and device.
                The result will be pushed onto this device and casted to this
                dtype if this is specified.

        Returns:
            list or torch.Tensor: the sampled values

        Notes:
            if the parameter ``tensor_like`` is given,
            it overwrites the parameters ``dtype`` and ``device``
        """
        n_samples = self._get_n_samples(size if size is not None else (1,))
        samples = self.sample(n_samples)
        if any([(s is None) for s in samples]):
            return None
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples).flatten()
        if size is not None:
            samples = reshape(samples, size)
        if isinstance(samples, torch.Tensor):
            if tensor_like is not None:
                samples = samples
            else:
                samples = samples
        return samples


class DiscreteParameter(AbstractParameter):
    """
    Samples parameters from a discrete population with or without
    replacement
    """

    def __init__(self, population: Sequence, replacement: bool=False, weights: Sequence=None, cum_weights: Sequence=None):
        """
        Args:
            population : the parameter population to sample from
            replacement : whether or not to sample with replacement
            weights : relative sampling weights
            cum_weights : cumulative sampling weights
        """
        super().__init__()
        if replacement:
            sample_fn = partial(sample_with_replacement, weights=weights, cum_weights=cum_weights)
        else:
            if weights is not None or cum_weights is not None:
                raise ValueError('weights and cum_weights should only be specified if replacement is set to True!')
            sample_fn = sample_without_replacement
        self.sample_fn = sample_fn
        self.population = population

    def sample(self, n_samples: int) ->list:
        """
        Samples from the discrete internal population

        Args:
            n_samples : the number of elements to sample

        Returns:
            list: the sampled values

        """
        return self.sample_fn(population=self.population, k=n_samples)


class AbstractTransform(torch.nn.Module):
    """Base class for all transforms"""

    def __init__(self, grad: bool=False, **kwargs):
        """
        Args:
            grad: enable gradient computation inside transformation
        """
        super().__init__()
        self.grad = grad
        self._registered_samplers = []
        for key, item in kwargs.items():
            setattr(self, key, item)

    def register_sampler(self, name: str, sampler: Union[Sequence, AbstractParameter], *args, **kwargs):
        """
        Registers a parameter sampler to the transform.
        Internally a property is created to forward calls to the attribute to
        calls of the sampler.

        Args:
            name : the property name
            sampler : the sampler. Will be wrapped to a sampler always returning
                the same element if not already a sampler
            *args : additional positional arguments (will be forwarded to
                sampler call)
            **kwargs : additional keyword arguments (will be forwarded to
                sampler call)
        """
        self._registered_samplers.append(name)
        if hasattr(self, name):
            raise NameError('Name %s already exists' % name)
        if not isinstance(sampler, (tuple, list)):
            sampler = [sampler]
        new_sampler = []
        for _sampler in sampler:
            if not isinstance(_sampler, AbstractParameter):
                _sampler = DiscreteParameter([_sampler], replacement=True)
            new_sampler.append(_sampler)
        sampler = new_sampler

        def sample(self):
            """
            Sample random values
            """
            sample_result = tuple([_sampler(*args, **kwargs) for _sampler in sampler])
            if len(sample_result) == 1:
                return sample_result[0]
            else:
                return sample_result
        setattr(self, name, property(sample))

    def __getattribute__(self, item) ->Any:
        """
        Automatically dereference registered samplers

        Args:
            item: name of attribute

        Returns:
            Any: attribute
        """
        res = super().__getattribute__(item)
        if isinstance(res, property) and item in self._registered_samplers:
            return res.__get__(self)
        else:
            return res

    def __call__(self, *args, **kwargs) ->Any:
        """
        Call super class with correct torch context

        Args:
            *args: forwarded positional arguments
            **kwargs: forwarded keyword arguments

        Returns:
            Any: transformed data

        """
        if self.grad:
            context = torch.enable_grad()
        else:
            context = torch.no_grad()
        with context:
            return super().__call__(*args, **kwargs)

    def forward(self, **data) ->dict:
        """
        Implement transform functionality here

        Args:
            **data: dict with data

        Returns:
            dict: dict with transformed data
        """
        raise NotImplementedError


class _TransformWrapper(torch.nn.Module):
    """
    Helper Class to wrap all non-module transforms into modules to use the
    torch.nn.ModuleList as container for the transforms. This enables
    forwarding of all model specific calls as ``.to()`` to all transforms
    """

    def __init__(self, trafo: Callable):
        """
        Args:
            trafo: the actual transform, which will be wrapped by this class.
                Since this transform is no subclass of ``torch.nn.Module``,
                its internal state won't be affected by module specific calls
        """
        super().__init__()
        self.trafo = trafo

    def forward(self, *args, **kwargs) ->Any:
        """
        Forwards calls to this wrapper to the internal transform

        Args:
            *args: positional arguments
            **kwargs: keyword arguments

        Returns:
            Any: trafo return
        """
        return self.trafo(*args, **kwargs)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (_TransformWrapper,
     lambda: ([], {'trafo': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_PhoenixDL_rising(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

