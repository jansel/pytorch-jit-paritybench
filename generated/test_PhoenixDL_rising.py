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
channel = _module
crop = _module
intensity = _module
spatial = _module
tensor = _module
utility = _module
intensity = _module
kernel = _module
spatial = _module
tensor = _module
utility = _module
utils = _module
affine = _module
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
_utils = _module
test_affine = _module
test_channel = _module
test_crop = _module
test_device = _module
test_intensity = _module
test_spatial = _module
test_tensor = _module
test_utility = _module
test_abstract_transform = _module
test_affine = _module
test_channel = _module
test_compose = _module
test_crop = _module
test_format_transforms = _module
test_intensity_transforms = _module
test_kernel_transforms = _module
test_spatial_transforms = _module
test_tensor = _module
test_utility_transforms = _module
test_affine = _module
test_checktype = _module
versioneer = _module

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


import collections.abc


from typing import Any


import numpy as np


import torch


import logging


from functools import partial


from typing import Sequence


from typing import Callable


from typing import Union


from typing import List


from typing import Iterator


from typing import Generator


from typing import Optional


from torch.multiprocessing import Pool


from torch.utils.data import Dataset as TorchDset


from torch.utils.data import Subset


import collections


import warnings


from typing import Mapping


from torch.utils.data import DataLoader as _DataLoader


from torch.utils.data import Sampler


from torch.utils.data import Dataset


from torch.utils.data._utils.collate import default_convert


from torch.utils.data.dataloader import _SingleProcessDataLoaderIter as __SingleProcessDataLoaderIter


from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter as __MultiProcessingDataLoaderIter


from abc import abstractmethod


from torch.distributions import Distribution as TorchDistribution


from typing import Tuple


from random import shuffle


from torch import Tensor


import random


from typing import Hashable


import math


from itertools import combinations


from torch.multiprocessing import Value


from typing import Dict


import itertools


from math import pi


from collections import namedtuple


from math import isclose


from copy import deepcopy


import copy


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


class ContinuousParameter(AbstractParameter):
    """Class to perform parameter sampling from torch distributions"""

    def __init__(self, distribution: TorchDistribution):
        """
        Args:
            distribution : the distribution to sample from
        """
        super().__init__()
        self.dist = distribution

    def sample(self, n_samples: int) ->torch.Tensor:
        """
        Samples from the internal distribution

        Args:
            n_samples : the number of elements to sample

        Returns
            torch.Tensor: samples
        """
        return self.dist.sample((n_samples,))


class NormalParameter(ContinuousParameter):
    """
    Samples Parameters from a normal distribution.
    For details have a look at :class:`torch.distributions.Normal`
    """

    def __init__(self, mu: Union[float, torch.Tensor], sigma: Union[float, torch.Tensor]):
        """
        Args:
            mu : the distributions mean
            sigma : the distributions standard deviation
        """
        super().__init__(torch.distributions.Normal(loc=mu, scale=sigma))


class UniformParameter(ContinuousParameter):
    """
    Samples Parameters from a uniform distribution.
    For details have a look at :class:`torch.distributions.Uniform`
    """

    def __init__(self, low: Union[float, torch.Tensor], high: Union[float, torch.Tensor]):
        """
        Args:
            low : the lower range (inclusive)
            high : the higher range (exclusive)
        """
        super().__init__(torch.distributions.Uniform(low=low, high=high))


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


augment_callable = Callable[[torch.Tensor], Any]


class BaseTransform(AbstractTransform):
    """
    Transform to apply a functional interface to given keys

    .. warning:: This transform should not be used
        with functions which have randomness build in because it will
        result in different augmentations per key.
    """

    def __init__(self, augment_fn: augment_callable, *args, keys: Sequence=('data',), grad: bool=False, property_names: Sequence[str]=(), **kwargs):
        """
        Args:
            augment_fn: function for augmentation
            *args: positional arguments passed to augment_fn
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            property_names: a tuple containing all the properties to call
                during forward pass
            **kwargs: keyword arguments passed to augment_fn
        """
        sampler_vals = [kwargs.pop(name) for name in property_names]
        super().__init__(grad=grad, **kwargs)
        self.augment_fn = augment_fn
        self.keys = keys
        self.property_names = property_names
        self.args = args
        self.kwargs = kwargs
        for name, val in zip(property_names, sampler_vals):
            self.register_sampler(name, val)

    def forward(self, **data) ->dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        kwargs = {}
        for k in self.property_names:
            kwargs[k] = getattr(self, k)
        kwargs.update(self.kwargs)
        for _key in self.keys:
            data[_key] = self.augment_fn(data[_key], *self.args, **kwargs)
        return data


class BaseTransformSeeded(BaseTransform):
    """
    Transform to apply a functional interface to given keys and use the same
    pytorch(!) seed for every key.
    """

    def forward(self, **data) ->dict:
        """
        Apply transformation and use same seed for every key

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        kwargs = {}
        for k in self.property_names:
            kwargs[k] = getattr(self, k)
        kwargs.update(self.kwargs)
        seed = torch.random.get_rng_state()
        for _key in self.keys:
            torch.random.set_rng_state(seed)
            data[_key] = self.augment_fn(data[_key], *self.args, **kwargs)
        return data


class PerSampleTransform(BaseTransform):
    """
    Apply transformation to each sample in batch individually
    :attr:`augment_fn` must be callable with option :attr:`out`
    where results are saved in.

    .. warning:: This transform should not be used
        with functions which have randomness build in because it will
        result in different augmentations per sample and key.
    """

    def forward(self, **data) ->dict:
        """
        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        kwargs = {}
        for k in self.property_names:
            kwargs[k] = getattr(self, k)
        kwargs.update(self.kwargs)
        for _key in self.keys:
            out = torch.empty_like(data[_key])
            for _i in range(data[_key].shape[0]):
                out[_i] = self.augment_fn(data[_key][_i], out=out[_i], **kwargs)
            data[_key] = out
        return data


class PerChannelTransform(BaseTransform):
    """
    Apply transformation per channel (but still to whole batch)

    .. warning:: This transform should not be used
        with functions which have randomness build in because it will
        result in different augmentations per channel and key.
    """

    def __init__(self, augment_fn: augment_callable, per_channel: bool=False, keys: Sequence=('data',), grad: bool=False, property_names: Tuple[str]=(), **kwargs):
        """
        Args:
            augment_fn: function for augmentation
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=augment_fn, keys=keys, grad=grad, property_names=property_names, **kwargs)
        self.per_channel = per_channel

    def forward(self, **data) ->dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        if self.per_channel:
            kwargs = {}
            for k in self.property_names:
                kwargs[k] = getattr(self, k)
            kwargs.update(self.kwargs)
            for _key in self.keys:
                out = torch.empty_like(data[_key])
                for _i in range(data[_key].shape[1]):
                    out[:, (_i)] = self.augment_fn(data[_key][:, (_i)], out=out[:, (_i)], **kwargs)
                data[_key] = out
            return data
        else:
            return super().forward(**data)


def matrix_to_cartesian(batch: torch.Tensor, keep_square: bool=False) ->torch.Tensor:
    """
    Transforms a matrix for a homogeneous transformation back to cartesian
    coordinates.

    Args:
        batch: the batch oif matrices to convert back
        keep_square: if False: returns a NDIM x NDIM+1 matrix to keep the
            translation part
            if True: returns a NDIM x NDIM matrix but looses the translation
            part. defaults to False.

    Returns:
        torch.Tensor: the given matrix in cartesian coordinates

    """
    batch = batch[:, :-1, (...)]
    if keep_square:
        batch = batch[(...), :-1]
    return batch


def matrix_to_homogeneous(batch: torch.Tensor) ->torch.Tensor:
    """
    Transforms a given transformation matrix to a homogeneous
    transformation matrix.

    Args:
        batch: the batch of matrices to convert [N, dim, dim]

    Returns:
        torch.Tensor: the converted batch of matrices

    """
    if batch.size(-1) == batch.size(-2):
        missing = batch.new_zeros(size=(*batch.shape[:-1], 1))
        batch = torch.cat([batch, missing], dim=-1)
    missing = torch.zeros((batch.size(0), *[(1) for tmp in batch.shape[1:-1]], batch.size(-1)), device=batch.device, dtype=batch.dtype)
    missing[..., -1] = 1
    return torch.cat([batch, missing], dim=-2)


def points_to_cartesian(batch: torch.Tensor) ->torch.Tensor:
    """
    Transforms a batch of points in homogeneous coordinates back to cartesian
    coordinates.

    Args:
        batch: batch of points in homogeneous coordinates. Should be of shape
            BATCHSIZE x NUMPOINTS x NDIM+1

    Returns:
        torch.Tensor: the batch of points in cartesian coordinates

    """
    return batch[(...), :-1] / batch[..., -1, None]


def points_to_homogeneous(batch: torch.Tensor) ->torch.Tensor:
    """
    Transforms points from cartesian to homogeneous coordinates

    Args:
        batch: the batch of points to transform. Should be of shape
            BATCHSIZE x NUMPOINTS x DIM.

    Returns:
        torch.Tensor: the batch of points in homogeneous coordinates

    """
    return torch.cat([batch, batch.new_ones((*batch.size()[:-1], 1))], dim=-1)


def affine_point_transform(point_batch: torch.Tensor, matrix_batch: torch.Tensor) ->torch.Tensor:
    """
    Function to perform an affine transformation onto point batches

    Args:
        point_batch: a point batch of shape [N, NP, NDIM]
            ``NP`` is the number of points,
            ``N`` is the batch size,
            ``NDIM`` is the number of spatial dimensions
        matrix_batch : torch.Tensor
            a batch of affine matrices with shape [N, NDIM, NDIM + 1],
            N is the batch size and NDIM is the number of spatial dimensions

    Returns:
        torch.Tensor: the batch of transformed points in cartesian coordinates)
            [N, NP, NDIM] ``NP`` is the number of points, ``N`` is the
            batch size, ``NDIM`` is the number of spatial dimensions
    """
    point_batch = points_to_homogeneous(point_batch)
    matrix_batch = matrix_to_homogeneous(matrix_batch)
    transformed_points = torch.bmm(point_batch, matrix_batch.permute(0, 2, 1))
    return points_to_cartesian(transformed_points)


def check_scalar(x: Union[Any, float, int]) ->bool:
    """
    Provide interface to check for scalars

    Args:
        x: object to check for scalar

    Returns:
        bool" True if input is scalar
    """
    return isinstance(x, (int, float)) or isinstance(x, torch.Tensor) and x.numel() == 1


def unit_box(n: int, scale: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
    Create a (scaled) version of a unit box

    Args:
        n: number of dimensions
        scale: scaling of each dimension

    Returns:
        torch.Tensor: scaled unit box
    """
    box = torch.tensor([list(i) for i in itertools.product([0, 1], repeat=n)])
    if scale is not None:
        box = box * scale[None]
    return box


def _check_new_img_size(curr_img_size, matrix: torch.Tensor, zero_border: bool=False) ->torch.Tensor:
    """
    Calculates the image size so that the whole image content fits the image.
    The resulting size will be the maximum size of the batch, so that the
    images can remain batched.

    Args:
        curr_img_size: the size of the current image.
            If int, it will be used as size for all image dimensions
        matrix: a batch of affine matrices with shape [N, NDIM, NDIM+1]
        zero_border: whether or not to have a fixed image border at zero

    Returns:
        torch.Tensor: the new image size
    """
    n_dim = matrix.size(-1) - 1
    if check_scalar(curr_img_size):
        curr_img_size = [curr_img_size] * n_dim
    possible_points = unit_box(n_dim, torch.tensor(curr_img_size))
    transformed_edges = affine_point_transform(possible_points[None].expand(matrix.size(0), *[(-1) for _ in possible_points.shape]).clone(), matrix)
    if zero_border:
        substr = 0
    else:
        substr = transformed_edges.min(1)[0]
    return (transformed_edges.max(1)[0] - substr).max(0)[0]


def matrix_revert_coordinate_order(batch: torch.Tensor) ->torch.Tensor:
    """
    Reverts the coordinate order of a matrix (e.g. from xyz to zyx).

    Args:
        batch: the batched transformation matrices; Should be of shape
            BATCHSIZE x NDIM x NDIM

    Returns:
        torch.Tensor: the matrix performing the same transformation on vectors with a
            reversed coordinate order
    """
    batch[:, :-1, :] = batch[:, :-1, :].flip(1).clone()
    batch[:, :-1, :-1] = batch[:, :-1, :-1].flip(2).clone()
    return batch


def affine_image_transform(image_batch: torch.Tensor, matrix_batch: torch.Tensor, output_size: Optional[tuple]=None, adjust_size: bool=False, interpolation_mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False, reverse_order: bool=False) ->torch.Tensor:
    """
    Performs an affine transformation on a batch of images

    Args:
        image_batch: the batch to transform. Should have shape of [N, C, NDIM]
        matrix_batch: a batch of affine matrices with shape [N, NDIM, NDIM+1]
        output_size: if given, this will be the resulting image size.
            Defaults to ``None``
        adjust_size: if True, the resulting image size will be calculated
            dynamically to ensure that the whole image fits.
        interpolation_mode: interpolation mode to calculate output values
            'bilinear' | 'nearest'. Default: 'bilinear'
        padding_mode: padding mode for outside grid values
            'zeros' | 'border' | 'reflection'. Default: 'zeros'
        align_corners:  Geometrically, we consider the pixels of the input as
            squares rather than points.
            If set to True, the extrema (-1 and 1) are
            considered as referring to the center points of the input’s corner
            pixels. If set to False, they are instead considered as referring
            to the corner points of the input’s corner pixels,
            making the sampling more resolution agnostic.

    Returns:
        torch.Tensor: transformed image

    Warnings:
        When align_corners = True, the grid positions depend on the pixel size
        relative to the input image size, and so the locations sampled by
        grid_sample() will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).

    Notes:
        :attr:`output_size` and :attr:`adjust_size` are mutually exclusive.
        If None of them is set, the resulting image will have the same size
        as the input image.
    """
    if len(matrix_batch.shape) < 3:
        matrix_batch = matrix_batch[None, ...].expand(image_batch.size(0), -1, -1).clone()
    image_size = image_batch.shape[2:]
    if output_size is not None:
        if check_scalar(output_size):
            output_size = tuple([output_size] * matrix_batch.size(-2))
        if adjust_size:
            warnings.warn('Adjust size is mutually exclusive with a given output size.', UserWarning)
        new_size = output_size
    elif adjust_size:
        new_size = tuple([int(tmp.item()) for tmp in _check_new_img_size(image_size, matrix_batch)])
    else:
        new_size = image_size
    if len(image_size) < len(image_batch.shape):
        missing_dims = len(image_batch.shape) - len(image_size)
        new_size = *image_batch.shape[:missing_dims], *new_size
    matrix_batch = matrix_batch
    if reverse_order:
        matrix_batch = matrix_revert_coordinate_order(matrix_batch)
    grid = torch.nn.functional.affine_grid(matrix_batch, size=new_size, align_corners=align_corners)
    return torch.nn.functional.grid_sample(image_batch, grid, mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners)


class Affine(BaseTransform):
    """
    Class Performing an Affine Transformation on a given sample dict.
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`.
    """

    def __init__(self, matrix: Optional[Union[torch.Tensor, Sequence[Sequence[float]]]]=None, keys: Sequence=('data',), grad: bool=False, output_size: Optional[tuple]=None, adjust_size: bool=False, interpolation_mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False, reverse_order: bool=False, per_sample: bool=True, **kwargs):
        """
        Args:
            matrix: if given, overwrites the parameters for :attr:`scale`,
                :attr:rotation` and :attr:`translation`.
                Should be a matrix of shape [(BATCHSIZE,) NDIM, NDIM(+1)]
                This matrix represents the whole transformation matrix
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be calculated
                dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode:padding mode for outside grid values
                ``'zeros``' | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            per_sample: sample different values for each element in the batch.
                The transform is still applied in a batched wise fashion.
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        super().__init__(augment_fn=affine_image_transform, keys=keys, grad=grad, **kwargs)
        self.matrix = matrix
        self.register_sampler('output_size', output_size)
        self.adjust_size = adjust_size
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.reverse_order = reverse_order
        self.per_sample = per_sample

    def assemble_matrix(self, **data) ->torch.Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Args:
            **data: the data to be transformed. Will be used to determine
                batchsize, dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix
        """
        if self.matrix is None:
            raise ValueError('Matrix needs to be initialized or overwritten.')
        if not torch.is_tensor(self.matrix):
            self.matrix = torch.tensor(self.matrix)
        self.matrix = self.matrix
        batchsize = data[self.keys[0]].shape[0]
        ndim = len(data[self.keys[0]].shape) - 2
        if len(self.matrix.shape) == 2:
            self.matrix = self.matrix[None].expand(batchsize, -1, -1).clone()
        if self.matrix.shape == (batchsize, ndim, ndim + 1):
            return self.matrix
        elif self.matrix.shape == (batchsize, ndim, ndim):
            return matrix_to_homogeneous(self.matrix)[:, :-1]
        elif self.matrix.shape == (batchsize, ndim + 1, ndim + 1):
            return matrix_to_cartesian(self.matrix)
        raise ValueError('Invalid Shape for affine transformation matrix. Got %s but expected %s' % (str(tuple(self.matrix.shape)), str((batchsize, ndim, ndim + 1))))

    def forward(self, **data) ->dict:
        """
        Assembles the matrix and applies it to the specified sample-entities.

        Args:
            **data: the data to transform

        Returns:
            dict: dictionary containing the transformed data
        """
        matrix = self.assemble_matrix(**data)
        for key in self.keys:
            data[key] = self.augment_fn(data[key], matrix_batch=matrix, output_size=self.output_size, adjust_size=self.adjust_size, interpolation_mode=self.interpolation_mode, padding_mode=self.padding_mode, align_corners=self.align_corners, reverse_order=self.reverse_order, **self.kwargs)
        return data

    def __add__(self, other: Any) ->BaseTransform:
        """
        Makes ``trafo + other_trafo work``
        (stacks them for dynamic assembling)

        Args:
            other: the other transformation

        Returns:
            StackedAffine: a stacked affine transformation
        """
        if not isinstance(other, Affine):
            other = Affine(matrix=other, keys=self.keys, grad=self.grad, output_size=self.output_size, adjust_size=self.adjust_size, interpolation_mode=self.interpolation_mode, padding_mode=self.padding_mode, align_corners=self.align_corners, **self.kwargs)
        return StackedAffine(self, other, keys=self.keys, grad=self.grad, output_size=self.output_size, adjust_size=self.adjust_size, interpolation_mode=self.interpolation_mode, padding_mode=self.padding_mode, align_corners=self.align_corners, **self.kwargs)

    def __radd__(self, other) ->BaseTransform:
        """
        Makes ``other_trafo + trafo`` work
        (stacks them for dynamic assembling)

        Args:
            other: the other transformation

        Returns:
            StackedAffine: a stacked affine transformation
        """
        if not isinstance(other, Affine):
            other = Affine(matrix=other, keys=self.keys, grad=self.grad, output_size=self.output_size, adjust_size=self.adjust_size, interpolation_mode=self.interpolation_mode, padding_mode=self.padding_mode, align_corners=self.align_corners, **self.kwargs)
        return StackedAffine(other, self, grad=other.grad, output_size=other.output_size, adjust_size=other.adjust_size, interpolation_mode=other.interpolation_mode, padding_mode=other.padding_mode, align_corners=other.align_corners, **other.kwargs)


AffineParamType = Union[int, Sequence[int], float, Sequence[float], torch.Tensor, AbstractParameter, Sequence[AbstractParameter]]


def create_rotation_2d(sin: Tensor, cos: Tensor) ->Tensor:
    """
    Create a 2d rotation matrix

   Args:
    sin: sin value to use for rotation matrix, [1]
    cos: cos value to use for rotation matrix, [1]

    Returns:
        torch.Tensor: rotation matrix, [2, 2]
    """
    return torch.tensor([[cos.clone(), -sin.clone()], [sin.clone(), cos.clone()]], device=sin.device, dtype=sin.dtype)


def create_rotation_3d_0(sin: Tensor, cos: Tensor) ->Tensor:
    """
    Create a rotation matrix around the zero-th axis

    Args:
        sin: sin value to use for rotation matrix, [1]
        cos: cos value to use for rotation matrix, [1]

    Returns:
        torch.Tensor: rotation matrix, [3, 3]
    """
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, cos.clone(), -sin.clone()], [0.0, sin.clone(), cos.clone()]], device=sin.device, dtype=sin.dtype)


def create_rotation_3d_1(sin: Tensor, cos: Tensor) ->Tensor:
    """
    Create a rotation matrix around the first axis

    Args:
        sin: sin value to use for rotation matrix, [1]
        cos: cos value to use for rotation matrix, [1]

    Returns:
        torch.Tensor: rotation matrix, [3, 3]
    """
    return torch.tensor([[cos.clone(), 0.0, sin.clone()], [0.0, 1.0, 0.0], [-sin.clone(), 0.0, cos.clone()]], device=sin.device, dtype=sin.dtype)


def create_rotation_3d_2(sin: Tensor, cos: Tensor) ->Tensor:
    """
    Create a rotation matrix around the second axis

    Args:
        sin: sin value to use for rotation matrix, [1]
        cos: cos value to use for rotation matrix, [1]

    Returns:
        torch.Tensor: rotation matrix, [3, 3]
    """
    return torch.tensor([[cos.clone(), -sin.clone(), 0.0], [sin.clone(), cos.clone(), 0.0], [0.0, 0.0, 1.0]], device=sin.device, dtype=sin.dtype)


def create_rotation_3d(sin: Tensor, cos: Tensor) ->Tensor:
    """
    Create a 3d rotation matrix which sequentially applies the rotation
    around axis (rot axis 0 -> rot axis 1 -> rot axis 2)

    Args:
        sin: sin values to use for the rotation, (axis 0, axis 1, axis 2)[3]
        cos: cos values to use for the rotation, (axis 0, axis 1, axis 2)[3]

    Returns:
        torch.Tensor: rotation matrix, [3, 3]
    """
    rot_0 = create_rotation_3d_0(sin[0], cos[0])
    rot_1 = create_rotation_3d_1(sin[1], cos[1])
    rot_2 = create_rotation_3d_2(sin[2], cos[2])
    return rot_2 @ (rot_1 @ rot_0)


def deg_to_rad(angles: Union[torch.Tensor, float, int]) ->Union[torch.Tensor, float, int]:
    """
    Converts from degree to radians.

    Args:
        angles: the (vectorized) angles to convert

    Returns:
        torch.Tensor: the transformed (vectorized) angles

    """
    return angles * pi / 180


def expand_scalar_param(param: AffineParamType, batchsize: int, ndim: int) ->Tensor:
    """
    Bring affine params to shape (batchsize, ndim)

    Args:
        param: affine parameter
        batchsize: size of batch
        ndim: number of spatial dimensions

    Returns:
        torch.Tensor: affine params in correct shape
    """
    if check_scalar(param):
        return torch.tensor([[param] * ndim] * batchsize).float()
    if not torch.is_tensor(param):
        param = torch.tensor(param)
    else:
        param = param.clone()
    if not param.ndimension() == 2:
        if param.shape[0] == ndim:
            param = param.reshape(1, -1).expand(batchsize, ndim)
        elif param.shape[0] == batchsize:
            param = param.reshape(-1, 1).expand(batchsize, ndim)
        else:
            raise ValueError(f'Unknown param for expanding. Found {param} for batchsize {batchsize} and ndim {ndim}')
    assert all([(i == j) for i, j in zip(param.shape, (batchsize, ndim))]), f'Affine param need to have shape (batchsize, ndim)({batchsize, ndim}) but found {param.shape}'
    return param.float()


def create_rotation(rotation: AffineParamType, batchsize: int, ndim: int, degree: bool=False, device: Optional[Union[torch.device, str]]=None, dtype: Optional[Union[torch.dtype, str]]=None) ->torch.Tensor:
    """
    Formats the given scale parameters to a homogeneous transformation matrix

    Args:
        rotation: the rotation factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
            for all dimensions and batch samples
            * a parameter per sample, which will be
            replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
            batch samples
            * a parameter per sampler per dimension
            * None will be treated as a rotation angle of 0
        batchsize: the number of samples per batch
        ndim : the dimensionality of the transform
        degree: whether the given rotation(s) are in degrees.
            Only valid for rotation parameters, which aren't passed as full
            transformation matrix.
        device: the device to put the resulting tensor to.
            Defaults to the torch default device
        dtype: the dtype of the resulting trensor.
            Defaults to the torch default dtype

    Returns:
        torch.Tensor: the homogeneous transformation matrix
            [N, NDIM + 1, NDIM + 1], N is the batch size and NDIM
            is the number of spatial dimensions

    """
    if rotation is None:
        rotation = 0
    num_rot_params = 1 if ndim == 2 else ndim
    rotation = expand_scalar_param(rotation, batchsize, num_rot_params)
    if degree:
        rotation = deg_to_rad(rotation)
    matrix_fn = create_rotation_2d if ndim == 2 else create_rotation_3d
    sin, cos = torch.sin(rotation), torch.cos(rotation)
    rotation_matrix = torch.stack([matrix_fn(s, c) for s, c in zip(sin, cos)])
    return matrix_to_homogeneous(rotation_matrix)


def get_batched_eye(batchsize: int, ndim: int, device: Optional[Union[torch.device, str]]=None, dtype: Optional[Union[torch.dtype, str]]=None) ->torch.Tensor:
    """
    Produces a batched matrix containing 1s on the diagonal

    Args:
        batchsize : int
            the batchsize (first dimension)
        ndim : int
            the dimensionality of the eyes (second and third dimension)
        device : torch.device, str, optional
            the device to put the resulting tensor to. Defaults to the default
            device
        dtype : torch.dtype, str, optional
            the dtype of the resulting trensor. Defaults to the default dtype

    Returns:
        torch.Tensor: batched eye matrix

    """
    return torch.eye(ndim, device=device, dtype=dtype).view(1, ndim, ndim).expand(batchsize, -1, -1).clone()


def create_scale(scale: AffineParamType, batchsize: int, ndim: int, device: Optional[Union[torch.device, str]]=None, dtype: Optional[Union[torch.dtype, str]]=None, image_transform: bool=True) ->torch.Tensor:
    """
    Formats the given scale parameters to a homogeneous transformation matrix

    Args:
        scale : the scale factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
            for all dimensions and batch samples
            * a parameter per sample, which will be
            replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
            batch samples
            * a parameter per sampler per dimension
            * None will be treated as a scaling factor of 1
        batchsize: the number of samples per batch
        ndim: the dimensionality of the transform
        device: the device to put the resulting tensor to.
            Defaults to the torch default device
        dtype: the dtype of the resulting trensor.
            Defaults to the torch default dtype
        image_transform:  inverts the scale matrix to match expected behavior
            when applied to an image, e.g. scale>1 increases the size of an
            image but decrease the size of an grid

    Returns:
        torch.Tensor: the homogeneous transformation matrix
            [N, NDIM + 1, NDIM + 1], N is the batch size and NDIM is the
            number of spatial dimensions
    """
    if scale is None:
        scale = 1
    scale = expand_scalar_param(scale, batchsize, ndim)
    if image_transform:
        scale = 1 / scale
    scale_matrix = torch.stack([(eye * s) for eye, s in zip(get_batched_eye(batchsize=batchsize, ndim=ndim, device=device, dtype=dtype), scale)])
    return matrix_to_homogeneous(scale_matrix)


def create_translation(offset: AffineParamType, batchsize: int, ndim: int, device: Optional[Union[torch.device, str]]=None, dtype: Optional[Union[torch.dtype, str]]=None, image_transform: bool=True) ->torch.Tensor:
    """
    Formats the given translation parameters to a homogeneous transformation
    matrix

    Args:
        offset: the translation offset(s). Supported are:
            * a single parameter (as float or int), which will be replicated
            for all dimensions and batch samples
            * a parameter per sample, which will be
            replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
            batch samples
            * a parameter per sampler per dimension
            * None will be treated as a translation offset of 0
        batchsize: the number of samples per batch
        ndim: the dimensionality of the transform
        device: the device to put the resulting tensor to.
            Defaults to the torch default device
        dtype: the dtype of the resulting trensor.
            Defaults to the torch default dtype
        image_transform: bool
            inverts the translation matrix to match expected behavior when
            applied to an image, e.g. translation > 0 should move the image
            in the positive direction of an axis but the grid in the negative
            direction

    Returns:
        torch.Tensor: the homogeneous transformation matrix [N, NDIM + 1, NDIM + 1],
            N is the batch size and NDIM is the number of spatial dimensions
    """
    if offset is None:
        offset = 0
    offset = expand_scalar_param(offset, batchsize, ndim)
    eye_batch = get_batched_eye(batchsize=batchsize, ndim=ndim, device=device, dtype=dtype)
    translation_matrix = torch.stack([torch.cat([eye, o.view(-1, 1)], dim=1) for eye, o in zip(eye_batch, offset)])
    if image_transform:
        translation_matrix[..., -1] = -translation_matrix[..., -1]
    return matrix_to_homogeneous(translation_matrix)


def parametrize_matrix(scale: AffineParamType, rotation: AffineParamType, translation: AffineParamType, batchsize: int, ndim: int, degree: bool=False, device: Optional[Union[torch.device, str]]=None, dtype: Optional[Union[torch.dtype, str]]=None, image_transform: bool=True) ->torch.Tensor:
    """
    Formats the given scale parameters to a homogeneous transformation matrix

    Args:
        scale: the scale factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
            for all dimensions and batch samples
            * a parameter per sample, which will be
            replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
            batch samples
            * a parameter per sampler per dimension
            * None will be treated as a scaling factor of 1
        rotation: the rotation factor(s). Supported are:
            * a single parameter (as float or int), which will be replicated
            for all dimensions and batch samples
            * a parameter per sample, which will be
            replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
            batch samples
            * a parameter per sampler per dimension
            * None will be treated as a rotation factor of 1
        translation: the translation offset(s). Supported are:
            * a single parameter (as float or int), which will be replicated
            for all dimensions and batch samples
            * a parameter per sample, which will be
            replicated for all dimensions
            * a parameter per dimension, which will be replicated for all
            batch samples
            * a parameter per sampler per dimension
            * None will be treated as a translation offset of 0
        batchsize: the number of samples per batch
        ndim: the dimensionality of the transform
        degree: whether the given rotation(s) are in degrees.
            Only valid for rotation parameters, which aren't passed as full
            transformation matrix.
        device: the device to put the resulting tensor to.
            Defaults to the torch default device
        dtype: the dtype of the resulting trensor.
            Defaults to the torch default dtype
        image_transform: bool
            adjusts transformation matrices such that they match the expected
            behavior on images (see :func:`create_scale` and
            :func:`create_translation` for more info)

    Returns:
        torch.Tensor: the transformation matrix [N, NDIM, NDIM+1], ``N`` is
            the batch size and ``NDIM`` is the number of spatial dimensions
    """
    scale = create_scale(scale, batchsize=batchsize, ndim=ndim, device=device, dtype=dtype, image_transform=image_transform)
    rotation = create_rotation(rotation, batchsize=batchsize, ndim=ndim, degree=degree, device=device, dtype=dtype)
    translation = create_translation(translation, batchsize=batchsize, ndim=ndim, device=device, dtype=dtype, image_transform=image_transform)
    return torch.bmm(torch.bmm(scale, rotation), translation)[:, :-1]


class BaseAffine(Affine):
    """
    Class performing a basic Affine Transformation on a given sample dict.
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`."""

    def __init__(self, scale: Optional[AffineParamType]=None, rotation: Optional[AffineParamType]=None, translation: Optional[AffineParamType]=None, degree: bool=False, image_transform: bool=True, keys: Sequence=('data',), grad: bool=False, output_size: Optional[tuple]=None, adjust_size: bool=False, interpolation_mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False, reverse_order: bool=False, per_sample: bool=True, **kwargs):
        """
        Args:
            scale: the scale factor(s). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * None will be treated as a scaling factor of 1
            rotation: the rotation factor(s). The rotation is performed in
                consecutive order axis0 -> axis1 (-> axis 2). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * None will be treated as a rotation angle of 0
            translation : torch.Tensor, int, float
                the translation offset(s) relative to image (should be in the
                range [0, 1]). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * None will be treated as a translation offset of 0
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            degree: whether the given rotation(s) are in degrees.
                Only valid for rotation parameters, which aren't passed
                as full transformation matrix.
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be
                calculated dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            per_sample: sample different values for each element in the batch.
                The transform is still applied in a batched wise fashion.
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        super().__init__(keys=keys, grad=grad, output_size=output_size, adjust_size=adjust_size, interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners, reverse_order=reverse_order, per_sample=per_sample, **kwargs)
        self.register_sampler('scale', scale)
        self.register_sampler('rotation', rotation)
        self.register_sampler('translation', translation)
        self.degree = degree
        self.image_transform = image_transform

    def assemble_matrix(self, **data) ->torch.Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Args:
            **data: the data to be transformed. Will be used to determine
                batchsize, dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix
        """
        batchsize = data[self.keys[0]].shape[0]
        ndim = len(data[self.keys[0]].shape) - 2
        device = data[self.keys[0]].device
        dtype = data[self.keys[0]].dtype
        self.matrix = parametrize_matrix(scale=self.sample_for_batch('scale', batchsize), rotation=self.sample_for_batch('rotation', batchsize), translation=self.sample_for_batch('translation', batchsize), batchsize=batchsize, ndim=ndim, degree=self.degree, device=device, dtype=dtype, image_transform=self.image_transform)
        return self.matrix

    def sample_for_batch(self, name: str, batchsize: int) ->Optional[Union[Any, Sequence[Any]]]:
        """
        Sample elements for batch

        Args:
            name: name of parameter
            batchsize: batch size

        Returns:
            Optional[Union[Any, Sequence[Any]]]: sampled elements
        """
        elem = getattr(self, name)
        if elem is not None and self.per_sample:
            return [elem] + [getattr(self, name) for _ in range(batchsize - 1)]
        else:
            return elem


class Rotate(BaseAffine):
    """
    Class Performing a Rotation-OnlyAffine Transformation on a given
    sample dict. The rotation is applied in consecutive order:
    rot axis 0 -> rot axis 1 -> rot axis 2
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`.
    """

    def __init__(self, rotation: AffineParamType, keys: Sequence=('data',), grad: bool=False, degree: bool=False, output_size: Optional[tuple]=None, adjust_size: bool=False, interpolation_mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False, reverse_order: bool=False, **kwargs):
        """
        Args:
            rotation: the rotation factor(s). The rotation is performed in
                consecutive order axis0 -> axis1 (-> axis 2). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * ``None`` will be treated as a rotation angle of 0
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            degree: whether the given rotation(s) are in degrees.
                Only valid for rotation parameters, which aren't passed
                as full transformation matrix.
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be
                calculated dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        super().__init__(scale=None, rotation=rotation, translation=None, matrix=None, keys=keys, grad=grad, degree=degree, output_size=output_size, adjust_size=adjust_size, interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners, reverse_order=reverse_order, **kwargs)


class Translate(BaseAffine):
    """
    Class Performing an Translation-Only
    Affine Transformation on a given sample dict.
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`.
    """

    def __init__(self, translation: AffineParamType, keys: Sequence=('data',), grad: bool=False, output_size: Optional[tuple]=None, adjust_size: bool=False, interpolation_mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False, unit: str='pixel', reverse_order: bool=False, **kwargs):
        """
        Args:
            translation : torch.Tensor, int, float
                the translation offset(s) relative to image (should be in the
                range [0, 1]). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for all
                batch samples
                * None will be treated as a translation offset of 0
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be
                calculated dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            unit: defines the unit of the translation. Either ```relative'``
                to the image size or in ```pixel'``
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the
                affine transform
        """
        super().__init__(scale=None, rotation=None, translation=translation, matrix=None, keys=keys, grad=grad, degree=False, output_size=output_size, adjust_size=adjust_size, interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners, reverse_order=reverse_order, **kwargs)
        self.unit = unit

    def assemble_matrix(self, **data) ->torch.Tensor:
        """
        Assembles the matrix (and takes care of batching and having it on the
        right device and in the correct dtype and dimensionality).

        Args:
            **data: the data to be transformed. Will be used to determine
                batchsize, dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix [N, NDIM, NDIM]
        """
        matrix = super().assemble_matrix(**data)
        if self.unit.lower() == 'pixel':
            img_size = torch.tensor(data[self.keys[0]].shape[2:])
            matrix[..., -1] = matrix[..., -1] / img_size
        return matrix


class Scale(BaseAffine):
    """Class Performing a Scale-Only Affine Transformation on a given
    sample dict.
    The transformation will be applied to all the dict-entries specified
    in :attr:`keys`.
    """

    def __init__(self, scale: AffineParamType, keys: Sequence=('data',), grad: bool=False, output_size: Optional[tuple]=None, adjust_size: bool=False, interpolation_mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False, reverse_order: bool=False, **kwargs):
        """
        Args:
            scale : torch.Tensor, int, float, optional
                the scale factor(s). Supported are:
                * a single parameter (as float or int), which will be
                replicated for all dimensions and batch samples
                * a parameter per dimension, which will be replicated for
                all batch samples
                * None will be treated as a scaling factor of 1
            keys: Sequence
                keys which should be augmented
            grad: bool
                enable gradient computation inside transformation
            degree : bool
                whether the given rotation(s) are in degrees.
                Only valid for rotation parameters, which aren't passed as full
                transformation matrix.
            output_size : Iterable
                if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size : bool
                if True, the resulting image size will be calculated
                dynamically to ensure that the whole image fits.
            interpolation_mode : str
                interpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'
            padding_mode :
                padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners : bool
                Geometrically, we consider the pixels of the input as
                squares rather than points. If set to True, the extrema
                (-1 and 1) are considered as referring to the center points of
                the input’s corner pixels. If set to False, they are instead
                considered as referring to the corner points of the input’s
                corner pixels, making the sampling more resolution agnostic.
            reverse_order: bool
                reverses the coordinate order of the transformation to conform
                to the pytorch convention: transformation params order
                [W,H(,D)] and batch order [(D,)H,W]
            **kwargs :
                additional keyword arguments passed to the affine transform
        """
        super().__init__(scale=scale, rotation=None, translation=None, matrix=None, keys=keys, grad=grad, degree=False, output_size=output_size, adjust_size=adjust_size, interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners, reverse_order=reverse_order, **kwargs)


class Resize(Scale):

    def __init__(self, size: Union[int, Tuple[int]], keys: Sequence=('data',), grad: bool=False, interpolation_mode: str='bilinear', padding_mode: str='zeros', align_corners: bool=False, reverse_order: bool=False, **kwargs):
        """
        Class Performing a Resizing Affine Transformation on a given
        sample dict.
        The transformation will be applied to all the dict-entries specified
        in :attr:`keys`.

        Args:
            size: the target size. If int, this will be repeated for all the
                dimensions
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            interpolation_mode: nterpolation mode to calculate output values
                'bilinear' | 'nearest'. Default: 'bilinear'
            padding_mode: padding mode for outside grid values
                'zeros' | 'border' | 'reflection'. Default: 'zeros'
            align_corners: Geometrically, we consider the pixels of the input
                as squares rather than points. If set to True, the extrema
                (-1 and 1) are considered as referring to the center points of
                the input’s corner pixels. If set to False, they are instead
                considered as referring to the corner points of the input’s
                corner pixels, making the sampling more resolution agnostic.
            reverse_order: reverses the coordinate order of the transformation
                to conform to the pytorch convention: transformation params
                order [W,H(,D)] and batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the affine
                transform

        Notes:
            The offsets for shifting back and to origin are calculated on the
            entry matching the first item iin :attr:`keys` for each batch
        """
        super().__init__(output_size=None, scale=None, keys=keys, grad=grad, adjust_size=False, interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners, reverse_order=reverse_order, **kwargs)
        self.register_sampler('size', size)

    def assemble_matrix(self, **data) ->torch.Tensor:
        """
        Handles the matrix assembly and calculates the scale factors for
        resizing

        Args:
            **data: the data to be transformed. Will be used to determine
                batchsize, dimensionality, dtype and device

        Returns:
            torch.Tensor: the (batched) transformation matrix

        """
        curr_img_size = data[self.keys[0]].shape[2:]
        output_size = self.size
        if torch.is_tensor(output_size):
            self.output_size = int(output_size.item())
        else:
            self.output_size = tuple(int(t.item()) for t in output_size)
        if check_scalar(output_size):
            output_size = [output_size] * len(curr_img_size)
        self.scale = [(float(output_size[i]) / float(curr_img_size[i])) for i in range(len(curr_img_size))]
        matrix = super().assemble_matrix(**data)
        return matrix


def torch_one_hot(target: torch.Tensor, num_classes: Optional[int]=None) ->torch.Tensor:
    """
    Compute one hot encoding of input tensor

    Args:
        target: tensor to be converted
        num_classes: number of classes. If :attr:`num_classes` is None,
            the maximum of target is used

    Returns:
        torch.Tensor: one hot encoded tensor
    """
    if num_classes is None:
        num_classes = int(target.max().detach().item() + 1)
    dtype, device = target.dtype, target.device
    target_onehot = torch.zeros(*target.shape, num_classes, dtype=dtype, device=device)
    return target_onehot.scatter_(1, target.unsqueeze_(1), 1.0)


def one_hot_batch(target: torch.Tensor, num_classes: Optional[int]=None, dtype: Optional[torch.dtype]=None) ->torch.Tensor:
    """
    Compute one hot for input tensor (assumed to a be batch and thus saved
    into first dimension -> input should only have one channel)

    Args:
        target: long tensor to be converted
        num_classes: number of classes.
            If :attr:`num_classes` is None, the maximum of target is used
        dtype: optionally changes the dtype of the onehot encoding

    Returns:
        torch.Tensor: one hot encoded tensor
    """
    if target.dtype != torch.long:
        raise TypeError(f'Target tensor needs to be of type torch.long, found {target.dtype}')
    if target.ndim in [0, 1]:
        return torch_one_hot(target, num_classes)
    else:
        if num_classes is None:
            num_classes = int(target.max().detach().item() + 1)
        _dtype, device, shape = target.dtype, target.device, target.shape
        if dtype is None:
            dtype = _dtype
        target_onehot = torch.zeros(shape[0], num_classes, *shape[2:], dtype=dtype, device=device)
        return target_onehot.scatter_(1, target, 1.0)


class OneHot(BaseTransform):
    """
    Convert to one hot encoding. One hot encoding is applied in first dimension
    which results in shape N x NumClasses x [same as input] while input is expected to
    have shape N x 1 x [arbitrary additional dimensions]
    """

    def __init__(self, num_classes: int, keys: Sequence=('seg',), dtype: Optional[torch.dtype]=None, grad: bool=False, **kwargs):
        """
        Args:
            num_classes: number of classes. If :attr:`num_classes` is None,
                the number of classes is automatically determined from the
                current batch (by using the max of the current batch and
                assuming a consecutive order from zero)
            dtype: optionally changes the dtype of the onehot encoding
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to :func:`one_hot_batch`

        Warnings:
            Input tensor needs to be of type torch.long. This could
            be achieved by applying `TenorOp("long", keys=("seg",))`.
        """
        super().__init__(augment_fn=one_hot_batch, keys=keys, grad=grad, num_classes=num_classes, dtype=dtype, **kwargs)


class ArgMax(BaseTransform):
    """
    Compute argmax along given dimension.
    Can be used to revert OneHot encoding.
    """

    def __init__(self, dim: int, keepdim: bool=True, keys: Sequence=('seg',), grad: bool=False, **kwargs):
        """
        Args:
            dim: dimension to apply argmax
            keepdim: whether the output tensor has dim retained or not
            dtype: optionally changes the dtype of the onehot encoding
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to :func:`one_hot_batch`

        Warnings
            The output of the argmax function is always a tensor of dtype long.
        """
        super().__init__(augment_fn=torch.argmax, keys=keys, grad=grad, dim=dim, keepdim=keepdim, **kwargs)


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


def dict_call(batch: dict, transform: Callable) ->Any:
    """
    Unpacks the dict for every transformation

    Args:
        batch: current batch which is passed to transform
        transform: transform to perform

    Returns:
        Any: transformed batch
    """
    return transform(**batch)


class Compose(AbstractTransform):
    """
    Compose multiple transforms
    """

    def __init__(self, *transforms: Union[AbstractTransform, Sequence[AbstractTransform]], shuffle: bool=False, transform_call: Callable[[Any, Callable], Any]=dict_call):
        """
        Args:
            transforms: one or multiple transformations which are applied
                in consecutive order
            shuffle: apply transforms in random order
            transform_call: function which determines how transforms are
                called. By default Mappings and Sequences are unpacked
                during the transform.

        """
        super().__init__(grad=True)
        if len(transforms) > 0 and isinstance(transforms[0], Sequence):
            transforms = transforms[0]
        if not transforms:
            raise ValueError('At least one transformation needs to be selected.')
        self.transforms = transforms
        self.transform_call = transform_call
        self.shuffle = shuffle

    def forward(self, *seq_like, **map_like) ->Union[Sequence, Mapping]:
        """
        Apply transforms in a consecutive order. Can either handle
        Sequence like or Mapping like data.

        Args:
            *seq_like: data which is unpacked like a Sequence
            **map_like: data which is unpacked like a dict

        Returns:
            Union[Sequence, Mapping]: transformed data
        """
        assert not (seq_like and map_like)
        assert len(self.transforms) == len(self.transform_order)
        data = seq_like if seq_like else map_like
        if self.shuffle:
            shuffle(self.transform_order)
        for idx in self.transform_order:
            data = self.transform_call(data, self.transforms[idx])
        return data

    @property
    def transforms(self) ->torch.nn.ModuleList:
        """
        Transforms getter

        Returns:
            torch.nn.ModuleList: transforms to compose
        """
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: Union[AbstractTransform, Sequence[AbstractTransform]]):
        """
        Transforms setter

        Args:
            transforms: one or multiple transformations which are applied in
                consecutive order

        """
        if isinstance(transforms, tuple):
            transforms = list(transforms)
        for idx, trafo in enumerate(transforms):
            if not isinstance(trafo, torch.nn.Module):
                transforms[idx] = _TransformWrapper(trafo)
        self._transforms = torch.nn.ModuleList(transforms)
        self.transform_order = list(range(len(self.transforms)))

    @property
    def shuffle(self) ->bool:
        """
        Getter for attribute shuffle

        Returns:
            bool: True if shuffle is enabled, False otherwise
        """
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle: bool):
        """
        Setter for shuffle

        Args:
            shuffle: new status of shuffle
        """
        self._shuffle = shuffle
        self.transform_order = list(range(len(self.transforms)))


class DropoutCompose(Compose):
    """
    Compose multiple transforms to one and randomly apply them
    """

    def __init__(self, *transforms: Union[AbstractTransform, Sequence[AbstractTransform]], dropout: Union[float, Sequence[float]]=0.5, shuffle: bool=False, random_sampler: ContinuousParameter=None, transform_call: Callable[[Any, Callable], Any]=dict_call, **kwargs):
        """
        Args:
            *transforms: one or multiple transformations which are applied in
                consecutive order
            dropout: if provided as float, each transform is skipped with the
                given probability
                if :attr:`dropout` is a sequence, it needs to specify the
                dropout probability for each given transform
            shuffle: apply transforms in random order
            random_sampler : a continuous parameter sampler. Samples a
                random value for each of the transforms.
            transform_call: function which determines how transforms are
                called. By default Mappings and Sequences are unpacked
                during the transform.

        Raises:
            ValueError: if dropout is a sequence it must have the same length
                as transforms
        """
        super().__init__(*transforms, transform_call=transform_call, shuffle=shuffle, **kwargs)
        if random_sampler is None:
            random_sampler = UniformParameter(0.0, 1.0)
        self.register_sampler('prob', random_sampler, size=(len(self.transforms),))
        if check_scalar(dropout):
            dropout = [dropout] * len(self.transforms)
        self.dropout = dropout
        if len(dropout) != len(self.transforms):
            raise TypeError(f'If dropout is a sequence it must specify the dropout probability for each transform, found {len(dropout)} probabilities and {len(self.transforms)} transforms.')

    def forward(self, *seq_like, **map_like) ->Union[Sequence, Mapping]:
        """
        Apply transforms in a consecutive order. Can either handle
        Sequence like or Mapping like data.

        Args:
            *seq_like: data which is unpacked like a Sequence
            **map_like: data which is unpacked like a dict

        Returns:
            Union[Sequence, Mapping]: dict with transformed data
        """
        assert not (seq_like and map_like)
        assert len(self.transforms) == len(self.transform_order)
        data = seq_like if seq_like else map_like
        rand = self.prob
        for idx in self.transform_order:
            if rand[idx] > self.dropout[idx]:
                data = self.transform_call(data, self.transforms[idx])
        return data


class OneOf(AbstractTransform):
    """
    Apply one of the given transforms.
    """

    def __init__(self, *transforms: Union[AbstractTransform, Sequence[AbstractTransform]], weights: Optional[Sequence[float]]=None, p: float=1.0, transform_call: Callable[[Any, Callable], Any]=dict_call):
        """
        Args:
            *transforms: transforms to choose from
            weights: additional weights for transforms
            p: probability that one transform i applied
            transform_call: function which determines how transforms are
                called. By default Mappings and Sequences are unpacked
                during the transform.
        """
        super().__init__(grad=True)
        if len(transforms) > 0 and isinstance(transforms[0], Sequence):
            transforms = transforms[0]
        if not transforms:
            raise ValueError('At least one transformation needs to be selected.')
        self.transforms = transforms
        if weights is not None and len(weights) != len(transforms):
            raise ValueError(f'If weights are porvided, every transform needs a weight. Found {len(weights)} weights and {len(transforms)} transforms')
        if weights is None:
            self.weights = torch.tensor([1 / len(self.transforms)] * len(self.transforms))
        else:
            self.weights = torch.tensor(weights)
        self.p = p
        self.transform_call = transform_call

    def forward(self, **data) ->dict:
        if torch.rand(1) < self.p:
            index = torch.multinomial(self.weights, 1)
            data = self.transform_call(data, self.transforms[int(index)])
        return data


def crop(data: torch.Tensor, corner: Sequence[int], size: Sequence[int]) ->torch.Tensor:
    """
    Extract crop from last dimensions of data

    Args:
    data: input tensor
    corner: top left corner point
    size: size of patch

    Returns:
        torch.Tensor: cropped data
    """
    _slices = []
    if len(corner) < data.ndim:
        for i in range(data.ndim - len(corner)):
            _slices.append(slice(0, data.shape[i]))
    _slices = _slices + [slice(c, c + s) for c, s in zip(corner, size)]
    return data[_slices]


def center_crop(data: torch.Tensor, size: Union[int, Sequence[int]]) ->torch.Tensor:
    """
    Crop patch from center

    Args:
    data: input tensor
    size: size of patch

    Returns:
        torch.Tensor: output tensor cropped from input tensor
    """
    if check_scalar(size):
        size = [size] * (data.ndim - 2)
    if not isinstance(size[0], int):
        size = [int(s) for s in size]
    corner = [int(round((img_dim - crop_dim) / 2.0)) for img_dim, crop_dim in zip(data.shape[2:], size)]
    return crop(data, corner, size)


class CenterCrop(BaseTransform):

    def __init__(self, size: Union[int, Sequence, AbstractParameter], keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            size: size of crop
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=center_crop, keys=keys, grad=grad, property_names=('size',), size=size, **kwargs)


def random_crop(data: torch.Tensor, size: Union[int, Sequence[int]], dist: Union[int, Sequence[int]]=0) ->torch.Tensor:
    """
    Crop random patch/volume from input tensor

    Args:
        data: input tensor
        size: size of patch/volume
        dist: minimum distance to border. By default zero

    Returns:
        torch.Tensor: cropped output
        List[int]: top left corner used for crop
    """
    if check_scalar(dist):
        dist = [dist] * (data.ndim - 2)
    if isinstance(dist[0], torch.Tensor):
        dist = [int(i) for i in dist]
    if check_scalar(size):
        size = [size] * (data.ndim - 2)
    if not isinstance(size[0], int):
        size = [int(s) for s in size]
    if any([(crop_dim + dist_dim >= img_dim) for img_dim, crop_dim, dist_dim in zip(data.shape[2:], size, dist)]):
        raise TypeError(f'Crop can not be realized with given size {size} and dist {dist}.')
    corner = [torch.randint(0, img_dim - crop_dim - dist_dim, (1,)).item() for img_dim, crop_dim, dist_dim in zip(data.shape[2:], size, dist)]
    return crop(data, corner, size)


class RandomCrop(BaseTransformSeeded):

    def __init__(self, size: Union[int, Sequence, AbstractParameter], dist: Union[int, Sequence, AbstractParameter]=0, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            size: size of crop
            dist: minimum distance to border. By default zero
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=random_crop, keys=keys, size=size, dist=dist, grad=grad, property_names=('size', 'dist'), **kwargs)


def clamp(data: torch.Tensor, min: float, max: float, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
    Clamp tensor to minimal and maximal value

    Args:
        data: tensor to clamp
        min: lower limit
        max: upper limit
        out: output tensor

    Returns:
        Tensor: clamped tensor
    """
    return torch.clamp(data, min=float(min), max=float(max), out=out)


class Clamp(BaseTransform):
    """Apply augment_fn to keys"""

    def __init__(self, min: Union[float, AbstractParameter], max: Union[float, AbstractParameter], keys: Sequence=('data',), grad: bool=False, **kwargs):
        """


        Args:
            min: minimal value
            max: maximal value
            keys: the keys corresponding to the values to clamp
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=clamp, keys=keys, grad=grad, min=min, max=max, property_names=('min', 'max'), **kwargs)


def norm_min_max(data: torch.Tensor, per_channel: bool=True, out: Optional[torch.Tensor]=None, eps: Optional[float]=1e-08) ->torch.Tensor:
    """
    Scale range to [0,1]

    Args:
        data: input data. Per channel option supports [C,H,W] and [C,H,W,D].
        per_channel: range is normalized per channel
        out:  if provided, result is saved in here
        eps: small constant for numerical stability.
            If None, no factor constant will be added

    Returns:
        torch.Tensor: scaled data
    """

    def _norm(_data: torch.Tensor, _out: torch.Tensor):
        _min = _data.min()
        _range = _data.max() - _min
        if eps is not None:
            _range = _range + eps
        _out = (_data - _min) / _range
        return _out
    if out is None:
        out = torch.zeros_like(data)
    if per_channel:
        for _c in range(data.shape[0]):
            out[_c] = _norm(data[_c], out[_c])
    else:
        out = _norm(data, out)
    return out


def norm_range(data: torch.Tensor, min: float, max: float, per_channel: bool=True, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
    Scale range of tensor

    Args:
        data: input data. Per channel option supports [C,H,W] and [C,H,W,D].
        min: minimal value
        max: maximal value
        per_channel: range is normalized per channel
        out: if provided, result is saved in here

    Returns:
        torch.Tensor: normalized data
    """
    if out is None:
        out = torch.zeros_like(data)
    out = norm_min_max(data, per_channel=per_channel, out=out)
    _range = max - min
    out = out * _range + min
    return out


class NormRange(PerSampleTransform):

    def __init__(self, min: Union[float, AbstractParameter], max: Union[float, AbstractParameter], keys: Sequence=('data',), per_channel: bool=True, grad: bool=False, **kwargs):
        """
        Args:
            min: minimal value
            max: maximal value
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_range, keys=keys, grad=grad, min=min, max=max, per_channel=per_channel, property_names=('min', 'max'), **kwargs)


class NormMinMax(PerSampleTransform):
    """Norm to [0, 1]"""

    def __init__(self, keys: Sequence=('data',), per_channel: bool=True, grad: bool=False, eps: Optional[float]=1e-08, **kwargs):
        """
        Args:
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            eps: small constant for numerical stability.
                If None, no factor constant will be added
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_min_max, keys=keys, grad=grad, per_channel=per_channel, eps=eps, **kwargs)


def norm_zero_mean_unit_std(data: torch.Tensor, per_channel: bool=True, out: Optional[torch.Tensor]=None, eps: Optional[float]=1e-08) ->torch.Tensor:
    """
    Normalize mean to zero and std to one

    Args:
        data: input data. Per channel option supports [C,H,W] and [C,H,W,D].
        per_channel: range is normalized per channel
        out: if provided, result is saved in here
        eps: small constant for numerical stability.
            If None, no factor constant will be added

    Returns:
        torch.Tensor: normalized data
    """

    def _norm(_data: torch.Tensor, _out: torch.Tensor):
        denom = _data.std()
        if eps is not None:
            denom = denom + eps
        _out = (_data - _data.mean()) / denom
        return _out
    if out is None:
        out = torch.zeros_like(data)
    if per_channel:
        for _c in range(data.shape[0]):
            out[_c] = _norm(data[_c], out[_c])
    else:
        out = _norm(data, out)
    return out


class NormZeroMeanUnitStd(PerSampleTransform):
    """Normalize mean to zero and std to one"""

    def __init__(self, keys: Sequence=('data',), per_channel: bool=True, grad: bool=False, eps: Optional[float]=1e-08, **kwargs):
        """
        Args:
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            eps: small constant for numerical stability.
                If None, no factor constant will be added
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_zero_mean_unit_std, keys=keys, grad=grad, per_channel=per_channel, eps=eps, **kwargs)


def norm_mean_std(data: torch.Tensor, mean: Union[float, Sequence], std: Union[float, Sequence], per_channel: bool=True, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
    Normalize mean and std with provided values

    Args:
        data:input data. Per channel option supports [C,H,W] and [C,H,W,D].
        mean: used for mean normalization
        std: used for std normalization
        per_channel: range is normalized per channel
        out: if provided, result is saved into out

    Returns:
        torch.Tensor: normalized data
    """
    if out is None:
        out = torch.zeros_like(data)
    if per_channel:
        if check_scalar(mean):
            mean = [mean] * data.shape[0]
        if check_scalar(std):
            std = [std] * data.shape[0]
        for _c in range(data.shape[0]):
            out[_c] = (data[_c] - mean[_c]) / std[_c]
    else:
        out = (data - mean) / std
    return out


class NormMeanStd(PerSampleTransform):
    """Normalize mean and std with provided values"""

    def __init__(self, mean: Union[float, Sequence[float]], std: Union[float, Sequence[float]], keys: Sequence[str]=('data',), per_channel: bool=True, grad: bool=False, **kwargs):
        """
        Args:
            mean: used for mean normalization
            std: used for std normalization
            keys: keys to normalize
            per_channel: normalize per channel
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to normalization function
        """
        super().__init__(augment_fn=norm_mean_std, keys=keys, grad=grad, mean=mean, std=std, per_channel=per_channel, **kwargs)


def add_noise(data: torch.Tensor, noise_type: str, out: Optional[torch.Tensor]=None, **kwargs) ->torch.Tensor:
    """
    Add noise to input

    Args:
        data: input data
        noise_type: supports all inplace functions of a pytorch tensor
        out: if provided, result is saved in here
        kwargs: keyword arguments passed to generating function

    Returns:
        torch.Tensor: data with added noise

    See Also:
        :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
    """
    if not noise_type.endswith('_'):
        noise_type = noise_type + '_'
    noise_tensor = torch.empty_like(data, requires_grad=False)
    getattr(noise_tensor, noise_type)(**kwargs)
    return torch.add(data, noise_tensor, out=out)


class Noise(PerChannelTransform):
    """
    Add noise to data

    .. warning:: This transform will apply different noise patterns to
        different keys.
    """

    def __init__(self, noise_type: str, per_channel: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            noise_type: supports all inplace functions of a
                :class:`torch.Tensor`
            per_channel: enable transformation per channel
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to noise function

        See Also:
            :func:`torch.Tensor.normal_`, :func:`torch.Tensor.exponential_`
        """
        super().__init__(augment_fn=add_noise, per_channel=per_channel, keys=keys, grad=grad, noise_type=noise_type, **kwargs)


class ExponentialNoise(Noise):
    """
    Add exponential noise to data

    .. warning:: This transform will apply different noise patterns to
        different keys.
    """

    def __init__(self, lambd: float, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            lambd: lambda of exponential distribution
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to noise function
        """
        super().__init__(noise_type='exponential_', lambd=lambd, keys=keys, grad=grad, **kwargs)


class GaussianNoise(Noise):
    """
    Add gaussian noise to data

    .. warning:: This transform will apply different noise patterns to
        different keys.
    """

    def __init__(self, mean: float, std: float, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            mean: mean of normal distribution
            std: std of normal distribution
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to noise function
        """
        super().__init__(noise_type='normal_', mean=mean, std=std, keys=keys, grad=grad, **kwargs)


def gamma_correction(data: torch.Tensor, gamma: float) ->torch.Tensor:
    """
    Apply gamma correction to data
    (currently this functions is intended as an interface in case
    additional functionality should be added to transform)

    Args:
        data: input data
        gamma: gamma for correction

    Returns:
        torch.Tensor: gamma corrected data
    """
    if torch.is_tensor(gamma):
        gamma = gamma
    return data.pow(gamma)


class GammaCorrection(BaseTransform):
    """Apply Gamma correction"""

    def __init__(self, gamma: Union[float, AbstractParameter], keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            gamma: define gamma
            keys: keys to normalize
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to superclass
        """
        super().__init__(augment_fn=gamma_correction, gamma=gamma, property_names=('gamma',), keys=keys, grad=grad, **kwargs)


class RandomValuePerChannel(PerChannelTransform):
    """
    Apply augmentations which take random values as input by keyword
    :attr:`value`

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(self, augment_fn: callable, random_sampler: AbstractParameter, per_channel: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            augment_fn: augmentation function
            random_mode: specifies distribution which should be used to
                sample additive value. All function from python's random
                module are supported
            random_args: positional arguments passed for random function
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=augment_fn, per_channel=per_channel, keys=keys, grad=grad, random_sampler=random_sampler, property_names=('random_sampler',), **kwargs)

    def forward(self, **data) ->dict:
        """
        Perform Augmentation.

        Args:
            data: dict with data

        Returns:
            dict: augmented data
        """
        if self.per_channel:
            seed = torch.random.get_rng_state()
            for _key in self.keys:
                torch.random.set_rng_state(seed)
                out = torch.empty_like(data[_key])
                for _i in range(data[_key].shape[1]):
                    rand_value = self.random_sampler
                    out[:, (_i)] = self.augment_fn(data[_key][:, (_i)], value=rand_value, out=out[:, (_i)], **self.kwargs)
                data[_key] = out
        else:
            rand_value = self.random_sampler
            for _key in self.keys:
                data[_key] = self.augment_fn(data[_key], value=rand_value, **self.kwargs)
        return data


def add_value(data: torch.Tensor, value: float, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
    Increase brightness additively by value
    (currently this functions is intended as an interface in case
    additional functionality should be added to transform)

    Args:
        data: input data
        value: additive value
        out: if provided, result is saved in here

    Returns:
        torch.Tensor: augmented data
    """
    return torch.add(data, value, out=out)


class RandomAddValue(RandomValuePerChannel):
    """
    Increase values additively

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(self, random_sampler: AbstractParameter, per_channel: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            random_sampler: specify values to add
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=add_value, random_sampler=random_sampler, per_channel=per_channel, keys=keys, grad=grad, **kwargs)


def scale_by_value(data: torch.Tensor, value: float, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
    Increase brightness scaled by value
    (currently this functions is intended as an interface in case
    additional functionality should be added to transform)

    Args:
        data: input data
        value: scaling value
        out: if provided, result is saved in here

    Returns:
        torch.Tensor: augmented data
    """
    return torch.mul(data, value, out=out)


class RandomScaleValue(RandomValuePerChannel):
    """
    Scale Values

    .. warning:: This transform will apply different values to different keys.
    """

    def __init__(self, random_sampler: AbstractParameter, per_channel: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            random_sampler: specify values to add
            per_channel: enable transformation per channel
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=scale_by_value, random_sampler=random_sampler, per_channel=per_channel, keys=keys, grad=grad, **kwargs)


class KernelTransform(AbstractTransform):
    """
    Baseclass for kernel based transformations (kernel is applied to
    each channel individually)
    """

    def __init__(self, in_channels: int, kernel_size: Union[int, Sequence], dim: int=2, stride: Union[int, Sequence]=1, padding: Union[int, Sequence]=0, padding_mode: str='zero', keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            in_channels: number of input channels
            kernel_size: size of kernel
            dim: number of spatial dimensions
            stride: stride of convolution
            padding: padding size for input
            padding_mode: padding mode for input. Supports all modes
                from :func:`torch.functional.pad` except ``circular``
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to superclass

        See Also:
            :func:`torch.functional.pad`
        """
        super().__init__(grad=grad, **kwargs)
        self.in_channels = in_channels
        if check_scalar(kernel_size):
            kernel_size = [kernel_size] * dim
        self.kernel_size = kernel_size
        if check_scalar(stride):
            stride = [stride] * dim
        self.stride = stride
        if check_scalar(padding):
            padding = [padding] * dim * 2
        self.padding = padding
        self.padding_mode = padding_mode
        self.keys = keys
        kernel = self.create_kernel()
        self.register_buffer('weight', kernel)
        self.groups = in_channels
        self.conv = self.get_conv(dim)

    @staticmethod
    def get_conv(dim) ->Callable:
        """
        Select convolution with regard to dimension

        Args:
            dim: spatial dimension of data

        Returns:
            Callable: the suitable convolutional function
        """
        if dim == 1:
            return torch.nn.functional.conv1d
        elif dim == 2:
            return torch.nn.functional.conv2d
        elif dim == 3:
            return torch.nn.functional.conv3d
        else:
            raise TypeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

    def create_kernel(self) ->torch.Tensor:
        """
        Create kernel for convolution
        """
        raise NotImplementedError

    def forward(self, **data) ->dict:
        """
        Apply kernel to selected keys

        Args:
            data: input data

        Returns:
            dict: dict with transformed data
        """
        for key in self.keys:
            inp_pad = torch.nn.functional.pad(data[key], self.padding, mode=self.padding_mode)
            data[key] = self.conv(inp_pad, weight=self.weight, groups=self.groups, stride=self.stride)
        return data


class GaussianSmoothing(KernelTransform):
    """
    Perform Gaussian Smoothing.
    Filtering is performed seperately for each channel in the input using a
    depthwise convolution.
    This code is adapted from:
    'https://discuss.pytorch.org/t/is-there-anyway-to-do-'
    'gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10'
    """

    def __init__(self, in_channels: int, kernel_size: Union[int, Sequence], std: Union[int, Sequence], dim: int=2, stride: Union[int, Sequence]=1, padding: Union[int, Sequence]=0, padding_mode: str='reflect', keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            in_channels: number of input channels
            kernel_size: size of kernel
            std: standard deviation of gaussian
            dim: number of spatial dimensions
            stride: stride of convolution
            padding: padding size for input
            padding_mode: padding mode for input. Supports all modes from
                :func:`torch.functional.pad` except ``circular``
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to superclass

        See Also:
            :func:`torch.functional.pad`
        """
        if check_scalar(std):
            std = [std] * dim
        self.std = std
        super().__init__(in_channels=in_channels, kernel_size=kernel_size, dim=dim, stride=stride, padding=padding, padding_mode=padding_mode, keys=keys, grad=grad, **kwargs)

    def create_kernel(self) ->torch.Tensor:
        """
        Create gaussian blur kernel
        """
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in self.kernel_size])
        for size, std, mgrid in zip(self.kernel_size, self.std, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(self.in_channels, *([1] * (kernel.dim() - 1)))
        kernel.requires_grad = False
        return kernel.contiguous()


def mirror(data: torch.Tensor, dims: Union[int, Sequence[int]]) ->torch.Tensor:
    """
    Mirror data at dims

    Args:
        data: input data
        dims: dimensions to mirror

    Returns:
        torch.Tensor: tensor with mirrored dimensions
    """
    if check_scalar(dims):
        dims = dims,
    dims = [(d + 2) for d in dims]
    return data.flip(dims)


class Mirror(BaseTransform):
    """Random mirror transform"""

    def __init__(self, dims: Union[int, DiscreteParameter, Sequence[Union[int, DiscreteParameter]]], keys: Sequence[str]=('data',), grad: bool=False, **kwargs):
        """
        Args:
            dims: axes which should be mirrored
            keys: keys which should be mirrored
            prob: probability for mirror. If float value is provided,
                it is used for all dims
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to superclass

        Examples:
            >>> # Use mirror transform for augmentations
            >>> from rising.random import DiscreteCombinationsParameter
            >>> # We sample from all possible mirror combination for
            >>> # volumetric data
            >>> trafo = Mirror(DiscreteCombinationsParameter((0, 1, 2)))
        """
        super().__init__(augment_fn=mirror, dims=dims, keys=keys, grad=grad, property_names=('dims',), **kwargs)


def rot90(data: torch.Tensor, k: int, dims: Union[int, Sequence[int]]):
    """
    Rotate 90 degrees around dims

    Args:
        data: input data
        k: number of times to rotate
        dims: dimensions to mirror

    Returns:
        torch.Tensor: tensor with mirrored dimensions
    """
    dims = [int(d + 2) for d in dims]
    return torch.rot90(data, int(k), dims)


class Rot90(AbstractTransform):
    """Rotate 90 degree around dims"""

    def __init__(self, dims: Union[Sequence[int], DiscreteParameter], keys: Sequence[str]=('data',), num_rots: Sequence[int]=(0, 1, 2, 3), prob: float=0.5, grad: bool=False, **kwargs):
        """
        Args:
            dims: dims/axis ro rotate. If more than two dims are
                provided, 2 dimensions are randomly chosen at each call
            keys: keys which should be rotated
            num_rots: possible values for number of rotations
            prob: probability for rotation
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to superclass

        See Also:
            :func:`torch.Tensor.rot90`
        """
        super().__init__(grad=grad, **kwargs)
        self.keys = keys
        self.prob = prob
        if not isinstance(dims, DiscreteParameter):
            if len(dims) > 2:
                dims = list(combinations(dims, 2))
            else:
                dims = dims,
            dims = DiscreteParameter(dims)
        self.register_sampler('dims', dims)
        self.register_sampler('num_rots', DiscreteParameter(num_rots))

    def forward(self, **data) ->dict:
        """
        Apply transformation

        Args:
            data: dict with tensors

        Returns:
            dict: dict with augmented data
        """
        if torch.rand(1) < self.prob:
            num_rots = self.num_rots
            rand_dims = self.dims
            for key in self.keys:
                data[key] = rot90(data[key], k=num_rots, dims=rand_dims)
        return data


def resize_native(data: torch.Tensor, size: Optional[Union[int, Sequence[int]]]=None, scale_factor: Optional[Union[float, Sequence[float]]]=None, mode: str='nearest', align_corners: Optional[bool]=None, preserve_range: bool=False):
    """
    Down/up-sample sample to either the given :attr:`size` or the given
    :attr:`scale_factor`
    The modes available for resizing are: nearest, linear (3D-only), bilinear,
    bicubic (4D-only), trilinear (5D-only), area

    Args:
        data: input tensor of shape batch x channels x height x width x [depth]
        size: spatial output size (excluding batch size and number of channels)
        scale_factor: multiplier for spatial size
        mode: one of ``nearest``, ``linear``, ``bilinear``, ``bicubic``,
            ``trilinear``, ``area``
            (for more inforamtion see :func:`torch.nn.functional.interpolate`)
        align_corners: input and output tensors are aligned by the center
            points of their corners pixels, preserving the values at the
            corner pixels.
        preserve_range:  output tensor has same range as input tensor

    Returns:
        torch.Tensor: interpolated tensor

    See Also:
        :func:`torch.nn.functional.interpolate`
    """
    if check_scalar(scale_factor):
        scale_factor = float(scale_factor)
    out = torch.nn.functional.interpolate(data, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    if preserve_range:
        out.clamp_(data.min(), data.max())
    return out


class ResizeNative(BaseTransform):
    """Resize data to given size"""

    def __init__(self, size: Union[int, Sequence[int]], mode: str='nearest', align_corners: Optional[bool]=None, preserve_range: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            size: spatial output size (excluding batch size and
                number of channels)
            mode: one of ``nearest``, ``linear``, ``bilinear``, ``bicubic``,
                ``trilinear``, ``area`` (for more inforamtion see
                :func:`torch.nn.functional.interpolate`)
            align_corners: input and output tensors are aligned by the center                 points of their corners pixels, preserving the values at the
                corner pixels.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=resize_native, size=size, mode=mode, align_corners=align_corners, preserve_range=preserve_range, keys=keys, grad=grad, **kwargs)


class Zoom(BaseTransform):
    """Apply augment_fn to keys. By default the scaling factor is sampled
       from a uniform distribution with the range specified by
       :attr:`random_args`
    """

    def __init__(self, scale_factor: Union[Sequence, AbstractParameter]=(0.75, 1.25), mode: str='nearest', align_corners: bool=None, preserve_range: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            scale_factor: positional arguments passed for random function.
                If Sequence[Sequence] is provided, a random value for each item
                in the outer Sequence is generated. This can be used to set
                different ranges for different axis.
            mode: one of `nearest`, `linear`, `bilinear`,
                `bicubic`, `trilinear`, `area` (for more
                inforamtion see :func:`torch.nn.functional.interpolate`)
            align_corners: input and output tensors are aligned by the center
                points of their corners pixels, preserving the values at the
                corner pixels.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn

        See Also:
            :func:`random.uniform`, :func:`torch.nn.functional.interpolate`
        """
        super().__init__(augment_fn=resize_native, scale_factor=scale_factor, mode=mode, align_corners=align_corners, preserve_range=preserve_range, keys=keys, grad=grad, property_names=('scale_factor',), **kwargs)


scheduler_type = Callable[[int], Union[int, Sequence[int]]]


class ProgressiveResize(ResizeNative):
    """Resize data to sizes specified by scheduler"""

    def __init__(self, scheduler: scheduler_type, mode: str='nearest', align_corners: bool=None, preserve_range: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            scheduler: scheduler which determined the current size.
                The scheduler is called with the current iteration of the
                transform
            mode: one of ``nearest``, ``linear``, ``bilinear``, ``bicubic``,
                    ``trilinear``, ``area`` (for more inforamtion see
                    :func:`torch.nn.functional.interpolate`)
            align_corners: input and output tensors are aligned by the center
                points of their corners pixels, preserving the values at the
                corner pixels.
            preserve_range: output tensor has same range as input tensor
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn

        Warnings:
            When this transformations is used in combination with
            multiprocessing, the step counter is not perfectly synchronized
            between multiple processes.
            As a result the step count my jump between values
            in a range of the number of processes used.
        """
        super().__init__(size=0, mode=mode, align_corners=align_corners, preserve_range=preserve_range, keys=keys, grad=grad, **kwargs)
        self.scheduler = scheduler
        self._step = Value('i', 0)

    def reset_step(self) ->ResizeNative:
        """
        Reset step to 0

        Returns:
            ResizeNative: returns self to allow chaining
        """
        with self._step.get_lock():
            self._step.value = 0
        return self

    def increment(self) ->ResizeNative:
        """
        Increment step by 1

        Returns:
            ResizeNative: returns self to allow chaining
        """
        with self._step.get_lock():
            self._step.value += 1
        return self

    @property
    def step(self) ->int:
        """
        Current step

        Returns:
            int: number of steps
        """
        return self._step.value

    def forward(self, **data) ->dict:
        """
        Resize data

        Args:
            **data: input batch

        Returns:
            dict: augmented batch
        """
        self.kwargs['size'] = self.scheduler(self.step)
        self.increment()
        return super().forward(**data)


class ToTensor(BaseTransform):
    """Transform Input Collection to Collection of :class:`torch.Tensor`"""

    def __init__(self, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            keys: keys which should be transformed
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to augment_fn
        """
        super().__init__(augment_fn=default_convert, keys=keys, grad=grad, **kwargs)


data_type = Union[Tensor, List[Tensor], Tuple[Tensor], Mapping[Hashable, Tensor]]


def to_device_dtype(data: data_type, dtype: Union[torch.dtype, str]=None, device: Union[torch.device, str]=None, **kwargs) ->data_type:
    """
    Pushes data to device

    Args:
        data: data which should be pushed to device. Sequence and mapping
            items are mapping individually to gpu
        device: target device
        kwargs: keyword arguments passed to assigning function

    Returns:
        Union[torch.Tensor, Sequence, Mapping]: data which was pushed to device
    """
    if torch.is_tensor(data):
        return data
    elif isinstance(data, Mapping):
        return {key: to_device_dtype(item, device=device, dtype=dtype, **kwargs) for key, item in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([to_device_dtype(item, device=device, dtype=dtype, **kwargs) for item in data])
    else:
        return data


class ToDeviceDtype(BaseTransform):
    """Push data to device and convert to tdype"""

    def __init__(self, device: Optional[Union[torch.device, str]]=None, dtype: Optional[torch.dtype]=None, non_blocking: bool=False, copy: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            device: target device
            dtype: target dtype
            non_blocking: if True and this copy is between CPU and GPU, the
                copy may occur asynchronously with respect to the host. For other
                cases, this argument has no effect.
            copy: create copy of data
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to function
        """
        super().__init__(augment_fn=to_device_dtype, keys=keys, grad=grad, device=device, dtype=dtype, non_blocking=non_blocking, copy=copy, **kwargs)


class ToDevice(ToDeviceDtype):
    """Push data to device"""

    def __init__(self, device: Optional[Union[torch.device, str]], non_blocking: bool=False, copy: bool=False, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            device: target device
            non_blocking: if True and this copy is between CPU and GPU,
                the copy may occur asynchronously with respect to the host.
                For other cases, this argument has no effect.
            copy: create copy of data
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to function
        """
        super().__init__(device=device, non_blocking=non_blocking, copy=copy, keys=keys, grad=grad, **kwargs)


class ToDtype(ToDeviceDtype):
    """Convert data to dtype"""

    def __init__(self, dtype: torch.dtype, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            dtype: target dtype
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            kwargs: keyword arguments passed to function
        """
        super().__init__(dtype=dtype, keys=keys, grad=grad, **kwargs)


def tensor_op(data: data_type, fn: str, *args, **kwargs) ->data_type:
    """
    Invokes a function form a tensor

    Args:
        data: data which should be pushed to device. Sequence and mapping items
            are mapping individually to gpu
        fn: tensor function
        *args: positional arguments passed to tensor function
        **kwargs: keyword arguments passed to tensor function

    Returns:
        Union[torch.Tensor, Sequence, Mapping]: data which was pushed to device
    """
    if torch.is_tensor(data):
        return getattr(data, fn)(*args, **kwargs)
    elif isinstance(data, Mapping):
        return {key: tensor_op(item, fn, *args, **kwargs) for key, item in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)([tensor_op(item, fn, *args, **kwargs) for item in data])
    else:
        return data


class TensorOp(BaseTransform):
    """Apply function which are supported by the `torch.Tensor` class"""

    def __init__(self, op_name: str, *args, keys: Sequence=('data',), grad: bool=False, **kwargs):
        """
        Args:
            op_name: name of tensor operation
            *args: positional arguments passed to function
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to function
        """
        super().__init__(tensor_op, op_name, *args, keys=keys, grad=grad, **kwargs)


class Permute(BaseTransform):
    """Permute dimensions of tensor"""

    def __init__(self, dims: Dict[str, Sequence[int]], grad: bool=False, **kwargs):
        """
        Args:
            dims: defines permutation sequence for respective key
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to permute function
        """
        super().__init__(tensor_op, 'permute', grad=grad)
        self.dims = dims
        self.kwargs = kwargs

    def forward(self, **data) ->dict:
        """
        Forward input

        Args:
        data: batch dict

        Returns:
            dict: augmented data
        """
        for key, item in self.dims.items():
            data[key] = tensor_op(data[key], 'permute', *item, **self.kwargs)
        return data


class DoNothing(AbstractTransform):
    """Transform that returns the input as is"""

    def __init__(self, grad: bool=False, **kwargs):
        """
        Args:
            grad: enable gradient computation inside transformation
            **kwargs: keyword arguments passed to superclass
        """
        super().__init__(grad=grad, **kwargs)

    def forward(self, **data) ->dict:
        """
        Forward input

        Args:
            data: input dict

        Returns:
            input dict
        """
        return data


def seg_to_box(seg: torch.Tensor, dim: int) ->List[torch.Tensor]:
    """
    Convert instance segmentation to bounding boxes

    Args:
        seg: segmentation of individual classes
            (index should start from one and be continuous)
        dim: number of spatial dimensions

    Returns:
        list: list of bounding boxes tuple with classes for
            bounding boxes
    """
    boxes = []
    _seg = seg.detach()
    for _idx in range(1, seg.max().detach().item() + 1):
        instance_map = (_seg == _idx).nonzero()
        _mins = instance_map.min(dim=0)[0]
        _maxs = instance_map.max(dim=0)[0]
        box = [_mins[-dim], _mins[-dim + 1], _maxs[-dim], _maxs[-dim + 1]]
        if dim > 2:
            box = box + [c for cv in zip(_mins[-dim + 2:], _maxs[-dim + 2:]) for c in cv]
        boxes.append(torch.tensor(box))
    return boxes


class SegToBox(AbstractTransform):
    """Convert instance segmentation to bounding boxes"""

    def __init__(self, keys: Mapping[Hashable, Hashable], grad: bool=False, **kwargs):
        """
        Args:
            keys: the key specifies which item to use as segmentation and the
                item specifies where the save the bounding boxes
            grad: enable gradient computation inside transformation
        """
        super().__init__(grad=grad, **kwargs)
        self.keys = keys

    def forward(self, **data) ->dict:
        """

        Args:
            **data: input data

        Returns:
            dict: transformed data

        """
        for source, target in self.keys.items():
            data[target] = [seg_to_box(s, s.ndim - 2) for s in data[source].split(1)]
        return data


def box_to_seg(boxes: Sequence[Sequence[int]], shape: Optional[Sequence[int]]=None, dtype: Optional[Union[torch.dtype, str]]=None, device: Optional[Union[torch.device, str]]=None, out: Optional[torch.Tensor]=None) ->torch.Tensor:
    """
    Convert a sequence of bounding boxes to a segmentation

    Args:
        boxes: sequence of bounding boxes encoded as
            (dim0_min, dim1_min, dim0_max, dim1_max, [dim2_min, dim2_max]).
            Supported bounding boxes for 2D (4 entries per box)
            and 3d (6 entries per box)
        shape: if :attr:`out` is not provided, shape of output tensor must
            be specified
        dtype: if :attr:`out` is not provided,
            dtype of output tensor must be specified
        device: if :attr:`out` is not provided,
            device of output tensor must be specified
        out: if not None, the segmentation will be saved inside this tensor

    Returns:
        torch.Tensor: bounding boxes encoded as a segmentation
    """
    if out is None:
        out = torch.zeros(*shape, dtype=dtype, device=device)
    for _idx, box in enumerate(boxes, 1):
        if len(box) == 4:
            out[(...), box[0]:box[2] + 1, box[1]:box[3] + 1] = _idx
        elif len(box) == 6:
            out[(...), box[0]:box[2] + 1, box[1]:box[3] + 1, box[4]:box[5] + 1] = _idx
        else:
            raise TypeError(f'Boxes must have length 4 (2D) or 6(3D) found len {len(box)}')
    return out


class BoxToSeg(AbstractTransform):
    """Convert bounding boxes to instance segmentation"""

    def __init__(self, keys: Mapping[Hashable, Hashable], shape: Sequence[int], dtype: torch.dtype, device: Union[torch.device, str], grad: bool=False, **kwargs):
        """
        Args:
            keys: the key specifies which item to use as the bounding boxes and
                the item specifies where the save the bounding boxes
            shape: spatial shape of output tensor (batchsize is derived from
                bounding boxes and has one channel)
            dtype: dtype of segmentation
            device: device of segmentation
            grad: enable gradient computation inside transformation
            **kwargs: Additional keyword arguments forwarded to the Base Class
        """
        super().__init__(grad=grad, **kwargs)
        self.keys = keys
        self.seg_shape = shape
        self.seg_dtype = dtype
        self.seg_device = device

    def forward(self, **data) ->dict:
        """
        Forward input

        Args:
            **data: input data

        Returns:
            dict: transformed data
        """
        for source, target in self.keys.items():
            out = torch.zeros((len(data[source]), 1, *self.seg_shape), dtype=self.seg_dtype, device=self.seg_device)
            for b in range(len(data[source])):
                box_to_seg(data[source][b], out=out[b])
            data[target] = out
        return data


def instance_to_semantic(instance: torch.Tensor, cls: Sequence[int]) ->torch.Tensor:
    """
    Convert an instance segmentation to a semantic segmentation

    Args:
        instance: instance segmentation of objects
            (objects need to start from 1, 0 background)
        cls: mapping from indices from instance segmentation to real classes.

    Returns:
        torch.Tensor: semantic segmentation

    Warnings:
        :attr:`instance` needs to encode objects starting from 1 and the
        indices need to be continuous (0 is interpreted as background)
    """
    seg = torch.zeros_like(instance)
    for idx, c in enumerate(cls, 1):
        seg[instance == idx] = c
    return seg


class InstanceToSemantic(AbstractTransform):
    """Convert an instance segmentation to a semantic segmentation"""

    def __init__(self, keys: Mapping[str, str], cls_key: Hashable, grad: bool=False, **kwargs):
        """
        Args:
            keys: the key specifies which item to use as instance segmentation
                and the item specifies where the save the semantic segmentation
            cls_key: key where the class mapping is saved. Mapping needs to
                be a Sequence{Sequence[int]].
            grad: enable gradient computation inside transformation
        """
        super().__init__(grad=grad, **kwargs)
        self.cls_key = cls_key
        self.keys = keys

    def forward(self, **data) ->dict:
        """
        Forward input

        Args:
            **data: input data

        Returns:
            dict: transformed data

        """
        for source, target in self.keys.items():
            data[target] = torch.cat([instance_to_semantic(data, mapping) for data, mapping in zip(data[source].split(1), data[self.cls_key])])
        return data


class AddTransform(AbstractTransform):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_tensor = torch.rand(1, 1, 32, 32, requires_grad=True)

    def forward(self, **data) ->dict:
        data['data'] = data['data'] + self.grad_tensor
        return data


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DoNothing,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (NormalParameter,
     lambda: ([], {'mu': 4, 'sigma': 4}),
     lambda: ([], {}),
     False),
    (UniformParameter,
     lambda: ([], {'low': 4, 'high': 4}),
     lambda: ([], {}),
     False),
    (_TransformWrapper,
     lambda: ([], {'trafo': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_PhoenixDL_rising(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

