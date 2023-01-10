import sys
_module = sys.modules[__name__]
del sys
sphinx_rtd_theme = _module
conf = _module
icpslam = _module
icpslam_scannet = _module
pointfusion = _module
pointfusion_scannet = _module
gradslam = _module
config = _module
cfgnode = _module
datasets = _module
datautils = _module
icl = _module
scannet = _module
tum = _module
tumutils = _module
geometry = _module
geometryutils = _module
projutils = _module
se3utils = _module
metrics = _module
odometry = _module
base = _module
gradicp = _module
groundtruth = _module
icp = _module
icputils = _module
slam = _module
fusionutils = _module
icpslam = _module
pointfusion = _module
structures = _module
pointclouds = _module
rgbdimages = _module
structutils = _module
utils = _module
version = _module
setup = _module
tests = _module
common = _module
common_testing = _module
test_cfgnode = _module
samplecfg = _module
samplecfg_dict = _module
test_datautils = _module
test_icl = _module
test_scannet = _module
test_tum = _module
test_projutils = _module
test_gradicp = _module
test_groundtruth = _module
test_icp = _module
test_icputils = _module
test_fusionutils = _module
test_pointclouds = _module
test_rgbdimages = _module
test_utils = _module

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


from torch.utils.data import DataLoader


import copy


import warnings


from collections import OrderedDict


from typing import List


from typing import Union


import numpy as np


from typing import Optional


from torch.utils import data


import torch.nn as nn


import math


import logging


from numpy.testing import assert_allclose as np_assert_allclose


from torch.testing import assert_allclose


from torch.autograd import gradcheck


class Pointclouds(object):
    """Batch of pointclouds (with varying numbers of points), enabling conversion between 2 representations:

    - List: Store points of each pointcloud of shape :math:`(N_b, 3)` in a list of length :math:`B`.
    - Padded: Store all points in a :math:`(B, max(N_b), 3)` tensor with zero padding as required.

    Args:
        points (torch.Tensor or list of torch.Tensor or None): :math:`(X, Y, Z)` coordinates of each point.
            Default: None
        normals (torch.Tensor or list of torch.Tensor or None): Normals :math:`(N_x, N_y, N_z)` of each point.
            Default: None
        colors (torch.Tensor or list of torch.Tensor or None): :math:`(R, G, B)` color of each point.
            Default: None
        features (torch.Tensor or list of torch.Tensor or None): :math:`C` features of each point.
            Default: None
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            same as `points` device. Default: None

    Shape:
        - points: Can either be a list of tensors of shape :math:`(N_b, 3)` or a padded tensor of shape
          :math:`(B, N, 3)`.
        - normals: Can either be a list of tensors of shape :math:`(N_b, 3)` or a padded tensor of shape
          :math:`(B, N, 3)`.
        - colors: Can either be a list of tensors of shape :math:`(N_b, 3)` or a padded tensor of shape
          :math:`(B, N, 3)`.
        - features: Can either be a list of tensors of shape :math:`(N_b, C)` or a padded tensor of shape
          :math:`(B, N, C)`.

    Examples::

        >>> points_list = [torch.rand(1, 3), torch.rand(4, 3)]
        >>> pcs1 = gradslam.Pointclouds(points_list)
        >>> print(pcs1.points_padded.shape)
        torch.Size([2, 4, 3])
        >>> print(len(pcs1.points_list))
        2
        >>> pcs2 = gradslam.Pointclouds(torch.rand((2, 4, 3)))
        >>> print(pcs2.points_padded.shape)
        torch.Size([2, 4, 3])
    """
    _INTERNAL_TENSORS = ['_points_padded', '_normals_padded', '_colors_padded', '_features_padded', '_nonpad_mask', '_num_points_per_pointcloud']

    def __init__(self, points: Union[List[torch.Tensor], torch.Tensor, None]=None, normals: Union[List[torch.Tensor], torch.Tensor, None]=None, colors: Union[List[torch.Tensor], torch.Tensor, None]=None, features: Union[List[torch.Tensor], torch.Tensor, None]=None, device: Union[torch.device, str, None]=None):
        super().__init__()
        if not (points is None or isinstance(points, list) or torch.is_tensor(points)):
            msg = 'Expected points to be of type list or tensor or None; got %r'
            raise TypeError(msg % type(points))
        if not (normals is None or isinstance(normals, type(points))):
            msg = 'Expected normals to be of same type as points (%r); got %r'
            raise TypeError(msg % (type(points), type(normals)))
        if not (colors is None or isinstance(colors, type(points))):
            msg = 'Expected colors to be of same type as points (%r); got %r'
            raise TypeError(msg % (type(points), type(colors)))
        if not (features is None or isinstance(features, type(points))):
            msg = 'Expected features to be of same type as points (%r); got %r'
            raise TypeError(msg % (type(points), type(features)))
        if points is not None and len(points) == 0:
            raise ValueError('len(points) (= 0) should be > 0')
        self._points_list = None
        self._normals_list = None
        self._colors_list = None
        self._features_list = None
        self._points_padded = None
        self._normals_padded = None
        self._colors_padded = None
        self._features_padded = None
        self._nonpad_mask = None
        self._has_points = None
        self._has_normals = None
        self._has_colors = None
        self._has_features = None
        self._num_points_per_pointcloud = None
        self.equisized = False
        if isinstance(points, list):
            points_shape_per_pointcloud = [p.shape for p in points]
            if any([(p.ndim != 2) for p in points]):
                raise ValueError('ndim of all tensors in points list should be 2')
            if any([(x[-1] != 3) for x in points_shape_per_pointcloud]):
                raise ValueError('last dim of all tensors in points should have shape 3 (X, Y, Z)')
            self.device = torch.Tensor().device if device is not None else points[0].device
            self._points_list = [p for p in points]
            num_points_per_pointcloud = [x[0] for x in points_shape_per_pointcloud]
            if not (normals is None or [n.shape for n in normals] == points_shape_per_pointcloud):
                raise ValueError("normals tensors should have same shape as points tensors, but didn't")
            if not (colors is None or [c.shape for c in colors] == points_shape_per_pointcloud):
                raise ValueError("colors tensors should have same shape as points tensors, but didn't")
            if not (features is None or all([(f.ndim == 2) for f in features])):
                raise ValueError('ndim of all tensors in features list should be 2')
            if not (features is None or [len(f) for f in features] == num_points_per_pointcloud):
                raise ValueError('number of features per pointcloud has to be equal to number of points')
            if not (features is None or len(set([f.shape[-1] for f in features])) == 1):
                raise ValueError('number of features per pointcloud has to be the same')
            self._normals_list = None if normals is None else [n for n in normals]
            self._colors_list = None if colors is None else [c for c in colors]
            self._features_list = None if features is None else [f for f in features]
            self._B = len(self._points_list)
            self._num_points_per_pointcloud = torch.tensor(num_points_per_pointcloud, device=self.device)
            self._N = self._num_points_per_pointcloud.max().item()
            self.equisized = len(self._num_points_per_pointcloud.unique()) == 1
        elif torch.is_tensor(points):
            self.device = torch.Tensor().device if device is not None else points.device
            if points.ndim != 3:
                msg = 'points should have ndim=3, but had ndim={}'.format(points.ndim)
                raise ValueError(msg)
            if points.shape[-1] != 3:
                msg = 'last dim of points should have shape 3 (X, Y, Z) but had shape %r'
                raise ValueError(msg % points.shape[-1])
            if points.shape[0] == 0:
                msg = 'Batch size of 0 not supported yet. Got input points shape {}.'.format(points.shape)
                raise ValueError(msg)
            if not (normals is None or normals.shape == points.shape):
                msg = "normals tensor should have same shape as points tensor, but didn't: %r != %r"
                raise ValueError(msg % (normals.shape, points.shape))
            if not (colors is None or colors.shape == points.shape):
                msg = "colors tensor should have same shape as points tensor, but didn't: %r != %r"
                raise ValueError(msg % (colors.shape, points.shape))
            if not (features is None or features.ndim == 3):
                msg = 'features should have ndim=3, but had ndim={}'.format(features.ndim)
                raise ValueError(msg)
            if not (features is None or features.shape[:-1] == points.shape[:-1]):
                msg = "first 2 dims of features tensor and points tensor should have same shape, but didn't: %r != %r"
                raise ValueError(msg % (features.shape[:-1], points.shape[:-1]))
            self._points_padded = points
            self._normals_padded = None if normals is None else normals
            self._colors_padded = None if colors is None else colors
            self._features_padded = None if features is None else features
            self._B = self._points_padded.shape[0]
            self._N = self._points_padded.shape[1]
            self._num_points_per_pointcloud = torch.tensor([self._N for _ in range(self._B)], device=self.device)
            self.equisized = True
        elif points is None:
            self.device = torch.Tensor().device if device is not None else torch.device('cpu')
            self._B = 0
            self._N = 0
            self._num_points_per_pointcloud = torch.tensor([0], device=self.device)
            self.equisized = None
        else:
            raise ValueError('points must either be None, a list, or a tensor with shape (batch_size, N, 3) where N is                     the maximum number of points.')

    def __len__(self):
        return self._B

    def __getitem__(self, index):
        """
        Args:
            index (int or slice or list of int or torch.Tensor): Specifying the index of the pointclouds to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            gradslam.Pointclouds: Selected pointclouds. The pointclouds tensors are not cloned.
        """
        if not self.has_points:
            raise IndexError('Cannot index empty pointclouds object')
        if isinstance(index, (int, slice)):
            points = self.points_list[index]
            normals = self.normals_list[index] if self.has_normals else None
            colors = self.colors_list[index] if self.has_colors else None
            features = self.features_list[index] if self.has_features else None
        elif isinstance(index, list):
            points = [self.points_list[i] for i in index]
            normals = [self.normals_list[i] for i in index] if self.has_normals else None
            colors = [self.colors_list[i] for i in index] if self.has_colors else None
            features = [self.features_list[i] for i in index] if self.has_features else None
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            if index.dtype == torch.bool:
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            points = [self.points_list[i] for i in index]
            normals = [self.normals_list[i] for i in index] if self.has_normals else None
            colors = [self.colors_list[i] for i in index] if self.has_colors else None
            features = [self.features_list[i] for i in index] if self.has_features else None
        else:
            raise IndexError(index)
        if isinstance(points, list):
            return Pointclouds(points=points, normals=normals, colors=colors, features=features)
        elif torch.is_tensor(points):
            points = [points]
            normals = None if normals is None else [normals]
            colors = None if colors is None else [colors]
            features = None if features is None else [features]
            return Pointclouds(points=points, normals=normals, colors=colors, features=features)
        else:
            raise ValueError('points not defined correctly')

    def __add__(self, other):
        """Out-of-place implementation of `Pointclouds.offset_`"""
        try:
            return self.clone().offset_(other)
        except TypeError:
            raise NotImplementedError('Pointclouds + {} currently not implemented.'.format(type(other)))

    def __sub__(self, other):
        """Subtracts `other` from all Pointclouds' points (`Pointclouds` - `other`).

        Args:
            other (torch.Tensor or float or int): Value(s) to subtract from all points.

        returns:
            gradslam.Pointclouds: Subtracted Pointclouds
        """
        try:
            return self.clone().offset_(other * -1)
        except TypeError:
            raise NotImplementedError('Pointclouds - {} currently not implemented.'.format(type(other)))

    def __mul__(self, other):
        """Out-of-place implementation of `Pointclouds.scale_`"""
        try:
            return self.clone().scale_(other)
        except TypeError:
            raise NotImplementedError('Pointclouds * {} currently not implemented.'.format(type(other)))

    def __truediv__(self, other):
        """Divides all Pointclouds' points by `other`.

        Args:
            other (torch.Tensor or float or int): Value(s) to divide all points by.

        Returns:
            self

        Shape:
            - other: Any. Must be compatible with :math:`(B, N, 3)`.
        """
        try:
            return self.__mul__(1.0 / other)
        except TypeError:
            raise NotImplementedError('Pointclouds / {} currently not implemented.'.format(type(other)))

    def __matmul__(self, other):
        """Post-multiplication :math:`SE(3)` transformation or :math:`SO(3)` rotation of Pointclouds' points and
        normals.

        Args:
            other (torch.Tensor): Either :math:`SE(3)` transformation or :math:`SO(3)` rotation

        Returns:
            self

        Shape:
            - other: Either :math:`SE(3)` transformation of shape :math:`(4, 4)` or :math:`(B, 4, 4)`, or :math:`SO(3)`
                rotation of shape :math:`(3, 3)` or :math:`(B, 3, 3)`
        """
        if not torch.is_tensor(other):
            raise NotImplementedError('Pointclouds @ {} currently not implemented.'.format(type(other)))
        if not ((other.ndim == 2 or other.ndim == 3) and (other.shape[-2:] == (3, 3) or other.shape[-2:] == (4, 4))):
            msg = 'Unsupported shape for Pointclouds @ operand: {}\n'.format(other.shape)
            msg += 'Use tensor of shape (3, 3) or (B, 3, 3) for rotations, or (4, 4) or (B, 4, 4) for transformations'
            raise ValueError(msg)
        if other.shape[-2:] == (3, 3):
            return self.clone().rotate_(other, pre_multiplication=False)
        if other.shape[-2:] == (4, 4):
            return self.clone().transform_(other, pre_multiplication=False)

    def rotate(self, rmat: torch.Tensor, *, pre_multiplication=True):
        """Out-of-place implementation of `Pointclouds.rotate_`"""
        return self.clone().rotate_(rmat, pre_multiplication=pre_multiplication)

    def transform(self, transform: torch.Tensor, *, pre_multiplication=True):
        """Out-of-place implementation of `Pointclouds.transform_`"""
        return self.clone().transform_(transform, pre_multiplication=pre_multiplication)

    def pinhole_projection(self, intrinsics: torch.Tensor):
        """Out-of-place implementation of `Pointclouds.pinhole_projection_`"""
        return self.clone().pinhole_projection_(intrinsics)

    def offset_(self, offset: Union[torch.Tensor, float, int]):
        """Adds :math:`offset` to all Pointclouds' points. In place operation.

        Args:
            offset (torch.Tensor or float or int): Value(s) to add to all points.

        Returns:
            self

        Shape:
            - offset: Any. Must be compatible with :math:`(B, N, 3)`.
        """
        if not (torch.is_tensor(offset) or isinstance(offset, float) or isinstance(offset, int)):
            raise TypeError('Operand should be tensor, float or int but was %r instead' % type(offset))
        if not self.has_points:
            return self
        self._points_padded = self.points_padded + offset * self.nonpad_mask.unsqueeze(-1)
        self._points_list = None
        return self

    def scale_(self, scale: Union[torch.Tensor, float, int]):
        """Scales all Pointclouds' points by `scale`. In place operation.

        Args:
            scale (torch.Tensor or float or int): Value(s) to scale all points by.

        Returns:
            self

        Shape:
            - scale: Any. Must be compatible with :math:`(B, N, 3)`.
        """
        if not (torch.is_tensor(scale) or isinstance(scale, float) or isinstance(scale, int)):
            raise TypeError('Operand should be tensor, float or int but was %r instead' % type(scale))
        if not self.has_points:
            return self
        self._points_padded = self.points_padded * scale * self.nonpad_mask.unsqueeze(-1)
        self._points_list = None
        return self

    def rotate_(self, rmat: torch.Tensor, *, pre_multiplication=True):
        """Applies batch or single :math:`SO(3)` rotation to all Pointclouds' points and normals. In place operation.

        Args:
            rmat (torch.Tensor): Either batch or single :math:`SO(3)` rotation matrix
            pre_multiplication (torch.Tensor): If True, will pre-multiply the rotation. Otherwise will
                post-multiply the rotation. Default: True

        Returns:
            self

        Shape:
            - rmat: :math:`(3, 3)` or :math:`(B, 3, 3)`
        """
        if not torch.is_tensor(rmat):
            raise TypeError('Rotation matrix should be tensor, but was %r instead' % type(rmat))
        if not ((rmat.ndim == 2 or rmat.ndim == 3) and rmat.shape[-2:] == (3, 3)):
            raise ValueError('Rotation matrix should be of shape (3, 3) or (B, 3, 3), but was {} instead.'.format(rmat.shape))
        if rmat.ndim == 3 and rmat.shape[0] != self._B:
            raise ValueError('Rotation matrix batch size ({}) != Pointclouds batch size ({})'.format(rmat.shape[0], self._B))
        if not self.has_points:
            return self
        if pre_multiplication:
            rmat = rmat.transpose(-1, -2)
        if rmat.ndim == 2:
            self._points_padded = torch.einsum('bij,jk->bik', self.points_padded, rmat)
            self._normals_padded = None if self.normals_padded is None else torch.einsum('bij,jk->bik', self.normals_padded, rmat)
        elif rmat.ndim == 3:
            self._points_padded = torch.einsum('bij,bjk->bik', self.points_padded, rmat)
            self._normals_padded = None if self.normals_padded is None else torch.einsum('bij,bjk->bik', self.normals_padded, rmat)
        self._points_list = None
        self._normals_list = None
        return self

    def transform_(self, transform: torch.Tensor, *, pre_multiplication=True):
        """Applies batch or single :math:`SE(3)` transformation to all Pointclouds' points and normals. In place
        operation.

        Args:
            transform (torch.Tensor): Either batch or single :math:`SE(3)` transformation tensor
            pre_multiplication (torch.Tensor): If True, will pre-multiply the transformation. Otherwise will
                post-multiply the transformation. Default: True

        Returns:
            self

        Shape:
            - transform: :math:`(4, 4)` or :math:`(B, 4, 4)`
        """
        if not torch.is_tensor(transform):
            raise TypeError('transform should be tensor, but was %r instead' % type(transform))
        if not ((transform.ndim == 2 or transform.ndim == 3) and transform.shape[-2:] == (4, 4)):
            raise ValueError('transform should be of shape (4, 4) or (B, 4, 4), but was {} instead.'.format(transform.shape))
        if transform.ndim == 3 and transform.shape[0] != self._B:
            raise ValueError('transform batch size ({}) != Pointclouds batch size ({})'.format(transform.shape[0], self._B))
        if not self.has_points:
            return self
        rmat = transform[..., :3, :3]
        tvec = transform[..., :3, 3]
        while tvec.ndim < self.points_padded.ndim:
            tvec = tvec.unsqueeze(-2)
        return self.rotate_(rmat, pre_multiplication=pre_multiplication).offset_(tvec)

    def pinhole_projection_(self, intrinsics: torch.Tensor):
        """Projects Pointclouds' points onto :math:`z=1` plane using intrinsics of a pinhole camera. In place
        operation.

        Args:
            intrinsics (torch.Tensor): Either batch or single intrinsics matrix

        Returns:
            self

        Shape:
            - intrinsics: :math:`(4, 4)` or :math:`(B, 4, 4)`
        """
        if not torch.is_tensor(intrinsics):
            raise TypeError('intrinsics should be tensor, but was {} instead'.format(type(intrinsics)))
        if not ((intrinsics.ndim == 2 or intrinsics.ndim == 3) and intrinsics.shape[-2:] == (4, 4)):
            msg = 'intrinsics should be of shape (4, 4) or (B, 4, 4), but was {} instead.'.format(intrinsics.shape)
            raise ValueError(msg)
        if not self.has_points:
            return self
        projected_2d = projutils.project_points(self.points_padded, intrinsics)
        self._points_padded = projutils.homogenize_points(projected_2d) * self.nonpad_mask.unsqueeze(-1)
        self._points_list = None
        return self

    @property
    def has_points(self):
        """Determines whether pointclouds have points or not

        Returns:
            bool
        """
        if self._has_points is None:
            self._has_points = self._points_list is not None or self._points_padded is not None
        return self._has_points

    @property
    def has_normals(self):
        """Determines whether pointclouds have normals or not

        Returns:
            bool
        """
        if self._has_normals is None:
            self._has_normals = self._normals_list is not None or self._normals_padded is not None
        return self._has_normals

    @property
    def has_colors(self):
        """Determines whether pointclouds have colors or not

        Returns:
            bool
        """
        if self._has_colors is None:
            self._has_colors = self._colors_list is not None or self._colors_padded is not None
        return self._has_colors

    @property
    def has_features(self):
        """Determines whether pointclouds have features or not

        Returns:
            bool
        """
        if self._has_features is None:
            self._has_features = self._features_list is not None or self._features_padded is not None
        return self._has_features

    @property
    def num_features(self):
        """Determines number of features in pointclouds

        Returns:
            int
        """
        if not self.has_features:
            return 0
        if self._features_padded is not None:
            return self._features_padded.shape[-1]
        if self._features_list is not None:
            return self._features_list[0].shape[-1]

    @property
    def points_list(self):
        """Gets the list representation of the points.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
        """
        if self._points_list is None and self._points_padded is not None:
            self._points_list = [p[0, :self._num_points_per_pointcloud[b]] for b, p in enumerate(self._points_padded.split([1] * self._B, 0))]
        return self._points_list

    @property
    def normals_list(self):
        """Gets the list representation of the point normals.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of normals of shape :math:`(N_b, 3)`.
        """
        if self._normals_list is None and self._normals_padded is not None:
            self._normals_list = [n[0, :self._num_points_per_pointcloud[b]] for b, n in enumerate(self._normals_padded.split([1] * self._B, 0))]
        return self._normals_list

    @property
    def colors_list(self):
        """Gets the list representation of the point colors.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of colors of shape :math:`(N_b, 3)`.
        """
        if self._colors_list is None and self._colors_padded is not None:
            self._colors_list = [c[0, :self._num_points_per_pointcloud[b]] for b, c in enumerate(self._colors_padded.split([1] * self._B, 0))]
        return self._colors_list

    @property
    def features_list(self):
        """Gets the list representation of the point features.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of features of shape :math:`(N_b, 3)`.
        """
        if self._features_list is None and self._features_padded is not None:
            self._features_list = [f[0, :self._num_points_per_pointcloud[b]] for b, f in enumerate(self._features_padded.split([1] * self._B, 0))]
        return self._features_list

    @property
    def points_padded(self):
        """Gets the padded representation of the points.

        Returns:
            torch.Tensor: tensor representation of points with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._points_padded

    @property
    def normals_padded(self):
        """Gets the padded representation of the normals.

        Returns:
            torch.Tensor: tensor representation of normals with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._normals_padded

    @property
    def colors_padded(self):
        """Gets the padded representation of the colors.

        Returns:
            torch.Tensor: tensor representation of colors with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._colors_padded

    @property
    def features_padded(self):
        """Gets the padded representation of the features.

        Returns:
            torch.Tensor: tensor representation of features with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), C)`
        """
        self._compute_padded()
        return self._features_padded

    @property
    def nonpad_mask(self):
        """Returns tensor of `bool` values which are True wherever points exist and False wherever there is padding.

        Returns:
            torch.Tensor: 2d `bool` mask

        Shape:
            - Output: :math:`(B, N)`
        """
        if self._nonpad_mask is None and self.has_points:
            self._nonpad_mask = torch.ones((self._B, self._N), dtype=torch.bool, device=self.device)
            if self.equisized:
                self._nonpad_mask[:, self._num_points_per_pointcloud[0]:] = 0
            else:
                for b in range(self._B):
                    self._nonpad_mask[b, self._num_points_per_pointcloud[b]:] = 0
        return self._nonpad_mask

    @property
    def num_points_per_pointcloud(self):
        """Returns a 1D tensor with length equal to the number of pointclouds giving the number of points in each
        pointcloud.

        Returns:
            torch.Tensor: 1D tensor of sizes

        Shape:
            - Output: tensor of shape :math:`(B)`.
        """
        return self._num_points_per_pointcloud

    @points_list.setter
    def points_list(self, value: List[torch.Tensor]):
        """Updates `points_list` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change.

        Args:
            value (list of torch.Tensor): list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
                Shape of tensors in `value` and `pointclouds.points_list` must match.

        """
        self._assert_set_list(value)
        self._points_list = [v.clone() for v in value]
        self._points_padded = None

    @normals_list.setter
    def normals_list(self, value: List[torch.Tensor]):
        """Updates `normals_list` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change.

        Args:
            value (list of torch.Tensor): list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
                Shape of tensors in `value` and `pointclouds.points_list` must match.

        """
        self._assert_set_list(value)
        self._normals_list = [v.clone() for v in value]
        self._noramls_padded = None

    @colors_list.setter
    def colors_list(self, value: List[torch.Tensor]):
        """Updates `colors_list` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change.

        Args:
            value (list of torch.Tensor): list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
                Shape of tensors in `value` and `pointclouds.points_list` must match.

        """
        self._assert_set_list(value)
        self._colors_list = [v.clone() for v in value]
        self._noramls_padded = None

    @features_list.setter
    def features_list(self, value: List[torch.Tensor]):
        """Updates `features_list` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change.

        Args:
            value (list of torch.Tensor): list of :math:`B` tensors of points of shape :math:`(N_b, C)`.
                Shape of tensors in `value` and `pointclouds.points_list` must match.

        """
        self._assert_set_list(value, first_dim_only=True)
        self._features_list = [v.clone() for v in value]
        self._noramls_padded = None

    @points_padded.setter
    def points_padded(self, value: torch.Tensor):
        """Updates `points_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `points_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) points with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b), 3)`
        """
        self._assert_set_padded(value)
        self._points_padded = value.clone()
        self._points_list = None

    @normals_padded.setter
    def normals_padded(self, value: torch.Tensor):
        """Updates `normals_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `normals_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) normals with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b), 3)`
        """
        self._assert_set_padded(value)
        self._normals_padded = value.clone()
        self._normals_list = None

    @colors_padded.setter
    def colors_padded(self, value: torch.Tensor):
        """Updates `colors_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `colors_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) colors with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b), 3)`
        """
        self._assert_set_padded(value)
        self._colors_padded = value.clone()
        self._colors_list = None

    @features_padded.setter
    def features_padded(self, value: torch.Tensor):
        """Updates `features_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `features_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) features with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b), C)`
        """
        self._assert_set_padded(value, first_2_dims_only=True)
        self._features_padded = value.clone()
        self._features_list = None

    def _compute_padded(self, refresh: bool=False):
        """Computes the padded version of pointclouds.

        Args:
            refresh (bool): If True, will recompute padded representation even if it already exists
        """
        if not self.has_points:
            return
        if not (refresh or self._points_padded is None):
            return
        self._points_padded = structutils.list_to_padded(self._points_list, (self._N, 3), pad_value=0.0, equisized=self.equisized)
        self._normals_padded = None if self._normals_list is None else structutils.list_to_padded(self._normals_list, (self._N, 3), pad_value=0.0, equisized=self.equisized)
        self._colors_padded = None if self._colors_list is None else structutils.list_to_padded(self._colors_list, (self._N, 3), pad_value=0.0, equisized=self.equisized)
        self._features_padded = None if self._features_list is None else structutils.list_to_padded(self._features_list, (self._N, self.num_features), pad_value=0.0, equisized=self.equisized)

    def clone(self):
        """Returns deep copy of Pointclouds object. All internal tensors are cloned individually.

        Returns:
            gradslam.Pointclouds: cloned gradslam.Pointclouds object
        """
        if not self.has_points:
            return Pointclouds(device=self.device)
        elif self._points_list is not None:
            new_points = [p.clone() for p in self.points_list]
            new_normals = None if self._normals_list is None else [n.clone() for n in self._normals_list]
            new_colors = None if self._colors_list is None else [c.clone() for c in self._colors_list]
            new_features = None if self._features_list is None else [f.clone() for f in self._features_list]
        elif self._points_padded is not None:
            new_points = self._points_padded.clone()
            new_normals = None if self._normals_padded is None else self._normals_padded.clone()
            new_colors = None if self._colors_padded is None else self._colors_padded.clone()
            new_features = None if self._features_padded is None else self._features_padded.clone()
        other = Pointclouds(points=new_points, normals=new_normals, colors=new_colors, features=new_features)
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def detach(self):
        """Detachs Pointclouds object. All internal tensors are detached individually.

        Returns:
            gradslam.Pointclouds: detached gradslam.Pointclouds object
        """
        other = self.clone()
        if other._points_list is not None:
            other._points_list = [p.detach() for p in other._points_list]
        if other._normals_list is not None:
            other._normals_list = [n.detach() for n in other._normals_list]
        if other._colors_list is not None:
            other._colors_list = [c.detach() for c in other._colors_list]
        if other._features_list is not None:
            other._features_list = [f.detach() for f in other._features_list]
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
        return other

    def to(self, device: Union[torch.device, str], copy: bool=False):
        """Match functionality of torch.Tensor.to(device)
        If copy = True or the self Tensor is on a different device, the returned tensor is a copy of self with the
        desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device, then self is returned.

        Args:
            device (torch.device or str): Device id for the new tensor.
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.Pointclouds
        """
        if not copy and self.device == device:
            return self
        other = self.clone()
        if self.device != device:
            other.device = torch.Tensor().device
            if other._points_list is not None:
                other._points_list = [p for p in other._points_list]
            if other._normals_list is not None:
                other._normals_list = [n for n in other._normals_list]
            if other._colors_list is not None:
                other._colors_list = [c for c in other._colors_list]
            if other._features_list is not None:
                other._features_list = [f for f in other._features_list]
            for k in self._INTERNAL_TENSORS:
                v = getattr(self, k)
                if torch.is_tensor(v):
                    setattr(other, k, v)
        return other

    def cpu(self):
        """Match functionality of torch.Tensor.cpu()

        Returns:
            gradslam.Pointclouds
        """
        return self

    def cuda(self):
        """Match functionality of torch.Tensor.cuda()

        Returns:
            gradslam.Pointclouds
        """
        return self

    def append_points(self, pointclouds: 'Pointclouds'):
        """Appends points, normals, colors and features of a gradslam.Pointclouds object to the current pointclouds.
        Both Pointclouds must have/not have the same attributes. In place operation.

        Args:
            pointclouds (gradslam.Pointclouds): Pointclouds to get appended to self. Must have same batch size as self.

        Returns:
            self
        """
        if not isinstance(pointclouds, type(self)):
            raise TypeError('Append object must be of type gradslam.Pointclouds, but was of type {}.'.format(type(pointclouds)))
        if not pointclouds.device == self.device:
            raise ValueError('Device of pointclouds to append and to be appended must match: ({0} != {1})'.format(pointclouds.device, self.device))
        if not pointclouds.has_points:
            return self
        if not self.has_points:
            if pointclouds.has_points:
                self._points_list = [p.clone() for p in pointclouds.points_list]
                if pointclouds.has_normals:
                    self._normals_list = [n.clone() for n in pointclouds.normals_list]
                if pointclouds.has_colors:
                    self._colors_list = [c.clone() for c in pointclouds.colors_list]
                if pointclouds.has_features:
                    self._features_list = [f.clone() for f in pointclouds.features_list]
                self._has_points = pointclouds._has_points
                self._has_normals = pointclouds._has_normals
                self._has_colors = pointclouds._has_colors
                self._has_features = pointclouds._has_features
                self._B = pointclouds._B
                self._N = pointclouds._N
                self.equisized = pointclouds.equisized
                for k in self._INTERNAL_TENSORS:
                    v = getattr(pointclouds, k)
                    if torch.is_tensor(v):
                        setattr(self, k, v.clone())
            return self
        if not len(pointclouds) == len(self):
            raise ValueError('Batch size of pointclouds to append and to be appended must match: ({0} != {1})'.format(len(pointclouds), len(self)))
        if self.has_normals != pointclouds.has_normals:
            raise ValueError('pointclouds to append and to be appended must either both have or not have normals: ({0} != {1})'.format(pointclouds.has_normals, self.has_normals))
        if self.has_colors != pointclouds.has_colors:
            raise ValueError('pointclouds to append and to be appended must either both have or not have colors: ({0} != {1})'.format(pointclouds.has_colors, self.has_colors))
        if self.has_features != pointclouds.has_features:
            raise ValueError('pointclouds to append and to be appended must either both have or not have features: ({0} != {1})'.format(pointclouds.has_features, self.has_features))
        if self.has_features and self.num_features != pointclouds.num_features:
            raise ValueError('pointclouds to append and to be appended must have the same number of features: ({0} != {1})'.format(pointclouds.num_features, self.num_features))
        self._points_list = [torch.cat([self.points_list[b], pointclouds.points_list[b]], 0) for b in range(self._B)]
        self._points_padded = None
        if self.has_normals:
            self._normals_list = [torch.cat([self.normals_list[b], pointclouds.normals_list[b]], 0) for b in range(self._B)]
            self._normals_padded = None
        if self.has_colors:
            self._colors_list = [torch.cat([self.colors_list[b], pointclouds.colors_list[b]], 0) for b in range(self._B)]
            self._colors_padded = None
        if self.has_features:
            self._features_list = [torch.cat([self.features_list[b], pointclouds.features_list[b]], 0) for b in range(self._B)]
            self._features_padded = None
        self._num_points_per_pointcloud = self._num_points_per_pointcloud + pointclouds._num_points_per_pointcloud
        self.equisized = len(self._num_points_per_pointcloud.unique()) == 1
        self._N = self._num_points_per_pointcloud.max()
        self._nonpad_mask = None
        return self

    def open3d(self, index: int, include_colors: bool=True, max_num_points: Optional[int]=None, include_normals: bool=False):
        """Converts `index`-th pointcloud to a `open3d.geometry.Pointcloud` object (e.g. for visualization).

        Args:
            index (int): Index of which pointcloud (from the batch of pointclouds) to convert to
                `open3d.geometry.Pointcloud`.
            include_colors (bool): If True, will include colors in the `o3d.geometry.Pointcloud`
                objects. Default: True
            max_num_points (int): Maximum number of points to include in the returned object. If None,
                will not set a max size (will not downsample). Default: None
            include_normals (bool): If True, will include normal vectors in the `o3d.geometry.Pointcloud`
                objects. Default: False

        Returns:
            pcd (open3d.geometry.Pointcloud): `open3d.geometry.Pointcloud` object from `index`-th pointcloud.
        """
        if not isinstance(index, int):
            raise TypeError('Index should be int, but was {}.'.format(type(index)))
        pcd = o3d.geometry.PointCloud()
        num_points = self.num_points_per_pointcloud[index]
        torch_points = self.points_list[index]
        subsample = max_num_points is not None and max_num_points < num_points
        if subsample:
            perm = torch.randperm(num_points)
            point_inds = perm[:max_num_points]
            torch_points = torch_points[point_inds]
        numpy_points = torch_points.detach().cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(numpy_points)
        if self.has_colors and include_colors:
            torch_colors = self.colors_list[index]
            if subsample:
                torch_colors = torch_colors[point_inds]
            if (torch_colors.max() > 1.1).item():
                torch_colors = torch_colors / 255
            torch_colors = torch.clamp(torch_colors, min=0.0, max=1.0)
            numpy_colors = torch_colors.detach().cpu().numpy()
            pcd.colors = o3d.utility.Vector3dVector(numpy_colors)
        if self.has_normals and include_normals:
            torch_normals = self.normals_list[index]
            if subsample:
                torch_normals = torch_normals[point_inds]
            numpy_normals = torch_normals.detach().cpu().numpy()
            pcd.normals = o3d.utility.Vector3dVector(numpy_normals)
        return pcd

    def plotly(self, index: int, include_colors: bool=True, max_num_points: Optional[int]=200000, as_figure: bool=True, point_size: int=2):
        """Converts `index`-th pointcloud to either a `plotly.graph_objects.Figure` or a
        `plotly.graph_objects.Scatter3d` object (for visualization).

        Args:
            index (int): Index of which pointcloud (from the batch of pointclouds) to convert to plotly
                representation.
            include_colors (bool): If True, will include point colors in the returned object. Default: True
            max_num_points (int): Maximum number of points to include in the returned object. If None,
                will not set a max size (will not downsample). Default: 200000
            as_figure (bool): If True, returns a `plotly.graph_objects.Figure` object which can easily
                be visualized by calling `.show()` on. Otherwise, returns a
                `plotly.graph_objects.Scatter3d` object. Default: True
            point_size (int): Point size radius (for visualization). Default: 2

        Returns:
            plotly.graph_objects.Figure or plotly.graph_objects.Scatter3d: If `as_figure` is True, will return
            `plotly.graph_objects.Figure` object from the `index`-th pointcloud. Else,
            returns `plotly.graph_objects.Scatter3d` object from the `index`-th pointcloud.
        """
        if not isinstance(index, int):
            raise TypeError('Index should be int, but was {}.'.format(type(index)))
        num_points = self.num_points_per_pointcloud[index]
        torch_points = self.points_list[index]
        subsample = max_num_points is not None and max_num_points < num_points
        if subsample:
            perm = torch.randperm(num_points)
            point_inds = perm[:max_num_points]
            torch_points = torch_points[point_inds]
        numpy_points = torch_points.detach().cpu().numpy()
        marker_dict = {'size': point_size}
        if self.has_colors and include_colors:
            torch_colors = self.colors_list[index]
            if subsample:
                torch_colors = torch_colors[point_inds]
            if (torch_colors.max() < 1.1).item():
                torch_colors = torch_colors * 255
            torch_colors = torch.clamp(torch_colors, min=0.0, max=255.0)
            numpy_colors = torch_colors.detach().cpu().numpy().astype('uint8')
            marker_dict['color'] = numpy_colors
        scatter3d = go.Scatter3d(x=numpy_points[..., 0], y=numpy_points[..., 1], z=numpy_points[..., 2], mode='markers', marker=marker_dict)
        if not as_figure:
            return scatter3d
        fig = go.Figure(data=[scatter3d])
        fig.update_layout(showlegend=False, scene=dict(xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False), yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False), zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, visible=False)))
        return fig

    def _assert_set_padded(self, value: torch.Tensor, first_2_dims_only: bool=False):
        """Checks if value can be set as a padded representation attribute

        Args:
            value (torch.Tensor): value we want to set as one of the padded representation attributes
            first_2_dims_only (bool): If True, will only check if first 2 dimensions of value are the same as
                `self.points_padded`. Otherwise will check the entire shape. Default: False
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError('value must be torch.Tensor. Got {}'.format(type(value)))
        if not self.has_points:
            raise ValueError('cannot set padded representation for an empty pointclouds object')
        if self.device != torch.device(value.device):
            raise ValueError('value must have the same device as pointclouds object: {} != {}'.format(value.device, torch.device(self.device)))
        if value.ndim != 3:
            raise ValueError('value.ndim should be 3. Got {}'.format(value.ndim))
        if first_2_dims_only and self.points_padded.shape[:2] != value.shape[:2]:
            raise ValueError("first 2 dims of value tensor and points tensor should have same shape, but didn't: {} != {}.".format(value.shape[:2], self.points_padded.shape[:2]))
        if not first_2_dims_only and self.points_padded.shape != value.shape:
            raise ValueError("value tensor and points tensor should have same shape, but didn't: {} != {}.".format(value.shape, self.points_padded.shape))
        if not all([value[b][N_b:].eq(0).all().item() for b, N_b in enumerate(self.num_points_per_pointcloud)]):
            raise ValueError('value must have zeros wherever pointclouds.points_padded has zero padding.')

    def _assert_set_list(self, value: List[torch.Tensor], first_dim_only: bool=False):
        """Checks if value can be set as a list representation attribute

        Args:
            value (list of torch.Tensor): value we want to set as one of the list representation attributes
            first_dim_only (bool): If True, will only check if first dimension of value is the same as
                `self.points_padded`. Otherwise will check the entire shape. Default: False
        """
        if not isinstance(value, list):
            raise TypeError('value must be list of torch.Tensors. Got {}'.format(type(value)))
        if not self.has_points:
            raise ValueError('cannot set list representation for an empty pointclouds object')
        if len(self) != len(value):
            raise ValueError('value must have same length as pointclouds.points_list. Got {} != {}.'.format(len(value), len(self)))
        if any([(v.ndim != 2) for v in value]):
            raise ValueError('ndim of all tensors in value list should be 2')
        if first_dim_only and any([(self.points_list[b].shape[:1] != value[b].shape[:1]) for b in range(len(self))]):
            raise ValueError('shape of first 2 dims of tensors in value and pointclouds.points_list must match')
        if not first_dim_only and any([(self.points_list[b].shape != value[b].shape) for b in range(len(self))]):
            raise ValueError('shape of tensors in value and pointclouds.points_list must match')


def gauss_newton_solve(src_pc: torch.Tensor, tgt_pc: torch.Tensor, tgt_normals: torch.Tensor, dist_thresh: Union[float, int, None]=None):
    """Computes Gauss Newton step by forming linear equation. Points from `src_pc` which have a distance greater
    than `dist_thresh` to the closest point in `tgt_pc` will be filtered.

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs warping).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the source pointcloud must be warped to).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point in the target pointcloud.
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
            Default: None

    Returns:
        tuple: tuple containing:

        - A (torch.Tensor): linear system equation
        - b (torch.Tensor): linear system residual
        - chamfer_indices (torch.Tensor): Index of the closest point in `tgt_pc` for each point in `src_pc`
            that was not filtered out.

    Shape:
        - src_pc: :math:`(1, N_s, 3)`
        - tgt_pc: :math:`(1, N_t, 3)`
        - tgt_normals: :math:`(1, N_t, 3)`
        - A: :math:`(N_sf, 6)` where :math:`N_sf \\leq N_s`
        - b: :math:`(N_sf, 1)` where :math:`N_sf \\leq N_s`
        - chamfer_indices: :math:`(1, N_sf)` where :math:`N_sf \\leq N_s`
    """
    if not torch.is_tensor(src_pc):
        raise TypeError('Expected src_pc to be of type torch.Tensor. Got {0}.'.format(type(src_pc)))
    if not torch.is_tensor(tgt_pc):
        raise TypeError('Expected tgt_pc to be of type torch.Tensor. Got {0}.'.format(type(tgt_pc)))
    if not torch.is_tensor(tgt_normals):
        raise TypeError('Expected tgt_normals to be of type torch.Tensor. Got {0}.'.format(type(tgt_normals)))
    if not (isinstance(dist_thresh, float) or isinstance(dist_thresh, int) or dist_thresh is None):
        raise TypeError('Expected dist_thresh to be of type float or int. Got {0}.'.format(type(dist_thresh)))
    if src_pc.ndim != 3:
        raise ValueError('src_pc should have ndim=3, but had ndim={}'.format(src_pc.ndim))
    if tgt_pc.ndim != 3:
        raise ValueError('tgt_pc should have ndim=3, but had ndim={}'.format(tgt_pc.ndim))
    if tgt_normals.ndim != 3:
        raise ValueError('tgt_normals should have ndim=3, but had ndim={}'.format(tgt_normals.ndim))
    if src_pc.shape[0] != 1:
        raise ValueError('src_pc.shape[0] should be 1, but was {} instead'.format(src_pc.shape[0]))
    if tgt_pc.shape[0] != 1:
        raise ValueError('tgt_pc.shape[0] should be 1, but was {} instead'.format(tgt_pc.shape[0]))
    if tgt_normals.shape[0] != 1:
        raise ValueError('tgt_normals.shape[0] should be 1, but was {} instead'.format(tgt_normals.shape[0]))
    if tgt_pc.shape[1] != tgt_normals.shape[1]:
        raise ValueError('tgt_pc.shape[1] and tgt_normals.shape[1] must be equal. Got {0}!={1}'.format(tgt_pc.shape[1], tgt_normals.shape[1]))
    if src_pc.shape[2] != 3:
        raise ValueError('src_pc.shape[2] should be 3, but was {} instead'.format(src_pc.shape[2]))
    if tgt_pc.shape[2] != 3:
        raise ValueError('tgt_pc.shape[2] should be 3, but was {} instead'.format(tgt_pc.shape[2]))
    if tgt_normals.shape[2] != 3:
        raise ValueError('tgt_normals.shape[2] should be 3, but was {} instead'.format(tgt_normals.shape[2]))
    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
    tgt_normals = tgt_normals.contiguous()
    _KNN = knn_points(src_pc, tgt_pc)
    dist1, idx1 = _KNN.dists.squeeze(-1), _KNN.idx.squeeze(-1)
    dist_filter = torch.ones_like(dist1[0], dtype=torch.bool) if dist_thresh is None else dist1[0] < dist_thresh
    chamfer_indices = idx1[0][dist_filter].long()
    sx = src_pc[0, dist_filter, 0].view(-1, 1)
    sy = src_pc[0, dist_filter, 1].view(-1, 1)
    sz = src_pc[0, dist_filter, 2].view(-1, 1)
    assoc_pts = torch.index_select(tgt_pc, 1, chamfer_indices)
    assoc_normals = torch.index_select(tgt_normals, 1, chamfer_indices)
    dx = assoc_pts[0, :, 0].view(-1, 1)
    dy = assoc_pts[0, :, 1].view(-1, 1)
    dz = assoc_pts[0, :, 2].view(-1, 1)
    nx = assoc_normals[0, :, 0].view(-1, 1)
    ny = assoc_normals[0, :, 1].view(-1, 1)
    nz = assoc_normals[0, :, 2].view(-1, 1)
    A = torch.cat([nx, ny, nz, nz * sy - ny * sz, nx * sz - nz * sx, ny * sx - nx * sy], 1)
    b = nx * (dx - sx) + ny * (dy - sy) + nz * (dz - sz)
    return A, b, chamfer_indices


_eps = 1e-06


def so3_hat(omega: torch.Tensor) ->torch.Tensor:
    """Implements the hat operator for SO(3), given an input axis-angle
    vector omega.

    """
    assert torch.is_tensor(omega), 'Input must be of type torch.tensor.'
    omega_hat = torch.zeros(3, 3).type(omega.dtype)
    omega_hat[0, 1] = -omega[2]
    omega_hat[1, 0] = omega[2]
    omega_hat[0, 2] = omega[1]
    omega_hat[2, 0] = -omega[1]
    omega_hat[1, 2] = -omega[0]
    omega_hat[2, 1] = omega[0]
    return omega_hat


def se3_exp(xi: torch.Tensor) ->torch.Tensor:
    """Computes the exponential map for the coordinate-vector xi.
    Returns a 4 x 4 SE(3) matrix.

    """
    assert torch.is_tensor(xi), 'Input must be of type torch.tensor.'
    v = xi[:3]
    omega = xi[3:]
    omega_hat = so3_hat(omega)
    if omega.norm() < _eps:
        R = torch.eye(3, 3).type(omega.dtype) + omega_hat
        V = torch.eye(3, 3).type(omega.dtype) + omega_hat
    else:
        theta = omega.norm()
        s = theta.sin()
        c = theta.cos()
        omega_hat_sq = omega_hat.mm(omega_hat)
        A = s / theta
        B = (1 - c) / torch.pow(theta, 2)
        C = (theta - s) / torch.pow(theta, 3)
        R = torch.eye(3, 3).type(omega.dtype) + A * omega_hat + B * omega_hat_sq
        V = torch.eye(3, 3).type(omega.dtype) + B * omega_hat + C * omega_hat_sq
    t = torch.mm(V, v.view(3, 1))
    last_row = torch.tensor([0, 0, 0, 1]).type(omega.dtype)
    return torch.cat((torch.cat((R, t), dim=1), last_row.unsqueeze(0)), dim=0)


def solve_linear_system(A: torch.Tensor, b: torch.Tensor, damp: Union[float, torch.Tensor]=1e-08):
    """Solves the normal equations of a linear system Ax = b, given the constraint matrix A and the coefficient vector
    b. Note that this solves the normal equations, not the linear system. That is, solves :math:`A^T A x = A^T b`,
    not :math:`Ax = b`.

    Args:
        A (torch.Tensor): The constraint matrix of the linear system.
        b (torch.Tensor): The coefficient vector of the linear system.
        damp (float or torch.Tensor): Damping coefficient to optionally condition the linear system (in practice,
            a damping coefficient of :math:`\\rho` means that we are solving a modified linear system that adds a tiny
            :math:`\\rho` to each diagonal element of the constraint matrix :math:`A`, so that the linear system
            becomes :math:`(A^TA + \\rho I)x = b`, where :math:`I` is the identity matrix of shape
            :math:`(\\text{num_of_variables}, \\text{num_of_variables})`. Default: 1e-8

    Returns:
        torch.Tensor: Solution vector of the normal equations of the linear system

    Shape:
        - A: :math:`(\\text{num_of_equations}, \\text{num_of_variables})`
        - b: :math:`(\\text{num_of_equations}, 1)`
        - Output: :math:`(\\text{num_of_variables}, 1)`
    """
    if not torch.is_tensor(A):
        raise TypeError('Expected A to be of type torch.Tensor. Got {0}.'.format(type(A)))
    if not torch.is_tensor(b):
        raise TypeError('Expected b to be of type torch.Tensor. Got {0}.'.format(type(b)))
    if not (isinstance(damp, float) or torch.is_tensor(damp)):
        raise TypeError('Expected damp to be of type float or torch.Tensor. Got {0}.'.format(type(damp)))
    if torch.is_tensor(damp) and damp.ndim != 0:
        raise ValueError('Expected torch.Tensor damp to have ndim=0 (scalar). Got {0}.'.format(damp.ndim))
    if A.ndim != 2:
        raise ValueError('A should have ndim=2, but had ndim={}'.format(A.ndim))
    if b.ndim != 2:
        raise ValueError('b should have ndim=2, but had ndim={}'.format(b.ndim))
    if b.shape[1] != 1:
        raise ValueError('b.shape[1] should 1, but was {0}'.format(b.shape[1]))
    if A.shape[0] != b.shape[0]:
        raise ValueError('A.shape[0] and b.shape[0] should be equal ({0} != {1})'.format(A.shape[0], b.shape[0]))
    damp = damp if torch.is_tensor(damp) else torch.tensor(damp, dtype=A.dtype, device=A.device)
    A_t = torch.transpose(A, 0, 1)
    damp_matrix = torch.eye(A.shape[1])
    At_A = torch.matmul(A_t, A) + damp_matrix * damp
    return torch.matmul(torch.inverse(At_A), torch.matmul(A_t, b))


def transform_pointcloud(pointcloud: torch.Tensor, transform: torch.Tensor):
    """Applies a rigid-body transformation to a pointcloud.

    Args:
        pointcloud (torch.Tensor): Pointcloud to be transformed
                                   (shape: numpts x 3)
        transform (torch.Tensor): An SE(3) rigid-body transform matrix
                                  (shape: 4 x 4)

    Returns:
        transformed_pointcloud (torch.Tensor): Rotated and translated cloud
                                               (shape: numpts x 3)

    """
    if not torch.is_tensor(pointcloud):
        raise TypeError('pointcloud should be tensor, but was %r instead' % type(pointcloud))
    if not torch.is_tensor(transform):
        raise TypeError('transform should be tensor, but was %r instead' % type(transform))
    if not pointcloud.ndim == 2:
        raise ValueError('pointcloud should have ndim of 2, but had {} instead.'.format(pointcloud.ndim))
    if not pointcloud.shape[1] == 3:
        raise ValueError('pointcloud.shape[1] should be 3 (x, y, z), but was {} instead.'.format(pointcloud.shape[1]))
    if not transform.shape[-2:] == (4, 4):
        raise ValueError('transform should be of shape (4, 4), but was {} instead.'.format(transform.shape))
    rmat = transform[:3, :3]
    tvec = transform[:3, 3]
    transposed_pointcloud = torch.transpose(pointcloud, 0, 1)
    transformed_pointcloud = torch.matmul(rmat, transposed_pointcloud) + tvec.unsqueeze(1)
    transformed_pointcloud = torch.transpose(transformed_pointcloud, 0, 1)
    return transformed_pointcloud


def point_to_plane_gradICP(src_pc: torch.Tensor, tgt_pc: torch.Tensor, tgt_normals: torch.Tensor, initial_transform: Optional[torch.Tensor]=None, numiters: int=20, damp: float=1e-08, dist_thresh: Union[float, int, None]=None, lambda_max: Union[float, int]=2.0, B: Union[float, int]=1.0, B2: Union[float, int]=1.0, nu: Union[float, int]=200.0):
    """Computes a rigid transformation between `tgt_pc` (target pointcloud) and `src_pc` (source pointcloud) using a
    point-to-plane error metric and gradLM (:math:`\\nabla LM`) solver (See gradLM section of 
    `the gradSLAM paper <https://arxiv.org/abs/1910.10672>`__).  The iterate and damping coefficient are updated by:

    .. math::

        lambda_1 = Q_\\lambda(r_0, r_1) & = \\lambda_{min} + \\frac{\\lambda_{max} -
        \\lambda_{min}}{1 + e^{-B (r_1 - r_0)}} \\\\
        Q_x(r_0, r_1) & = x_0 + \\frac{\\delta x_0}{\\sqrt[nu]{1 + e^{-B2*(r_1 - r_0)}}}`

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs warping).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the source pointcloud must be warped to).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point in the target pointcloud.
        initial_transform (torch.Tensor or None): The initial estimate of the transformation between 'src_pc' 
            and 'tgt_pc'. If None, will use the identity matrix as the initial transform. Default: None
        numiters (int): Number of iterations to run the optimization for. Default: 20
        damp (float): Damping coefficient for nonlinear least-squares. Default: 1e-8
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
            Default: None
        lambda_max (float or int): Maximum value the damping function can assume (`lambda_min` will be 
            :math:`\\frac{1}{\\text{lambda_max}}`)
        B (float or int): gradLM falloff control parameter
        B2 (float or int): gradLM control parameter
        nu (float or int): gradLM control parameter

    Returns:
        tuple: tuple containing:

        - transform (torch.Tensor): linear system residual
        - chamfer_indices (torch.Tensor): Index of the closest point in `tgt_pc` for each point in `src_pc` that was not
          filtered out.

    Shape:
        - src_pc: :math:`(1, N_s, 3)`
        - tgt_pc: :math:`(1, N_t, 3)`
        - tgt_normals: :math:`(1, N_t, 3)`
        - initial_transform: :math:`(4, 4)`
        - transform: :math:`(4, 4)`
        - chamfer_indices: :math:`(1, N_sf)` where :math:`N_sf \\leq N_s`

    """
    if not torch.is_tensor(src_pc):
        raise TypeError('Expected src_pc to be of type torch.Tensor. Got {0}.'.format(type(src_pc)))
    if not torch.is_tensor(tgt_pc):
        raise TypeError('Expected tgt_pc to be of type torch.Tensor. Got {0}.'.format(type(tgt_pc)))
    if not torch.is_tensor(tgt_normals):
        raise TypeError('Expected tgt_normals to be of type torch.Tensor. Got {0}.'.format(type(tgt_normals)))
    if not (torch.is_tensor(initial_transform) or initial_transform is None):
        raise TypeError('Expected initial_transform to be of type torch.Tensor. Got {0}.'.format(type(initial_transform)))
    if not isinstance(numiters, int):
        raise TypeError('Expected numiters to be of type int. Got {0}.'.format(type(numiters)))
    if not (isinstance(lambda_max, float) or isinstance(lambda_max, int)):
        raise TypeError('Expected lambda_max to be of type float or int; got {0}'.format(type(lambda_max)))
    if not (isinstance(B, float) or isinstance(B, int)):
        raise TypeError('Expected B to be of type float or int; got {0}'.format(type(B)))
    if not (isinstance(B2, float) or isinstance(B2, int)):
        raise TypeError('Expected B2 to be of type float or int; got {0}'.format(type(B2)))
    if not (isinstance(nu, float) or isinstance(nu, int)):
        raise TypeError('Expected nu to be of type float or int; got {0}'.format(type(nu)))
    if initial_transform.ndim != 2:
        raise ValueError('Expected initial_transform.ndim to be 2. Got {0}.'.format(initial_transform.ndim))
    if not (initial_transform.shape[0] == 4 and initial_transform.shape[1] == 4):
        raise ValueError('Expected initial_transform.shape to be (4, 4). Got {0}.'.format(initial_transform.shape))
    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
    tgt_normals = tgt_normals.contiguous()
    dtype = src_pc.dtype
    device = src_pc.device
    damp = torch.tensor(damp, dtype=dtype, device=device)
    lambda_min = 1 / lambda_max
    initial_transform = torch.eye(4, dtype=dtype, device=device) if initial_transform is None else initial_transform
    src_pc = transform_pointcloud(src_pc[0], initial_transform).unsqueeze(0)
    transform = initial_transform
    for it in range(numiters):
        A, b, chamfer_indices = gauss_newton_solve(src_pc, tgt_pc, tgt_normals, dist_thresh)
        residual = b[:, 0]
        xi = solve_linear_system(A, b, damp)
        residual_transform = se3_exp(xi)
        err = torch.dot(residual.t(), residual)
        pc_error = torch.sqrt(torch.sum((torch.mm(A, xi) - b) ** 2))
        one_step_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)
        _, one_step_b, chamfer_indices_onestep = gauss_newton_solve(one_step_pc, tgt_pc, tgt_normals, dist_thresh)
        one_step_residual = one_step_b[:, 0]
        new_err = torch.dot(one_step_residual.t(), one_step_residual)
        errdiff = new_err - err
        errdiff = errdiff.clamp(-70.0, 70.0)
        damp_new = lambda_min + (lambda_max - lambda_min) / (1 + torch.exp(-B * errdiff))
        damp = damp * damp_new
        sigmoid = 1 / (1 + torch.exp(-B2 * errdiff)) ** (1 / nu)
        residual_transform = se3_exp(sigmoid * xi)
        src_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)
        transform = torch.mm(residual_transform, transform)
    return transform, chamfer_indices


def point_to_plane_ICP(src_pc: torch.Tensor, tgt_pc: torch.Tensor, tgt_normals: torch.Tensor, initial_transform: Optional[torch.Tensor]=None, numiters: int=20, damp: float=1e-08, dist_thresh: Union[float, int, None]=None):
    """Computes a rigid transformation between `tgt_pc` (target pointcloud) and `src_pc` (source pointcloud) using a
    point-to-plane error metric and the LM (Levenberg–Marquardt) solver.

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs warping).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the source pointcloud must be warped to).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point in the target pointcloud.
        initial_transform (torch.Tensor or None): The initial estimate of the transformation between 'src_pc'
            and 'tgt_pc'. If None, will use the identity matrix as the initial transform. Default: None
        numiters (int): Number of iterations to run the optimization for. Default: 20
        damp (float): Damping coefficient for nonlinear least-squares. Default: 1e-8
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
            Default: None

    Returns:
        tuple: tuple containing:

        - transform (torch.Tensor): linear system residual
        - chamfer_indices (torch.Tensor): Index of the closest point in `tgt_pc` for each point in `src_pc` that was not
          filtered out.

    Shape:
        - src_pc: :math:`(1, N_s, 3)`
        - tgt_pc: :math:`(1, N_t, 3)`
        - tgt_normals: :math:`(1, N_t, 3)`
        - initial_transform: :math:`(4, 4)`
        - transform: :math:`(4, 4)`
        - chamfer_indices: :math:`(1, N_sf)` where :math:`N_sf \\leq N_s`

    """
    if not torch.is_tensor(src_pc):
        raise TypeError('Expected src_pc to be of type torch.Tensor. Got {0}.'.format(type(src_pc)))
    if not torch.is_tensor(tgt_pc):
        raise TypeError('Expected tgt_pc to be of type torch.Tensor. Got {0}.'.format(type(tgt_pc)))
    if not torch.is_tensor(tgt_normals):
        raise TypeError('Expected tgt_normals to be of type torch.Tensor. Got {0}.'.format(type(tgt_normals)))
    if not (torch.is_tensor(initial_transform) or initial_transform is None):
        raise TypeError('Expected initial_transform to be of type torch.Tensor. Got {0}.'.format(type(initial_transform)))
    if not isinstance(numiters, int):
        raise TypeError('Expected numiters to be of type int. Got {0}.'.format(type(numiters)))
    if initial_transform.ndim != 2:
        raise ValueError('Expected initial_transform.ndim to be 2. Got {0}.'.format(initial_transform.ndim))
    if not (initial_transform.shape[0] == 4 and initial_transform.shape[1] == 4):
        raise ValueError('Expected initial_transform.shape to be (4, 4). Got {0}.'.format(initial_transform.shape))
    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
    tgt_normals = tgt_normals.contiguous()
    dtype = src_pc.dtype
    device = src_pc.device
    damp = torch.tensor(damp, dtype=dtype, device=device)
    initial_transform = torch.eye(4, dtype=dtype, device=device) if initial_transform is None else initial_transform
    src_pc = transform_pointcloud(src_pc[0], initial_transform).unsqueeze(0)
    transform = initial_transform
    for it in range(numiters):
        A, b, chamfer_indices = gauss_newton_solve(src_pc, tgt_pc, tgt_normals, dist_thresh)
        residual = b[:, 0]
        xi = solve_linear_system(A, b, damp)
        residual_transform = se3_exp(xi)
        err = torch.dot(residual.t(), residual)
        pc_error = torch.sqrt(torch.sum((torch.mm(A, xi) - b) ** 2))
        one_step_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)
        _, one_step_b, chamfer_indices_onestep = gauss_newton_solve(one_step_pc, tgt_pc, tgt_normals, dist_thresh)
        one_step_residual = one_step_b[:, 0]
        new_err = torch.dot(one_step_residual.t(), one_step_residual)
        if new_err < err:
            src_pc = one_step_pc
            damp = damp / 2
            transform = torch.mm(residual_transform, transform)
        else:
            damp = damp * 2
    return transform, chamfer_indices


def create_meshgrid(height: int, width: int, normalized_coords: Optional[bool]=True) ->torch.Tensor:
    """Generates a coordinate grid for an image.

    When `normalized_coords` is set to True, the grid is normalized to
    be in the range [-1, 1] (to be consistent with the pytorch function
    `grid_sample`.)

    https://kornia.readthedocs.io/en/latest/utils.html#kornia.utils.create_meshgrid

    Args:
        height (int): Height of the image (number of rows).
        width (int): Width of the image (number of columns).
        normalized_coords (optional, bool): whether or not to
            normalize the coordinates to be in the range [-1, 1].

    Returns:
        (torch.Tensor): grid tensor (shape: :math:`1 \\times H \\times W \\times 2`).

    """
    xs: Optional[torch.Tensor] = None
    ys: Optional[torch.Tensor] = None
    if normalized_coords:
        xs = torch.linspace(-1, 1, height)
        ys = torch.linspace(-1, 1, width)
    else:
        xs = torch.linspace(0, height - 1, height)
        ys = torch.linspace(0, width - 1, width)
    base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]))
    return base_grid.permute(1, 2, 0).unsqueeze(0)


def inverse_intrinsics(K: torch.Tensor, eps: float=1e-06) ->torch.Tensor:
    """Efficient inversion of intrinsics matrix

    Args:
        K (torch.Tensor): Intrinsics matrix
        eps (float): Epsilon for numerical stability

    Returns:
        torch.Tensor: Inverse of intrinsics matrices

    Shape:
        - K: :math:`(*, 4, 4)` or :math:`(*, 3, 3)`
        - Kinv: Matches shape of `K` (:math:`(*, 4, 4)` or :math:`(*, 3, 3)`)
    """
    if not torch.is_tensor(K):
        raise TypeError('Expected K to be of type torch.Tensor. Got {0} instead.'.format(type(K)))
    if K.dim() < 2:
        raise ValueError('Input K must have at least 2 dims. Got {0} instead.'.format(K.dim()))
    if not (K.shape[-1] == 3 and K.shape[-2] == 3 or K.shape[-1] == 4 and K.shape[-2] == 4):
        raise ValueError('Input K must have shape (*, 4, 4) or (*, 3, 3). Got {0} instead.'.format(K.shape))
    Kinv = torch.zeros_like(K)
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    Kinv[..., 0, 0] = 1.0 / (fx + eps)
    Kinv[..., 1, 1] = 1.0 / (fy + eps)
    Kinv[..., 0, 2] = -1.0 * cx / (fx + eps)
    Kinv[..., 1, 2] = -1.0 * cy / (fy + eps)
    Kinv[..., 2, 2] = 1
    Kinv[..., -1, -1] = 1
    return Kinv


def img_to_b64str(img, quality=95):
    """Converts a numpy array of uint8 into a base64 jpeg string.

    Args
        img (np.ndarray): RGB or greyscale image array
        quality (int): Image quality from 0 to 100 (the higher is the better). Default: 95

    Returns:
        str: base64 jpeg string
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f'img must be of type np.ndarray, but was {type(img)}')
    if img.ndim != 2 and img.ndim != 3:
        raise ValueError(f'img.ndim must be 2 or 3, but was {img.ndim}')
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    retval, buffer = cv2.imencode('.jpg', img, encode_param)
    imstr = base64.b64encode(buffer).decode('utf-8')
    prefix = 'data:image/jpeg;base64,'
    base64_string = prefix + imstr
    return base64_string


def numpy_to_plotly_image(img, name=None, is_depth=False, scale=None, quality=95):
    """Converts a numpy array img to a `plotly.graph_objects.Image` object.

    Args
        img (np.ndarray): RGB image array
        name (str): Name for the returned `plotly.graph_objects.Image` object
        is_depth (bool): Bool indicating whether input `img` is depth image. Default: False
        scale (int or None): Scale factor to display on hover. If None, will not display `scale: ...`. Default: None
        quality (int): Image quality from 0 to 100 (the higher is the better). Default: 95

    Returns:
        `plotly.graph_objects.Image`
    """
    img_str = img_to_b64str(img, quality)
    hovertemplate = 'x: %%{x}<br>y: %%{y}<br>%s: %s'
    if not is_depth:
        hover_name = '[%{z[0]}, %{z[1]}, %{z[2]}]'
        hovertemplate = hovertemplate % ('color', hover_name)
    else:
        hover_name = '%{z[0]}'
        hovertemplate = hovertemplate % ('depth', hover_name)
    if scale is not None:
        scale = int(scale) if int(scale) == scale else scale
        hovertemplate += f'<br>scale: x{scale}<br>'
    hovertemplate += '<extra></extra>'
    return go.Image(source=img_str, hovertemplate=hovertemplate, name=name)


class RGBDImages(object):
    """Initializes an RGBDImage object consisting of a batch of a sequence of rgb images, depth maps,
    camera intrinsics, and (optionally) poses.

    Args:
        rgb_image (torch.Tensor): 3-channel rgb image
        depth_image (torch.Tensor): 1-channel depth map
        intrinsics (torch.Tensor): camera intrinsics
        poses (torch.Tensor or None): camera extrinsics. Default: None
        channels_first(bool): indicates whether `rgb_image` and `depth_image` have channels first or channels last
            representation (i.e. rgb_image.shape is :math:`(B, L, H, W, 3)` or :math:`(B, L, 3, H, W)`.
            Default: False
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            same as `rgb_image` device. Default: None
        pixel_pos (torch.Tensor or None): Similar to meshgrid but with extra channel of 1s at the end. If provided,
            can save computations when computing vertex maps. Default: None

    Shape:
        - rgb_image: :math:`(B, L, H, W, 3)` if `channels_first` is False, else :math:`(B, L, 3, H, W)`
        - depth_image: :math:`(B, L, H, W, 1)` if `channels_first` is False, else :math:`(B, L, 1, H, W)`
        - intrinsics: :math:`(B, 1, 4, 4)`
        - poses: :math:`(B, L, 4, 4)`
        - pixel_pos: :math:`(B, L, H, W, 3)` if `channels_first` is False, else :math:`(B, L, 3, H, W)`

    Examples::

        >>> colors = torch.rand([2, 8, 32, 32, 3])
        >>> depths = torch.rand([2, 8, 32, 32, 1])
        >>> intrinsics = torch.rand([2, 1, 4, 4])
        >>> poses = torch.rand([2, 8, 4, 4])
        >>> rgbdimages = gradslam.RGBDImages(colors, depths, intrinsics, poses)
        >>> print(rgbdimages.shape)
        (2, 8, 32, 32)
        >>> rgbd_select = rgbd_frame[1, 4:8]
        >>> print(rgbd_select.shape)
        (1, 4, 32, 32)
        >>> print(rgbdimages.vertex_map.shape)
        (2, 8, 32, 32, 3)
        >>> print(rgbdimages.normal_map.shape)
        (2, 8, 32, 32, 3)
    """
    _INTERNAL_TENSORS = ['_rgb_image', '_depth_image', '_intrinsics', '_poses', '_pixel_pos', '_vertex_map', '_normal_map', '_global_vertex_map', '_global_normal_map']

    def __init__(self, rgb_image: torch.Tensor, depth_image: torch.Tensor, intrinsics: torch.Tensor, poses: Optional[torch.Tensor]=None, channels_first: bool=False, device: Union[torch.device, str, None]=None, *, pixel_pos: Optional[torch.Tensor]=None):
        super().__init__()
        if not torch.is_tensor(rgb_image):
            msg = 'Expected rgb_image to be of type tensor; got {}'
            raise TypeError(msg.format(type(rgb_image)))
        if not torch.is_tensor(depth_image):
            msg = 'Expected depth_image to be of type tensor; got {}'
            raise TypeError(msg.format(type(depth_image)))
        if not torch.is_tensor(intrinsics):
            msg = 'Expected intrinsics to be of type tensor; got {}'
            raise TypeError(msg.format(type(intrinsics)))
        if not (poses is None or torch.is_tensor(poses)):
            msg = 'Expected poses to be of type tensor or None; got {}'
            raise TypeError(msg.format(type(poses)))
        if not isinstance(channels_first, bool):
            msg = 'Expected channels_first to be of type bool; got {}'
            raise TypeError(msg.format(type(channels_first)))
        if not (pixel_pos is None or torch.is_tensor(pixel_pos)):
            msg = 'Expected pixel_pos to be of type tensor or None; got {}'
            raise TypeError(msg.format(type(pixel_pos)))
        self._channels_first = channels_first
        if rgb_image.ndim != 5:
            msg = 'rgb_image should have ndim=5, but had ndim={}'.format(rgb_image.ndim)
            raise ValueError(msg)
        if depth_image.ndim != 5:
            msg = 'depth_image should have ndim=5, but had ndim={}'.format(depth_image.ndim)
            raise ValueError(msg)
        if intrinsics.ndim != 4:
            msg = 'intrinsics should have ndim=4, but had ndim={}'.format(intrinsics.ndim)
            raise ValueError(msg)
        if poses is not None and poses.ndim != 4:
            msg = 'poses should have ndim=4, but had ndim={}'.format(poses.ndim)
            raise ValueError(msg)
        self._rgb_image_shape = rgb_image.shape
        self._depth_shape = tuple(v if i != self.cdim else 1 for i, v in enumerate(rgb_image.shape))
        self._intrinsics_shape = rgb_image.shape[0], 1, 4, 4
        self._poses_shape = *rgb_image.shape[:2], 4, 4
        self._pixel_pos_shape = *rgb_image.shape[:self.cdim], *rgb_image.shape[self.cdim + 1:], 3
        if rgb_image.shape[self.cdim] != 3:
            msg = 'Expected rgb_image to have 3 channels on dimension {0}. Got {1} instead'
            raise ValueError(msg.format(self.cdim, rgb_image.shape[self.cdim]))
        if depth_image.shape != self._depth_shape:
            msg = 'Expected depth_image to have shape {0}. Got {1} instead'
            raise ValueError(msg.format(self._depth_shape, depth_image.shape))
        if intrinsics.shape != self._intrinsics_shape:
            msg = 'Expected intrinsics to have shape {0}. Got {1} instead'
            raise ValueError(msg.format(self._intrinsics_shape, intrinsics.shape))
        if poses is not None and poses.shape != self._poses_shape:
            msg = 'Expected poses to have shape {0}. Got {1} instead'
            raise ValueError(msg.format(self._poses_shape, poses.shape))
        if pixel_pos is not None and pixel_pos.shape != self._pixel_pos_shape:
            msg = 'Expected pixel_pos to have shape {0}. Got {1} instead'
            raise ValueError(msg.format(self._pixel_pos_shape, pixel_pos.shape))
        inputs = [rgb_image, depth_image, intrinsics, poses, pixel_pos]
        devices = [x.device for x in inputs if x is not None]
        if len(set(devices)) != 1:
            raise ValueError('All inputs must be on same device, but got more than 1 device: {}'.format(set(devices)))
        self._rgb_image = rgb_image if device is None else rgb_image
        self.device = self._rgb_image.device
        self._depth_image = depth_image
        self._intrinsics = intrinsics
        self._poses = poses if poses is not None else None
        self._pixel_pos = pixel_pos if pixel_pos is not None else None
        self._vertex_map = None
        self._global_vertex_map = None
        self._normal_map = None
        self._global_normal_map = None
        self._valid_depth_mask = None
        self._B, self._L = self._rgb_image.shape[:2]
        self.h = self._rgb_image.shape[3] if self._channels_first else self._rgb_image.shape[2]
        self.w = self._rgb_image.shape[4] if self._channels_first else self._rgb_image.shape[3]
        self.shape = self._B, self._L, self.h, self.w

    def __getitem__(self, index):
        """
        Args:
            index (int or slice or list of int): Specifying the index of the rgbdimages to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            gradslam.RGBDImages: Selected rgbdimages. The rgbdimages tensors are not cloned.
        """
        if isinstance(index, tuple) or isinstance(index, int):
            _index_slices = ()
            if isinstance(index, int):
                _index_slices += (slice(index, index + 1),) + (slice(None, None),)
            elif len(index) > 2:
                raise IndexError('Only batch and sequences can be indexed')
            elif isinstance(index, tuple):
                for x in index:
                    if isinstance(x, int):
                        _index_slices += slice(x, x + 1),
                    else:
                        _index_slices += x,
            new_rgb = self._rgb_image[_index_slices[0], _index_slices[1]]
            if new_rgb.shape[0] == 0:
                raise IndexError('Incorrect indexing at dimension 0, make sure range is within 0 and {0}'.format(self._B))
            if new_rgb.shape[1] == 0:
                raise IndexError('Incorrect indexing at dimension 1, make sure range is within 0 and {0}'.format(self._L))
            new_depth = self._depth_image[_index_slices[0], _index_slices[1]]
            new_intrinsics = self._intrinsics[_index_slices[0], :]
            other = RGBDImages(new_rgb, new_depth, new_intrinsics, channels_first=self.channels_first)
            for k in self._INTERNAL_TENSORS:
                if k in ['_rgb_image', '_depth_image', '_intrinsics']:
                    continue
                v = getattr(self, k)
                if torch.is_tensor(v):
                    setattr(other, k, v[_index_slices[0], _index_slices[1]])
            return other
        else:
            raise IndexError(index)

    def __len__(self):
        return self._B

    @property
    def channels_first(self):
        """Gets bool indicating whether RGBDImages representation is channels first or not

        Returns:
            bool: True if RGBDImages representation is channels first, else False.
        """
        return self._channels_first

    @property
    def cdim(self):
        """Gets the channel dimension

        Returns:
            int: :math:`2` if self.channels_first is True, else :math:`4`.
        """
        return 2 if self.channels_first else 4

    @property
    def rgb_image(self):
        """Gets the rgb image

        Returns:
            torch.Tensor: tensor representation of `rgb_image`

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        return self._rgb_image

    @property
    def depth_image(self):
        """Gets the depth image

        Returns:
            torch.Tensor: tensor representation of `depth_image`

        Shape:
            - Output: :math:`(B, L, H, W, 1)` if self.channels_first is False, else :math:`(B, L, 1, H, W)`
        """
        return self._depth_image

    @property
    def intrinsics(self):
        """Gets the `intrinsics`

        Returns:
            torch.Tensor: tensor representation of `intrinsics`

        Shape:
            - Output: :math:`(B, 1, 4, 4)`
        """
        return self._intrinsics

    @property
    def poses(self):
        """Gets the `poses`

        Returns:
            torch.Tensor: tensor representation of `poses`

        Shape:
            - Output: :math:`(B, L, 4, 4)`
        """
        return self._poses

    @property
    def pixel_pos(self):
        """Gets the `pixel_pos`

        Returns:
            torch.Tensor: tensor representation of `pixel_pos`

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        return self._pixel_pos

    @property
    def valid_depth_mask(self):
        """Gets a mask which is True wherever `self.dept_image` is :math:`>0`

        Returns:
            torch.Tensor: Tensor of dtype bool with same shape as `self.depth_image`. Tensor is True wherever
            `self.depth_image` > 0, and False otherwise.

        Shape:
            - Output: :math:`(B, L, H, W, 1)` if self.channels_first is False, else :math:`(B, L, 1, H, W)`
        """
        if self._valid_depth_mask is None:
            self._valid_depth_mask = self._depth_image > 0
        return self._valid_depth_mask

    @property
    def has_poses(self):
        """Determines whether self has `poses` or not

        Returns:
            bool
        """
        return self._poses is not None

    @property
    def vertex_map(self):
        """Gets the local vertex maps

        Returns:
            torch.Tensor: tensor representation of local coordinated vertex maps

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if self._vertex_map is None:
            self._compute_vertex_map()
        return self._vertex_map

    @property
    def normal_map(self):
        """Gets the local normal maps

        Returns:
            torch.Tensor: tensor representation of local coordinated normal maps

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if self._normal_map is None:
            self._compute_normal_map()
        return self._normal_map

    @property
    def global_vertex_map(self):
        """Gets the global vertex maps

        Returns:
            torch.Tensor: tensor representation of global coordinated vertex maps

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if self._global_vertex_map is None:
            self._compute_global_vertex_map()
        return self._global_vertex_map

    @property
    def global_normal_map(self):
        """Gets the global normal maps

        Returns:
            torch.Tensor: tensor representation of global coordinated normal maps

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if self._global_normal_map is None:
            self._compute_global_normal_map()
        return self._global_normal_map

    @rgb_image.setter
    def rgb_image(self, value):
        """Updates `rgb_image` of self.

        Args:
            value (torch.Tensor): New rgb image values

        Shape:
            - value: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if value is not None:
            self._assert_shape(value, self._rgb_image_shape)
        self._rgb_image = value

    @depth_image.setter
    def depth_image(self, value):
        """Updates `depth_image` of self.

        Args:
            value (torch.Tensor): New depth image values

        Shape:
            - value: :math:`(B, L, H, W, 1)` if self.channels_first is False, else :math:`(B, L, 1, H, W)`
        """
        if value is not None:
            self._assert_shape(value, self._depth_image_shape)
        self._depth_image = value
        self._vertex_map = None
        self._normal_map = None
        self._global_vertex_map = None
        self._global_normal_map = None

    @intrinsics.setter
    def intrinsics(self, value):
        """Updates `intrinsics` of self.

        Args:
            value (torch.Tensor): New intrinsics values

        Shape:
            - value: :math:`(B, 1, 4, 4)`
        """
        if value is not None:
            self._assert_shape(value, self._intrinsics_shape)
        self._intrinsics = value
        self._vertex_map = None
        self._normal_map = None
        self._global_vertex_map = None
        self._global_normal_map = None

    @poses.setter
    def poses(self, value):
        """Updates `poses` of self.

        Args:
            value (torch.Tensor): New pose values

        Shape:
            - value: :math:`(B, L, 4, 4)`
        """
        if value is not None:
            self._assert_shape(value, self._poses_shape)
        self._poses = value
        self._global_vertex_map = None
        self._global_normal_map = None

    def detach(self):
        """Detachs RGBDImages object. All internal tensors are detached individually.

        Returns:
            gradslam.RGBDImages: detached gradslam.RGBDImages object
        """
        other = self.clone()
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
        return other

    def clone(self):
        """Returns deep copy of RGBDImages object. All internal tensors are cloned individually.

        Returns:
            gradslam.RGBDImages: cloned gradslam.RGBDImages object
        """
        other = RGBDImages(rgb_image=self._rgb_image.clone(), depth_image=self._depth_image.clone(), intrinsics=self._intrinsics.clone(), channels_first=self.channels_first)
        for k in self._INTERNAL_TENSORS:
            if k in ['_rgb_image', '_depth_image', '_intrinsics']:
                continue
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device: Union[torch.device, str], copy: bool=False):
        """Match functionality of torch.Tensor.to(device)
        If copy = True or the self Tensor is on a different device, the returned tensor is a copy of self with the
        desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device, then self is returned.

        Args:
            device (torch.device or str): Device id for the new tensor.
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.RGBDImages
        """
        device = torch.Tensor().device
        if not copy and self.device == device:
            return self
        other = self.clone()
        other.device = device
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v)
        return other

    def cpu(self):
        """Match functionality of torch.Tensor.cpu()

        Returns:
            gradslam.RGBDImages
        """
        return self

    def cuda(self):
        """Match functionality of torch.Tensor.cuda()

        Returns:
            gradslam.RGBDImages
        """
        return self

    def to_channels_last(self, copy: bool=False):
        """Converts to channels last representation
        If copy = True or self channels_first is True, the returned RGBDImages object is a copy of self with
        channels last representation.
        If copy = False and self channels_first is already False, then self is returned.

        Args:
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.RGBDImages
        """
        if not (copy or self.channels_first):
            return self
        return self.clone().to_channels_last_()

    def to_channels_first(self, copy: bool=False):
        """Converts to channels first representation
        If copy = True or self channels_first is False, the returned RGBDImages object is a copy of self with
        channels first representation.
        If copy = False and self channels_first is already True, then self is returned.

        Args:
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.RGBDImages
        """
        if not copy and self.channels_first:
            return self
        return self.clone().to_channels_first_()

    def to_channels_last_(self):
        """Converts to channels last representation. In place operation.

        Returns:
            gradslam.RGBDImages
        """
        if not self.channels_first:
            return self
        ordering = 0, 1, 3, 4, 2
        permute = RGBDImages._permute_if_not_None
        self._rgb_image = permute(self._rgb_image, ordering)
        self._depth_image = permute(self._depth_image, ordering)
        self._vertex_map = permute(self._vertex_map, ordering)
        self._global_vertex_map = permute(self._global_vertex_map, ordering)
        self._normal_map = permute(self._normal_map, ordering)
        self._global_normal_map = permute(self._global_normal_map, ordering)
        self._channels_first = False
        self._rgb_image_shape = tuple(self._rgb_image.shape)
        self._depth_image_shape = tuple(self._depth_image.shape)
        return self

    def to_channels_first_(self):
        """Converts to channels first representation. In place operation.

        Returns:
            gradslam.RGBDImages
        """
        if self.channels_first:
            return self
        ordering = 0, 1, 4, 2, 3
        permute = RGBDImages._permute_if_not_None
        self._rgb_image = permute(self._rgb_image, ordering)
        self._depth_image = permute(self._depth_image, ordering)
        self._vertex_map = permute(self._vertex_map, ordering)
        self._global_vertex_map = permute(self._global_vertex_map, ordering)
        self._normal_map = permute(self._normal_map, ordering)
        self._global_normal_map = permute(self._global_normal_map, ordering)
        self._channels_first = True
        self._rgb_image_shape = tuple(self._rgb_image.shape)
        self._depth_image_shape = tuple(self._depth_image.shape)
        return self

    @staticmethod
    def _permute_if_not_None(tensor: Optional[torch.Tensor], ordering: tuple, contiguous: bool=True):
        """Permutes input if it is not None based on given ordering

        Args:
            tensor (torch.Tensor or None): Tensor to be permuted, or None
            ordering (tuple): The desired ordering of dimensions
            contiguous (bool): Whether to call `.contiguous()` on permuted tensor before returning.
                Default: True

        Returns:
            torch.Tensor or None: Permuted tensor or None
        """
        if tensor is None:
            return None
        assert torch.is_tensor(tensor)
        return tensor.permute(*ordering).contiguous() if contiguous else tensor.permute(*ordering)

    def _compute_vertex_map(self):
        """Coverts a batch of depth images into a batch of vertex maps."""
        B, L = self.shape[:2]
        device = self._depth_image.device
        if self._pixel_pos is None:
            meshgrid = create_meshgrid(self.h, self.w, normalized_coords=False).view(1, 1, self.h, self.w, 2).repeat(B, L, 1, 1, 1)
            self._pixel_pos = torch.cat([meshgrid[..., 1:], meshgrid[..., 0:1], torch.ones_like(meshgrid[..., 0].unsqueeze(-1))], -1)
        Kinv = inverse_intrinsics(self._intrinsics)[..., :3, :3]
        Kinv = Kinv.repeat(1, L, 1, 1)
        if self.channels_first:
            self._vertex_map = torch.einsum('bsjc,bshwc->bsjhw', Kinv, self._pixel_pos) * self._depth_image
        else:
            self._vertex_map = torch.einsum('bsjc,bshwc->bshwj', Kinv, self._pixel_pos) * self._depth_image
        self._vertex_map = self._vertex_map * self.valid_depth_mask

    def _compute_global_vertex_map(self):
        """Coverts a batch of local vertex maps into a batch of global vertex maps."""
        if self._poses is None:
            self._global_vertex_map = self.vertex_map.clone()
            return
        local_vertex_map = self.vertex_map
        B, L = self.shape[:2]
        rmat = self._poses[..., :3, :3]
        tvec = self._poses[..., :3, 3]
        if self.channels_first:
            self._global_vertex_map = torch.einsum('bsjc,bschw->bsjhw', rmat, local_vertex_map)
            self._global_vertex_map = self._global_vertex_map + tvec.view(B, L, 3, 1, 1)
        else:
            self._global_vertex_map = torch.einsum('bsjc,bshwc->bshwj', rmat, local_vertex_map)
            self._global_vertex_map = self._global_vertex_map + tvec.view(B, L, 1, 1, 3)
        self._global_vertex_map = self._global_vertex_map * self.valid_depth_mask

    def _compute_normal_map(self):
        """Converts a batch of vertex maps to a batch of normal maps."""
        dhoriz: torch.Tensor = torch.zeros_like(self.vertex_map)
        dverti: torch.Tensor = torch.zeros_like(self.vertex_map)
        if self.channels_first:
            dhoriz[..., :-1] = self.vertex_map[..., 1:] - self.vertex_map[..., :-1]
            dverti[..., :-1, :] = self.vertex_map[..., 1:, :] - self.vertex_map[..., :-1, :]
            dhoriz[..., -1] = dhoriz[..., -2]
            dverti[..., -1, :] = dverti[..., -2, :]
            dim = 2
        else:
            dhoriz[..., :-1, :] = self.vertex_map[..., 1:, :] - self.vertex_map[..., :-1, :]
            dverti[..., :-1, :, :] = self.vertex_map[..., 1:, :, :] - self.vertex_map[..., :-1, :, :]
            dhoriz[..., -1, :] = dhoriz[..., -2, :]
            dverti[..., -1, :, :] = dverti[..., -2, :, :]
            dim = -1
        normal_map: torch.Tensor = torch.cross(dhoriz, dverti, dim=dim)
        norm: torch.Tensor = normal_map.norm(dim=dim).unsqueeze(dim)
        self._normal_map: torch.Tensor = normal_map / torch.where(norm == 0, torch.ones_like(norm), norm)
        self._normal_map = self._normal_map * self.valid_depth_mask

    def _compute_global_normal_map(self):
        """Coverts a batch of local noraml maps into a batch of global normal maps."""
        if self._poses is None:
            self._global_normal_map = self.normal_map.clone()
            return
        local_normal_map = self.normal_map
        B, L = self.shape[:2]
        rmat = self._poses[..., :3, :3]
        if self.channels_first:
            self._global_normal_map = torch.einsum('bsjc,bschw->bsjhw', rmat, local_normal_map)
        else:
            self._global_normal_map = torch.einsum('bsjc,bshwc->bshwj', rmat, local_normal_map)

    def plotly(self, index: int, include_depth: bool=True, as_figure: bool=True, ms_per_frame: int=50):
        """Converts `index`-th sequence of rgbd images to either a `plotly.graph_objects.Figure` or a
        list of dicts containing `plotly.graph_objects.Image` objects of rgb and (optionally) depth images:

        .. code-block:: python


            frames = [
                {'name': 0, 'data': [rgbImage0, depthImage0], 'traces': [0, 1]},
                {'name': 1, 'data': [rgbImage1, depthImage1], 'traces': [0, 1]},
                {'name': 2, 'data': [rgbImage2, depthImage2], 'traces': [0, 1]},
                ...
            ]

        Returned `frames` can be passed to `go.Figure(frames=frames)`.

        Args:
            index (int): Index of which rgbd image (from the batch of rgbd images) to convert to plotly
                representation.
            include_depth (bool): If True, will include depth images in the returned object. Default: True
            as_figure (bool): If True, returns a `plotly.graph_objects.Figure` object which can easily
                be visualized by calling `.show()` on. Otherwise, returns a list of dicts (`frames`)
                which can be passed to `go.Figure(frames=frames)`. Default: True
            ms_per_frame (int): Milliseconds per frame when play button is hit. Only applicable if `as_figure=True`.
                Default: 50

        Returns:
            plotly.graph_objects.Figure or list of dict: If `as_figure` is True, will return
            `plotly.graph_objects.Figure` object from the `index`-th sequence of rgbd images. Else,
            returns a list of dicts (`frames`).
        """
        if not isinstance(index, int):
            raise TypeError('Index should be int, but was {}.'.format(type(index)))

        def frame_args(duration):
            return {'frame': {'duration': duration, 'redraw': True}, 'mode': 'immediate', 'fromcurrent': True, 'transition': {'duration': duration, 'easing': 'linear'}}
        torch_rgb = self.rgb_image[index]
        if (torch_rgb.max() < 1.1).item():
            torch_rgb = torch_rgb * 255
        torch_rgb = torch.clamp(torch_rgb, min=0.0, max=255.0)
        numpy_rgb = torch_rgb.detach().cpu().numpy().astype('uint8')
        Image_rgb = [numpy_to_plotly_image(rgb, i) for i, rgb in enumerate(numpy_rgb)]
        if not include_depth:
            frames = [{'data': [frame], 'name': i} for i, frame in enumerate(Image_rgb)]
        else:
            torch_depth = self.depth_image[index, ..., 0]
            scale = 10 ** torch.log10(255.0 / torch_depth.detach().max()).floor().item()
            numpy_depth = (torch_depth * scale).detach().cpu().numpy().astype('uint8')
            Image_depth = [numpy_to_plotly_image(d, i, True, scale) for i, d in enumerate(numpy_depth)]
            frames = [{'name': i, 'data': list(frame), 'traces': [0, 1]} for i, frame in enumerate(zip(Image_rgb, Image_depth))]
        if not as_figure:
            return frames
        steps = [{'args': [[i], frame_args(0)], 'label': i, 'method': 'animate'} for i in range(self._L)]
        sliders = [{'active': 0, 'yanchor': 'top', 'xanchor': 'left', 'currentvalue': {'prefix': 'Frame: '}, 'pad': {'b': 10, 't': 60}, 'len': 0.9, 'x': 0.1, 'y': 0, 'steps': steps}]
        updatemenus = [{'buttons': [{'args': [None, frame_args(ms_per_frame)], 'label': '&#9654;', 'method': 'animate'}, {'args': [[None], frame_args(0)], 'label': '&#9724;', 'method': 'animate'}], 'direction': 'left', 'pad': {'r': 10, 't': 70}, 'showactive': False, 'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0, 'yanchor': 'top'}]
        if not include_depth:
            fig = make_subplots(rows=1, cols=1, subplot_titles=('RGB',))
            fig.add_traces(frames[0]['data'][0])
        else:
            fig = make_subplots(rows=2, cols=1, subplot_titles=('RGB', 'Depth'), shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.1)
            fig.add_trace(frames[0]['data'][0], row=1, col=1)
            fig.add_trace(frames[0]['data'][1], row=2, col=1)
            fig.update_layout(scene=dict(aspectmode='data'))
            fig.update_layout(autosize=False, height=1080)
        fig.update(frames=frames)
        fig.update_layout(updatemenus=updatemenus, sliders=sliders)
        return fig

    def _assert_shape(self, value: torch.Tensor, shape: tuple):
        """Asserts if value is a tensor with same shape as `shape`

        Args:
            value (torch.Tensor): Tensor to check shape of
            shape (tuple): Expected shape of value
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError('value must be torch.Tensor. Got {}'.format(type(value)))
        if value.shape != shape:
            msg = 'Expected value to have shape {0}. Got {1} instead'
            raise ValueError(msg.format(shape, value.shape))


def downsample_pointclouds(pointclouds: Pointclouds, pc2im_bnhw: torch.Tensor, ds_ratio: int) ->Pointclouds:
    """Downsamples active points of pointclouds (points that project inside the live frame) and removes non-active
    points.

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds to downsample
        pc2im_bnhw (torch.Tensor): Active map points lookup table. Each row contains batch index `b`, point
            index (in pointclouds) `n`, and height and width index after projection to live frame `h` and `w`
            respectively.
        ds_ratio (int): Downsampling ratio

    Returns:
        gradslam.Pointclouds: Downsampled pointclouds

    Shape:
        - pc2im_bnhw: :math:`(\\text{num_active_map_points}, 4)`

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError('Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.'.format(type(pointclouds)))
    if not torch.is_tensor(pc2im_bnhw):
        raise TypeError('Expected pc2im_bnhw to be of type torch.Tensor. Got {0}.'.format(type(pc2im_bnhw)))
    if not isinstance(ds_ratio, int):
        raise TypeError('Expected ds_ratio to be of type int. Got {0}.'.format(type(ds_ratio)))
    if pc2im_bnhw.ndim != 2:
        raise ValueError('Expected pc2im_bnhw to have ndim=2. Got {0}.'.format(pc2im_bnhw.ndim))
    if pc2im_bnhw.shape[1] != 4:
        raise ValueError('pc2im_bnhw.shape[1] must be 4, but was {0}.'.format(pc2im_bnhw.shape[1]))
    B = len(pointclouds)
    pc2im_bnhw = pc2im_bnhw[pc2im_bnhw[..., 2] % ds_ratio == 0]
    pc2im_bnhw = pc2im_bnhw[pc2im_bnhw[..., 3] % ds_ratio == 0]
    maps_points = [pointclouds.points_list[b][pc2im_bnhw[pc2im_bnhw[..., 0] == b][..., 1]] for b in range(B)]
    maps_normals = None if pointclouds.normals_list is None else [pointclouds.normals_list[b][pc2im_bnhw[pc2im_bnhw[..., 0] == b][..., 1]] for b in range(B)]
    maps_colors = None if pointclouds.colors_list is None else [pointclouds.colors_list[b][pc2im_bnhw[pc2im_bnhw[..., 0] == b][..., 1]] for b in range(B)]
    return Pointclouds(points=maps_points, normals=maps_normals, colors=maps_colors)


def downsample_rgbdimages(rgbdimages: RGBDImages, ds_ratio: int) ->Pointclouds:
    """Downsamples points and normals of RGBDImages and returns a gradslam.Pointclouds object

    Args:
        rgbdimages (gradslam.RGBDImages): RGBDImages to downsample
        ds_ratio (int): Downsampling ratio

    Returns:
        gradslam.Pointclouds: Downsampled points and normals

    """
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError('Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.'.format(type(rgbdimages)))
    if not isinstance(ds_ratio, int):
        raise TypeError('Expected ds_ratio to be of type int. Got {0}.'.format(type(ds_ratio)))
    if rgbdimages.shape[1] != 1:
        raise ValueError('Sequence length of rgbdimages must be 1, but was {0}.'.format(rgbdimages.shape[1]))
    B = len(rgbdimages)
    mask = rgbdimages.valid_depth_mask.squeeze(-1)[..., ::ds_ratio, ::ds_ratio]
    points = [rgbdimages.global_vertex_map[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]] for b in range(B)]
    normals = [rgbdimages.global_normal_map[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]] for b in range(B)]
    colors = [rgbdimages.rgb_image[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]] for b in range(B)]
    return Pointclouds(points=points, normals=normals, colors=colors)


def find_active_map_points(pointclouds: Pointclouds, rgbdimages: RGBDImages) ->torch.Tensor:
    """Returns lookup table for indices of active global map points and their position inside the live frames.
    (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Batch of `B` global maps
        rgbdimages (gradslam.RGBDImages): Batch of `B` live frames from the latest sequence. `poses`, `intrinsics`,
            heights and widths of frames are used.

    Returns:
        pc2im_bnhw (torch.Tensor): Active map points lookup table. Each row contains batch index `b`, point index (in
            pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively.

    Shape:
        - pc2im_bnhw: :math:`(\\text{num_active_map_points}, 4)`

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError('Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.'.format(type(pointclouds)))
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError('Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.'.format(type(rgbdimages)))
    if rgbdimages.shape[1] != 1:
        raise ValueError('Expected rgbdimages to have sequence length of 1. Got {0}.'.format(rgbdimages.shape[1]))
    device = pointclouds.device
    if not pointclouds.has_points:
        return torch.empty((0, 4), dtype=torch.int64, device=device)
    if len(rgbdimages) != len(pointclouds):
        raise ValueError('Expected equal batch sizes for pointclouds and rgbdimages. Got {0} and {1} respectively.'.format(len(pointclouds), len(rgbdimages)))
    batch_size, seq_len, height, width = rgbdimages.shape
    tinv = inverse_transformation(rgbdimages.poses.squeeze(1))
    pointclouds_transformed = pointclouds.transform(tinv)
    is_front_of_plane = pointclouds_transformed.points_padded[..., -1] > 0
    pointclouds_transformed.pinhole_projection_(rgbdimages.intrinsics.squeeze(1))
    img_plane_points = pointclouds_transformed.points_padded[..., :-1]
    is_in_frame = (img_plane_points[..., 0] > -0.001) & (img_plane_points[..., 0] < width - 0.999) & (img_plane_points[..., 1] > -0.001) & (img_plane_points[..., 1] < height - 0.999) & is_front_of_plane & pointclouds.nonpad_mask
    in_plane_pos = img_plane_points.round().long()
    in_plane_pos = torch.cat([in_plane_pos[..., 1:2].clamp(0, height - 1), in_plane_pos[..., 0:1].clamp(0, width - 1)], -1)
    batch_size, num_points = in_plane_pos.shape[:2]
    batch_point_idx = create_meshgrid(batch_size, num_points, normalized_coords=False).squeeze(0)
    idx_and_plane_pos = torch.cat([batch_point_idx.long(), in_plane_pos], -1)
    pc2im_bnhw = idx_and_plane_pos[is_in_frame]
    if pc2im_bnhw.shape[0] == 0:
        warnings.warn('No active map points were found')
    return pc2im_bnhw


def pointclouds_from_rgbdimages(rgbdimages: RGBDImages, *, global_coordinates: bool=True, filter_missing_depths: bool=True) ->Pointclouds:
    """Converts gradslam.RGBDImages containing batch of RGB-D images with sequence length of 1 to gradslam.Pointclouds

    Args:
        rgbdimages (gradslam.RGBDImages): Can contain a batch of RGB-D images but must have sequence length of 1.
        global_coordinates (bool): If True, will create pointclouds object based on :math:`(X, Y, Z)` coordinates
            in the global coordinates (based on `rgbdimages.poses`). Otherwise, will use the local frame coordinates.
        filter_missing_depths (bool): If True, will not include vertices corresponding to missing depth values
            in the output pointclouds.

    Returns:
        gradslam.Pointclouds: Output pointclouds
    """
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError('Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.'.format(type(rgbdimages)))
    if not rgbdimages.shape[1] == 1:
        raise ValueError('Expected rgbdimages to have sequence length of 1. Got {0}.'.format(rgbdimages.shape[1]))
    B = rgbdimages.shape[0]
    rgbdimages = rgbdimages.to_channels_last()
    vertex_map = rgbdimages.global_vertex_map if global_coordinates else rgbdimages.vertex_map
    normal_map = rgbdimages.global_normal_map if global_coordinates else rgbdimages.normal_map
    if filter_missing_depths:
        mask = rgbdimages.valid_depth_mask.squeeze(-1)
        points = [vertex_map[b][mask[b]] for b in range(B)]
        normals = [normal_map[b][mask[b]] for b in range(B)]
        colors = [rgbdimages.rgb_image[b][mask[b]] for b in range(B)]
    else:
        points = vertex_map.reshape(B, -1, 3).contiguous()
        normals = normal_map.reshape(B, -1, 3).contiguous()
        colors = rgbdimages.rgb_image.reshape(B, -1, 3).contiguous()
    return Pointclouds(points=points, normals=normals, colors=colors)


def update_map_aggregate(pointclouds: Pointclouds, rgbdimages: RGBDImages, inplace: bool=False) ->Pointclouds:
    """Aggregate points from live frames with global maps by appending the live frame points.

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of global maps. Must have points, colors, normals and features
            (ccounts).
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        inplace (bool): Can optionally update the pointclouds in-place. Default: False

    Returns:
        gradslam.Pointclouds: Updated Pointclouds object containing global maps.

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError('Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.'.format(type(pointclouds)))
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError('Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.'.format(type(rgbdimages)))
    new_pointclouds = pointclouds_from_rgbdimages(rgbdimages, global_coordinates=True)
    if not inplace:
        pointclouds = pointclouds.clone()
    pointclouds.append_points(new_pointclouds)
    return pointclouds


class ICPSLAM(nn.Module):
    """ICP-SLAM for batched sequences of RGB-D images.

    Args:
        odom (str): Odometry method to be used from {'gt', 'icp', 'gradicp'}. Default: 'gradicp'
        dsratio (int): Downsampling ratio to apply to input frames before ICP. Only used if `odom` is
            'icp' or 'gradicp'. Default: 4
        numiters (int): Number of iterations to run the optimization for. Only used if `odom` is
            'icp' or 'gradicp'. Default: 20
        damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Only used if `odom` is
            'icp' or 'gradicp'. Default: 1e-8
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
                Only used if `odom` is 'icp' or 'gradicp'. Default: None
        lambda_max (float or int): Maximum value the damping function can assume (`lambda_min` will be
            :math:`\\frac{1}{\\text{lambda_max}}`). Only used if `odom` is 'gradicp'.
        B (float or int): gradLM falloff control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        B2 (float or int): gradLM control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        nu (float or int): gradLM control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            the CPU. Default: None


    Examples::

        >>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
        >>> slam = ICPSLAM(odom='gt')
        >>> pointclouds, poses = slam(rgbdimages)
        >>> o3d.visualization.draw_geometries([pointclouds.o3d(0)])

        >>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
        >>> slam = ICPSLAM(odom='gt')
        >>> pointclouds = Pointclouds()
        >>> pointclouds, new_poses = self.step(pointclouds, frames[:, 0], None)
        >>> frames.poses[:, :1] = new_poses
        >>> pointclouds, new_poses = self.step(pointclouds, frames[:, 1], frames[:, 0])

        >>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
        >>> slam = ICPSLAM(odom='gradicp')
        >>> pointclouds = Pointclouds()
        >>> pointclouds, new_poses = self.step(pointclouds, frames[:, 0], None)
        >>> frames.poses[:, :1] = new_poses
        >>> pointclouds, new_poses = self.step(pointclouds, frames[:, 1], frames[:, 0])
    """

    def __init__(self, *, odom: str='gradicp', dsratio: int=4, numiters: int=20, damp: float=1e-08, dist_thresh: Union[float, int, None]=None, lambda_max: Union[float, int]=2.0, B: Union[float, int]=1.0, B2: Union[float, int]=1.0, nu: Union[float, int]=200.0, device: Union[torch.device, str, None]=None):
        super().__init__()
        if odom not in ['gt', 'icp', 'gradicp']:
            msg = 'odometry method ({}) not supported for PointFusion. '.format(odom)
            msg += "Currently supported odometry modules for PointFusion are: 'gt', 'icp', 'gradicp'"
            raise ValueError(msg)
        odomprov = None
        if odom == 'icp':
            odomprov = ICPOdometryProvider(numiters, damp, dist_thresh)
        elif odom == 'gradicp':
            odomprov = GradICPOdometryProvider(numiters, damp, dist_thresh, lambda_max, B, B2, nu)
        self.odom = odom
        self.odomprov = odomprov
        self.dsratio = dsratio
        device = torch.device(device) if device is not None else torch.device('cpu')
        self.device = torch.Tensor().device

    def forward(self, frames: RGBDImages):
        """Builds global map pointclouds from a batch of input RGBDImages with a batch size
        of :math:`B` and sequence length of :math:`L`.

        Args:
            frames (gradslam.RGBDImages): Input batch of frames with a sequence length of `L`.

        Returns:
            tuple: tuple containing:

            - pointclouds (gradslam.Pointclouds): Pointclouds object containing :math:`B` global maps
            - poses (torch.Tensor): Poses computed by the odometry method

        Shape:
            - poses: :math:`(B, L, 4, 4)`
        """
        if not isinstance(frames, RGBDImages):
            raise TypeError('Expected frames to be of type gradslam.RGBDImages. Got {0}.'.format(type(frames)))
        pointclouds = Pointclouds(device=self.device)
        batch_size, seq_len = frames.shape[:2]
        recovered_poses = torch.empty(batch_size, seq_len, 4, 4)
        prev_frame = None
        for s in range(seq_len):
            live_frame = frames[:, s]
            if s == 0 and live_frame.poses is None:
                live_frame.poses = torch.eye(4, dtype=torch.float, device=self.device).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
            pointclouds, live_frame.poses = self.step(pointclouds, live_frame, prev_frame, inplace=True)
            prev_frame = live_frame if self.odom != 'gt' else None
            recovered_poses[:, s] = live_frame.poses[:, 0]
        return pointclouds, recovered_poses

    def step(self, pointclouds: Pointclouds, live_frame: RGBDImages, prev_frame: Optional[RGBDImages]=None, inplace: bool=False):
        """Updates global map pointclouds with a SLAM step on `live_frame`.
        If `prev_frame` is not None, computes the relative transformation between `live_frame`
        and `prev_frame` using the selected odometry provider. If `prev_frame` is None,
        use the pose from `live_frame`.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            prev_frame (gradslam.RGBDImages or None): Input batch of previous frames (at time step :math:`t-1`).
                Must have sequence length of 1. If None, will (skip calling odometry provider and) use the pose
                from `live_frame`. Default: None
            inplace (bool): Can optionally update the pointclouds and live_frame poses in-place. Default: False

        Returns:
            tuple: tuple containing:

            - pointclouds (gradslam.Pointclouds): Updated :math:`B` global maps
            - poses (torch.Tensor): Poses for the live_frame batch

        Shape:
            - poses: :math:`(B, 1, 4, 4)`
        """
        if not isinstance(live_frame, RGBDImages):
            raise TypeError('Expected live_frame to be of type gradslam.RGBDImages. Got {0}.'.format(type(live_frame)))
        live_frame.poses = self._localize(pointclouds, live_frame, prev_frame)
        pointclouds = self._map(pointclouds, live_frame, inplace)
        return pointclouds, live_frame.poses

    def _localize(self, pointclouds: Pointclouds, live_frame: RGBDImages, prev_frame: RGBDImages):
        """Compute the poses for `live_frame`. If `prev_frame` is not None, computes the relative
        transformation between `live_frame` and `prev_frame` using the selected odometry provider.
        If `prev_frame` is None, use the pose from `live_frame`.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            prev_frame (gradslam.RGBDImages or None): Input batch of previous frames (at time step :math:`t-1`).
                Must have sequence length of 1. If None, will (skip calling odometry provider and) use the pose
                from `live_frame`. Default: None

        Returns:
            torch.Tensor: Poses for the live_frame batch

        Shape:
            - Output: :math:`(B, 1, 4, 4)`
        """
        if not isinstance(pointclouds, Pointclouds):
            raise TypeError('Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.'.format(type(pointclouds)))
        if not isinstance(live_frame, RGBDImages):
            raise TypeError('Expected live_frame to be of type gradslam.RGBDImages. Got {0}.'.format(type(live_frame)))
        if not isinstance(prev_frame, (RGBDImages, type(None))):
            raise TypeError('Expected prev_frame to be of type gradslam.RGBDImages or None. Got {0}.'.format(type(prev_frame)))
        if prev_frame is not None:
            if self.odom == 'gt':
                warnings.warn("`prev_frame` is not used when using `odom='gt'` (should be None)")
            elif not prev_frame.has_poses:
                raise ValueError('`prev_frame` should have poses, but did not.')
        if prev_frame is None and pointclouds.has_points and self.odom != 'gt':
            msg = '`prev_frame` was None despite `{}` odometry method. Using `live_frame` poses.'.format(self.odom)
            warnings.warn(msg)
        if prev_frame is None or self.odom == 'gt':
            if not live_frame.has_poses:
                raise ValueError("`live_frame` must have poses when `prev_frame` is None or `odom='gt'`.")
            return live_frame.poses
        if self.odom in ['icp', 'gradicp']:
            live_frame.poses = prev_frame.poses
            frames_pc = downsample_rgbdimages(live_frame, self.dsratio)
            pc2im_bnhw = find_active_map_points(pointclouds, prev_frame)
            maps_pc = downsample_pointclouds(pointclouds, pc2im_bnhw, self.dsratio)
            transform = self.odomprov.provide(maps_pc, frames_pc)
            return compose_transformations(transform.squeeze(1), prev_frame.poses.squeeze(1)).unsqueeze(1)

    def _map(self, pointclouds: Pointclouds, live_frame: RGBDImages, inplace: bool=False):
        """Updates global map pointclouds by aggregating them with points from `live_frame`.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            inplace (bool): Can optionally update the pointclouds in-place. Default: False

        Returns:
            gradslam.Pointclouds: Updated :math:`B` global maps

        """
        return update_map_aggregate(pointclouds, live_frame, inplace)


def find_best_unique_correspondences(pointclouds: Pointclouds, rgbdimages: RGBDImages, pc2im_bnhw: torch.Tensor) ->torch.Tensor:
    """Amongst global map points which project to the same frame pixel, find the ones which have the highest
    confidence counter (and if confidence counter is equal then find the closest one to viewing ray).
    (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of globalmaps
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        pc2im_bnhw (torch.Tensor): Similar map points lookup table. Each row contains batch index `b`, point index (in
            pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively. This
            table can have different points (`b`s and `n`s) projecting to the same live frame pixel (same `h` and `w`)

    Returns:
        pc2im_bnhw_unique (torch.Tensor): Lookup table of one-to-one correspondences between points from the global map
            and live frames' points (pixels).

    Shape:
        - pc2im_bnhw: :math:`(\\text{num_similar_map_points}, 4)`
        - pc2im_bnhw_unique: :math:`(\\text{num_unique_correspondences}, 4)` where
            :math:`\\text{num_unique_correspondences}\\leq\\text{num_similar_map_points}`

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError('Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.'.format(type(pointclouds)))
    if not torch.is_tensor(pc2im_bnhw):
        raise TypeError('Expected input pc2im_bnhw to be of type torch.Tensor. Got {0} instead.'.format(type(pc2im_bnhw)))
    if pc2im_bnhw.dtype != torch.int64:
        raise TypeError('Expected input pc2im_bnhw to have dtype of torch.int64 (torch.long), not {0}.'.format(pc2im_bnhw.dtype))
    if rgbdimages.shape[1] != 1:
        raise ValueError('Expected rgbdimages to have sequence length of 1. Got {0}.'.format(rgbdimages.shape[1]))
    if pc2im_bnhw.ndim != 2:
        raise ValueError('Expected pc2im_bnhw.ndim of 2. Got {0}.'.format(pc2im_bnhw.ndim))
    if pc2im_bnhw.shape[1] != 4:
        raise ValueError('Expected pc2im_bnhw.shape[1] to be 4. Got {0}.'.format(pc2im_bnhw.shape[1]))
    device = pointclouds.device
    if not pointclouds.has_points or pc2im_bnhw.shape[0] == 0:
        return torch.empty((0, 4), dtype=torch.int64, device=device)
    if len(rgbdimages) != len(pointclouds):
        raise ValueError('Expected equal batch sizes for pointclouds and rgbdimages. Got {0} and {1} respectively.'.format(len(pointclouds), len(rgbdimages)))
    if not pointclouds.has_features:
        raise ValueError('Pointclouds must have features for finding best unique correspondences, but did not.')
    inv_ccounts = 1 / (pointclouds.features_padded[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] + 1e-20)
    frame_points = rgbdimages.global_vertex_map[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]]
    ray_dists = ((pointclouds.points_padded[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] - frame_points) ** 2).sum(-1).unsqueeze(1)
    unique_criteria_bhwcrn = [pc2im_bnhw[:, 0:1].float(), pc2im_bnhw[:, 2:4].float(), inv_ccounts, ray_dists, pc2im_bnhw[:, 1:2].float()]
    unique_criteria_bhwcrn = torch.cat(unique_criteria_bhwcrn, -1)
    sorted_criteria = torch.unique(unique_criteria_bhwcrn.detach(), dim=0)
    indices = sorted_criteria[:, -1].long()
    sorted_nonunique_inds = sorted_criteria[:, :3]
    first_unique_mask = torch.ones(sorted_nonunique_inds.shape[0], dtype=bool, device=device)
    first_unique_mask[1:] = (sorted_nonunique_inds[1:, :3] - sorted_nonunique_inds[:-1, :3] != 0).any(-1)
    first_unique = sorted_criteria[first_unique_mask]
    pc2im_bnhw_unique = torch.cat([first_unique[:, 0:1].long(), first_unique[:, -1:].long(), first_unique[:, 1:3].long()], -1)
    return pc2im_bnhw_unique


def are_normals_similar(tensor1: torch.Tensor, tensor2: torch.Tensor, dot_th: Union[float, int], dim: int=-1) ->torch.Tensor:
    """Returns bool tensor indicating dot product of two tensors containing normals along given dimension `dim` is
    greater than the given threshold value `dot_th`.

    Args:
        tensor1 (torch.Tensor): Input to compute dot product on. `dim` must be of length 3 :math:`(N_x, N_y, N_z)`.
        tensor2 (torch.Tensor): Input to compute dot product on. `dim` must be of length 3 :math:`(N_x, N_y, N_z)`.
        dot_th (float or int): Dot product threshold.
        dim (int): The dimension to compute product along. Default: -1

    Returns:
        Output (torch.Tensor): Tensor of bool

    Shape:
        - tensor1: :math:`(*, 3, *)`
        - tensor2: :math:`(*, 3, *)`
        - dot_th: Scalar
        - Output: Similar dimensions to `tensor1` except `dim` is squeezed and output tensor has 1 fewer dimension.
    """
    if not torch.is_tensor(tensor1):
        raise TypeError('Expected input tensor1 to be of type torch.Tensor. Got {0} instead.'.format(type(tensor1)))
    if not torch.is_tensor(tensor2):
        raise TypeError('Expected input tensor2 to be of type torch.Tensor. Got {0} instead.'.format(type(tensor2)))
    if not (isinstance(dot_th, float) or isinstance(dot_th, int)):
        raise TypeError('Expected input dot_th to be of type float or int. Got {0} instead.'.format(type(dot_th)))
    if tensor1.shape != tensor2.shape:
        raise ValueError('tensor1 and tensor2 should have the same shape, but had shapes {0} and {1} respectively.'.format(tensor1.shape, tensor2.shape))
    if tensor1.shape[dim] != 3:
        raise ValueError("Expected length of input tensors' dim-th ({0}th) dimension to be 3. Got {1} instead.".format(dim, tensor1.shape[dim]))
    dot_res = (tensor1 * tensor2).sum(dim)
    if dot_res.max() > 1.001:
        warnings.warn('Max of dot product was {0} > 1. Inputs were not normalized along dim ({1}). Was this intentional?'.format(dot_res.max(), dim), RuntimeWarning)
    return dot_res > dot_th


def are_points_close(tensor1: torch.Tensor, tensor2: torch.Tensor, dist_th: Union[float, int], dim: int=-1) ->torch.Tensor:
    """Returns bool tensor indicating the euclidean distance between two tensors of vertices along given dimension
    `dim` is smaller than the given threshold value `dist_th`.

    Args:
        tensor1 (torch.Tensor): Input to compute distance on. `dim` must be of length 3 :math:`(X, Y, Z)`.
        tensor2 (torch.Tensor): Input to compute distance on. `dim` must be of length 3 :math:`(X, Y, Z)`.
        dist_th (float or int): Distance threshold.
        dim (int): The dimension to compute distance along. Default: -1

    Returns:
        Output (torch.Tensor): Tensor of bool

    Shape:
        - tensor1: :math:`(*, 3, *)`
        - tensor2: :math:`(*, 3, *)`
        - dist_th: Scalar
        - Output: Similar dimensions to `tensor1` except `dim` is squeezed and output tensor has 1 fewer dimension.
    """
    if not torch.is_tensor(tensor1):
        raise TypeError('Expected input tensor1 to be of type torch.Tensor. Got {0} instead.'.format(type(tensor1)))
    if not torch.is_tensor(tensor2):
        raise TypeError('Expected input tensor2 to be of type torch.Tensor. Got {0} instead.'.format(type(tensor2)))
    if not (isinstance(dist_th, float) or isinstance(dist_th, int)):
        raise TypeError('Expected input dist_th to be of type float or int. Got {0} instead.'.format(type(dist_th)))
    if tensor1.shape != tensor2.shape:
        raise ValueError('tensor1 and tensor2 should have the same shape, but had shapes {0} and {1} respectively.'.format(tensor1.shape, tensor2.shape))
    if tensor1.shape[dim] != 3:
        raise ValueError("Expected length of input tensors' dim-th ({0}th) dimension to be 3. Got {1} instead.".format(dim, tensor1.shape[dim]))
    return (tensor1 - tensor2).norm(dim=dim) < dist_th


def find_similar_map_points(pointclouds: Pointclouds, rgbdimages: RGBDImages, pc2im_bnhw: torch.Tensor, dist_th: Union[float, int], dot_th: Union[float, int]) ->torch.Tensor:
    """Returns lookup table for points from global maps that are close and have similar normals to points from live
    frames occupying the same pixel as their projection (onto that live frame).
    (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of globalmaps
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        pc2im_bnhw (torch.Tensor): Active map points lookup table. Each row contains batch index `b`, point index (in
            pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively.
        dist_th (float or int): Distance threshold.
        dot_th (float or int): Dot product threshold.

    Returns:
        pc2im_bnhw_similar (torch.Tensor): Lookup table of points from global map that are close and have have normals
            that are similar to the live frame points.
        is_similar_mask (torch.Tensor): bool mask indicating which rows from input `pc2im_bnhw` are retained.

    Shape:
        - pc2im_bnhw: :math:`(\\text{num_active_map_points}, 4)`
        - dist_th: Scalar
        - dot_th: Scalar
        - pc2im_bnhw_similar: :math:`(\\text{num_similar_map_points}, 4)` where
            :math:`\\text{num_similar_map_points}\\leq\\text{num_active_map_points}`
        - is_similar_mask: :math:`(\\text{num_active_map_points})` where
            :math:`\\text{num_similar_map_points}\\leq\\text{num_active_map_points}

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError('Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.'.format(type(pointclouds)))
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError('Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.'.format(type(rgbdimages)))
    if not torch.is_tensor(pc2im_bnhw):
        raise TypeError('Expected input pc2im_bnhw to be of type torch.Tensor. Got {0} instead.'.format(type(pc2im_bnhw)))
    if pc2im_bnhw.dtype != torch.int64:
        raise TypeError('Expected input pc2im_bnhw to have dtype of torch.int64 (torch.long), not {0}.'.format(pc2im_bnhw.dtype))
    if rgbdimages.shape[1] != 1:
        raise ValueError('Expected rgbdimages to have sequence length of 1. Got {0}.'.format(rgbdimages.shape[1]))
    if pc2im_bnhw.ndim != 2:
        raise ValueError('Expected pc2im_bnhw.ndim of 2. Got {0}.'.format(pc2im_bnhw.ndim))
    if pc2im_bnhw.shape[1] != 4:
        raise ValueError('Expected pc2im_bnhw.shape[1] to be 4. Got {0}.'.format(pc2im_bnhw.shape[1]))
    device = pointclouds.device
    if not pointclouds.has_points or pc2im_bnhw.shape[0] == 0:
        return torch.empty((0, 4), dtype=torch.int64, device=device), torch.empty(0, dtype=torch.bool, device=device)
    if len(rgbdimages) != len(pointclouds):
        raise ValueError('Expected equal batch sizes for pointclouds and rgbdimages. Got {0} and {1} respectively.'.format(len(pointclouds), len(rgbdimages)))
    if not pointclouds.has_normals:
        raise ValueError('Pointclouds must have normals for finding similar map points, but did not.')
    vertex_maps = rgbdimages.global_vertex_map
    normal_maps = rgbdimages.global_normal_map
    frame_points = torch.zeros_like(pointclouds.points_padded)
    frame_normals = torch.zeros_like(pointclouds.normals_padded)
    frame_points[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = vertex_maps[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]]
    frame_normals[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = normal_maps[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]]
    is_close = are_points_close(frame_points, pointclouds.points_padded, dist_th)
    is_similar = are_normals_similar(frame_normals, pointclouds.normals_padded, dot_th)
    mask = is_close & is_similar
    is_similar_mask = mask[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]]
    pc2im_bnhw_similar = pc2im_bnhw[is_similar_mask]
    if len(pc2im_bnhw_similar) == 0:
        warnings.warn('No similar map points were found (despite total {0} active points across the batch)'.format(pc2im_bnhw.shape[0]), RuntimeWarning)
    return pc2im_bnhw_similar, is_similar_mask


def find_correspondences(pointclouds: Pointclouds, rgbdimages: RGBDImages, dist_th: Union[float, int], dot_th: Union[float, int]) ->torch.Tensor:
    """Returns a lookup table for inferring unique correspondences between points from the live frame and the global
    map (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of global maps
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        dist_th (float or int): Distance threshold.
        dot_th (float or int): Dot product threshold.

    Returns:
        pc2im_bnhw (torch.Tensor): Unique correspondence lookup table. Each row contains batch index `b`, point index
            (in pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively.

    Shape:
        - pc2im_bnhw: :math:`(\\text{num_unique_correspondences}, 4)`

    """
    pc2im_bnhw = find_active_map_points(pointclouds, rgbdimages)
    pc2im_bnhw, _ = find_similar_map_points(pointclouds, rgbdimages, pc2im_bnhw, dist_th, dot_th)
    pc2im_bnhw = find_best_unique_correspondences(pointclouds, rgbdimages, pc2im_bnhw)
    return pc2im_bnhw


def get_alpha(points: torch.Tensor, sigma: Union[torch.Tensor, float, int], dim: int=-1, keepdim: bool=False, eps: float=1e-07) ->torch.Tensor:
    """Computes sample confidence alpha.
    (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        points (torch.Tensor): Tensor of points.
        sigma (torch.Tensor or float or int): Standard deviation of the Gaussian. Original paper uses 0.6 emperically.
        dim (int): Dimension along which :math:`(X, Y, Z)` of points is stored. Default: -1
        keepdim (bool): Whether the output tensor has `dim` retained or not. Default: False
        eps (float): Minimum value for alpha (to avoid numerical instability). Default: 1e-7

    Returns:
        alpha (torch.Tensor): Sample confidence.

    Shape:
        - points: :math:`(*, 3, *)`
        - sigma: Scalar
        - alpha: Same shape as input points without the `dim`-th dimension.
    """
    if not torch.is_tensor(points):
        raise TypeError('Expected input points to be of type torch.Tensor. Got {0} instead.'.format(type(points)))
    if not (torch.is_tensor(sigma) or isinstance(sigma, float) or isinstance(sigma, int)):
        raise TypeError('Expected input sigma to be of type torch.Tensor or float or int. Got {0} instead.'.format(type(sigma)))
    if not isinstance(eps, float):
        raise TypeError('Expected input eps to be of type float. Got {0} instead.'.format(type(eps)))
    if points.shape[dim] != 3:
        raise ValueError('Expected length of dim-th ({0}th) dimension to be 3. Got {1} instead.'.format(dim, points.shape[dim]))
    if torch.is_tensor(sigma) and sigma.ndim != 0:
        raise ValueError('Expected sigma.ndim to be 0 (scalar). Got {0}.'.format(sigma.ndim))
    alpha = torch.exp(-torch.sum(points ** 2, dim, keepdim=keepdim) / (2 * sigma ** 2))
    alpha = torch.clamp(alpha, min=eps, max=1.01)
    return alpha


def fuse_with_map(pointclouds: Pointclouds, rgbdimages: RGBDImages, pc2im_bnhw: torch.Tensor, sigma: Union[torch.Tensor, float, int], inplace: bool=False) ->Pointclouds:
    """Fuses points from live frames with global maps by merging corresponding points and appending new points.
    (See section 4.2 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of global maps. Must have points, colors, normals and features
            (ccounts).
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        pc2im_bnhw (torch.Tensor): Unique correspondence lookup table. Each row contains batch index `b`, point index
            (in pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively.
        sigma (torch.Tensor or float or int): Standard deviation of the Gaussian. Original paper uses 0.6 emperically.
        inplace (bool): Can optionally update the pointclouds in-place. Default: False

    Returns:
        pointclouds (gradslam.Pointclouds): Updated Pointclouds object containing global maps.

    Shape:
        - pc2im_bnhw: :math:`(\\text{num_unique_correspondences}, 4)`
        - sigma: Scalar

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError('Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.'.format(type(pointclouds)))
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError('Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.'.format(type(rgbdimages)))
    if not torch.is_tensor(pc2im_bnhw):
        raise TypeError('Expected input pc2im_bnhw to be of type torch.Tensor. Got {0} instead.'.format(type(pc2im_bnhw)))
    if pc2im_bnhw.dtype != torch.int64:
        raise TypeError('Expected input pc2im_bnhw to have dtype of torch.int64 (torch.long), not {0}.'.format(pc2im_bnhw.dtype))
    if pc2im_bnhw.ndim != 2:
        raise ValueError('Expected pc2im_bnhw.ndim of 2. Got {0}.'.format(pc2im_bnhw.ndim))
    if pc2im_bnhw.shape[1] != 4:
        raise ValueError('Expected pc2im_bnhw.shape[1] to be 4. Got {0}.'.format(pc2im_bnhw.shape[1]))
    if pointclouds.has_points:
        if not pointclouds.has_normals:
            raise ValueError('Pointclouds must have normals for map fusion, but did not.')
        if not pointclouds.has_colors:
            raise ValueError('Pointclouds must have colors for map fusion, but did not.')
        if not pointclouds.has_features:
            raise ValueError('Pointclouds must have features (ccounts) for map fusion, but did not.')
    vertex_maps = rgbdimages.global_vertex_map
    normal_maps = rgbdimages.global_normal_map
    rgb_image = rgbdimages.rgb_image
    alpha_image = get_alpha(rgbdimages.vertex_map, dim=4, keepdim=True, sigma=sigma)
    if pointclouds.has_points and pc2im_bnhw.shape[0] != 0:
        frame_points = torch.zeros_like(pointclouds.points_padded)
        frame_normals = torch.zeros_like(pointclouds.normals_padded)
        frame_colors = torch.zeros_like(pointclouds.colors_padded)
        frame_alphas = torch.zeros_like(pointclouds.features_padded)
        frame_points[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = vertex_maps[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]]
        frame_normals[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = normal_maps[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]]
        frame_colors[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = rgb_image[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]]
        frame_alphas[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = alpha_image[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]]
        map_ccounts = pointclouds.features_padded
        updated_ccounts = map_ccounts + frame_alphas
        updated_points = map_ccounts * pointclouds.points_padded + frame_alphas * frame_points
        updated_normals = map_ccounts * pointclouds.normals_padded + frame_alphas * frame_normals
        updated_colors = map_ccounts * pointclouds.colors_padded + frame_alphas * frame_colors
        inv_updated_ccounts = 1 / torch.where(updated_ccounts == 0, torch.ones_like(updated_ccounts), updated_ccounts)
        pointclouds.points_padded = updated_points * inv_updated_ccounts
        pointclouds.normals_padded = updated_normals * inv_updated_ccounts
        pointclouds.colors_padded = updated_colors * inv_updated_ccounts
        pointclouds.features_padded = updated_ccounts
    new_mask = torch.ones_like(vertex_maps[..., 0], dtype=bool)
    if pointclouds.has_points and pc2im_bnhw.shape[0] != 0:
        new_mask[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]] = 0
    new_mask = new_mask * rgbdimages.valid_depth_mask.squeeze(-1)
    B = new_mask.shape[0]
    new_points = [vertex_maps[b][new_mask[b]] for b in range(B)]
    new_normals = [normal_maps[b][new_mask[b]] for b in range(B)]
    new_colors = [rgb_image[b][new_mask[b]] for b in range(B)]
    new_features = [alpha_image[b][new_mask[b]] for b in range(B)]
    new_pointclouds = Pointclouds(points=new_points, normals=new_normals, colors=new_colors, features=new_features)
    if not inplace:
        pointclouds = pointclouds.clone()
    pointclouds.append_points(new_pointclouds)
    return pointclouds


def update_map_fusion(pointclouds: Pointclouds, rgbdimages: RGBDImages, dist_th: Union[float, int], dot_th: Union[float, int], sigma: Union[torch.Tensor, float, int], inplace: bool=False) ->Pointclouds:
    """Updates pointclouds in-place given the live frame RGB-D images using PointFusion.
    (See Point-based Fusion `paper <http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf>`__).

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of global maps. Must have points, colors, normals and features
            (ccounts).
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        dist_th (float or int): Distance threshold.
        dot_th (float or int): Dot product threshold.
        sigma (torch.Tensor or float or int): Standard deviation of the Gaussian. Original paper uses 0.6 emperically.
        inplace (bool): Can optionally update the pointclouds in-place. Default: False

    Returns:
        gradslam.Pointclouds: Updated Pointclouds object containing global maps.

    """
    batch_size, seq_len, height, width = rgbdimages.shape
    pc2im_bnhw = find_correspondences(pointclouds, rgbdimages, dist_th, dot_th)
    pointclouds = fuse_with_map(pointclouds, rgbdimages, pc2im_bnhw, sigma, inplace)
    return pointclouds


class PointFusion(ICPSLAM):
    """Point-based Fusion (PointFusion for short) SLAM for batched sequences of RGB-D images
    (See Point-based Fusion `paper <http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf>`__).

    Args:
        odom (str): Odometry method to be used from {'gt', 'icp', 'gradicp'}. Default: 'gradicp'
        dist_th (float or int): Distance threshold.
        dot_th (float or int): Dot product threshold.
        sigma (torch.Tensor or float or int): Width of the gaussian bell. Original paper uses 0.6 emperically.
        dsratio (int): Downsampling ratio to apply to input frames before ICP. Only used if `odom` is
            'icp' or 'gradicp'. Default: 4
        numiters (int): Number of iterations to run the optimization for. Only used if `odom` is
            'icp' or 'gradicp'. Default: 20
        damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Only used if `odom` is
            'icp' or 'gradicp'. Default: 1e-8
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
                Only used if `odom` is 'icp' or 'gradicp'. Default: None
        lambda_max (float or int): Maximum value the damping function can assume (`lambda_min` will be
            :math:`\\frac{1}{\\text{lambda_max}}`). Only used if `odom` is 'gradicp'.
        B (float or int): gradLM falloff control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        B2 (float or int): gradLM control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        nu (float or int): gradLM control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            the CPU. Default: None


    Examples::

    >>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
    >>> slam = PointFusion(odom='gt')
    >>> pointclouds, poses = slam(rgbdimages)
    >>> o3d.visualization.draw_geometries([pointclouds.o3d(0)])
    """

    def __init__(self, *, odom: str='gradicp', dist_th: Union[float, int]=0.05, angle_th: Union[float, int]=20, sigma: Union[float, int]=0.6, dsratio: int=4, numiters: int=20, damp: float=1e-08, dist_thresh: Union[float, int, None]=None, lambda_max: Union[float, int]=2.0, B: Union[float, int]=1.0, B2: Union[float, int]=1.0, nu: Union[float, int]=200.0, device: Union[torch.device, str, None]=None):
        super().__init__(odom=odom, dsratio=dsratio, numiters=numiters, damp=damp, dist_thresh=dist_thresh, lambda_max=lambda_max, B=B, B2=B2, nu=nu, device=device)
        if not (isinstance(dist_th, float) or isinstance(dist_th, int)):
            raise TypeError('Distance threshold must be of type float or int; but was of type {}.'.format(type(dist_th)))
        if not (isinstance(angle_th, float) or isinstance(angle_th, int)):
            raise TypeError('Angle threshold must be of type float or int; but was of type {}.'.format(type(angle_th)))
        if dist_th < 0:
            warnings.warn('Distance threshold ({}) should be non-negative.'.format(dist_th))
        if not (0 <= angle_th and angle_th <= 90):
            warnings.warn('Angle threshold ({}) should be non-negative and <=90.'.format(angle_th))
        self.dist_th = dist_th
        rad_th = angle_th * math.pi / 180
        self.dot_th = torch.cos(rad_th) if torch.is_tensor(rad_th) else math.cos(rad_th)
        self.sigma = sigma

    def _map(self, pointclouds: Pointclouds, live_frame: RGBDImages, inplace: bool=False):
        return update_map_fusion(pointclouds, live_frame, self.dist_th, self.dot_th, self.sigma, inplace)

