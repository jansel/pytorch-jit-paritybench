import sys
_module = sys.modules[__name__]
del sys
conf = _module
distance = _module
distance2plane = _module
gyrovector_parallel_transport = _module
mobius_add = _module
mobius_matvec = _module
mobius_sigmoid_apply = _module
parallel_transport = _module
mobius_linear_example = _module
geoopt = _module
docutils = _module
linalg = _module
_expm = _module
batch_linalg = _module
manifolds = _module
base = _module
birkhoff_polytope = _module
euclidean = _module
product = _module
scaled = _module
sphere = _module
stereographic = _module
manifold = _module
math = _module
stiefel = _module
optim = _module
mixin = _module
radam = _module
rsgd = _module
sparse_radam = _module
sparse_rsgd = _module
samplers = _module
base = _module
rhmc = _module
rsgld = _module
sgrhmc = _module
tensor = _module
utils = _module
setup = _module
test_adam = _module
test_birkhoff = _module
test_euclidean = _module
test_gyrovector_math = _module
test_manifold_basic = _module
test_origin = _module
test_product_manifold = _module
test_random = _module
test_rhmc = _module
test_rsgd = _module
test_scaling = _module
test_sparse_adam = _module
test_sparse_rsgd = _module
test_tensor_api = _module
test_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import math


import torch.nn


import torch.jit


from typing import List


import abc


import itertools


from typing import Optional


from typing import Tuple


from typing import Union


import functools


import inspect


import types


import torch.optim


import torch.optim.optimizer


from torch import optim as optim


from typing import Any


import re


import random


import warnings


import collections


def create_ball(ball=None, c=None):
    """
    Helper to create a PoincareBall.

    Sometimes you may want to share a manifold across layers, e.g. you are using scaled PoincareBall.
    In this case you will require same curvature parameters for different layers or end up with nans.

    Parameters
    ----------
    ball : geoopt.PoincareBall
    c : float

    Returns
    -------
    geoopt.PoincareBall
    """
    if ball is None:
        assert c is not None, 'curvature of the ball should be explicitly specified'
        ball = geoopt.PoincareBall(c)
    return ball


class MobiusLinear(torch.nn.Linear):

    def __init__(self, *args, nonlin=None, ball=None, c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ball = create_ball(ball, c)
        if self.bias is not None:
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
        self.nonlin = nonlin
        self.reset_parameters()

    def forward(self, input):
        return mobius_linear(input, weight=self.weight, bias=self.bias, nonlin=self.nonlin, ball=self.ball)

    @torch.no_grad()
    def reset_parameters(self):
        torch.nn.init.eye_(self.weight)
        self.weight.add_(torch.rand_like(self.weight).mul_(0.001))
        if self.bias is not None:
            self.bias.zero_()


class ScalingInfo(object):
    """
    Scaling info for each argument that requires rescaling.

    .. code:: python

        scaled_value = value * scaling ** power if power != 0 else value

    For results it is not always required to set powers of scaling, then it is no-op.

    The convention for this info is the following. The output of a function is either a tuple or a single object.
    In any case, outputs are treated as positionals. Function inputs, in contrast, are treated by keywords.
    It is a common practice to maintain function signature when overriding, so this way may be considered
    as a sufficient in this particular scenario. The only required info for formula above is ``power``.
    """
    NotCompatible = object()
    __slots__ = ['kwargs', 'results']

    def __init__(self, *results: float, **kwargs: float):
        self.results = results
        self.kwargs = kwargs


class ScalingStorage(dict):
    """
    Helper class to make implementation transparent.

    This is just a dictionary with additional overriden ``__call__``
    for more explicit and elegant API to declare members. A usage example may be found in :class:`Manifold`.

    Methods that require rescaling when wrapped into :class:`Scaled` should be defined as follows

    1. Regular methods like ``dist``, ``dist2``, ``expmap``, ``retr`` etc. that are already present in the base class
    do not require registration, it has already happened in the base :class:`Manifold` class.

    2. New methods (like in :class:`PoincareBall`) should be treated with care.

    .. code-block:: python

        class PoincareBall(Manifold):
            # make a class copy of __scaling__ info. Default methods are already present there
            __scaling__ = Manifold.__scaling__.copy()
            ... # here come regular implementation of the required methods

            @__scaling__(ScalingInfo(1))  # rescale output according to rule `out * scaling ** 1`
            def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False):
                return math.dist0(x, c=self.c, dim=dim, keepdim=keepdim)

            @__scaling__(ScalingInfo(u=-1))  # rescale argument `u` according to the rule `out * scaling ** -1`
            def expmap0(self, u: torch.Tensor, *, dim=-1, project=True):
                res = math.expmap0(u, c=self.c, dim=dim)
                if project:
                    return math.project(res, c=self.c, dim=dim)
                else:
                    return res
            ... # other special methods implementation

    3. Some methods are not compliant with the above rescaling rules. We should mark them as `NotCompatible`

    .. code-block:: python

            # continuation of the PoincareBall definition
            @__scaling__(ScalingInfo.NotCompatible)
            def mobius_fn_apply(
                self, fn: callable, x: torch.Tensor, *args, dim=-1, project=True, **kwargs
            ):
                res = math.mobius_fn_apply(fn, x, *args, c=self.c, dim=dim, **kwargs)
                if project:
                    return math.project(res, c=self.c, dim=dim)
                else:
                    return res
    """

    def __call__(self, scaling_info: ScalingInfo, *aliases):

        def register(fn):
            self[fn.__name__] = scaling_info
            for alias in aliases:
                self[alias] = scaling_info
            return fn
        return register

    def copy(self):
        return self.__class__(self)


class Manifold(torch.nn.Module, metaclass=abc.ABCMeta):
    __scaling__ = ScalingStorage()
    name = None
    ndim = None
    reversible = None
    forward = NotImplemented

    def __init__(self, **kwargs):
        super().__init__()

    @property
    def device(self) ->Optional[torch.device]:
        """
        Manifold device.

        Returns
        -------
        Optional[torch.device]
        """
        p = next(itertools.chain(self.buffers(), self.parameters()), None)
        if p is not None:
            return p.device
        else:
            return None

    @property
    def dtype(self) ->Optional[torch.dtype]:
        """
        Manifold dtype.

        Returns
        -------
        Optional[torch.dtype]
        """
        p = next(itertools.chain(self.buffers(), self.parameters()), None)
        if p is not None:
            return p.dtype
        else:
            return None

    def check_point(self, x: torch.Tensor, *, explain=False) ->Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if point is valid to be used with the manifold.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(x.shape, 'x')
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point(self, x: torch.Tensor):
        """
        Check if point is valid to be used with the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(x.shape, 'x')
        if not ok:
            raise ValueError('`x` seems to be not valid tensor for {} manifold.\nerror: {}'.format(self.name, reason))

    def check_vector(self, u: torch.Tensor, *, explain=False):
        """
        Check if vector is valid to be used with the manifold.

        Parameters
        ----------
        u : torch.Tensor
            vector on the tangent plane
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(u.shape, 'u')
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector(self, u: torch.Tensor):
        """
        Check if vector is valid to be used with the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        u : torch.Tensor
            vector on the tangent plane

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(u.shape, 'u')
        if not ok:
            raise ValueError('`u` seems to be not valid tensor for {} manifold.\nerror: {}'.format(self.name, reason))

    def check_point_on_manifold(self, x: torch.Tensor, *, explain=False, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if point :math:`x` is lying on the manifold.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        explain: bool
            return an additional information on check

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False

        Notes
        -----
        This check is compatible to what optimizer expects, last dimensions are treated as manifold dimensions
        """
        ok, reason = self._check_shape(x.shape, 'x')
        if ok:
            ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-05, rtol=1e-05):
        """
        Check if point :math`x` is lying on the manifold and raise an error with informative message on failure.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        """
        self.assert_check_point(x)
        ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError('`x` seems to be a tensor not lying on {} manifold.\nerror: {}'.format(self.name, reason))

    def check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, ok_point=False, explain=False, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        """
        Check if :math:`u` is lying on the tangent space to x.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            vector on the tangent space to :math:`x`
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        explain: bool
            return an additional information on check
        ok_point: bool
            is a check for point required?

        Returns
        -------
        bool
            boolean indicating if tensor is valid and reason of failure if False
        """
        if not ok_point:
            ok, reason = self._check_shape(x.shape, 'x')
            if ok:
                ok, reason = self._check_shape(u.shape, 'u')
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if explain:
            return ok, reason
        else:
            return ok

    def assert_check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, ok_point=False, atol=1e-05, rtol=1e-05):
        """
        Check if u :math:`u` is lying on the tangent space to x and raise an error on fail.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            vector on the tangent space to :math:`x`
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`
        ok_point: bool
            is a check for point required?
        """
        if not ok_point:
            ok, reason = self._check_shape(x.shape, 'x')
            if ok:
                ok, reason = self._check_shape(u.shape, 'u')
            if ok:
                ok, reason = self._check_point_on_manifold(x, atol=atol, rtol=rtol)
        else:
            ok = True
            reason = None
        if ok:
            ok, reason = self._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)
        if not ok:
            raise ValueError('`u` seems to be a tensor not lying on tangent space to `x` for {} manifold.\nerror: {}'.format(self.name, reason))

    @__scaling__(ScalingInfo(1))
    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) ->torch.Tensor:
        """
        Compute distance between 2 points on the manifold that is the shortest path along geodesics.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            distance between two points
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(2))
    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) ->torch.Tensor:
        """
        Compute squared distance between 2 points on the manifold that is the shortest path along geodesics.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            squared distance between two points
        """
        return self.dist(x, y, keepdim=keepdim) ** 2

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
    def retr(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        """
        Perform a retraction from point :math:`x` with given direction :math:`u`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            transported point
        """
        raise NotImplementedError

    @abc.abstractmethod
    @__scaling__(ScalingInfo(u=-1))
    def expmap(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        """
        Perform an exponential map :math:`\\operatorname{Exp}_x(u)`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            transported point
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(1))
    def logmap(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        """
        Perform an logarithmic map :math:`\\operatorname{Log}_{x}(y)`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        y : torch.Tensor
            point on the manifold

        Returns
        -------
        torch.Tensor
            tangent vector
        """
        raise NotImplementedError

    @__scaling__(ScalingInfo(u=-1))
    def expmap_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform an exponential map and vector transport from point :math:`x` with given direction :math:`u`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported point
        """
        y = self.expmap(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def retr_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a retraction + vector transport at once.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            transported point and vectors

        Notes
        -----
        Sometimes this is a far more optimal way to preform retraction + vector transport
        """
        y = self.retr(x, u)
        v_transp = self.transp(x, y, v)
        return y, v_transp

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_retr(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        """
        Perform vector transport following :math:`u`: :math:`\\mathfrak{T}_{x\\to\\operatorname{retr}(x, u)}(v)`.

        This operation is sometimes is much more simpler and can be optimized.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported tensor
        """
        y = self.retr(x, u)
        return self.transp(x, y, v)

    @__scaling__(ScalingInfo(u=-1))
    def transp_follow_expmap(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        """
        Perform vector transport following :math:`u`: :math:`\\mathfrak{T}_{x\\to\\operatorname{Exp}(x, u)}(v)`.

        Here, :math:`\\operatorname{Exp}` is the best possible approximation of the true exponential map.
        There are cases when the exact variant is hard or impossible implement, therefore a
        fallback, non-exact, implementation is used.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : torch.Tensor
            tangent vector at point :math:`x` to be transported

        Returns
        -------
        torch.Tensor
            transported tensor
        """
        y = self.expmap(x, u)
        return self.transp(x, y, v)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        """
        Perform vector transport :math:`\\mathfrak{T}_{x\\to y}(v)`.

        Parameters
        ----------
        x : torch.Tensor
            start point on the manifold
        y : torch.Tensor
            target point on the manifold
        v : torch.Tensor
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
           transported tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inner(self, x: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False) ->torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`x`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : Optional[torch.Tensor]
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)
        """
        raise NotImplementedError

    def component_inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor=None) ->torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`x` according to components of the manifold.

        The result of the function is same as ``inner`` with ``keepdim=True`` for
        all the manifolds except ProductManifold. For this manifold it acts different way
        computing inner product for each component and then building an output correctly
        tiling and reshaping the result.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        v : Optional[torch.Tensor]
            tangent vector at point :math:`x`

        Returns
        -------
        torch.Tensor
            inner product component wise (broadcasted)

        Notes
        -----
        The purpose of this method is better adaptive properties in optimization since ProductManifold
        will "hide" the structure in public API.
        """
        return self.inner(x, u, v, keepdim=True)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False) ->torch.Tensor:
        """
        Norm of a tangent vector at point :math:`x`.

        Parameters
        ----------
        x : torch.Tensor
            point on the manifold
        u : torch.Tensor
            tangent vector at point :math:`x`
        keepdim : bool
            keep the last dim?

        Returns
        -------
        torch.Tensor
            inner product (broadcasted)
        """
        return self.inner(x, u, keepdim=keepdim) ** 0.5

    @abc.abstractmethod
    def proju(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        """
        Project vector :math:`u` on a tangent space for :math:`x`, usually is the same as :meth:`egrad2rgrad`.

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        u torch.Tensor
            vector to be projected

        Returns
        -------
        torch.Tensor
            projected vector
        """
        raise NotImplementedError

    @abc.abstractmethod
    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        u torch.Tensor
            gradient to be projected

        Returns
        -------
        torch.Tensor
            grad vector in the Riemannian manifold
        """
        raise NotImplementedError

    @abc.abstractmethod
    def projx(self, x: torch.Tensor) ->torch.Tensor:
        """
        Project point :math:`x` on the manifold.

        Parameters
        ----------
        x torch.Tensor
            point to be projected

        Returns
        -------
        torch.Tensor
            projected point
        """
        raise NotImplementedError

    def _check_shape(self, shape: Tuple[int], name: str) ->Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check shape.

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It should return boolean and
        a reason of failure if check is not passed

        Parameters
        ----------
        shape : Tuple[int]
            shape of point on the manifold
        name : str
            name to be present in errors

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        ok = len(shape) >= self.ndim
        if not ok:
            reason = "'{}' on the {} requires more than {} dim".format(name, self, self.ndim)
        else:
            reason = None
        return ok, reason

    def _assert_check_shape(self, shape: Tuple[int], name: str):
        """
        Util to check shape and raise an error if needed.

        Exhaustive implementation for checking if
        a given point has valid dimension size,
        shape, etc. It will raise a ValueError if check is not passed

        Parameters
        ----------
        shape : tuple
            shape of point on the manifold
        name : str
            name to be present in errors

        Raises
        ------
        ValueError
        """
        ok, reason = self._check_shape(shape, name)
        if not ok:
            raise ValueError(reason)

    @abc.abstractmethod
    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check point lies on the manifold.

        Exhaustive implementation for checking if
        a given point lies on the manifold. It
        should return boolean and a reason of
        failure if check is not passed. You can
        assume assert_check_point is already
        passed beforehand

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check a vector belongs to the tangent space of a point.

        Exhaustive implementation for checking if
        a given point lies in the tangent space at x
        of the manifold. It should return a boolean
        indicating whether the test was passed
        and a reason of failure if check is not passed.
        You can assume assert_check_point is already
        passed beforehand

        Parameters
        ----------
        x torch.Tensor
        u torch.Tensor
        atol : float
            absolute tolerance
        rtol :
            relative tolerance

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        raise NotImplementedError

    def extra_repr(self):
        return ''

    def __repr__(self):
        extra = self.extra_repr()
        if extra:
            return self.name + '({}) manifold'.format(extra)
        else:
            return self.name + ' manifold'

    def unpack_tensor(self, tensor: torch.Tensor) ->torch.Tensor:
        """
        Construct a point on the manifold.

        This method should help to work with product and compound manifolds.
        Internally all points on the manifold are stored in an intuitive format.
        However, there might be cases, when this representation is simpler or more efficient to store in
        a different way that is hard to use in practice.

        Parameters
        ----------
        tensor : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return tensor

    def pack_point(self, *tensors: torch.Tensor) ->torch.Tensor:
        """
        Construct a tensor representation of a manifold point.

        In case of regular manifolds this will return the same tensor. However, for e.g. Product manifold
        this function will pack all non-batch dimensions.

        Parameters
        ----------
        tensors : Tuple[torch.Tensor]

        Returns
        -------
        torch.Tensor
        """
        if len(tensors) != 1:
            raise ValueError('1 tensor expected, got {}'.format(len(tensors)))
        return tensors[0]

    def random(self, *size, dtype=None, device=None, **kwargs) ->torch.Tensor:
        """
        Random sampling on the manifold.

        The exact implementation depends on manifold and usually does not follow all
        assumptions about uniform measure, etc.
        """
        raise NotImplementedError

    def origin(self, *size: Union[int, Tuple[int]], dtype=None, device=None, seed: Optional[int]=42) ->torch.Tensor:
        """
        Create some reasonable point on the manifold in a deterministic way.

        For some manifolds there may exist e.g. zero vector or some analogy.
        In case it is possible to define this special point, this point is returned with the desired size.
        In other case, the returned point is sampled on the manifold in a deterministic way.

        Parameters
        ----------
        size : Union[int, Tuple[int]]
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : Optional[int]
            A parameter controlling deterministic randomness for manifolds that do not provide ``.origin``,
            but provide ``.random``. (default: 42)

        Returns
        -------
        torch.Tensor
        """
        if seed is not None:
            state = torch.random.get_rng_state()
            torch.random.manual_seed(seed)
            try:
                return self.random(*size, dtype=dtype, device=device)
            finally:
                torch.random.set_rng_state(state)
        else:
            return self.random(*size, dtype=dtype, device=device)


def broadcast_shapes(*shapes: Tuple[int]) ->Tuple[int]:
    """Apply numpy broadcasting rules to shapes."""
    result = []
    for dims in itertools.zip_longest(*map(reversed, shapes), fillvalue=1):
        dim: int = 1
        for d in dims:
            if dim != 1 and d != 1 and d != dim:
                raise ValueError("Shapes can't be broadcasted")
            elif d > dim:
                dim = d
        result.append(dim)
    return tuple(reversed(result))


def make_tuple(obj: Union[Tuple, List, Any]) ->Tuple:
    if isinstance(obj, list):
        obj = tuple(obj)
    if not isinstance(obj, tuple):
        return obj,
    else:
        return obj


def strip_tuple(tup: Tuple) ->Union[Tuple, Any]:
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def size2shape(*size: Union[Tuple[int], int]) ->Tuple[int]:
    return make_tuple(strip_tuple(size))


class Euclidean(Manifold):
    """
    Simple Euclidean manifold, every coordinate is treated as an independent element.

    Parameters
    ----------
    ndim : int
        number of trailing dimensions treated as manifold dimensions. All the operations acting on cuch
        as inner products, etc will respect the :attr:`ndim`.
    """
    __scaling__ = Manifold.__scaling__.copy()
    name = 'Euclidean'
    ndim = 0
    reversible = True

    def __init__(self, ndim=0):
        super().__init__()
        self.ndim = ndim

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        return True, None

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        return True, None

    def retr(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        return x + u

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor=None, *, keepdim=False) ->torch.Tensor:
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v
        if self.ndim > 0:
            inner = inner.sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
            x_shape = x.shape[:-self.ndim] + (1,) * self.ndim * keepdim
        else:
            x_shape = x.shape
        i_shape = inner.shape
        target_shape = broadcast_shapes(x_shape, i_shape)
        return inner.expand(target_shape)

    def component_inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor=None) ->torch.Tensor:
        if v is None:
            inner = u.pow(2)
        else:
            inner = u * v
        target_shape = broadcast_shapes(x.shape, inner.shape)
        return inner.expand(target_shape)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False):
        if self.ndim > 0:
            return u.norm(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return u.abs()

    def proju(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def projx(self, x: torch.Tensor) ->torch.Tensor:
        return x

    def logmap(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        return y - x

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) ->torch.Tensor:
        if self.ndim > 0:
            return (x - y).norm(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return (x - y).abs()

    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) ->torch.Tensor:
        if self.ndim > 0:
            return (x - y).pow(2).sum(dim=tuple(range(-self.ndim, 0)), keepdim=keepdim)
        else:
            return (x - y).pow(2)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        return x + u

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        target_shape = broadcast_shapes(x.shape, y.shape, v.shape)
        return v.expand(target_shape)

    @__scaling__(ScalingInfo(std=-1), 'random')
    def random_normal(self, *size, mean=0.0, std=1.0, device=None, dtype=None) ->'geoopt.ManifoldTensor':
        """
        Create a point on the manifold, measure is induced by Normal distribution.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype

        Returns
        -------
        ManifoldTensor
            random point on the manifold
        """
        self._assert_check_shape(size2shape(*size), 'x')
        mean = torch.as_tensor(mean, device=device, dtype=dtype)
        std = torch.as_tensor(std, device=device, dtype=dtype)
        tens = std.new_empty(*size).normal_() * std + mean
        return geoopt.ManifoldTensor(tens, manifold=self)
    random = random_normal

    def origin(self, *size, dtype=None, device=None, seed=42) ->'geoopt.ManifoldTensor':
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
        """
        self._assert_check_shape(size2shape(*size), 'x')
        return geoopt.ManifoldTensor(torch.zeros(*size, dtype=dtype, device=device), manifold=self)

    def extra_repr(self):
        return 'ndim={}'.format(self.ndim)


def _rebuild_manifold_parameter(cls, *args):
    import torch._utils
    tensor = torch._utils._rebuild_tensor_v2(*args[:-1])
    return cls(tensor, manifold=args[-1], requires_grad=args[-3])


def copy_or_set_(dest: torch.Tensor, source: torch.Tensor) ->torch.Tensor:
    """
    Copy or inplace set from :code:`source` to :code:`dest`.

    A workaround to respect strides of :code:`dest` when copying :code:`source`.
    The original issue was raised `here <https://github.com/geoopt/geoopt/issues/70>`_
    when working with matrix manifolds. Inplace set operation is mode efficient,
    but the resulting storage might be incompatible after. To avoid the issue we refer to
    the safe option and use :code:`copy_` if strides do not match.

    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data
    source : torch.Tensor
        Source data to put in the new tensor

    Returns
    -------
    dest
        torch.Tensor, modified inplace
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


def insert_docs(doc, pattern=None, repl=None):

    def wrapper(fn):
        if pattern is not None:
            if repl is None:
                raise RuntimeError('need repl parameter')
            fn.__doc__ = re.sub(pattern, repl, doc)
        else:
            fn.__doc__ = doc
        return fn
    return wrapper


class ManifoldTensor(torch.Tensor):
    """Same as :class:`torch.Tensor` that has information about its manifold.

    Other Parameters
    ----------------
    manifold : :class:`geoopt.Manifold`
        A manifold for the tensor, (default: :class:`geoopt.Euclidean`)
    """
    manifold: Manifold

    def __new__(cls, *args, manifold: Manifold=Euclidean(), requires_grad=False, **kwargs):
        if len(args) == 1 and isinstance(args[0], torch.Tensor):
            data = args[0].data
        else:
            data = torch.Tensor(*args, **kwargs)
        if kwargs.get('device') is not None:
            data.data = data.data
        with torch.no_grad():
            manifold.assert_check_point(data)
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.manifold = manifold
        return instance

    def proj_(self) ->torch.Tensor:
        """
        Inplace projection to the manifold.

        Returns
        -------
        tensor
            same instance
        """
        return copy_or_set_(self, self.manifold.projx(self))

    @insert_docs(Manifold.retr.__doc__, '\\s+x : .+\\n.+', '')
    def retr(self, u: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.manifold.retr(self, u=u, **kwargs)

    @insert_docs(Manifold.expmap.__doc__, '\\s+x : .+\\n.+', '')
    def expmap(self, u: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.manifold.expmap(self, u=u, **kwargs)

    @insert_docs(Manifold.inner.__doc__, '\\s+x : .+\\n.+', '')
    def inner(self, u: torch.Tensor, v: torch.Tensor=None, **kwargs) ->torch.Tensor:
        return self.manifold.inner(self, u=u, v=v, **kwargs)

    @insert_docs(Manifold.proju.__doc__, '\\s+x : .+\\n.+', '')
    def proju(self, u: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.manifold.proju(self, u, **kwargs)

    @insert_docs(Manifold.transp.__doc__, '\\s+x : .+\\n.+', '')
    def transp(self, y: torch.Tensor, v: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.manifold.transp(self, y, v, **kwargs)

    @insert_docs(Manifold.retr_transp.__doc__, '\\s+x : .+\\n.+', '')
    def retr_transp(self, u: torch.Tensor, v: torch.Tensor, **kwargs) ->Tuple[torch.Tensor, torch.Tensor]:
        return self.manifold.retr_transp(self, u, v, **kwargs)

    @insert_docs(Manifold.expmap_transp.__doc__, '\\s+x : .+\\n.+', '')
    def expmap_transp(self, u: torch.Tensor, v: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.manifold.expmap_transp(self, u, v, **kwargs)

    @insert_docs(Manifold.transp_follow_expmap.__doc__, '\\s+x : .+\\n.+', '')
    def transp_follow_expmap(self, u: torch.Tensor, v: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.manifold.transp_follow_expmap(self, u, v, **kwargs)

    @insert_docs(Manifold.transp_follow_retr.__doc__, '\\s+x : .+\\n.+', '')
    def transp_follow_retr(self, u: torch.Tensor, v: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.manifold.transp_follow_retr(self, u, v, **kwargs)

    def dist(self, other: torch.Tensor, p: Union[int, float, bool, str]=2, **kwargs) ->torch.Tensor:
        """
        Return euclidean  or geodesic distance between points on the manifold. Allows broadcasting.

        Parameters
        ----------
        other : tensor
        p : str|int
            The norm to use. The default behaviour is not changed and is just euclidean distance.
            To compute geodesic distance, :attr:`p` should be set to ``"g"``

        Returns
        -------
        scalar
        """
        if p == 'g':
            return self.manifold.dist(self, other, **kwargs)
        else:
            return super().dist(other)

    @insert_docs(Manifold.logmap.__doc__, '\\s+x : .+\\n.+', '')
    def logmap(self, y: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.manifold.logmap(self, y, **kwargs)

    def __repr__(self):
        return 'Tensor on {} containing:\n'.format(self.manifold) + torch.Tensor.__repr__(self)

    def __reduce_ex__(self, proto):
        proto = self.__class__, self.storage(), self.storage_offset(), self.size(), self.stride(), self.requires_grad, dict()
        return _rebuild_manifold_parameter, proto + (self.manifold,)

    @insert_docs(Manifold.unpack_tensor.__doc__, '\\s+tensor : .+\\n.+', '')
    def unpack_tensor(self) ->Union[torch.Tensor, Tuple[torch.Tensor]]:
        return self.manifold.unpack_tensor(self)


@torch.jit.script
def proj_doubly_stochastic(x, max_iter: int=300, eps: float=1e-05, tol: float=1e-05):
    iter = 0
    c = 1.0 / (x.sum(dim=-2, keepdim=True) + eps)
    r = 1.0 / (x @ c.transpose(-1, -2) + eps)
    while iter < max_iter:
        iter += 1
        cinv = torch.matmul(r.transpose(-1, -2), x)
        if torch.max(torch.abs(cinv * c - 1)) <= tol:
            break
        c = 1.0 / (cinv + eps)
        r = 1.0 / (x @ c.transpose(-1, -2) + eps)
    return x * (r @ c)


class BirkhoffPolytope(Manifold):
    """
    Birkhoff Polytope Manifold.

    Manifold induced by the Doubly Stochastic matrices as described in
    A. Douik and B. Hassibi, "Manifold Optimization Over the Set
    of Doubly Stochastic Matrices: A Second-Order Geometry"
    ArXiv:1802.02628, 2018.
    Link to the paper: https://arxiv.org/abs/1802.02628.

    @Techreport{Douik2018Manifold,
       Title   = {Manifold Optimization Over the Set of Doubly Stochastic
                  Matrices: {A} Second-Order Geometry},
       Author  = {Douik, A. and Hassibi, B.},
       Journal = {Arxiv preprint ArXiv:1802.02628},
       Year    = {2018}
    }

    Please also cite:
    Tolga Birdal, Umut Şimşekli,
    "Probabilistic Permutation Synchronization using the Riemannian Structure of the BirkhoffPolytope Polytope"
    IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2019
    Link to the paper: https://arxiv.org/abs/1904.05814

    @inproceedings{birdal2019probabilistic,
    title={Probabilistic Permutation Synchronization using the Riemannian Structure of the Birkhoff Polytope},
    author={Birdal, Tolga and Simsekli, Umut},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    pages={11105--11116},
    year={2019}
    }

    This implementation is by Tolga Birdal and Haowen Deng.
    """
    name = 'BirkhoffPolytope'
    reversible = False
    ndim = 2

    def __init__(self, max_iter=100, tol=1e-05, eps=1e-12):
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.eps = eps

    def _check_shape(self, shape, name):
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason
        shape_is_ok = shape[-1] == shape[-2]
        if not shape_is_ok:
            return False, '`{}` should have shape[-1] == shape[-2], got {} != {}'.format(name, shape[-1], shape[-2])
        return True, None

    def _check_point_on_manifold(self, x, *, atol=0.0001, rtol=0.0001):
        row_sum = x.sum(dim=-1)
        col_sum = x.sum(dim=-2)
        row_ok = torch.allclose(row_sum, row_sum.new((1,)).fill_(1), atol=atol, rtol=rtol)
        col_ok = torch.allclose(col_sum, col_sum.new((1,)).fill_(1), atol=atol, rtol=rtol)
        if row_ok and col_ok:
            return True, None
        else:
            return False, 'illegal doubly stochastic matrix with atol={}, rtol={}'.format(atol, rtol)

    def _check_vector_on_tangent(self, x, u, *, atol=1e-05, rtol=1e-05):
        diff1 = u.sum(-1)
        diff2 = u.sum(-2)
        ok1 = torch.allclose(diff1, diff1.new((1,)).fill_(0), atol=atol, rtol=rtol)
        ok2 = torch.allclose(diff2, diff2.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok1:
            return False, '`u 1 !=0` with atol={}, rtol={}'.format(atol, rtol)
        if not ok2:
            return False, '`u^T 1 !=0` with atol={}, rtol={}'.format(atol, rtol)
        return True, None

    def projx(self, x):
        return proj_doubly_stochastic(x=x, max_iter=self.max_iter, eps=self.eps, tol=self.tol)

    def proju(self, x, u):
        return proj_tangent(x, u)
    egrad2rgrad = proju

    def retr(self, x, u):
        k = u / x
        y = x * torch.exp(k)
        y = self.projx(y)
        y = torch.max(y, y.new(1).fill_(1e-12))
        return y
    expmap = retr

    def inner(self, x, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        n = x.shape[-1]
        return torch.sum(u * v / x, dim=(-1, -2), keepdim=keepdim) / n

    def transp(self, x, y, v):
        return self.proju(y, v)

    def retr_transp(self, x, u, v):
        y = self.retr(x, u)
        vs = self.transp(x, y, v)
        return (y,) + make_tuple(vs)
    expmap_transp = retr_transp

    def transp_follow_retr(self, x, u, v):
        y = self.retr(x, u)
        return self.transp(x, y, v)

    def transp_follow_expmap(self, x, u, v):
        y = self.expmap(x, u)
        return self.transp(x, y, v)

    def random_naive(self, *size, dtype=None, device=None) ->torch.Tensor:
        """
        Naive approach to get random matrix on Birkhoff Polytope manifold.

        A helper function to sample a random point on the Birkhoff Polytope manifold.
        The measure is non-uniform for this method, but fast to compute.

        Parameters
        ----------
        size : shape
            the desired output shape
        dtype : torch.dtype
            desired dtype
        device : torch.device
            desired device

        Returns
        -------
        ManifoldTensor
            random point on Birkhoff Polytope manifold
        """
        self._assert_check_shape(size2shape(*size), 'x')
        tens = torch.randn(*size, device=device, dtype=dtype).abs_()
        return ManifoldTensor(self.projx(tens), manifold=self)
    random = random_naive

    def origin(self, *size, dtype=None, device=None, seed=42) ->torch.Tensor:
        """
        Identity matrix point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
        """
        shape = size2shape(*size)
        self._assert_check_shape(shape, 'x')
        eye = torch.eye(*shape[-2:], dtype=dtype, device=device)
        eye = eye.expand(shape)
        return ManifoldTensor(eye, manifold=self)


def _calculate_target_batch_dim(*dims: int):
    return max(dims) - 1


def _shape2size(shape: Tuple[int]):
    return functools.reduce(operator.mul, shape, 1)


class ProductManifold(Manifold):
    """
    Product Manifold.

    Examples
    --------
    A Torus

    >>> import geoopt
    >>> sphere = geoopt.Sphere()
    >>> torus = ProductManifold((sphere, 2), (sphere, 2))
    """
    ndim = 1

    def __init__(self, *manifolds_with_shape: Tuple[Manifold, Union[Tuple[int, ...], int]]):
        if len(manifolds_with_shape) < 1:
            raise ValueError('There should be at least one manifold in a product manifold')
        super().__init__()
        self.shapes = []
        self.slices = []
        name_parts = []
        manifolds = []
        dtype = None
        device = None
        pos0 = 0
        for i, (manifold, shape) in enumerate(manifolds_with_shape):
            shape = geoopt.utils.size2shape(shape)
            ok, reason = manifold._check_shape(shape, str("{}'th shape".format(i)))
            if not ok:
                raise ValueError(reason)
            if manifold.device is not None and device is not None:
                if device != manifold.device:
                    raise ValueError('Not all manifold share the same device')
            elif device is None:
                device = manifold.device
            if manifold.dtype is not None and dtype is not None:
                if dtype != manifold.dtype:
                    raise ValueError('Not all manifold share the same dtype')
            elif dtype is None:
                dtype = manifold.dtype
            name_parts.append(manifold.name)
            manifolds.append(manifold)
            self.shapes.append(shape)
            pos1 = pos0 + _shape2size(shape)
            self.slices.append(slice(pos0, pos1))
            pos0 = pos1
        self.name = 'x'.join(['({})'.format(name) for name in name_parts])
        self.n_elements = pos0
        self.n_manifolds = len(manifolds)
        self.manifolds = torch.nn.ModuleList(manifolds)

    @property
    def reversible(self) ->bool:
        return all(m.reversible for m in self.manifolds)

    def take_submanifold_value(self, x: torch.Tensor, i: int, reshape=True) ->torch.Tensor:
        """
        Take i'th slice of the ambient tensor and possibly reshape.

        Parameters
        ----------
        x : tensor
            Ambient tensor
        i : int
            submanifold index
        reshape : bool
            reshape the slice?

        Returns
        -------
        torch.Tensor
        """
        slc = self.slices[i]
        part = x.narrow(-1, slc.start, slc.stop - slc.start)
        if reshape:
            part = part.reshape((*part.shape[:-1], *self.shapes[i]))
        return part

    def _check_shape(self, shape: Tuple[int], name: str) ->Tuple[bool, Optional[str]]:
        ok = shape[-1] == self.n_elements
        if not ok:
            return ok, 'The last dimension should be equal to {}, but got {}'.format(self.n_elements, shape[-1])
        return ok, None

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Tuple[bool, Optional[str]]:
        ok, reason = True, None
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            ok, reason = manifold.check_point_on_manifold(point, atol=atol, rtol=rtol, explain=True)
            if not ok:
                break
        return ok, reason

    def _check_vector_on_tangent(self, x, u, *, atol=1e-05, rtol=1e-05) ->Tuple[bool, Optional[str]]:
        ok, reason = True, None
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            tangent = self.take_submanifold_value(u, i)
            ok, reason = manifold.check_vector_on_tangent(point, tangent, atol=atol, rtol=rtol, explain=True)
            if not ok:
                break
        return ok, reason

    def inner(self, x: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False) ->torch.Tensor:
        if v is not None:
            target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim(), v.dim())
        else:
            target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim())
        products = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            u_vec = self.take_submanifold_value(u, i)
            if v is not None:
                v_vec = self.take_submanifold_value(v, i)
            else:
                v_vec = None
            inner = manifold.inner(point, u_vec, v_vec, keepdim=True)
            inner = inner.view(*inner.shape[:target_batch_dim], -1).sum(-1)
            products.append(inner)
        result = sum(products)
        if keepdim:
            result = torch.unsqueeze(result, -1)
        return result

    def component_inner(self, x: torch.Tensor, u: torch.Tensor, v=None) ->torch.Tensor:
        products = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            u_vec = self.take_submanifold_value(u, i)
            target_shape = geoopt.utils.broadcast_shapes(point.shape, u_vec.shape)
            if v is not None:
                v_vec = self.take_submanifold_value(v, i)
            else:
                v_vec = None
            inner = manifold.component_inner(point, u_vec, v_vec)
            inner = inner.expand(target_shape)
            products.append(inner)
        result = self.pack_point(*products)
        return result

    def projx(self, x: torch.Tensor) ->torch.Tensor:
        projected = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            proj = manifold.projx(point)
            proj = proj.view(*x.shape[:len(x.shape) - 1], -1)
            projected.append(proj)
        return torch.cat(projected, -1)

    def proju(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim())
        projected = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            tangent = self.take_submanifold_value(u, i)
            proj = manifold.proju(point, tangent)
            proj = proj.reshape((*proj.shape[:target_batch_dim], -1))
            projected.append(proj)
        return torch.cat(projected, -1)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim())
        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            tangent = self.take_submanifold_value(u, i)
            mapped = manifold.expmap(point, tangent)
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)
        return torch.cat(mapped_tensors, -1)

    def retr(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim())
        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            tangent = self.take_submanifold_value(u, i)
            mapped = manifold.retr(point, tangent)
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)
        return torch.cat(mapped_tensors, -1)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim(), v.dim())
        transported_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            tangent = self.take_submanifold_value(v, i)
            transported = manifold.transp(point, point1, tangent)
            transported = transported.reshape((*transported.shape[:target_batch_dim], -1))
            transported_tensors.append(transported)
        return torch.cat(transported_tensors, -1)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim())
        logmapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            logmapped = manifold.logmap(point, point1)
            logmapped = logmapped.reshape((*logmapped.shape[:target_batch_dim], -1))
            logmapped_tensors.append(logmapped)
        return torch.cat(logmapped_tensors, -1)

    def transp_follow_retr(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim(), v.dim())
        results = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            direction = self.take_submanifold_value(u, i)
            vector = self.take_submanifold_value(v, i)
            transported = manifold.transp_follow_retr(point, direction, vector)
            transported = transported.reshape((*transported.shape[:target_batch_dim], -1))
            results.append(transported)
        return torch.cat(results, -1)

    def transp_follow_expmap(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim(), v.dim())
        results = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            direction = self.take_submanifold_value(u, i)
            vector = self.take_submanifold_value(v, i)
            transported = manifold.transp_follow_expmap(point, direction, vector)
            transported = transported.reshape((*transported.shape[:target_batch_dim], -1))
            results.append(transported)
        return torch.cat(results, -1)

    def expmap_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim(), v.dim())
        results = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            direction = self.take_submanifold_value(u, i)
            vector = self.take_submanifold_value(v, i)
            new_point, transported = manifold.expmap_transp(point, direction, vector)
            transported = transported.reshape((*transported.shape[:target_batch_dim], -1))
            new_point = new_point.reshape((*new_point.shape[:target_batch_dim], -1))
            results.append((new_point, transported))
        points, vectors = zip(*results)
        return torch.cat(points, -1), torch.cat(vectors, -1)

    def retr_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim(), v.dim())
        results = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            direction = self.take_submanifold_value(u, i)
            vector = self.take_submanifold_value(v, i)
            new_point, transported = manifold.retr_transp(point, direction, vector)
            transported = transported.reshape((*transported.shape[:target_batch_dim], -1))
            new_point = new_point.reshape((*new_point.shape[:target_batch_dim], -1))
            results.append((new_point, transported))
        points, vectors = zip(*results)
        return torch.cat(points, -1), torch.cat(vectors, -1)

    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False):
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim())
        mini_dists2 = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            mini_dist2 = manifold.dist2(point, point1, keepdim=True)
            mini_dist2 = mini_dist2.reshape((*mini_dist2.shape[:target_batch_dim], -1)).sum(-1)
            mini_dists2.append(mini_dist2)
        result = sum(mini_dists2)
        if keepdim:
            result = torch.unsqueeze(result, -1)
        return result

    def dist(self, x, y, *, keepdim=False):
        return self.dist2(x, y, keepdim=keepdim).clamp_min_(1e-15) ** 0.5

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor):
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim())
        transformed_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            grad = self.take_submanifold_value(u, i)
            transformed = manifold.egrad2rgrad(point, grad)
            transformed = transformed.reshape((*transformed.shape[:target_batch_dim], -1))
            transformed_tensors.append(transformed)
        return torch.cat(transformed_tensors, -1)

    def unpack_tensor(self, tensor: torch.Tensor) ->Tuple[torch.Tensor]:
        parts = []
        for i in range(self.n_manifolds):
            part = self.take_submanifold_value(tensor, i)
            parts.append(part)
        return tuple(parts)

    def pack_point(self, *tensors: torch.Tensor) ->torch.Tensor:
        if len(tensors) != len(self.manifolds):
            raise ValueError('{} tensors expected, got {}'.format(len(self.manifolds), len(tensors)))
        flattened = []
        for i in range(self.n_manifolds):
            part = tensors[i]
            shape = self.shapes[i]
            if len(shape) > 0:
                if part.shape[-len(shape):] != shape:
                    raise ValueError('last shape dimension does not seem to be valid. {} required, but got {}'.format(part.shape[-len(shape):], shape))
                new_shape = *part.shape[:-len(shape)], -1
            else:
                new_shape = *part.shape, -1
            flattened.append(part.reshape(new_shape))
        return torch.cat(flattened, -1)

    @classmethod
    def from_point(cls, *parts: 'geoopt.ManifoldTensor', batch_dims=0):
        """
        Construct Product manifold from given points.

        Parameters
        ----------
        parts : tuple[geoopt.ManifoldTensor]
            Manifold tensors to construct Product manifold from
        batch_dims : int
            number of first dims to treat as batch dims and not include in the Product manifold

        Returns
        -------
        ProductManifold
        """
        batch_shape = None
        init = []
        for tens in parts:
            manifold = tens.manifold
            if batch_shape is None:
                batch_shape = tens.shape[:batch_dims]
            elif not batch_shape == tens.shape[:batch_dims]:
                raise ValueError('Not all parts have same batch shape')
            init.append((manifold, tens.shape[batch_dims:]))
        return cls(*init)

    def random_combined(self, *size, dtype=None, device=None) ->'geoopt.ManifoldTensor':
        shape = geoopt.utils.size2shape(*size)
        self._assert_check_shape(shape, 'x')
        batch_shape = shape[:-1]
        points = []
        for manifold, shape in zip(self.manifolds, self.shapes):
            points.append(manifold.random(batch_shape + shape, dtype=dtype, device=device))
        tensor = self.pack_point(*points)
        return geoopt.ManifoldTensor(tensor, manifold=self)
    random = random_combined

    def origin(self, *size, dtype=None, device=None, seed=42) ->'geoopt.ManifoldTensor':
        shape = geoopt.utils.size2shape(*size)
        self._assert_check_shape(shape, 'x')
        batch_shape = shape[:-1]
        points = []
        for manifold, shape in zip(self.manifolds, self.shapes):
            points.append(manifold.origin(batch_shape + shape, dtype=dtype, device=device, seed=seed))
        tensor = self.pack_point(*points)
        return geoopt.ManifoldTensor(tensor, manifold=self)


def rescale_value(value, scaling, power):
    return value * scaling ** power if power != 0 else value


def rescale(function, scaling_info):
    if scaling_info is ScalingInfo.NotCompatible:

        @functools.wraps(functools)
        def stub(self, *args, **kwargs):
            raise NotImplementedError("Scaled version of '{}' is not available".format(function.__name__))
        return stub
    signature = inspect.signature(function)

    @functools.wraps(function)
    def rescaled_function(self, *args, **kwargs):
        params = signature.bind(self.base, *args, **kwargs)
        params.apply_defaults()
        arguments = params.arguments
        for k, power in scaling_info.kwargs.items():
            arguments[k] = rescale_value(arguments[k], self.scale, power)
        params = params.__class__(signature, arguments)
        results = function(*params.args, **params.kwargs)
        if not scaling_info.results:
            return results
        wrapped_results = []
        is_tuple = isinstance(results, tuple)
        results = geoopt.utils.make_tuple(results)
        for i, (res, power) in enumerate(itertools.zip_longest(results, scaling_info.results, fillvalue=0)):
            wrapped_results.append(rescale_value(res, self.scale, power))
        if not is_tuple:
            wrapped_results = wrapped_results[0]
        else:
            wrapped_results = results.__class__(wrapped_results)
        return wrapped_results
    return rescaled_function


class Scaled(Manifold):
    """
    Scaled manifold.

    Scales all the distances on tha manifold by a constant factor. Scaling may be learnable
    since the underlying representation is canonical.

    Examples
    --------
    Here is a simple example of radius 2 Sphere

    >>> import geoopt, torch, numpy as np
    >>> sphere = geoopt.Sphere()
    >>> radius_2_sphere = Scaled(sphere, 2)
    >>> p1 = torch.tensor([-1., 0.])
    >>> p2 = torch.tensor([0., 1.])
    >>> np.testing.assert_allclose(sphere.dist(p1, p2), np.pi / 2)
    >>> np.testing.assert_allclose(radius_2_sphere.dist(p1, p2), np.pi)
    """

    def __init__(self, manifold: Manifold, scale=1.0, learnable=False):
        super().__init__()
        self.base = manifold
        scale = torch.as_tensor(scale, dtype=torch.get_default_dtype())
        scale = scale.requires_grad_(False)
        if not learnable:
            self.register_buffer('_scale', scale)
            self.register_buffer('_log_scale', None)
        else:
            self.register_buffer('_scale', None)
            self.register_parameter('_log_scale', torch.nn.Parameter(scale.log()))
        for method, scaling_info in self.base.__scaling__.items():
            unbound_method = getattr(self.base, method).__func__
            self.__setattr__(method, types.MethodType(rescale(unbound_method, scaling_info), self))

    @property
    def scale(self) ->torch.Tensor:
        if self._scale is None:
            return self._log_scale.exp()
        else:
            return self._scale

    @property
    def log_scale(self) ->torch.Tensor:
        if self._log_scale is None:
            return self._scale.log()
        else:
            return self._log_scale
    reversible = property(lambda self: self.base.reversible)
    ndim = property(lambda self: self.base.ndim)
    name = 'Scaled'
    __scaling__ = property(lambda self: self.base.__scaling__)
    retr = NotImplemented
    expmap = NotImplemented

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as original:
            try:
                if isinstance(self.base, Scaled) and item in self._base_attributes:
                    return self.base.__getattr__(item)
                else:
                    return self.base.__getattribute__(item)
            except AttributeError as e:
                raise original from e

    @property
    def _base_attributes(self):
        if isinstance(self.base, Scaled):
            return self.base._base_attributes
        else:
            base_attributes = set(dir(self.base.__class__))
            base_attributes |= set(self.base.__dict__.keys())
            return base_attributes

    def __dir__(self):
        return list(set(super().__dir__()) | self._base_attributes)

    def __repr__(self):
        extra = self.base.extra_repr()
        if extra:
            return self.name + '({})({}) manifold'.format(self.base.name, extra)
        else:
            return self.name + '({}) manifold'.format(self.base.name)

    def _check_shape(self, shape: Tuple[int], name: str) ->Tuple[bool, Optional[str]]:
        return self.base._check_shape(shape, name)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        return self.base._check_point_on_manifold(x, atol=atol, rtol=rtol)

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        return self.base._check_vector_on_tangent(x, u, atol=atol, rtol=rtol)

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor=None, *, keepdim=False, **kwargs) ->torch.Tensor:
        return self.base.inner(x, u, v, keepdim=keepdim, **kwargs)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, **kwargs) ->torch.Tensor:
        return self.base.norm(x, u, keepdim=keepdim, **kwargs)

    def proju(self, x: torch.Tensor, u: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.base.proju(x, u, **kwargs)

    def projx(self, x: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.base.projx(x, **kwargs)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.base.egrad2rgrad(x, u, **kwargs)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, **kwargs) ->torch.Tensor:
        return self.base.transp(x, y, v, **kwargs)

    def random(self, *size, dtype=None, device=None, **kwargs) ->torch.Tensor:
        return self.base.random(*size, dtype=dtype, device=device, **kwargs)


EPS = {torch.float32: 0.0001, torch.float64: 1e-07}


_sphere_doc = """
    Sphere manifold induced by the following constraint

    .. math::

        \\|x\\|=1\\\\
        x \\in \\mathbb{span}(U)

    where :math:`U` can be parametrized with compliment space or intersection.

    Parameters
    ----------
    intersection : tensor
        shape ``(..., dim, K)``, subspace to intersect with
    complement : tensor
        shape ``(..., dim, K)``, subspace to compliment
"""


class Sphere(Manifold):
    __doc__ = """{}

    See Also
    --------
    :class:`SphereExact`
    """.format(_sphere_doc)
    ndim = 1
    name = 'Sphere'
    reversible = False

    def __init__(self, intersection: torch.Tensor=None, complement: torch.Tensor=None):
        super().__init__()
        if intersection is not None and complement is not None:
            raise TypeError("Can't initialize with both intersection and compliment arguments, please specify only one")
        elif intersection is not None:
            self._configure_manifold_intersection(intersection)
        elif complement is not None:
            self._configure_manifold_complement(complement)
        else:
            self._configure_manifold_no_constraints()
        if self.projector is not None and (geoopt.linalg.batch_linalg.matrix_rank(self.projector) == 1).any():
            raise ValueError('Manifold only consists of isolated points when subspace is 1-dimensional.')

    def _check_shape(self, shape: Tuple[int], name: str) ->Union[Tuple[bool, Optional[str]], bool]:
        ok, reason = super()._check_shape(shape, name)
        if ok and self.projector is not None:
            ok = len(shape) < self.projector.dim() - 1
            if not ok:
                reason = '`{}` should have at least {} dimensions but has {}'.format(name, self.projector.dim() - 1, len(shape))
            ok = shape[-1] == self.projector.shape[-2]
            if not ok:
                reason = 'The [-2] shape of `span` does not match `{}`: {}, {}'.format(name, shape[-1], self.projector.shape[-1])
        elif ok:
            ok = shape[-1] != 1
            if not ok:
                reason = 'Manifold only consists of isolated points when subspace is 1-dimensional.'
        return ok, reason

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Tuple[bool, Optional[str]]:
        norm = x.norm(dim=-1)
        ok = torch.allclose(norm, norm.new((1,)).fill_(1), atol=atol, rtol=rtol)
        if not ok:
            return False, '`norm(x) != 1` with atol={}, rtol={}'.format(atol, rtol)
        ok = torch.allclose(self._project_on_subspace(x), x, atol=atol, rtol=rtol)
        if not ok:
            return False, '`x` is not in the subspace of the manifold with atol={}, rtol={}'.format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Tuple[bool, Optional[str]]:
        inner = self.inner(x, x, u, keepdim=True)
        ok = torch.allclose(inner, inner.new_zeros((1,)), atol=atol, rtol=rtol)
        if not ok:
            return False, '`<x, u> != 0` with atol={}, rtol={}'.format(atol, rtol)
        return True, None

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor=None, *, keepdim=False) ->torch.Tensor:
        if v is None:
            v = u
        inner = (u * v).sum(-1, keepdim=keepdim)
        target_shape = broadcast_shapes(x.shape[:-1] + (1,) * keepdim, inner.shape)
        return inner.expand(target_shape)

    def projx(self, x: torch.Tensor) ->torch.Tensor:
        x = self._project_on_subspace(x)
        return x / x.norm(dim=-1, keepdim=True)

    def proju(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        u = u - (x * u).sum(dim=-1, keepdim=True) * x
        return self._project_on_subspace(u)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        norm_u = u.norm(dim=-1, keepdim=True)
        exp = x * torch.cos(norm_u) + u * torch.sin(norm_u) / norm_u
        retr = self.projx(x + u)
        cond = norm_u > EPS[norm_u.dtype]
        return torch.where(cond, exp, retr)

    def retr(self, x: torch.Tensor, u: torch.Tensor) ->torch.Tensor:
        return self.projx(x + u)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) ->torch.Tensor:
        return self.proju(y, v)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) ->torch.Tensor:
        u = self.proju(x, y - x)
        dist = self.dist(x, y, keepdim=True)
        cond = dist.gt(EPS[x.dtype])
        result = torch.where(cond, u * dist / u.norm(dim=-1, keepdim=True).clamp_min(EPS[x.dtype]), u)
        return result

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) ->torch.Tensor:
        inner = self.inner(x, x, y, keepdim=keepdim).clamp(-1 + EPS[x.dtype], 1 - EPS[x.dtype])
        return torch.acos(inner)
    egrad2rgrad = proju

    def _configure_manifold_complement(self, complement: torch.Tensor):
        Q, _ = geoopt.linalg.batch_linalg.qr(complement)
        P = -Q @ Q.transpose(-1, -2)
        P[..., torch.arange(P.shape[-2]), torch.arange(P.shape[-2])] += 1
        self.register_buffer('projector', P)

    def _configure_manifold_intersection(self, intersection: torch.Tensor):
        Q, _ = geoopt.linalg.batch_linalg.qr(intersection)
        self.register_buffer('projector', Q @ Q.transpose(-1, -2))

    def _configure_manifold_no_constraints(self):
        self.register_buffer('projector', None)

    def _project_on_subspace(self, x: torch.Tensor) ->torch.Tensor:
        if self.projector is not None:
            return x @ self.projector.transpose(-1, -2)
        else:
            return x

    def random_uniform(self, *size, dtype=None, device=None) ->torch.Tensor:
        """
        Uniform random measure on Sphere manifold.

        Parameters
        ----------
        size : shape
            the desired output shape
        dtype : torch.dtype
            desired dtype
        device : torch.device
            desired device

        Returns
        -------
        ManifoldTensor
            random point on Sphere manifold

        Notes
        -----
        In case of projector on the manifold, dtype and device are set automatically and shouldn't be provided.
        If you provide them, they are checked to match the projector device and dtype
        """
        self._assert_check_shape(size2shape(*size), 'x')
        if self.projector is None:
            tens = torch.randn(*size, device=device, dtype=dtype)
        else:
            if device is not None and device != self.projector.device:
                raise ValueError('`device` does not match the projector `device`, set the `device` argument to None')
            if dtype is not None and dtype != self.projector.dtype:
                raise ValueError('`dtype` does not match the projector `dtype`, set the `dtype` arguement to None')
            tens = torch.randn(*size, device=self.projector.device, dtype=self.projector.dtype)
        return ManifoldTensor(self.projx(tens), manifold=self)
    random = random_uniform


class SphereExact(Sphere):
    __doc__ = """{}

    See Also
    --------
    :class:`Sphere`

    Notes
    -----
    The implementation of retraction is an exact exponential map, this retraction will be used in optimization
    """.format(_sphere_doc)
    retr_transp = Sphere.expmap_transp
    transp_follow_retr = Sphere.transp_follow_expmap
    retr = Sphere.expmap

    def extra_repr(self):
        return 'exact'


_references = """References
    ----------
    The functions for the mathematics in gyrovector spaces are taken from the
    following resources:

    [1] Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic 
           neural networks." Advances in neural information processing systems. 
           2018.
    [2] Bachmann, Gregor, Gary Bécigneul, and Octavian-Eugen Ganea. "Constant
           Curvature Graph Convolutional Networks." arXiv preprint 
           arXiv:1911.05076 (2019).
    [3] Skopek, Ondrej, Octavian-Eugen Ganea, and Gary Bécigneul. 
           "Mixed-curvature Variational Autoencoders." arXiv preprint 
           arXiv:1911.08411 (2019).
    [4] Ungar, Abraham A. Analytic hyperbolic geometry: Mathematical 
           foundations and applications. World Scientific, 2005.
    [5] Albert, Ungar Abraham. Barycentric calculus in Euclidean and 
           hyperbolic geometry: A comparative introduction. World Scientific, 
           2010.
"""


_stereographic_doc = """
    :math:`\\kappa`-Stereographic model.

    Parameters
    ----------
    k : float|tensor 
        sectional curvature :math:`\\kappa` of the manifold
        - k<0: Poincaré ball (stereographic projection of hyperboloid)
        - k>0: Stereographic projection of sphere
        - k=0: Euclidean geometry

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision.

    Documentation & Illustration
    ---------------------------- 
    http://andbloch.github.io/K-Stereographic-Model/ or :doc:`/extended/stereographic`
"""


class Stereographic(Manifold):
    __doc__ = """{}

    {}

    See Also
    --------
    :class:`StereographicExact`
    :class:`PoincareBall`
    :class:`PoincareBallExact`
    :class:`SphereProjection`
    :class:`SphereProjectionExact`
    """.format(_stereographic_doc, _references)
    ndim = 1
    reversible = False
    name = property(lambda self: self.__class__.__name__)
    __scaling__ = Manifold.__scaling__.copy()

    @property
    def radius(self):
        return self.k.abs().sqrt().reciprocal()

    def __init__(self, k=0.0, learnable=False):
        super().__init__()
        k = torch.as_tensor(k)
        if not torch.is_floating_point(k):
            k = k
        self.k = torch.nn.Parameter(k, requires_grad=learnable)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-05, rtol=1e-05, dim=-1) ->Tuple[bool, Optional[str]]:
        px = math.project(x, k=self.k, dim=dim)
        ok = torch.allclose(x, px, atol=atol, rtol=rtol)
        if not ok:
            reason = "'x' norm lies out of the bounds [-1/sqrt(c)+eps, 1/sqrt(c)-eps]"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-05, rtol=1e-05, dim=-1) ->Tuple[bool, Optional[str]]:
        return True, None

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1) ->torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim)

    def dist2(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1) ->torch.Tensor:
        return math.dist(x, y, k=self.k, keepdim=keepdim, dim=dim) ** 2

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.egrad2rgrad(x, u, k=self.k, dim=dim)

    def retr(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) ->torch.Tensor:
        approx = x + u
        return math.project(approx, k=self.k, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) ->torch.Tensor:
        target_shape = broadcast_shapes(x.shape, u.shape)
        return u.expand(target_shape)

    def inner(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor=None, *, keepdim=False, dim=-1) ->torch.Tensor:
        if v is None:
            v = u
        return math.inner(x, u, v, k=self.k, keepdim=keepdim, dim=dim)

    def norm(self, x: torch.Tensor, u: torch.Tensor, *, keepdim=False, dim=-1) ->torch.Tensor:
        return math.norm(x, u, k=self.k, keepdim=keepdim, dim=dim)

    def expmap(self, x: torch.Tensor, u: torch.Tensor, *, project=True, dim=-1) ->torch.Tensor:
        res = math.expmap(x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.logmap(x, y, k=self.k, dim=dim)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1):
        return math.parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp_follow_retr(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1) ->torch.Tensor:
        y = self.retr(x, u, dim=dim)
        return self.transp(x, y, v, dim=dim)

    def transp_follow_expmap(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        y = self.expmap(x, u, dim=dim, project=project)
        return self.transp(x, y, v, dim=dim)

    def expmap_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, project=True) ->Tuple[torch.Tensor, torch.Tensor]:
        y = self.expmap(x, u, dim=dim, project=project)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def retr_transp(self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1) ->Tuple[torch.Tensor, torch.Tensor]:
        y = self.retr(x, u, dim=dim)
        v_transp = self.transp(x, y, v, dim=dim)
        return y, v_transp

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.mobius_add(x, y, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_sub(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.mobius_sub(x, y, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_coadd(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.mobius_coadd(x, y, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_cosub(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.mobius_cosub(x, y, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_scalar_mul(self, r: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.mobius_scalar_mul(r, x, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_pointwise_mul(self, w: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.mobius_pointwise_mul(w, x, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def mobius_matvec(self, m: torch.Tensor, x: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.mobius_matvec(m, x, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def geodesic(self, t: torch.Tensor, x: torch.Tensor, y: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.geodesic(t, x, y, k=self.k, dim=dim)

    @__scaling__(ScalingInfo(t=-1))
    def geodesic_unit(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.geodesic_unit(t, x, u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    def lambda_x(self, x: torch.Tensor, *, dim=-1, keepdim=False) ->torch.Tensor:
        return math.lambda_x(x, k=self.k, dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(1))
    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) ->torch.Tensor:
        return math.dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    @__scaling__(ScalingInfo(u=-1))
    def expmap0(self, u: torch.Tensor, *, dim=-1, project=True) ->torch.Tensor:
        res = math.expmap0(u, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(1))
    def logmap0(self, x: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.logmap0(x, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.parallel_transport0back(y, u, k=self.k, dim=dim)

    def gyration(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.gyration(x, y, z, k=self.k, dim=dim)

    def antipode(self, x: torch.Tensor, *, dim=-1) ->torch.Tensor:
        return math.antipode(x, k=self.k, dim=dim)

    @__scaling__(ScalingInfo(1))
    def dist2plane(self, x: torch.Tensor, p: torch.Tensor, a: torch.Tensor, *, dim=-1, keepdim=False, signed=False, scaled=False) ->torch.Tensor:
        return math.dist2plane(x, p, a, dim=dim, k=self.k, keepdim=keepdim, signed=signed, scaled=scaled)

    @__scaling__(ScalingInfo.NotCompatible)
    def mobius_fn_apply(self, fn: callable, x: torch.Tensor, *args, dim=-1, project=True, **kwargs) ->torch.Tensor:
        res = math.mobius_fn_apply(fn, x, *args, k=self.k, dim=dim, **kwargs)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo.NotCompatible)
    def mobius_fn_apply_chain(self, x: torch.Tensor, *fns: callable, project=True, dim=-1) ->torch.Tensor:
        res = math.mobius_fn_apply_chain(x, *fns, k=self.k, dim=dim)
        if project:
            return math.project(res, k=self.k, dim=dim)
        else:
            return res

    @__scaling__(ScalingInfo(std=-1), 'random')
    def random_normal(self, *size, mean=0, std=1, dtype=None, device=None) ->'geoopt.ManifoldTensor':
        """
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random point on the PoincareBall manifold

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        size = size2shape(*size)
        self._assert_check_shape(size, 'x')
        if device is not None and device != self.k.device:
            raise ValueError('`device` does not match the manifold `device`, set the `device` argument to None')
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError('`dtype` does not match the manifold `dtype`, set the `dtype` argument to None')
        tens = torch.randn(size, device=self.k.device, dtype=self.k.dtype) * std / size[-1] ** 0.5 + mean
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)
    random = random_normal

    def origin(self, *size, dtype=None, device=None, seed=42) ->'geoopt.ManifoldTensor':
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
            random point on the manifold
        """
        return geoopt.ManifoldTensor(torch.zeros(*size, dtype=dtype, device=device), manifold=self)

    def weighted_midpoint(self, xs: torch.Tensor, weights: Optional[torch.Tensor]=None, *, reducedim: Optional[List[int]]=None, dim: int=-1, keepdim: bool=False, lincomb: bool=False, posweight=False, project=True):
        mid = math.weighted_midpoint(xs=xs, weights=weights, k=self.k, reducedim=reducedim, dim=dim, keepdim=keepdim, lincomb=lincomb, posweight=posweight)
        if project:
            return math.project(mid, k=self.k, dim=dim)
        else:
            return mid

    def sproj(self, x: torch.Tensor, *, dim: int=-1):
        return math.sproj(x, k=self.k, dim=dim)

    def inv_sproj(self, x: torch.Tensor, *, dim: int=-1):
        return math.inv_sproj(x, k=self.k, dim=dim)


class StereographicExact(Stereographic):
    __doc__ = """{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization.

    See Also
    --------
    :class:`Stereographic`
    :class:`PoincareBall`
    :class:`PoincareBallExact`
    :class:`SphereProjection`
    :class:`SphereProjectionExact`
    """.format(_stereographic_doc)
    reversible = True
    retr_transp = Stereographic.expmap_transp
    transp_follow_retr = Stereographic.transp_follow_expmap
    retr = Stereographic.expmap

    def extra_repr(self):
        return 'exact'


_poincare_ball_doc = """
    Poincare ball model.

    See more in :doc:`/extended/stereographic`

    Parameters
    ----------
    c : float|tensor
        ball's negative curvature. The parametrization is constrained to have positive c

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
"""


class PoincareBall(Stereographic):
    __doc__ = """{}

    See Also
    --------
    :class:`Stereographic`
    :class:`StereographicExact`
    :class:`PoincareBallExact`
    :class:`SphereProjection`
    :class:`SphereProjectionExact`
    """.format(_poincare_ball_doc)

    @property
    def k(self):
        return -self.c

    @property
    def c(self):
        return torch.nn.functional.softplus(self.isp_c)

    def __init__(self, c=1.0, learnable=False):
        super().__init__(k=c, learnable=learnable)
        k = self._parameters.pop('k')
        with torch.no_grad():
            self.isp_c = k.exp_().sub_(1).log_()


class PoincareBallExact(PoincareBall, StereographicExact):
    __doc__ = """{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization.

    See Also
    --------
    :class:`Stereographic`
    :class:`StereographicExact`
    :class:`PoincareBall`
    :class:`SphereProjection`
    :class:`SphereProjectionExact`
    """.format(_poincare_ball_doc)


_sphere_projection_doc = """
    Stereographic Projection Spherical model.

    See more in :doc:`/extended/stereographic`

    Parameters
    ----------
    k : float|tensor
        sphere's positive curvature. The parametrization is constrained to have positive k

    Notes
    -----
    It is extremely recommended to work with this manifold in double precision
"""


class SphereProjection(Stereographic):
    __doc__ = """{}

    See Also
    --------
    :class:`Stereographic`
    :class:`StereographicExact`
    :class:`PoincareBall`
    :class:`PoincareBallExact`
    :class:`SphereProjectionExact`
    :class:`Sphere`
    """.format(_sphere_projection_doc)

    @property
    def k(self):
        return torch.nn.functional.softplus(self.isp_k)

    def __init__(self, k=1.0, learnable=False):
        super().__init__(k=k, learnable=learnable)
        k = self._parameters.pop('k')
        with torch.no_grad():
            self.isp_k = k.exp_().sub_(1).log_()


class SphereProjectionExact(SphereProjection, StereographicExact):
    __doc__ = """{}

    The implementation of retraction is an exact exponential map, this retraction will be used in optimization.

    See Also
    --------
    :class:`Stereographic`
    :class:`StereographicExact`
    :class:`PoincareBall`
    :class:`PoincareBallExact`
    :class:`SphereProjectionExact`
    :class:`Sphere`
    """.format(_sphere_projection_doc)


_stiefel_doc = """
    Manifold induced by the following matrix constraint:

    .. math::

        X^\\top X = I\\\\
        X \\in \\mathrm{R}^{n\\times m}\\\\
        n \\ge m
"""


class Stiefel(Manifold):
    __doc__ = """
    {}

    Parameters
    ----------
    canonical : bool
        Use canonical inner product instead of euclidean one (defaults to canonical)

    See Also
    --------
    :class:`CanonicalStiefel`, :class:`EuclideanStiefel`, :class:`EuclideanStiefelExact`
    """.format(_stiefel_doc)
    ndim = 2

    def __new__(cls, canonical=True):
        if cls is Stiefel:
            if canonical:
                return super().__new__(CanonicalStiefel)
            else:
                return super().__new__(EuclideanStiefel)
        else:
            return super().__new__(cls)

    def _check_shape(self, shape: Tuple[int], name: str) ->Union[Tuple[bool, Optional[str]], bool]:
        ok, reason = super()._check_shape(shape, name)
        if not ok:
            return False, reason
        shape_is_ok = shape[-1] <= shape[-2]
        if not shape_is_ok:
            return False, '`{}` should have shape[-1] <= shape[-2], got {} </= {}'.format(name, shape[-1], shape[-2])
        return True, None

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        xtx = x.transpose(-1, -2) @ x
        xtx[..., torch.arange(x.shape[-1]), torch.arange(x.shape[-1])] -= 1
        ok = torch.allclose(xtx, xtx.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, '`X^T X != I` with atol={}, rtol={}'.format(atol, rtol)
        return True, None

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-05, rtol=1e-05) ->Union[Tuple[bool, Optional[str]], bool]:
        diff = u.transpose(-1, -2) @ x + x.transpose(-1, -2) @ u
        ok = torch.allclose(diff, diff.new((1,)).fill_(0), atol=atol, rtol=rtol)
        if not ok:
            return False, '`u^T x + x^T u !=0` with atol={}, rtol={}'.format(atol, rtol)
        return True, None

    def projx(self, x: torch.Tensor) ->torch.Tensor:
        U, _, V = linalg.batch_linalg.svd(x)
        return torch.einsum('...ik,...jk->...ij', U, V)

    def random_naive(self, *size, dtype=None, device=None) ->torch.Tensor:
        """
        Naive approach to get random matrix on Stiefel manifold.

        A helper function to sample a random point on the Stiefel manifold.
        The measure is non-uniform for this method, but fast to compute.

        Parameters
        ----------
        size : shape
            the desired output shape
        dtype : torch.dtype
            desired dtype
        device : torch.device
            desired device

        Returns
        -------
        ManifoldTensor
            random point on Stiefel manifold
        """
        self._assert_check_shape(size2shape(*size), 'x')
        tens = torch.randn(*size, device=device, dtype=dtype)
        return ManifoldTensor(linalg.qr(tens)[0], manifold=self)
    random = random_naive

    def origin(self, *size, dtype=None, device=None, seed=42) ->torch.Tensor:
        """
        Identity matrix point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
        """
        self._assert_check_shape(size2shape(*size), 'x')
        eye = torch.zeros(*size, dtype=dtype, device=device)
        eye[..., torch.arange(eye.shape[-1]), torch.arange(eye.shape[-1])] += 1
        return ManifoldTensor(eye, manifold=self)

