import sys
_module = sys.modules[__name__]
del sys
conf = _module
discrete_hmm = _module
eeg_slds = _module
kalman_filter = _module
minipyro = _module
mixed_hmm = _module
experiment = _module
model = _module
seal_data = _module
pcfg = _module
sensor = _module
slds = _module
vae = _module
funsor = _module
adjoint = _module
affine = _module
cnf = _module
compat = _module
ops = _module
delta = _module
distribution = _module
domains = _module
einsum = _module
numpy_log = _module
numpy_map = _module
util = _module
gaussian = _module
integrate = _module
interpreter = _module
jax = _module
distributions = _module
ops = _module
joint = _module
memoize = _module
minipyro = _module
montecarlo = _module
ops = _module
optimizer = _module
pyro = _module
convert = _module
distribution = _module
hmm = _module
registry = _module
sum_product = _module
tensor = _module
terms = _module
testing = _module
distributions = _module
ops = _module
util = _module
update_headers = _module
setup = _module
test = _module
conftest = _module
test_bart = _module
test_sensor_fusion = _module
test_convert = _module
test_distribution = _module
test_hmm = _module
test_pyroapi = _module
test_adjoint = _module
test_affine = _module
test_alpha_conversion = _module
test_cnf = _module
test_delta = _module
test_distribution = _module
test_einsum = _module
test_gaussian = _module
test_import = _module
test_integrate = _module
test_joint = _module
test_memoize = _module
test_minipyro = _module
test_optimizer = _module
test_samplers = _module
test_sum_product = _module
test_tensor = _module
test_terms = _module

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


from collections import OrderedDict


import torch


import time


import numpy as np


import torch.nn as nn


from torch.distributions import constraints


import functools


import uuid


import math


import itertools


from torch.optim import Adam


import torch.utils.data


from torch import nn


from torch import optim


from torch.nn import functional as F


from torchvision import datasets


from torchvision import transforms


from collections import defaultdict


from functools import reduce


from typing import Tuple


from typing import Union


from torch import ones


from torch import randn


from torch import tensor


from torch import zeros


import inspect


import typing


import numpy as onp


import warnings


from collections import namedtuple


from numbers import Number


import numbers


from collections import Hashable


from functools import singledispatch


import re


def get_tracing_state():
    if _FUNSOR_BACKEND == 'torch':
        import torch
        return torch._C._get_tracing_state()
    else:
        return None


class lazy_property(object):

    def __init__(self, fn):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.fn(obj)
        setattr(obj, self.fn.__name__, value)
        return value


class Domain(namedtuple('Domain', ['shape', 'dtype'])):
    """
    An object representing the type and shape of a :class:`Funsor` input or
    output.
    """

    def __new__(cls, shape, dtype):
        assert isinstance(shape, tuple)
        if get_tracing_state():
            shape = tuple(map(int, shape))
        assert all(isinstance(size, int) for size in shape), shape
        if isinstance(dtype, int):
            assert not shape
        elif isinstance(dtype, str):
            assert dtype == 'real'
        else:
            raise ValueError(repr(dtype))
        return super(Domain, cls).__new__(cls, shape, dtype)

    def __repr__(self):
        shape = tuple(self.shape)
        if isinstance(self.dtype, int):
            if not shape:
                return 'bint({})'.format(self.dtype)
            return 'bint({}, {})'.format(self.dtype, shape)
        if not shape:
            return 'reals()'
        return 'reals{}'.format(shape)

    def __iter__(self):
        if isinstance(self.dtype, int) and not self.shape:
            return (Number(i, self.dtype) for i in range(self.dtype))
        raise NotImplementedError

    @lazy_property
    def num_elements(self):
        return reduce(operator.mul, self.shape, 1)

    @property
    def size(self):
        assert isinstance(self.dtype, int)
        return self.dtype


def broadcast_shape(*shapes, **kwargs):
    """
    Similar to ``np.broadcast()`` but for shapes.
    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.
    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop('strict', False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError('shape mismatch: objects cannot be broadcast to a single shape: {}'.format(' vs '.join(map(str, shapes))))
    return tuple(reversed(reversed_shape))


def find_domain(op, *domains):
    """
    Finds the :class:`Domain` resulting when applying ``op`` to ``domains``.
    :param callable op: An operation.
    :param Domain \\*domains: One or more input domains.
    """
    assert callable(op), op
    assert all(isinstance(arg, Domain) for arg in domains)
    if len(domains) == 1:
        dtype = domains[0].dtype
        shape = domains[0].shape
        if op is ops.log or op is ops.exp:
            dtype = 'real'
        elif isinstance(op, ops.ReshapeOp):
            shape = op.shape
        elif isinstance(op, ops.AssociativeOp):
            shape = ()
        return Domain(shape, dtype)
    lhs, rhs = domains
    if isinstance(op, ops.GetitemOp):
        dtype = lhs.dtype
        shape = lhs.shape[:op.offset] + lhs.shape[1 + op.offset:]
        return Domain(shape, dtype)
    elif op == ops.matmul:
        assert lhs.shape and rhs.shape
        if len(rhs.shape) == 1:
            assert lhs.shape[-1] == rhs.shape[-1]
            shape = lhs.shape[:-1]
        elif len(lhs.shape) == 1:
            assert lhs.shape[-1] == rhs.shape[-2]
            shape = rhs.shape[:-2] + rhs.shape[-1:]
        else:
            assert lhs.shape[-1] == rhs.shape[-2]
            shape = broadcast_shape(lhs.shape[:-1], rhs.shape[:-2] + (1,)) + rhs.shape[-1:]
        return Domain(shape, 'real')
    if lhs.dtype == 'real' or rhs.dtype == 'real':
        dtype = 'real'
    elif op in (ops.add, ops.mul, ops.pow, ops.max, ops.min):
        dtype = op(lhs.dtype - 1, rhs.dtype - 1) + 1
    elif op in (ops.and_, ops.or_, ops.xor):
        dtype = 2
    elif lhs.dtype == rhs.dtype:
        dtype = lhs.dtype
    else:
        raise NotImplementedError('TODO')
    if lhs.shape == rhs.shape:
        shape = lhs.shape
    else:
        shape = broadcast_shape(lhs.shape, rhs.shape)
    return Domain(shape, dtype)


def _issubclass_tuple(subcls, cls):
    """
    utility for pattern matching with tuple subexpressions
    """
    cls_is_union = hasattr(cls, '__origin__') and (cls.__origin__ or cls) is typing.Union
    if isinstance(cls, tuple) or cls_is_union:
        return any(_issubclass_tuple(subcls, option) for option in (getattr(cls, '__args__', []) if cls_is_union else cls))
    subcls_is_union = hasattr(subcls, '__origin__') and (subcls.__origin__ or subcls) is typing.Union
    if isinstance(subcls, tuple) or subcls_is_union:
        return any(_issubclass_tuple(option, cls) for option in (getattr(subcls, '__args__', []) if subcls_is_union else subcls))
    subcls_is_tuple = hasattr(subcls, '__origin__') and (subcls.__origin__ or subcls) in (tuple, typing.Tuple)
    cls_is_tuple = hasattr(cls, '__origin__') and (cls.__origin__ or cls) in (tuple, typing.Tuple)
    if subcls_is_tuple != cls_is_tuple:
        return False
    if not cls_is_tuple:
        return issubclass(subcls, cls)
    if not cls.__args__:
        return True
    if not subcls.__args__ or len(subcls.__args__) != len(cls.__args__):
        return False
    return all(_issubclass_tuple(a, b) for a, b in zip(subcls.__args__, cls.__args__))


def getargspec(fn):
    """
    Similar to Python 2's :py:func:`inspect.getargspec` but:
    - In Python 3 uses ``getfullargspec`` to avoid ``DeprecationWarning``.
    - For builtin functions like ``torch.matmul`` or ``numpy.matmul``, falls back to
      attempting to parse the function docstring, assuming torch-style or numpy-style.
    """
    assert callable(fn)
    try:
        args, vargs, kwargs, defaults, _, _, _ = inspect.getfullargspec(fn)
    except TypeError:
        match = re.match('\\s*{}\\(([^)]*)\\)'.format(fn.__name__), fn.__doc__)
        if match is None:
            raise
        parts = re.sub('[[\\]]', '', match.group(1)).split(', ')
        args = [a.split('=')[0] for a in parts if a not in ['/', '*']]
        if not all(re.match('^[^\\d\\W]\\w*\\Z', arg) for arg in args):
            raise
        vargs = None
        kwargs = None
        defaults = ()
    return args, vargs, kwargs, defaults


_INTERPRETATION = None


class Interpreter:

    @property
    def __call__(self):
        return _INTERPRETATION


def _classname(cls):
    return getattr(cls, 'classname', cls.__name__)


_STACK_SIZE = 0


def _indent():
    result = u'    │' * (_STACK_SIZE // 4 + 3)
    return result[:_STACK_SIZE]


def debug_interpret(cls, *args):
    global _STACK_SIZE
    indent = _indent()
    if _DEBUG > 1:
        typenames = [_classname(cls)] + [_classname(type(arg)) for arg in args]
    else:
        typenames = [cls.__name__] + [type(arg).__name__ for arg in args]
    None
    _STACK_SIZE += 1
    try:
        result = _INTERPRETATION(cls, *args)
    finally:
        _STACK_SIZE -= 1
    if _DEBUG > 1:
        result_str = re.sub('\n', '\n          ' + indent, str(result))
    else:
        result_str = type(result).__name__
    None
    return result


class FunsorMeta(type):
    """
    Metaclass for Funsors to perform four independent tasks:

    1.  Fill in default kwargs and convert kwargs to args before deferring to a
        nonstandard interpretation. This allows derived metaclasses to fill in
        defaults and do type conversion, thereby simplifying logic of
        interpretations.
    2.  Ensure each Funsor class has an attribute ``._ast_fields`` describing
        its input args and each Funsor instance has an attribute ``._ast_args``
        with values corresponding to its input args. This allows the instance
        to be reflectively reconstructed under a different interpretation, and
        is used by :func:`funsor.interpreter.reinterpret`.
    3.  Cons-hash construction, so that repeatedly calling the constructor
        with identical args will product the same object. This enables cheap
        syntactic equality testing using the ``is`` operator, which is
        is important both for hashing (e.g. for memoizing funsor functions)
        and for unit testing, since ``.__eq__()`` is overloaded with
        elementwise semantics. Cons hashing differs from memoization in that
        it incurs no memory overhead beyond the cons hash dict.
    4.  Support subtyping with parameters for pattern matching, e.g. Number[int, int].
    """

    def __init__(cls, name, bases, dct):
        super(FunsorMeta, cls).__init__(name, bases, dct)
        if not hasattr(cls, '__args__'):
            cls.__args__ = ()
        if cls.__args__:
            base, = bases
            cls.__origin__ = base
        else:
            cls._ast_fields = getargspec(cls.__init__)[0][1:]
            cls._cons_cache = WeakValueDictionary()
            cls._type_cache = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls.__args__:
            cls = cls.__origin__
        if kwargs:
            args = list(args)
            for name in cls._ast_fields[len(args):]:
                args.append(kwargs.pop(name))
            assert not kwargs, kwargs
            args = tuple(args)
        return interpret(cls, *args)

    def __getitem__(cls, arg_types):
        if not isinstance(arg_types, tuple):
            arg_types = arg_types,
        assert not any(isvariadic(arg_type) for arg_type in arg_types), 'nested variadic types not supported'
        arg_types = tuple(typing.Tuple if arg_type is tuple else arg_type for arg_type in arg_types)
        if arg_types not in cls._type_cache:
            assert not cls.__args__, 'cannot subscript a subscripted type {}'.format(cls)
            assert len(arg_types) == len(cls._ast_fields), 'must provide types for all params'
            new_dct = cls.__dict__.copy()
            new_dct.update({'__args__': arg_types})
            cls._type_cache[arg_types] = type(cls)(cls.__name__, (cls,), new_dct)
        return cls._type_cache[arg_types]

    def __subclasscheck__(cls, subcls):
        if cls is subcls:
            return True
        if not isinstance(subcls, FunsorMeta):
            return super(FunsorMeta, getattr(cls, '__origin__', cls)).__subclasscheck__(subcls)
        cls_origin = getattr(cls, '__origin__', cls)
        subcls_origin = getattr(subcls, '__origin__', subcls)
        if not super(FunsorMeta, cls_origin).__subclasscheck__(subcls_origin):
            return False
        if cls.__args__:
            if not subcls.__args__:
                return False
            if len(cls.__args__) != len(subcls.__args__):
                return False
            for subcls_param, param in zip(subcls.__args__, cls.__args__):
                if not _issubclass_tuple(subcls_param, param):
                    return False
        return True

    @lazy_property
    def classname(cls):
        return cls.__name__ + '[{}]'.format(', '.join(str(getattr(t, 'classname', t)) for t in cls.__args__))


class GetitemMeta(type):
    _cache = {}

    def __call__(cls, offset):
        try:
            return GetitemMeta._cache[offset]
        except KeyError:
            instance = super(GetitemMeta, cls).__call__(offset)
            GetitemMeta._cache[offset] = instance
            return instance


@singledispatch
def to_funsor(x, output=None, dim_to_name=None, **kwargs):
    """
    Convert to a :class:`Funsor` .
    Only :class:`Funsor` s and scalars are accepted.

    :param x: An object.
    :param funsor.domains.Domain output: An optional output hint.
    :param OrderedDict dim_to_name: An optional mapping from negative batch dimensions to name strings.
    :return: A Funsor equivalent to ``x``.
    :rtype: Funsor
    :raises: ValueError
    """
    raise ValueError('Cannot convert to Funsor: {}'.format(repr(x)))


class SubsMeta(FunsorMeta):
    """
    Wrapper to call :func:`to_funsor` and check types.
    """

    def __call__(cls, arg, subs):
        subs = tuple((k, to_funsor(v, arg.inputs[k])) for k, v in subs if k in arg.inputs)
        return super().__call__(arg, subs)


def substitute(expr, subs):
    if isinstance(subs, (dict, OrderedDict)):
        subs = tuple(subs.items())
    assert isinstance(subs, tuple)
    assert all(isinstance(v, Funsor) for k, v in subs)

    @interpreter.interpretation(interpreter._INTERPRETATION)
    def subs_interpreter(cls, *args):
        expr = cls(*args)
        fresh_subs = tuple((k, v) for k, v in subs if k in expr.fresh)
        if fresh_subs:
            expr = interpreter.debug_logged(expr.eager_subs)(fresh_subs)
        return expr
    with interpreter.interpretation(subs_interpreter):
        return interpreter.reinterpret(expr)


def _convert_reduced_vars(reduced_vars):
    """
    Helper to convert the reduced_vars arg of ``.reduce()`` and friends.

    :param reduced_vars:
    :type reduced_vars: str, Variable, or set or frozenset thereof.
    :rtype: frozenset of str
    """
    if isinstance(reduced_vars, frozenset) and all(isinstance(var, str) for var in reduced_vars):
        return reduced_vars
    if isinstance(reduced_vars, (str, Variable)):
        reduced_vars = {reduced_vars}
    assert isinstance(reduced_vars, (frozenset, set))
    assert all(isinstance(var, (str, Variable)) for var in reduced_vars)
    return frozenset(var if isinstance(var, str) else var.name for var in reduced_vars)


def bint(size):
    """
    Construct a bounded integer domain of scalar shape.
    """
    if get_tracing_state():
        size = int(size)
    assert isinstance(size, int) and size >= 0
    return Domain((), size)


def reals(*shape):
    """
    Construct a real domain of given shape.
    """
    return Domain(shape, 'real')


_builtin_max = max


def _logaddexp(x, y):
    if hasattr(x, '__logaddexp__'):
        return x.__logaddexp__(y)
    if hasattr(y, '__rlogaddexp__'):
        return y.__logaddexp__(x)
    shift = max(x, y)
    return log(exp(x - shift) + exp(y - shift)) + shift


_builtin_min = min


def _find_intervals(intervals, end):
    """
    Finds a complete set of intervals partitioning [0, end), given a partial
    set of non-overlapping intervals.
    """
    cuts = list(sorted({0, end}.union(*intervals)))
    return list(zip(cuts[:-1], cuts[1:]))


def _parse_slices(index, value):
    if not isinstance(index, tuple):
        index = index,
    if index[0] is Ellipsis:
        index = index[1:]
    start_stops = []
    for pos, i in reversed(list(enumerate(index))):
        if isinstance(i, slice):
            start_stops.append((i.start, i.stop))
        elif isinstance(i, int):
            start_stops.append((i, i + 1))
            value = ops.unsqueeze(value, pos - len(index))
        else:
            raise ValueError('invalid index: {}'.format(i))
    start_stops.reverse()
    return start_stops, value


class BlockMatrix(object):
    """
    Jit-compatible helper to build blockwise matrices.
    Syntax is similar to :func:`torch.zeros` ::

        x = BlockMatrix((100, 20, 20))
        x[..., 0:4, 0:4] = x11
        x[..., 0:4, 6:10] = x12
        x[..., 6:10, 0:4] = x12.transpose(-1, -2)
        x[..., 6:10, 6:10] = x22
        x = x.as_tensor()
        assert x.shape == (100, 20, 20)
    """

    def __init__(self, shape):
        self.shape = shape
        self.parts = defaultdict(dict)

    def __setitem__(self, index, value):
        (i, j), value = _parse_slices(index, value)
        self.parts[i][j] = value

    def as_tensor(self):
        arbitrary_row = next(iter(self.parts.values()))
        prototype = next(iter(arbitrary_row.values()))
        js = set().union(*(part.keys() for part in self.parts.values()))
        rows = _find_intervals(self.parts.keys(), self.shape[-2])
        cols = _find_intervals(js, self.shape[-1])
        for i in rows:
            for j in cols:
                if j not in self.parts[i]:
                    shape = self.shape[:-2] + (i[1] - i[0], j[1] - j[0])
                    self.parts[i][j] = ops.new_zeros(prototype, shape)
        columns = {i: ops.cat(-1, *[v for j, v in sorted(part.items())]) for i, part in self.parts.items()}
        result = ops.cat(-2, *[v for i, v in sorted(columns.items())])
        if not get_tracing_state():
            assert result.shape == self.shape
        return result


class BlockVector(object):
    """
    Jit-compatible helper to build blockwise vectors.
    Syntax is similar to :func:`torch.zeros` ::

        x = BlockVector((100, 20))
        x[..., 0:4] = x1
        x[..., 6:10] = x2
        x = x.as_tensor()
        assert x.shape == (100, 20)
    """

    def __init__(self, shape):
        self.shape = shape
        self.parts = {}

    def __setitem__(self, index, value):
        (i,), value = _parse_slices(index, value)
        self.parts[i] = value

    def as_tensor(self):
        prototype = next(iter(self.parts.values()))
        for i in _find_intervals(self.parts.keys(), self.shape[-1]):
            if i not in self.parts:
                self.parts[i] = ops.new_zeros(prototype, self.shape[:-1] + (i[1] - i[0],))
        parts = [v for k, v in sorted(self.parts.items())]
        result = ops.cat(-1, *parts)
        if not get_tracing_state():
            assert result.shape == self.shape
        return result


class DeltaMeta(FunsorMeta):
    """
    Makes Delta less of a pain to use by supporting Delta(name, point, log_density)
    """

    def __call__(cls, *args):
        if len(args) > 1:
            assert len(args) == 2 or len(args) == 3
            assert isinstance(args[0], str) and isinstance(args[1], Funsor)
            args = args + (Number(0.0),) if len(args) == 2 else args
            args = ((args[0], (to_funsor(args[1]), to_funsor(args[2]))),),
        assert isinstance(args[0], tuple)
        return super().__call__(args[0])


def solve(expr, value):
    """
    Tries to solve for free inputs of an ``expr`` such that ``expr == value``,
    and computes the log-abs-det-Jacobian of the resulting substitution.

    :param Funsor expr: An expression with a free variable.
    :param Funsor value: A target value.
    :return: A tuple ``(name, point, log_abs_det_jacobian)``
    :rtype: tuple
    :raises: ValueError
    """
    assert isinstance(expr, Funsor)
    assert isinstance(value, Funsor)
    result = solve.dispatch(type(expr), *(expr._ast_values + (value,)))
    if result is None:
        raise ValueError('Cannot substitute into a Delta: {}'.format(value))
    return result


class GaussianMeta(FunsorMeta):
    """
    Wrapper to convert between OrderedDict and tuple.
    """

    def __call__(cls, info_vec, precision, inputs):
        if isinstance(inputs, OrderedDict):
            inputs = tuple(inputs.items())
        assert isinstance(inputs, tuple)
        return super(GaussianMeta, cls).__call__(info_vec, precision, inputs)


class SliceMeta(FunsorMeta):
    """
    Wrapper to fill in ``start``, ``stop``, ``step``, ``dtype`` following
    Python conventions.
    """

    def __call__(cls, name, *args, **kwargs):
        start = 0
        step = 1
        dtype = None
        if len(args) == 1:
            stop = args[0]
            dtype = kwargs.pop('dtype', stop)
        elif len(args) == 2:
            start, stop = args
            dtype = kwargs.pop('dtype', stop)
        elif len(args) == 3:
            start, stop, step = args
            dtype = kwargs.pop('dtype', stop)
        elif len(args) == 4:
            start, stop, step, dtype = args
        else:
            raise ValueError
        if step <= 0:
            raise ValueError
        stop = min(dtype, max(start, stop))
        return super().__call__(name, start, stop, step, dtype)


def _compute_offsets(inputs):
    """
    Compute offsets of real inputs into the concatenated Gaussian dims.
    This ignores all int inputs.

    :param OrderedDict inputs: A schema mapping variable name to domain.
    :return: a pair ``(offsets, total)``, where ``offsets`` is an OrderedDict
        mapping input name to integer offset, and ``total`` is the total event
        size.
    :rtype: tuple
    """
    assert isinstance(inputs, OrderedDict)
    offsets = OrderedDict()
    total = 0
    for key, domain in inputs.items():
        if domain.dtype == 'real':
            offsets[key] = total
            total += domain.num_elements
    return offsets, total


def _log_det_tri(x):
    return ops.log(ops.diagonal(x, -1, -2)).sum(-1)


def _mv(mat, vec):
    return ops.matmul(mat, ops.unsqueeze(vec, -1)).squeeze(-1)


def _vv(vec1, vec2):
    """
    Computes the inner product ``< vec1 | vec 2 >``.
    """
    return ops.matmul(ops.unsqueeze(vec1, -2), ops.unsqueeze(vec2, -1)).squeeze(-1).squeeze(-1)


@singledispatch
def _affine_inputs(fn):
    assert isinstance(fn, Funsor)
    return frozenset()


def affine_inputs(fn):
    """
    Returns a [sound sub]set of real inputs of ``fn``
    wrt which ``fn`` is known to be affine.

    :param Funsor fn: A funsor.
    :return: A set of input names wrt which ``fn`` is affine.
    :rtype: frozenset
    """
    result = getattr(fn, '_affine_inputs', None)
    if result is None:
        result = fn._affine_inputs = _affine_inputs(fn)
    return result


def align_tensor(new_inputs, x, expand=False):
    """
    Permute and add dims to a tensor to match desired ``new_inputs``.

    :param OrderedDict new_inputs: A target set of inputs.
    :param funsor.terms.Funsor x: A :class:`Tensor` or
        :class:`~funsor.terms.Number` .
    :param bool expand: If False (default), set result size to 1 for any input
        of ``x`` not in ``new_inputs``; if True expand to ``new_inputs`` size.
    :return: a number or :class:`torch.Tensor` or :class:`np.ndarray` that can be broadcast to other
        tensors with inputs ``new_inputs``.
    :rtype: int or float or torch.Tensor or np.ndarray
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(x, (Number, Tensor))
    assert all(isinstance(d.dtype, int) for d in x.inputs.values())
    data = x.data
    if isinstance(x, Number):
        return data
    old_inputs = x.inputs
    if old_inputs == new_inputs:
        return data
    x_keys = tuple(old_inputs)
    data = ops.permute(data, tuple(x_keys.index(k) for k in new_inputs if k in old_inputs) + tuple(range(len(old_inputs), len(data.shape))))
    data = data.reshape(tuple(old_inputs[k].dtype if k in old_inputs else 1 for k in new_inputs) + x.output.shape)
    if expand:
        data = ops.expand(data, tuple(d.dtype for d in new_inputs.values()) + x.output.shape)
    return data


def align_gaussian(new_inputs, old):
    """
    Align data of a Gaussian distribution to a new ``inputs`` shape.
    """
    assert isinstance(new_inputs, OrderedDict)
    assert isinstance(old, Gaussian)
    info_vec = old.info_vec
    precision = old.precision
    new_ints = OrderedDict((k, d) for k, d in new_inputs.items() if d.dtype != 'real')
    old_ints = OrderedDict((k, d) for k, d in old.inputs.items() if d.dtype != 'real')
    if new_ints != old_ints:
        info_vec = align_tensor(new_ints, Tensor(info_vec, old_ints))
        precision = align_tensor(new_ints, Tensor(precision, old_ints))
    new_offsets, new_dim = _compute_offsets(new_inputs)
    old_offsets, old_dim = _compute_offsets(old.inputs)
    assert info_vec.shape[-1:] == (old_dim,)
    assert precision.shape[-2:] == (old_dim, old_dim)
    if new_offsets != old_offsets:
        old_info_vec = info_vec
        old_precision = precision
        info_vec = BlockVector(old_info_vec.shape[:-1] + (new_dim,))
        precision = BlockMatrix(old_info_vec.shape[:-1] + (new_dim, new_dim))
        for k1, new_offset1 in new_offsets.items():
            if k1 not in old_offsets:
                continue
            offset1 = old_offsets[k1]
            num_elements1 = old.inputs[k1].num_elements
            old_slice1 = slice(offset1, offset1 + num_elements1)
            new_slice1 = slice(new_offset1, new_offset1 + num_elements1)
            info_vec[..., new_slice1] = old_info_vec[..., old_slice1]
            for k2, new_offset2 in new_offsets.items():
                if k2 not in old_offsets:
                    continue
                offset2 = old_offsets[k2]
                num_elements2 = old.inputs[k2].num_elements
                old_slice2 = slice(offset2, offset2 + num_elements2)
                new_slice2 = slice(new_offset2, new_offset2 + num_elements2)
                precision[..., new_slice1, new_slice2] = old_precision[..., old_slice1, old_slice2]
        info_vec = info_vec.as_tensor()
        precision = precision.as_tensor()
    return info_vec, precision


def align_tensors(*args, **kwargs):
    """
    Permute multiple tensors before applying a broadcasted op.

    This is mainly useful for implementing eager funsor operations.

    :param funsor.terms.Funsor \\*args: Multiple :class:`Tensor` s and
        :class:`~funsor.terms.Number` s.
    :param bool expand: Whether to expand input tensors. Defaults to False.
    :return: a pair ``(inputs, tensors)`` where tensors are all
        :class:`torch.Tensor` s or :class:`np.ndarray` s
        that can be broadcast together to a single data
        with given ``inputs``.
    :rtype: tuple
    """
    expand = kwargs.pop('expand', False)
    assert not kwargs
    inputs = OrderedDict()
    for x in args:
        inputs.update(x.inputs)
    tensors = [align_tensor(inputs, x, expand=expand) for x in args]
    return inputs, tensors


def gensym(x=None):
    global _GENSYM_COUNTER
    _GENSYM_COUNTER += 1
    sym = _GENSYM_COUNTER
    if x is not None:
        if isinstance(x, str):
            return x + '_' + str(sym)
        return id(x)
    return 'V' + str(sym)


def get_backend():
    """
    Get the current backend of Funsor.

    :return: either "numpy", "torch", or "jax".
    :rtype: str
    """
    return _FUNSOR_BACKEND


def get_default_prototype():
    backend = get_backend()
    if backend == 'torch':
        import torch
        return torch.tensor([])
    else:
        return np.array([])


def extract_affine(fn):
    """
    Extracts an affine representation of a funsor, satisfying::

        x = ...
        const, coeffs = extract_affine(x)
        y = sum(Einsum(eqn, (coeff, Variable(var, coeff.output)))
                for var, (coeff, eqn) in coeffs.items())
        assert_close(y, x)
        assert frozenset(coeffs) == affine_inputs(x)

    The ``coeffs`` will have one key per input wrt which ``fn`` is known to be
    affine (via :func:`affine_inputs` ), and ``const`` and ``coeffs.values``
    will all be constant wrt these inputs.

    The affine approximation is computed by ev evaluating ``fn`` at
    zero and each basis vector. To improve performance, users may want to run
    under the :func:`~funsor.memoize.memoize` interpretation.

    :param Funsor fn: A funsor that is affine wrt the (add,mul) semiring in
        some subset of its inputs.
    :return: A pair ``(const, coeffs)`` where const is a funsor with no real
        inputs and ``coeffs`` is an OrderedDict mapping input name to a
        ``(coefficient, eqn)`` pair in einsum form.
    :rtype: tuple
    """
    prototype = get_default_prototype()
    inputs = affine_inputs(fn)
    inputs = OrderedDict((k, v) for k, v in fn.inputs.items() if k in inputs)
    zeros = {k: Tensor(ops.new_zeros(prototype, v.shape)) for k, v in inputs.items()}
    const = fn(**zeros)
    name = gensym('probe')
    coeffs = OrderedDict()
    for k, v in inputs.items():
        dim = v.num_elements
        var = Variable(name, bint(dim))
        subs = zeros.copy()
        subs[k] = Tensor(ops.new_eye(prototype, (dim,)).reshape((dim,) + v.shape))[var]
        coeff = Lambda(var, fn(**subs) - const).reshape(v.shape + const.shape)
        inputs1 = ''.join(map(opt_einsum.get_symbol, range(len(coeff.shape))))
        inputs2 = inputs1[:len(v.shape)]
        output = inputs1[len(v.shape):]
        eqn = f'{inputs1},{inputs2}->{output}'
        coeffs[k] = coeff, eqn
    return const, coeffs


def _real_inputs(fn):
    return frozenset(k for k, d in fn.inputs.items() if d.dtype == 'real')


def is_affine(fn):
    """
    A sound but incomplete test to determine whether a funsor is affine with
    respect to all of its real inputs.

    :param Funsor fn: A funsor.
    :rtype: bool
    """
    return affine_inputs(fn) == _real_inputs(fn)


def _alpha_mangle(expr):
    """
    Rename bound variables in expr to avoid conflict with any free variables.

    FIXME this does not avoid conflict with other bound variables.
    """
    alpha_subs = {name: interpreter.gensym(name + '__BOUND') for name in expr.bound if '__BOUND' not in name}
    if not alpha_subs:
        return expr
    ast_values = expr._alpha_convert(alpha_subs)
    return reflect(type(expr), *ast_values)


def reflect(cls, *args, **kwargs):
    """
    Construct a funsor, populate ``._ast_values``, and cons hash.
    This is the only interpretation allowed to construct funsors.
    """
    if len(args) > len(cls._ast_fields):
        new_args = tuple(args[:len(cls._ast_fields) - 1]) + (args[len(cls._ast_fields) - 1 - len(args):],)
        assert len(new_args) == len(cls._ast_fields)
        _, args = args, new_args
    cache_key = tuple(id(arg) if type(arg).__name__ == 'DeviceArray' or not isinstance(arg, Hashable) else arg for arg in args)
    if cache_key in cls._cons_cache:
        return cls._cons_cache[cache_key]
    arg_types = tuple(typing.Tuple[tuple(map(type, arg))] if type(arg) is tuple and all(isinstance(a, Funsor) for a in arg) else typing.Tuple if type(arg) is tuple and not arg else type(arg) for arg in args)
    cls_specific = (cls.__origin__ if cls.__args__ else cls)[arg_types]
    result = super(FunsorMeta, cls_specific).__call__(*args)
    result._ast_values = args
    result = _alpha_mangle(result)
    cls._cons_cache[cache_key] = result
    return result


DIM_TO_NAME = tuple(map('_pyro_dim_{}'.format, range(-100, 0)))


NAME_TO_DIM = dict(zip(DIM_TO_NAME, range(-100, 0)))


def default_name_to_dim(event_inputs):
    if not event_inputs:
        return NAME_TO_DIM
    dim_to_name = DIM_TO_NAME + event_inputs
    return dict(zip(dim_to_name, range(-len(dim_to_name), 0)))


@singledispatch
def to_data(x, name_to_dim=None, **kwargs):
    """
    Extract a python object from a :class:`Funsor`.

    Raises a ``ValueError`` if free variables remain or if the funsor is lazy.

    :param x: An object, possibly a :class:`Funsor`.
    :param OrderedDict name_to_dim: An optional inputs hint.
    :return: A non-funsor equivalent to ``x``.
    :raises: ValueError if any free variables remain.
    :raises: PatternMissingError if funsor is not fully evaluated.
    """
    return x


def funsor_to_cat_and_mvn(funsor_, ndims, event_inputs):
    """
    Converts a labeled gaussian mixture model to a pair of distributions.

    :param funsor.joint.Joint funsor_: A Gaussian mixture funsor.
    :param int ndims: The number of batch dimensions in the result.
    :return: A pair ``(cat, mvn)``, where ``cat`` is a
        :class:`~pyro.distributions.Categorical` distribution over mixture
        components and ``mvn`` is a
        :class:`~pyro.distributions.MultivariateNormal` with rightmost batch
        dimension ranging over mixture components.
    """
    assert isinstance(funsor_, Contraction), funsor_
    assert sum(1 for d in funsor_.inputs.values() if d.dtype == 'real') == 1
    assert event_inputs, 'no components name found'
    assert not any(isinstance(v, Delta) for v in funsor_.terms)
    cat, mvn = to_data(funsor_, name_to_dim=default_name_to_dim(event_inputs))
    if ndims != len(cat.batch_shape):
        cat = cat.expand((1,) * (ndims - len(cat.batch_shape)) + cat.batch_shape)
    if ndims + 1 != len(mvn.batch_shape):
        mvn = mvn.expand((1,) * (ndims + 1 - len(mvn.batch_shape)) + mvn.batch_shape)
    return cat, mvn


def funsor_to_mvn(gaussian, ndims, event_inputs=()):
    """
    Convert a :class:`~funsor.terms.Funsor` to a
    :class:`pyro.distributions.MultivariateNormal` , dropping the normalization
    constant.

    :param gaussian: A Gaussian funsor.
    :type gaussian: funsor.gaussian.Gaussian or funsor.joint.Joint
    :param int ndims: The number of batch dimensions in the result.
    :param tuple event_inputs: A tuple of names to assign to rightmost
        dimensions.
    :return: a multivariate normal distribution.
    :rtype: pyro.distributions.MultivariateNormal
    """
    assert sum(1 for d in gaussian.inputs.values() if d.dtype == 'real') == 1
    if isinstance(gaussian, Contraction):
        gaussian = [v for v in gaussian.terms if isinstance(v, Gaussian)][0]
    assert isinstance(gaussian, Gaussian)
    result = to_data(gaussian, name_to_dim=default_name_to_dim(event_inputs))
    if ndims != len(result.batch_shape):
        result = result.expand((1,) * (ndims - len(result.batch_shape)) + result.batch_shape)
    return result


@to_funsor.register(np.ndarray)
@to_funsor.register(np.generic)
def tensor_to_funsor(x, output=None, dim_to_name=None):
    if not dim_to_name:
        output = output if output is not None else reals(*x.shape)
        result = Tensor(x, dtype=output.dtype)
        if result.output != output:
            raise ValueError('Invalid shape: expected {}, actual {}'.format(output.shape, result.output.shape))
        return result
    else:
        assert all(isinstance(k, int) and k < 0 and isinstance(v, str) for k, v in dim_to_name.items())
        if output is None:
            batch_ndims = min(-min(dim_to_name.keys()), len(x.shape))
            output = reals(*x.shape[batch_ndims:])
        packed_inputs = OrderedDict()
        for dim, size in zip(range(len(x.shape) - len(output.shape)), x.shape):
            name = dim_to_name.get(dim + len(output.shape) - len(x.shape), None)
            if name is not None and size > 1:
                packed_inputs[name] = bint(size)
        shape = tuple(d.size for d in packed_inputs.values()) + output.shape
        if x.shape != shape:
            x = x.reshape(shape)
        return Tensor(x, packed_inputs, dtype=output.dtype)


def matrix_and_mvn_to_funsor(matrix, mvn, event_dims=(), x_name='value_x', y_name='value_y'):
    """
    Convert a noisy affine function to a Gaussian. The noisy affine function is
    defined as::

        y = x @ matrix + mvn.sample()

    The result is a non-normalized Gaussian funsor with two real inputs,
    ``x_name`` and ``y_name``, corresponding to a conditional distribution of
    real vector ``y` given real vector ``x``.

    :param torch.Tensor matrix: A matrix with rightmost shape ``(x_size, y_size)``.
    :param mvn: A multivariate normal distribution with
        ``event_shape == (y_size,)``.
    :type mvn: torch.distributions.MultivariateNormal or
        torch.distributions.Independent of torch.distributions.Normal
    :param tuple event_dims: A tuple of names for rightmost dimensions.
        These will be assigned to ``result.inputs`` of type ``bint``.
    :param str x_name: The name of the ``x`` random variable.
    :param str y_name: The name of the ``y`` random variable.
    :return: A funsor with given ``real_inputs`` and possibly additional
        bint inputs.
    :rtype: funsor.terms.Funsor
    """
    assert isinstance(mvn, torch.distributions.MultivariateNormal) or isinstance(mvn, torch.distributions.Independent) and isinstance(mvn.base_dist, torch.distributions.Normal)
    assert isinstance(matrix, torch.Tensor)
    x_size, y_size = matrix.shape[-2:]
    assert mvn.event_shape == (y_size,)
    if isinstance(mvn, torch.distributions.Independent):
        return AffineNormal(tensor_to_funsor(matrix, event_dims, 2), tensor_to_funsor(mvn.base_dist.loc, event_dims, 1), tensor_to_funsor(mvn.base_dist.scale, event_dims, 1), Variable(x_name, reals(x_size)), Variable(y_name, reals(y_size)))
    info_vec = mvn.loc.unsqueeze(-1).cholesky_solve(mvn.scale_tril).squeeze(-1)
    log_prob = -0.5 * y_size * math.log(2 * math.pi) - mvn.scale_tril.diagonal(dim1=-1, dim2=-2).log().sum(-1) - 0.5 * (info_vec * mvn.loc).sum(-1)
    batch_shape = broadcast_shape(matrix.shape[:-2], mvn.batch_shape)
    P_yy = mvn.precision_matrix.expand(batch_shape + (y_size, y_size))
    neg_P_xy = matrix.matmul(P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = neg_P_xy.matmul(matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1), torch.cat([P_yx, P_yy], -1)], -2)
    info_y = info_vec.expand(batch_shape + (y_size,))
    info_x = -matrix.matmul(info_y.unsqueeze(-1)).squeeze(-1)
    info_vec = torch.cat([info_x, info_y], -1)
    info_vec = tensor_to_funsor(info_vec, event_dims, 1)
    precision = tensor_to_funsor(precision, event_dims, 2)
    inputs = info_vec.inputs.copy()
    inputs[x_name] = reals(x_size)
    inputs[y_name] = reals(y_size)
    return tensor_to_funsor(log_prob, event_dims) + Gaussian(info_vec.data, precision.data, inputs)


def default_dim_to_name(inputs_shape, event_inputs):
    dim_to_name_list = DIM_TO_NAME + event_inputs if event_inputs else DIM_TO_NAME
    return OrderedDict(zip(range(-len(inputs_shape), 0), dim_to_name_list[len(dim_to_name_list) - len(inputs_shape):]))


def mvn_to_funsor(pyro_dist, event_inputs=(), real_inputs=OrderedDict()):
    """
    Convert a joint :class:`torch.distributions.MultivariateNormal`
    distribution into a :class:`~funsor.terms.Funsor` with multiple real
    inputs.

    This should satisfy::

        sum(d.num_elements for d in real_inputs.values())
          == pyro_dist.event_shape[0]

    :param torch.distributions.MultivariateNormal pyro_dist: A
        multivariate normal distribution over one or more variables
        of real or vector or tensor type.
    :param tuple event_inputs: A tuple of names for rightmost dimensions.
        These will be assigned to ``result.inputs`` of type ``bint``.
    :param OrderedDict real_inputs: A dict mapping real variable name
        to appropriately sized ``reals()``. The sum of all ``.numel()``
        of all real inputs should be equal to the ``pyro_dist`` dimension.
    :return: A funsor with given ``real_inputs`` and possibly additional
        bint inputs.
    :rtype: funsor.terms.Funsor
    """
    assert isinstance(pyro_dist, torch.distributions.MultivariateNormal)
    assert isinstance(event_inputs, tuple)
    assert isinstance(real_inputs, OrderedDict)
    dim_to_name = default_dim_to_name(pyro_dist.batch_shape, event_inputs)
    return to_funsor(pyro_dist, reals(), dim_to_name, real_inputs=real_inputs)


NCV_PROCESS_NOISE = torch.tensor([[1 / 3, 0.0, 1 / 2, 0.0], [0.0, 1 / 3, 0.0, 1 / 2], [1 / 2, 0.0, 1.0, 0.0], [0.0, 1 / 2, 0.0, 1.0]])


NCV_TRANSITION_MATRIX = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])


def dist_to_funsor(pyro_dist, event_inputs=()):
    """
    Convert a PyTorch distribution to a Funsor.

    :param torch.distribution.Distribution: A PyTorch distribution.
    :return: A funsor.
    :rtype: funsor.terms.Funsor
    """
    assert isinstance(pyro_dist, torch.distributions.Distribution)
    assert isinstance(event_inputs, tuple)
    return to_funsor(pyro_dist, reals(), default_dim_to_name(pyro_dist.batch_shape, event_inputs))


class Model(nn.Module):

    def __init__(self, num_sensors):
        super(Model, self).__init__()
        self.num_sensors = num_sensors
        self.log_bias_scale = nn.Parameter(torch.tensor(0.0))
        self.log_obs_noise = nn.Parameter(torch.tensor(0.0))
        self.log_trans_noise = nn.Parameter(torch.tensor(0.0))

    def forward(self, observations, add_bias=True):
        obs_dim = 2 * self.num_sensors
        bias_scale = self.log_bias_scale.exp()
        obs_noise = self.log_obs_noise.exp()
        trans_noise = self.log_trans_noise.exp()
        bias = Variable('bias', reals(obs_dim))
        assert not torch.isnan(bias_scale), 'bias scales was nan'
        bias_dist = dist_to_funsor(dist.MultivariateNormal(torch.zeros(obs_dim), scale_tril=bias_scale * torch.eye(2 * self.num_sensors)))(value=bias)
        init_dist = torch.distributions.MultivariateNormal(torch.zeros(4), scale_tril=100.0 * torch.eye(4))
        self.init = dist_to_funsor(init_dist)(value='state')
        prev = Variable('prev', reals(4))
        curr = Variable('curr', reals(4))
        self.trans_dist = f_dist.MultivariateNormal(loc=prev @ NCV_TRANSITION_MATRIX, scale_tril=trans_noise * NCV_PROCESS_NOISE.cholesky(), value=curr)
        state = Variable('state', reals(4))
        obs = Variable('obs', reals(obs_dim))
        observation_matrix = Tensor(torch.eye(4, 2).unsqueeze(-1).expand(-1, -1, self.num_sensors).reshape(4, -1))
        assert observation_matrix.output.shape == (4, obs_dim), observation_matrix.output.shape
        obs_loc = state @ observation_matrix
        if add_bias:
            obs_loc += bias
        self.observation_dist = f_dist.MultivariateNormal(loc=obs_loc, scale_tril=obs_noise * torch.eye(obs_dim), value=obs)
        logp = bias_dist
        curr = 'state_init'
        logp += self.init(state=curr)
        for t, x in enumerate(observations):
            prev, curr = curr, f'state_{t}'
            logp += self.trans_dist(prev=prev, curr=curr)
            logp += self.observation_dist(state=curr, obs=x)
            logp = logp.reduce(ops.logaddexp, prev)
        logp = logp.reduce(ops.logaddexp, 'bias')
        assert set(logp.inputs) == {f'state_{len(observations) - 1}'}
        posterior = funsor_to_mvn(logp, ndims=0)
        logp = logp.reduce(ops.logaddexp)
        assert isinstance(logp, Tensor) and logp.shape == (), logp.pretty()
        return logp.data, posterior


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

    def forward(self, image):
        image = image.reshape(image.shape[:-2] + (-1,))
        h1 = F.relu(self.fc1(image))
        loc = self.fc21(h1)
        scale = self.fc22(h1).exp()
        return loc, scale


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        out = torch.sigmoid(self.fc4(h3))
        return out.reshape(out.shape[:-1] + (28, 28))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {}),
     lambda: ([torch.rand([20, 20])], {}),
     True),
]

class Test_pyro_ppl_funsor(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

