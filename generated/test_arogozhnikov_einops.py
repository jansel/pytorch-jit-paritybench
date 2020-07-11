import sys
_module = sys.modules[__name__]
del sys
converter = _module
einops = _module
_backends = _module
einops = _module
layers = _module
chainer = _module
gluon = _module
keras = _module
tensorflow = _module
setup = _module
test = _module
tests = _module
test_layers = _module
test_notebooks = _module
test_ops = _module
test_other = _module

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


import warnings


import functools


import itertools


from collections import OrderedDict


from typing import Tuple


from typing import List


from typing import Set


from typing import Dict


import math


import torch


from collections import namedtuple


import numpy


class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass


def _optimize_transformation(init_shapes, reduced_axes, axes_reordering, final_shapes):
    assert len(axes_reordering) + len(reduced_axes) == len(init_shapes)
    reduced_axes = tuple(sorted(reduced_axes))
    for i in range(len(reduced_axes) - 1)[::-1]:
        if reduced_axes[i] + 1 == reduced_axes[i + 1]:
            removed_axis = reduced_axes[i + 1]
            removed_length = init_shapes[removed_axis]
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            reduced_axes = reduced_axes[:i + 1] + tuple(axis - 1 for axis in reduced_axes[i + 2:])

    def build_mapping():
        init_to_final = {}
        for axis in range(len(init_shapes)):
            if axis in reduced_axes:
                init_to_final[axis] = None
            else:
                after_reduction = sum(x is not None for x in init_to_final.values())
                init_to_final[axis] = list(axes_reordering).index(after_reduction)
        return init_to_final
    init_axis_to_final_axis = build_mapping()
    for init_axis in range(len(init_shapes) - 1)[::-1]:
        if init_axis_to_final_axis[init_axis] is None:
            continue
        if init_axis_to_final_axis[init_axis + 1] is None:
            continue
        if init_axis_to_final_axis[init_axis] + 1 == init_axis_to_final_axis[init_axis + 1]:
            removed_axis = init_axis + 1
            removed_length = init_shapes[removed_axis]
            removed_axis_after_reduction = sum(x not in reduced_axes for x in range(removed_axis))
            reduced_axes = tuple(axis if axis < removed_axis else axis - 1 for axis in reduced_axes)
            init_shapes = init_shapes[:removed_axis] + init_shapes[removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            old_reordering = axes_reordering
            axes_reordering = []
            for axis in old_reordering:
                if axis == removed_axis_after_reduction:
                    pass
                elif axis < removed_axis_after_reduction:
                    axes_reordering.append(axis)
                else:
                    axes_reordering.append(axis - 1)
            init_axis_to_final_axis = build_mapping()
    return init_shapes, reduced_axes, axes_reordering, final_shapes


def _product(sequence):
    result = 1
    for element in sequence:
        result *= element
    return result


_reductions = 'min', 'max', 'sum', 'mean', 'prod'


def _reduce_axes(tensor, reduction_type: str, reduced_axes: Tuple[int], backend):
    reduced_axes = tuple(reduced_axes)
    if len(reduced_axes) == 0:
        return tensor
    assert reduction_type in _reductions
    if reduction_type == 'mean':
        if not backend.is_float_type(tensor):
            raise NotImplementedError('reduce_mean is not available for non-floating tensors')
    return backend.reduce(tensor, reduction_type, reduced_axes)


class AbstractBackend:
    """ Base backend class, major part of methods are only for debugging purposes. """
    framework_name = None

    def is_appropriate_type(self, tensor):
        """ helper method should recognize tensors it can handle """
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop):
        raise NotImplementedError("framework doesn't implement arange")

    def shape(self, x):
        """shape should return a tuple with integers or "shape symbols" (which will evaluate to actual size)"""
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def reduce(self, x, operation, axes):
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        raise NotImplementedError()

    def add_axis(self, x, new_position):
        raise NotImplementedError()

    def add_axes(self, x, n_axes, pos2len):
        repeats = [1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return self.tile(x, tuple(repeats))

    def tile(self, x, repeats):
        """repeats is a number of  """
        raise NotImplementedError()

    def is_float_type(self, x):
        raise NotImplementedError()

    def layers(self, x):
        raise NotImplementedError('backend does not provide layers')

    def __repr__(self):
        return '<einops backend for {}>'.format(self.framework_name)


_backends = {}


_debug_importing = False


def get_backend(tensor) ->'AbstractBackend':
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
    """
    for framework_name, backend in _backends.items():
        if backend.is_appropriate_type(tensor):
            return backend
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)
    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            None
        if BackendSubclass.framework_name not in _backends:
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    None
                backend = BackendSubclass()
                _backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    return backend
    raise RuntimeError('Tensor type unknown to einops {}'.format(type(tensor)))


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """

    def __init__(self, elementary_axes_lengths: List, input_composite_axes: List[Tuple[List[int], List[int]]], reduced_elementary_axes: Tuple[int], axes_permutation: Tuple[int], added_axes: Dict[int, int], output_composite_axes: List[List[int]], reduction_type: str='rearrange', ellipsis_positions: Tuple[int, int]=(math.inf, math.inf)):
        self.elementary_axes_lengths = elementary_axes_lengths
        self.input_composite_axes = input_composite_axes
        self.output_composite_axes = output_composite_axes
        self.axes_permutation = axes_permutation
        self.added_axes = added_axes
        self.reduction_type = reduction_type
        self.reduced_elementary_axes = reduced_elementary_axes
        self.ellipsis_positions = ellipsis_positions

    @functools.lru_cache(maxsize=1024)
    def reconstruct_from_shape(self, shape, optimize=False):
        """
        Reconstruct all actual parameters using shape.
        Shape is a tuple that may contain integers, shape symbols (tf, keras, theano) and UnknownSize (keras, mxnet)
        known axes can be integers or symbols, but not Nones.
        """
        axes_lengths = list(self.elementary_axes_lengths)
        if self.ellipsis_positions != (math.inf, math.inf):
            if len(shape) < len(self.input_composite_axes) - 1:
                raise EinopsError('Expected at least {} dimensions, got {}'.format(len(self.input_composite_axes) - 1, len(shape)))
        elif len(shape) != len(self.input_composite_axes):
            raise EinopsError('Expected {} dimensions, got {}'.format(len(self.input_composite_axes), len(shape)))
        for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composite_axes):
            before_ellipsis = input_axis
            after_ellipsis = input_axis + len(shape) - len(self.input_composite_axes)
            if input_axis == self.ellipsis_positions[0]:
                assert len(known_axes) == 0 and len(unknown_axes) == 1
                unknown_axis, = unknown_axes
                ellipsis_shape = shape[before_ellipsis:after_ellipsis + 1]
                if any(d is None for d in ellipsis_shape):
                    raise EinopsError("Couldn't infer shape for one or more axes represented by ellipsis")
                axes_lengths[unknown_axis] = _product(ellipsis_shape)
            else:
                if input_axis < self.ellipsis_positions[0]:
                    length = shape[before_ellipsis]
                else:
                    length = shape[after_ellipsis]
                known_product = 1
                for axis in known_axes:
                    known_product *= axes_lengths[axis]
                if len(unknown_axes) == 0:
                    if isinstance(length, int) and isinstance(known_product, int) and length != known_product:
                        raise EinopsError('Shape mismatch, {} != {}'.format(length, known_product))
                else:
                    if isinstance(length, int) and isinstance(known_product, int) and length % known_product != 0:
                        raise EinopsError("Shape mismatch, can't divide axis of length {} in chunks of {}".format(length, known_product))
                    unknown_axis, = unknown_axes
                    axes_lengths[unknown_axis] = length // known_product
        init_shapes = axes_lengths[:len(axes_lengths) - len(self.added_axes)]
        final_shapes = []
        for output_axis, grouping in enumerate(self.output_composite_axes):
            if output_axis == self.ellipsis_positions[1]:
                final_shapes.extend(ellipsis_shape)
            else:
                lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
                final_shapes.append(_product(lengths))
        reduced_axes = self.reduced_elementary_axes
        axes_reordering = self.axes_permutation
        added_axes = {pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()}
        if optimize:
            assert len(self.added_axes) == 0
            return _optimize_transformation(init_shapes, reduced_axes, axes_reordering, final_shapes)
        else:
            return init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes

    def apply(self, tensor):
        backend = get_backend(tensor)
        init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes = self.reconstruct_from_shape(backend.shape(tensor))
        tensor = backend.reshape(tensor, init_shapes)
        tensor = _reduce_axes(tensor, reduction_type=self.reduction_type, reduced_axes=reduced_axes, backend=backend)
        tensor = backend.transpose(tensor, axes_reordering)
        if len(added_axes) > 0:
            tensor = backend.add_axes(tensor, n_axes=len(axes_reordering) + len(added_axes), pos2len=added_axes)
        return backend.reshape(tensor, final_shapes)


def _check_elementary_axis_name(name: str) ->bool:
    """
    Valid elementary axes contain only lower latin letters and digits and start with a letter.
    """
    if len(name) == 0:
        return False
    if not 'a' <= name[0] <= 'z':
        return False
    for letter in name:
        if not letter.isdigit() and not 'a' <= letter <= 'z':
            return False
    return True


_ellipsis = '…'


CompositeAxis = List[str]


def parse_expression(expression: str) ->Tuple[Set[str], List[CompositeAxis]]:
    """
    Parses an indexing expression (for a single tensor).
    Checks uniqueness of names, checks usage of '...' (allowed only once)
    Returns set of all used identifiers and a list of axis groups.
    """
    identifiers = set()
    composite_axes = []
    if '.' in expression:
        if '...' not in expression:
            raise EinopsError('Expression may contain dots only inside ellipsis (...)')
        if str.count(expression, '...') != 1 or str.count(expression, '.') != 3:
            raise EinopsError('Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ')
        expression = expression.replace('...', _ellipsis)
    bracket_group = None

    def add_axis_name(x):
        if x is not None:
            if x in identifiers:
                raise ValueError('Indexing expression contains duplicate dimension "{}"'.format(x))
            identifiers.add(x)
            if bracket_group is None:
                composite_axes.append([x])
            else:
                bracket_group.append(x)
    current_identifier = None
    for char in expression:
        if char in '() ' + _ellipsis:
            add_axis_name(current_identifier)
            current_identifier = None
            if char == _ellipsis:
                if bracket_group is not None:
                    raise EinopsError("Ellipsis can't be used inside the composite axis (inside brackets)")
                composite_axes.append(_ellipsis)
                identifiers.add(_ellipsis)
            elif char == '(':
                if bracket_group is not None:
                    raise EinopsError('Axis composition is one-level (brackets inside brackets not allowed)')
                bracket_group = []
            elif char == ')':
                if bracket_group is None:
                    raise EinopsError('Brackets are not balanced')
                composite_axes.append(bracket_group)
                bracket_group = None
        elif '0' <= char <= '9':
            if current_identifier is None:
                raise EinopsError("Axis name can't start with a digit")
            current_identifier += char
        elif 'a' <= char <= 'z':
            if current_identifier is None:
                current_identifier = char
            else:
                current_identifier += char
        else:
            if 'A' <= char <= 'Z':
                raise EinopsError("Only lower-case latin letters allowed in names, not '{}'".format(char))
            raise EinopsError("Unknown character '{}'".format(char))
    if bracket_group is not None:
        raise EinopsError('Imbalanced parentheses in expression: "{}"'.format(expression))
    add_axis_name(current_identifier)
    return identifiers, composite_axes


@functools.lru_cache(256)
def _prepare_transformation_recipe(pattern: str, operation: str, axes_lengths: Tuple[Tuple]) ->TransformRecipe:
    """ Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left, right = pattern.split('->')
    identifiers_left, composite_axes_left = parse_expression(left)
    identifiers_rght, composite_axes_rght = parse_expression(right)
    if operation == 'rearrange':
        difference = set.symmetric_difference(identifiers_left, identifiers_rght)
        if len(difference) > 0:
            raise EinopsError('Identifiers only on one side of expression (should be on both): {}'.format(difference))
    elif operation == 'repeat':
        difference = set.difference(identifiers_left, identifiers_rght)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the left side of repeat: {}'.format(difference))
    elif operation in _reductions:
        difference = set.difference(identifiers_rght, identifiers_left)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the right side of reduce {}: {}'.format(operation, difference))
    else:
        raise EinopsError('Unknown reduction {}. Expect one of {}.'.format(operation, _reductions))
    axis_name2known_length = OrderedDict()
    for composite_axis in composite_axes_left:
        for axis_name in composite_axis:
            axis_name2known_length[axis_name] = None
    repeat_axes_names = []
    for axis_name in identifiers_rght:
        if axis_name not in axis_name2known_length:
            axis_name2known_length[axis_name] = None
            repeat_axes_names.append(axis_name)
    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}
    reduced_axes = [position for axis, position in axis_name2position.items() if axis not in identifiers_rght]
    for elementary_axis, axis_length in axes_lengths:
        if not _check_elementary_axis_name(elementary_axis):
            raise EinopsError('Invalid name for an axis', elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise EinopsError('Axis {} is not used in transform'.format(elementary_axis))
        assert axis_name2known_length[elementary_axis] is None
        axis_name2known_length[elementary_axis] = axis_length
    input_axes_known_unknown = []
    for composite_axis in composite_axes_left:
        known = {axis for axis in composite_axis if axis_name2known_length[axis] is not None}
        unknown = {axis for axis in composite_axis if axis_name2known_length[axis] is None}
        if len(unknown) > 1:
            raise EinopsError('Could not infer sizes for {}'.format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown]))
    axis_position_after_reduction = {}
    for axis_name in itertools.chain(*composite_axes_left):
        if axis_name in identifiers_rght:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)
    result_axes_grouping = [[axis_name2position[axis] for axis in composite_axis] for composite_axis in composite_axes_rght]
    ordered_axis_right = list(itertools.chain(*composite_axes_rght))
    axes_permutation = tuple(axis_position_after_reduction[axis] for axis in ordered_axis_right if axis in identifiers_left)
    added_axes = {i: axis_name2position[axis_name] for i, axis_name in enumerate(ordered_axis_right) if axis_name not in identifiers_left}
    ellipsis_left = math.inf if _ellipsis not in composite_axes_left else composite_axes_left.index(_ellipsis)
    ellipsis_rght = math.inf if _ellipsis not in composite_axes_rght else composite_axes_rght.index(_ellipsis)
    return TransformRecipe(elementary_axes_lengths=list(axis_name2known_length.values()), input_composite_axes=input_axes_known_unknown, reduced_elementary_axes=tuple(reduced_axes), axes_permutation=axes_permutation, added_axes=added_axes, output_composite_axes=result_axes_grouping, reduction_type=operation, ellipsis_positions=(ellipsis_left, ellipsis_rght))


class RearrangeMixin:
    """
    Rearrange layer behaves identically to einops.rearrange operation.

    :param pattern: str, rearrangement pattern
    :param axes_lengths: any additional specification of dimensions

    See einops.rearrange for source_examples.
    """

    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        self.recipe()

    def __repr__(self):
        params = repr(self.pattern)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) ->TransformRecipe:
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(self.pattern, operation='rearrange', axes_lengths=hashable_lengths)
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        try:
            return self.recipe().apply(x)
        except EinopsError as e:
            raise EinopsError(' Error while computing {!r}\n {}'.format(self, e))


class Rearrange(RearrangeMixin, torch.nn.Module):

    def forward(self, input):
        return self._apply_recipe(input)


class ReduceMixin:
    """
    Reduce layer behaves identically to einops.reduce operation.

    :param pattern: str, rearrangement pattern
    :param reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
    :param axes_lengths: any additional specification of dimensions

    See einops.reduce for source_examples.
    """

    def __init__(self, pattern, reduction, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths
        self.recipe()

    def __repr__(self):
        params = '{!r}, {!r}'.format(self.pattern, self.reduction)
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)

    @functools.lru_cache(maxsize=1024)
    def recipe(self) ->TransformRecipe:
        try:
            hashable_lengths = tuple(sorted(self.axes_lengths.items()))
            return _prepare_transformation_recipe(self.pattern, operation=self.reduction, axes_lengths=hashable_lengths)
        except EinopsError as e:
            raise EinopsError(' Error while preparing {!r}\n {}'.format(self, e))

    def _apply_recipe(self, x):
        try:
            return self.recipe().apply(x)
        except EinopsError as e:
            raise EinopsError(' Error while computing {!r}\n {}'.format(self, e))


class Reduce(ReduceMixin, torch.nn.Module):

    def forward(self, input):
        return self._apply_recipe(input)

