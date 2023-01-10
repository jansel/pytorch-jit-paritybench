import sys
_module = sys.modules[__name__]
del sys
converter = _module
utils = _module
einops = _module
_backends = _module
_torch_specific = _module
einops = _module
experimental = _module
data_api_packing = _module
indexing = _module
layers = _module
_einmix = _module
chainer = _module
flax = _module
gluon = _module
keras = _module
oneflow = _module
tensorflow = _module
packing = _module
parsing = _module
convert_readme = _module
setup = _module
test = _module
tests = _module
test_einsum = _module
test_examples = _module
test_layers = _module
test_notebooks = _module
test_ops = _module
test_other = _module
test_packing = _module
test_parsing = _module

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


from typing import Dict


from typing import List


import torch


import functools


import itertools


import string


import typing


from collections import OrderedDict


from typing import Set


from typing import Tuple


from typing import Union


from typing import Callable


from typing import Optional


from typing import TypeVar


from typing import cast


from typing import Any


import numpy as np


import numpy


from collections import namedtuple


import torch.jit


class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass


class TransformRecipe:
    """
    Recipe describes actual computation pathway.
    Recipe can be applied to a tensor or variable.
    """

    def __init__(self, elementary_axes_lengths: List[int], input_composite_axes: List[Tuple[List[int], List[int]]], reduced_elementary_axes: List[int], axes_permutation: List[int], added_axes: Dict[int, int], output_composite_axes: List[List[int]], ellipsis_position_in_lhs: Optional[int]=None):
        self.elementary_axes_lengths: List[int] = elementary_axes_lengths
        self.input_composite_axes: List[Tuple[List[int], List[int]]] = input_composite_axes
        self.output_composite_axes: List[List[int]] = output_composite_axes
        self.axes_permutation: List[int] = axes_permutation
        self.added_axes: Dict[int, int] = added_axes
        self.reduced_elementary_axes: List[int] = reduced_elementary_axes
        self.ellipsis_position_in_lhs: int = ellipsis_position_in_lhs if ellipsis_position_in_lhs is not None else 10000


class AnonymousAxis(object):
    """Important thing: all instances of this class are not equal to each other """

    def __init__(self, value: str):
        self.value = int(value)
        if self.value <= 1:
            if self.value == 1:
                raise EinopsError('No need to create anonymous axis of length 1. Report this as an issue')
            else:
                raise EinopsError('Anonymous axis should have positive length, not {}'.format(self.value))

    def __repr__(self):
        return '{}-axis'.format(str(self.value))


class ParsedExpression:
    """
    non-mutable structure that contains information about one side of expression (e.g. 'b c (h w)')
    and keeps some information important for downstream
    """

    def __init__(self, expression: str, *, allow_underscore: bool=False, allow_duplicates: bool=False):
        self.has_ellipsis: bool = False
        self.has_ellipsis_parenthesized: Optional[bool] = None
        self.identifiers: Set[str] = set()
        self.has_non_unitary_anonymous_axes: bool = False
        self.composition: List[Union[List[str], str]] = []
        if '.' in expression:
            if '...' not in expression:
                raise EinopsError('Expression may contain dots only inside ellipsis (...)')
            if str.count(expression, '...') != 1 or str.count(expression, '.') != 3:
                raise EinopsError('Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ')
            expression = expression.replace('...', _ellipsis)
            self.has_ellipsis = True
        bracket_group: Optional[List[str]] = None

        def add_axis_name(x):
            if x in self.identifiers:
                if not (allow_underscore and x == '_') and not allow_duplicates:
                    raise EinopsError('Indexing expression contains duplicate dimension "{}"'.format(x))
            if x == _ellipsis:
                self.identifiers.add(_ellipsis)
                if bracket_group is None:
                    self.composition.append(_ellipsis)
                    self.has_ellipsis_parenthesized = False
                else:
                    bracket_group.append(_ellipsis)
                    self.has_ellipsis_parenthesized = True
            else:
                is_number = str.isdecimal(x)
                if is_number and int(x) == 1:
                    if bracket_group is None:
                        self.composition.append([])
                    else:
                        pass
                    return
                is_axis_name, reason = self.check_axis_name_return_reason(x, allow_underscore=allow_underscore)
                if not (is_number or is_axis_name):
                    raise EinopsError('Invalid axis identifier: {}\n{}'.format(x, reason))
                if is_number:
                    x = AnonymousAxis(x)
                self.identifiers.add(x)
                if is_number:
                    self.has_non_unitary_anonymous_axes = True
                if bracket_group is None:
                    self.composition.append([x])
                else:
                    bracket_group.append(x)
        current_identifier = None
        for char in expression:
            if char in '() ':
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == '(':
                    if bracket_group is not None:
                        raise EinopsError('Axis composition is one-level (brackets inside brackets not allowed)')
                    bracket_group = []
                elif char == ')':
                    if bracket_group is None:
                        raise EinopsError('Brackets are not balanced')
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ['_', _ellipsis]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise EinopsError("Unknown character '{}'".format(char))
        if bracket_group is not None:
            raise EinopsError('Imbalanced parentheses in expression: "{}"'.format(expression))
        if current_identifier is not None:
            add_axis_name(current_identifier)

    def flat_axes_order(self) ->List:
        result = []
        for composed_axis in self.composition:
            assert isinstance(composed_axis, list), 'does not work with ellipsis'
            for axis in composed_axis:
                result.append(axis)
        return result

    def has_composed_axes(self) ->bool:
        for axes in self.composition:
            if isinstance(axes, list) and len(axes) > 1:
                return True
        return False

    @staticmethod
    def check_axis_name_return_reason(name: str, allow_underscore: bool=False) ->Tuple[bool, str]:
        if not str.isidentifier(name):
            return False, 'not a valid python identifier'
        elif name[0] == '_' or name[-1] == '_':
            if name == '_' and allow_underscore:
                return True, ''
            return False, 'axis name should should not start or end with underscore'
        else:
            if keyword.iskeyword(name):
                warnings.warn('It is discouraged to use axes names that are keywords: {}'.format(name), RuntimeWarning)
            if name in ['axis']:
                warnings.warn("It is discouraged to use 'axis' as an axis name and will raise an error in future", FutureWarning)
            return True, ''

    @staticmethod
    def check_axis_name(name: str) ->bool:
        """
        Valid axes names are python identifiers except keywords,
        and additionally should not start or end with underscore
        """
        is_valid, _reason = ParsedExpression.check_axis_name_return_reason(name)
        return is_valid


ReductionCallable = Callable[[Tensor, Tuple[int, ...]], Tensor]


Reduction = Union[str, ReductionCallable]


_reductions = 'min', 'max', 'sum', 'mean', 'prod'


_unknown_axis_length = -999999


@functools.lru_cache(256)
def _prepare_transformation_recipe(pattern: str, operation: Reduction, axes_lengths: Tuple[Tuple, ...]) ->TransformRecipe:
    """ Perform initial parsing of pattern and provided supplementary info
    axes_lengths is a tuple of tuples (axis_name, axis_length)
    """
    left_str, rght_str = pattern.split('->')
    left = ParsedExpression(left_str)
    rght = ParsedExpression(rght_str)
    if not left.has_ellipsis and rght.has_ellipsis:
        raise EinopsError('Ellipsis found in right side, but not left side of a pattern {}'.format(pattern))
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise EinopsError('Ellipsis is parenthesis in the left side is not allowed: {}'.format(pattern))
    if operation == 'rearrange':
        difference = set.symmetric_difference(left.identifiers, rght.identifiers)
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise EinopsError('Non-unitary anonymous axes are not supported in rearrange (exception is length 1)')
        if len(difference) > 0:
            raise EinopsError('Identifiers only on one side of expression (should be on both): {}'.format(difference))
    elif operation == 'repeat':
        difference = set.difference(left.identifiers, rght.identifiers)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the left side of repeat: {}'.format(difference))
        axes_without_size = set.difference({ax for ax in rght.identifiers if not isinstance(ax, AnonymousAxis)}, {*left.identifiers, *(ax for ax, _ in axes_lengths)})
        if len(axes_without_size) > 0:
            raise EinopsError('Specify sizes for new axes in repeat: {}'.format(axes_without_size))
    elif operation in _reductions or callable(operation):
        difference = set.difference(rght.identifiers, left.identifiers)
        if len(difference) > 0:
            raise EinopsError('Unexpected identifiers on the right side of reduce {}: {}'.format(operation, difference))
    else:
        raise EinopsError('Unknown reduction {}. Expect one of {}.'.format(operation, _reductions))
    axis_name2known_length: Dict[Union[str, AnonymousAxis], int] = OrderedDict()
    for composite_axis in left.composition:
        for axis_name in composite_axis:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
    repeat_axes_names = []
    for axis_name in rght.identifiers:
        if axis_name not in axis_name2known_length:
            if isinstance(axis_name, AnonymousAxis):
                axis_name2known_length[axis_name] = axis_name.value
            else:
                axis_name2known_length[axis_name] = _unknown_axis_length
            repeat_axes_names.append(axis_name)
    axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}
    reduced_axes: List[int] = [position for axis, position in axis_name2position.items() if axis not in rght.identifiers]
    reduced_axes = list(sorted(reduced_axes))
    for elementary_axis, axis_length in axes_lengths:
        if not ParsedExpression.check_axis_name(elementary_axis):
            raise EinopsError('Invalid name for an axis', elementary_axis)
        if elementary_axis not in axis_name2known_length:
            raise EinopsError('Axis {} is not used in transform'.format(elementary_axis))
        axis_name2known_length[elementary_axis] = axis_length
    input_axes_known_unknown = []
    for composite_axis in left.composition:
        known: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
        unknown: Set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length}
        if len(unknown) > 1:
            raise EinopsError('Could not infer sizes for {}'.format(unknown))
        assert len(unknown) + len(known) == len(composite_axis)
        input_axes_known_unknown.append(([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown]))
    axis_position_after_reduction: Dict[str, int] = {}
    for axis_name in itertools.chain(*left.composition):
        if axis_name in rght.identifiers:
            axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)
    result_axes_grouping: List[List[int]] = []
    for composite_axis in rght.composition:
        if composite_axis == _ellipsis:
            result_axes_grouping.append(_ellipsis_not_in_parenthesis)
        else:
            result_axes_grouping.append([axis_name2position[axis] for axis in composite_axis])
    ordered_axis_right = list(itertools.chain(*rght.composition))
    axes_permutation = [axis_position_after_reduction[axis] for axis in ordered_axis_right if axis in left.identifiers]
    added_axes = {i: axis_name2position[axis_name] for i, axis_name in enumerate(ordered_axis_right) if axis_name not in left.identifiers}
    ellipsis_left = None if _ellipsis not in left.composition else left.composition.index(_ellipsis)
    return TransformRecipe(elementary_axes_lengths=list(axis_name2known_length.values()), input_composite_axes=input_axes_known_unknown, reduced_elementary_axes=reduced_axes, axes_permutation=axes_permutation, added_axes=added_axes, output_composite_axes=result_axes_grouping, ellipsis_position_in_lhs=ellipsis_left)


class ReduceMixin:
    """
    Reduce layer behaves identically to einops.reduce operation.

    :param pattern: str, rearrangement pattern
    :param reduction: one of available reductions ('min', 'max', 'sum', 'mean', 'prod'), case-sensitive
    :param axes_lengths: any additional specification of dimensions

    See einops.reduce for source_examples.
    """

    def __init__(self, pattern: str, reduction: str, **axes_lengths: Any):
        super().__init__()
        self.pattern = pattern
        self.reduction = reduction
        self.axes_lengths = axes_lengths
        self._recipe = self.recipe()

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
        return _apply_recipe(self._recipe, x, reduction_type=self.reduction)


class TorchJitBackend:
    """
    Completely static backend that mimics part of normal backend functionality
    but restricted to torch stuff only
    """

    @staticmethod
    def reduce(x: torch.Tensor, operation: str, reduced_axes: List[int]):
        if operation == 'min':
            return x.amin(dim=reduced_axes)
        elif operation == 'max':
            return x.amax(dim=reduced_axes)
        elif operation == 'sum':
            return x.sum(dim=reduced_axes)
        elif operation == 'mean':
            return x.mean(dim=reduced_axes)
        elif operation == 'prod':
            for i in list(sorted(reduced_axes))[::-1]:
                x = x.prod(dim=i)
            return x
        else:
            raise NotImplementedError('Unknown reduction ', operation)

    @staticmethod
    def transpose(x, axes: List[int]):
        return x.permute(axes)

    @staticmethod
    def stack_on_zeroth_dimension(tensors: List[torch.Tensor]):
        return torch.stack(tensors)

    @staticmethod
    def tile(x, repeats: List[int]):
        return x.repeat(repeats)

    @staticmethod
    def add_axes(x, n_axes: int, pos2len: Dict[int, int]):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = torch.unsqueeze(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    @staticmethod
    def is_float_type(x):
        return x.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]

    @staticmethod
    def shape(x):
        return x.shape

    @staticmethod
    def reshape(x, shape: List[int]):
        return x.reshape(shape)


CookedRecipe = Tuple[List[int], List[int], List[int], Dict[int, int], List[int]]


def _product(sequence: List[int]) ->int:
    """ minimalistic product that works both with numbers and symbols. Supports empty lists """
    result = 1
    for element in sequence:
        result *= element
    return result


def is_ellipsis_not_in_parenthesis(group: List[int]) ->bool:
    if len(group) != 1:
        return False
    return group[0] == -999


def _reconstruct_from_shape_uncached(self: TransformRecipe, shape: List[int]) ->CookedRecipe:
    """
    Reconstruct all actual parameters using shape.
    Shape is a tuple that may contain integers, shape symbols (tf, keras, theano) and UnknownSize (keras, mxnet)
    known axes can be integers or symbols, but not Nones.
    """
    axes_lengths: List[int] = list(self.elementary_axes_lengths)
    if self.ellipsis_position_in_lhs != 10000:
        if len(shape) < len(self.input_composite_axes) - 1:
            raise EinopsError('Expected at least {} dimensions, got {}'.format(len(self.input_composite_axes) - 1, len(shape)))
    elif len(shape) != len(self.input_composite_axes):
        raise EinopsError('Expected {} dimensions, got {}'.format(len(self.input_composite_axes), len(shape)))
    ellipsis_shape: List[int] = []
    for input_axis, (known_axes, unknown_axes) in enumerate(self.input_composite_axes):
        before_ellipsis = input_axis
        after_ellipsis = input_axis + len(shape) - len(self.input_composite_axes)
        if input_axis == self.ellipsis_position_in_lhs:
            assert len(known_axes) == 0 and len(unknown_axes) == 1
            unknown_axis: int = unknown_axes[0]
            ellipsis_shape = shape[before_ellipsis:after_ellipsis + 1]
            for d in ellipsis_shape:
                if d is None:
                    raise EinopsError("Couldn't infer shape for one or more axes represented by ellipsis")
            total_dim_size: int = _product(ellipsis_shape)
            axes_lengths[unknown_axis] = total_dim_size
        else:
            if input_axis < self.ellipsis_position_in_lhs:
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
                unknown_axis = unknown_axes[0]
                inferred_length: int = length // known_product
                axes_lengths[unknown_axis] = inferred_length
    init_shapes = axes_lengths[:len(axes_lengths) - len(self.added_axes)]
    final_shapes: List[int] = []
    for output_axis, grouping in enumerate(self.output_composite_axes):
        if is_ellipsis_not_in_parenthesis(grouping):
            final_shapes.extend(ellipsis_shape)
        else:
            lengths = [axes_lengths[elementary_axis] for elementary_axis in grouping]
            final_shapes.append(_product(lengths))
    reduced_axes = self.reduced_elementary_axes
    axes_reordering = self.axes_permutation
    added_axes: Dict[int, int] = {pos: axes_lengths[pos_in_elementary] for pos, pos_in_elementary in self.added_axes.items()}
    return init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes


def apply_for_scriptable_torch(recipe: TransformRecipe, tensor: torch.Tensor, reduction_type: str) ->torch.Tensor:
    backend = TorchJitBackend
    init_shapes, reduced_axes, axes_reordering, added_axes, final_shapes = _reconstruct_from_shape_uncached(recipe, backend.shape(tensor))
    tensor = backend.reshape(tensor, init_shapes)
    if len(reduced_axes) > 0:
        tensor = backend.reduce(tensor, operation=reduction_type, reduced_axes=reduced_axes)
    tensor = backend.transpose(tensor, axes_reordering)
    if len(added_axes) > 0:
        tensor = backend.add_axes(tensor, n_axes=len(axes_reordering) + len(added_axes), pos2len=added_axes)
    return backend.reshape(tensor, final_shapes)


class Reduce(ReduceMixin, torch.nn.Module):

    def forward(self, input):
        return apply_for_scriptable_torch(self._recipe, input, reduction_type=self.reduction)

    def _apply_recipe(self, x):
        pass


class RearrangeMixin:
    """
    Rearrange layer behaves identically to einops.rearrange operation.

    :param pattern: str, rearrangement pattern
    :param axes_lengths: any additional specification of dimensions

    See einops.rearrange for source_examples.
    """

    def __init__(self, pattern: str, **axes_lengths: Any) ->None:
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        self._recipe = self.recipe()

    def __repr__(self) ->str:
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
        return _apply_recipe(self._recipe, x, reduction_type='rearrange')


class Rearrange(RearrangeMixin, torch.nn.Module):

    def forward(self, input):
        return apply_for_scriptable_torch(self._recipe, input, reduction_type='rearrange')

    def _apply_recipe(self, x):
        pass


def _report_axes(axes: set, report_message: str):
    if len(axes) > 0:
        raise EinopsError(report_message.format(axes))


class _EinmixMixin:

    def __init__(self, pattern: str, weight_shape: str, bias_shape: Optional[str]=None, **axes_lengths: Any):
        """
        EinMix - Einstein summation with automated tensor management and axis packing/unpacking.

        EinMix is an advanced tool, helpful tutorial:
        https://github.com/arogozhnikov/einops/blob/master/docs/3-einmix-layer.ipynb

        Imagine taking einsum with two arguments, one of each input, and one - tensor with weights
        >>> einsum('time batch channel_in, channel_in channel_out -> time batch channel_out', input, weight)

        This layer manages weights for you, syntax highlights separate role of weight matrix
        >>> EinMix('time batch channel_in -> time batch channel_out', weight_shape='channel_in channel_out')
        But otherwise it is the same einsum under the hood.

        Simple linear layer with bias term (you have one like that in your framework)
        >>> EinMix('t b cin -> t b cout', weight_shape='cin cout', bias_shape='cout', cin=10, cout=20)
        There is restriction to mix the last axis. Let's mix along height
        >>> EinMix('h w c-> hout w c', weight_shape='h hout', bias_shape='hout', h=32, hout=32)
        Channel-wise multiplication (like one used in normalizations)
        >>> EinMix('t b c -> t b c', weight_shape='c', c=128)
        Separate dense layer within each head, no connection between different heads
        >>> EinMix('t b (head cin) -> t b (head cout)', weight_shape='head cin cout', ...)

        ... ah yes, you need to specify all dimensions of weight shape/bias shape in parameters.

        Use cases:
        - when channel dimension is not last, use EinMix, not transposition
        - patch/segment embeddings
        - when need only within-group connections to reduce number of weights and computations
        - perfect as a part of sequential models
        - next-gen MLPs (follow tutorial to learn more)

        Uniform He initialization is applied to weight tensor and encounters for number of elements mixed.

        Parameters
        :param pattern: transformation pattern, left side - dimensions of input, right side - dimensions of output
        :param weight_shape: axes of weight. A tensor of this shape is created, stored, and optimized in a layer
        :param bias_shape: axes of bias added to output. Weights of this shape are created and stored. If `None` (the default), no bias is added.
        :param axes_lengths: dimensions of weight tensor
        """
        super().__init__()
        self.pattern = pattern
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.axes_lengths = axes_lengths
        self.initialize_einmix(pattern=pattern, weight_shape=weight_shape, bias_shape=bias_shape, axes_lengths=axes_lengths)

    def initialize_einmix(self, pattern: str, weight_shape: str, bias_shape: Optional[str], axes_lengths: dict):
        left_pattern, right_pattern = pattern.split('->')
        left = ParsedExpression(left_pattern)
        right = ParsedExpression(right_pattern)
        weight = ParsedExpression(weight_shape)
        _report_axes(set.difference(right.identifiers, {*left.identifiers, *weight.identifiers}), 'Unrecognized identifiers on the right side of EinMix {}')
        if left.has_ellipsis or right.has_ellipsis or weight.has_ellipsis:
            raise EinopsError('Ellipsis is not supported in EinMix (right now)')
        if any(x.has_non_unitary_anonymous_axes for x in [left, right, weight]):
            raise EinopsError('Anonymous axes (numbers) are not allowed in EinMix')
        if '(' in weight_shape or ')' in weight_shape:
            raise EinopsError(f'Parenthesis is not allowed in weight shape: {weight_shape}')
        pre_reshape_pattern = None
        pre_reshape_lengths = None
        post_reshape_pattern = None
        if any(len(group) != 1 for group in left.composition):
            names: List[str] = []
            for group in left.composition:
                names += group
            composition = ' '.join(names)
            pre_reshape_pattern = f'{left_pattern}->{composition}'
            pre_reshape_lengths = {name: length for name, length in axes_lengths.items() if name in names}
        if any(len(group) != 1 for group in right.composition):
            names = []
            for group in right.composition:
                names += group
            composition = ' '.join(names)
            post_reshape_pattern = f'{composition}->{right_pattern}'
        self._create_rearrange_layers(pre_reshape_pattern, pre_reshape_lengths, post_reshape_pattern, {})
        for axis in weight.identifiers:
            if axis not in axes_lengths:
                raise EinopsError('Dimension {} of weight should be specified'.format(axis))
        _report_axes(set.difference(set(axes_lengths), {*left.identifiers, *weight.identifiers}), 'Axes {} are not used in pattern')
        _report_axes(set.difference(weight.identifiers, {*left.identifiers, *right.identifiers}), 'Weight axes {} are redundant')
        if len(weight.identifiers) == 0:
            warnings.warn('EinMix: weight has no dimensions (means multiplication by a number)')
        _weight_shape = [axes_lengths[axis] for axis, in weight.composition]
        _fan_in = _product([axes_lengths[axis] for axis, in weight.composition if axis not in right.identifiers])
        if bias_shape is not None:
            if not isinstance(bias_shape, str):
                raise EinopsError('bias shape should be string specifying which axes bias depends on')
            bias = ParsedExpression(bias_shape)
            _report_axes(set.difference(bias.identifiers, right.identifiers), 'Bias axes {} not present in output')
            _report_axes(set.difference(bias.identifiers, set(axes_lengths)), 'Sizes not provided for bias axes {}')
            _bias_shape = []
            for axes in right.composition:
                for axis in axes:
                    if axis in bias.identifiers:
                        _bias_shape.append(axes_lengths[axis])
                    else:
                        _bias_shape.append(1)
        else:
            _bias_shape = None
        weight_bound = (3 / _fan_in) ** 0.5
        bias_bound = (1 / _fan_in) ** 0.5
        self._create_parameters(_weight_shape, weight_bound, _bias_shape, bias_bound)
        mapped_identifiers = {*left.identifiers, *right.identifiers, *weight.identifiers}
        mapping2letters = {k: letter for letter, k in zip(string.ascii_lowercase, mapped_identifiers)}

        def write_flat(axes: list):
            return ''.join(mapping2letters[axis] for axis in axes)
        self.einsum_pattern: str = '{},{}->{}'.format(write_flat(left.flat_axes_order()), write_flat(weight.flat_axes_order()), write_flat(right.flat_axes_order()))

    def _create_rearrange_layers(self, pre_reshape_pattern: Optional[str], pre_reshape_lengths: Optional[Dict], post_reshape_pattern: Optional[str], post_reshape_lengths: Optional[Dict]):
        raise NotImplementedError('Should be defined in framework implementations')

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        """ Shape and implementations """
        raise NotImplementedError('Should be defined in framework implementations')

    def __repr__(self):
        params = repr(self.pattern)
        params += f", '{self.weight_shape}'"
        if self.bias_shape is not None:
            params += f", '{self.bias_shape}'"
        for axis, length in self.axes_lengths.items():
            params += ', {}={}'.format(axis, length)
        return '{}({})'.format(self.__class__.__name__, params)


class EinMix(_EinmixMixin, torch.nn.Module):

    def _create_parameters(self, weight_shape, weight_bound, bias_shape, bias_bound):
        self.weight = torch.nn.Parameter(torch.zeros(weight_shape).uniform_(-weight_bound, weight_bound), requires_grad=True)
        if bias_shape is not None:
            self.bias = torch.nn.Parameter(torch.zeros(bias_shape).uniform_(-bias_bound, bias_bound), requires_grad=True)
        else:
            self.bias = None

    def _create_rearrange_layers(self, pre_reshape_pattern: Optional[str], pre_reshape_lengths: Optional[Dict], post_reshape_pattern: Optional[str], post_reshape_lengths: Optional[Dict]):
        self.pre_rearrange = None
        if pre_reshape_pattern is not None:
            self.pre_rearrange = Rearrange(pre_reshape_pattern, **cast(dict, pre_reshape_lengths))
        self.post_rearrange = None
        if post_reshape_pattern is not None:
            self.post_rearrange = Rearrange(post_reshape_pattern, **cast(dict, post_reshape_lengths))

    def forward(self, input):
        if self.pre_rearrange is not None:
            input = self.pre_rearrange(input)
        result = torch.einsum(self.einsum_pattern, input, self.weight)
        if self.bias is not None:
            result += self.bias
        if self.post_rearrange is not None:
            result = self.post_rearrange(result)
        return result

