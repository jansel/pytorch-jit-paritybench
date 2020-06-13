import sys
_module = sys.modules[__name__]
del sys
converter = _module
einops = _module
_backends = _module
layers = _module
chainer = _module
gluon = _module
keras = _module
tensorflow = _module
torch = _module
setup = _module
test = _module
tests = _module
test_layers = _module
test_notebooks = _module
test_ops = _module
test_other = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from collections import namedtuple


import numpy


def _optimize_transformation(init_shapes, reduced_axes, axes_reordering,
    final_shapes):
    assert len(axes_reordering) + len(reduced_axes) == len(init_shapes)
    reduced_axes = tuple(sorted(reduced_axes))
    for i in range(len(reduced_axes) - 1)[::-1]:
        if reduced_axes[i] + 1 == reduced_axes[i + 1]:
            removed_axis = reduced_axes[i + 1]
            removed_length = init_shapes[removed_axis]
            init_shapes = init_shapes[:removed_axis] + init_shapes[
                removed_axis + 1:]
            init_shapes[removed_axis - 1] *= removed_length
            reduced_axes = reduced_axes[:i + 1] + tuple(axis - 1 for axis in
                reduced_axes[i + 2:])

    def build_mapping():
        init_to_final = {}
        for axis in range(len(init_shapes)):
            if axis in reduced_axes:
                init_to_final[axis] = None
            else:
                after_reduction = sum(x is not None for x in init_to_final.
                    values())
                init_to_final[axis] = list(axes_reordering).index(
                    after_reduction)
        return init_to_final
    init_axis_to_final_axis = build_mapping()
    for init_axis in range(len(init_shapes) - 1)[::-1]:
        if init_axis_to_final_axis[init_axis] is None:
            continue
        if init_axis_to_final_axis[init_axis + 1] is None:
            continue
        if init_axis_to_final_axis[init_axis] + 1 == init_axis_to_final_axis[
            init_axis + 1]:
            removed_axis = init_axis + 1
            removed_length = init_shapes[removed_axis]
            removed_axis_after_reduction = sum(x not in reduced_axes for x in
                range(removed_axis))
            reduced_axes = tuple(axis if axis < removed_axis else axis - 1 for
                axis in reduced_axes)
            init_shapes = init_shapes[:removed_axis] + init_shapes[
                removed_axis + 1:]
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


_backends = {}


_debug_importing = False


class AbstractBackend:
    """ Base backend class, major part of methods are only for debugging purposes. """
    framework_name = None

    def is_appropriate_type(self, tensor):
        """ helper method should recognize tensors it can handle """
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError(
            "framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError(
            "framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError(
            "framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        raise NotImplementedError(
            "framework doesn't support symbolic computations")

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
            print('Testing for subclass of ', BackendSubclass)
        if BackendSubclass.framework_name not in _backends:
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    print('Imported backend for ', BackendSubclass.
                        framework_name)
                backend = BackendSubclass()
                _backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    return backend
    raise RuntimeError('Tensor type unknown to einops {}'.format(type(tensor)))


_reductions = 'min', 'max', 'sum', 'mean', 'prod'


_ellipsis = '…'


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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_arogozhnikov_einops(_paritybench_base):
    pass
