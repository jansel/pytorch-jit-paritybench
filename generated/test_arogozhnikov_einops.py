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


class EinopsError(RuntimeError):
    """ Runtime error thrown by einops """
    pass


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


def _product(sequence):
    result = 1
    for element in sequence:
        result *= element
    return result


_reductions = 'min', 'max', 'sum', 'mean', 'prod'


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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_arogozhnikov_einops(_paritybench_base):
    pass
