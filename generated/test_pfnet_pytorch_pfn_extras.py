import sys
_module = sys.modules[__name__]
del sys
mnist = _module
pytorch_pfn_extras = _module
_version = _module
config = _module
nn = _module
modules = _module
extended_sequential = _module
lazy = _module
lazy_conv = _module
lazy_linear = _module
reporting = _module
training = _module
extension = _module
extensions = _module
_snapshot = _module
evaluator = _module
fail_on_non_number = _module
log_report = _module
micro_average = _module
parameter_statistics = _module
plot_report = _module
print_report = _module
progress_bar = _module
snapshot_writers = _module
util = _module
value_observation = _module
variable_statistics_plot = _module
manager = _module
trainer = _module
trigger = _module
trigger_util = _module
triggers = _module
early_stopping_trigger = _module
interval_trigger = _module
manual_schedule_trigger = _module
minmax_value_trigger = _module
once_trigger = _module
time_trigger = _module
writing = _module
setup = _module
tests = _module
pytorch_pfn_extras_tests = _module
nn_tests = _module
modules_tests = _module
test_extended_sequential = _module
test_lazy = _module
test_lazy_conv = _module
test_lazy_linear = _module
test_config = _module
test_reporter = _module
test_distributed_snapshot = _module
test_evaluator = _module
test_fail_on_non_number = _module
test_micro_average = _module
test_plot_report = _module
test_progress_bar = _module
test_snapshot = _module
test_snapshot_writers = _module
test_value_observation = _module
test_variable_statistics_plot = _module
test_extension = _module
test_manager = _module
test_trigger_util = _module
test_early_stopping_trigger = _module
test_interval_trigger = _module
test_minmax_value_trigger = _module
test_once_trigger = _module
test_schedule_trigger = _module
test_time_trigger = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch import nn


from torch.optim import SGD


from torch.utils.data import DataLoader


import torch


import torch.nn.functional as F


from torchvision.transforms import Compose


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torchvision.datasets import MNIST


import torch.nn as nn


import torch.optim as optim


from torchvision import datasets


from torchvision import transforms


import copy


import inspect


import warnings


import collections


import typing as tp


import numpy


import torch.distributed


import time


from torch.nn import functional as F


import itertools


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class Net(nn.Module):

    def __init__(self, lazy):
        super().__init__()
        if lazy:
            self.conv1 = ppe.nn.LazyConv2d(None, 20, 5, 1)
            self.conv2 = ppe.nn.LazyConv2d(None, 50, 5, 1)
            self.fc1 = ppe.nn.LazyLinear(None, 500)
            self.fc2 = ppe.nn.LazyLinear(None, 10)
        else:
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _reset_parameters(model):
    if isinstance(model, torch.nn.Sequential) or isinstance(model, torch.nn
        .ModuleList):
        for submodel in model:
            _reset_parameters(submodel)
    elif isinstance(model, torch.nn.ModuleDict):
        for submodel in model.values():
            _reset_parameters(submodel)
    elif isinstance(model, torch.nn.Module):
        model.reset_parameters()
    return model


class ExtendedSequential(torch.nn.Sequential):
    """Sequential module with extended features from chainer.

    """

    def _copy_model(self, mode):
        if mode == 'init':
            return _reset_parameters(copy.deepcopy(self))
        elif mode == 'copy':
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def repeat(self, n_repeat: int, mode: 'str'='init'):
        """Repeats this Sequential multiple times.

        This method returns a :class:`~torch.nn.Sequential` object which has
        original `Sequential` multiple times repeatedly. The ``mode``
        argument means how to copy this sequential to repeat.

        The functions is supposed to behave the same way as `repeat`
        in `chainer`.

        Args:
            n_repeat (int): Number of times to repeat.
            mode (str): It should be either ``init``, ``copy``, or ``share``.
                ``init`` means parameters of each repeated element in the
                returned :class:`~torch.nn.Sequential` will be re-initialized,
                so that all elements have different initial parameters.
                ``copy`` means that the parameters will not be re-initialized
                but object itself will be deep-copied, so that all elements
                have same initial parameters but can be changed independently.
                ``share`` means all the elements which consist the resulting
                :class:`~torch.nn.Sequential` object are same object because
                they are shallow-copied, so that all parameters of elements
                are shared with each other.
        """
        if n_repeat <= 0:
            return ExtendedSequential()
        if mode not in ['copy', 'share', 'init']:
            raise ValueError(
                "The 'mode' argument should be either 'init','copy', or 'share'. But {} was given."
                .format(mode))
        model_list = []
        for _ in range(n_repeat):
            model_list.append(self._copy_model(mode))
        return ExtendedSequential(*model_list)


class UninitializedParameter(torch.nn.Parameter):

    def __repr__(self):
        return 'Uninitialized lazy parameter'

    @property
    def is_leaf(self):
        frame = inspect.currentframe()
        if frame.f_back.f_globals['__package__'].startswith('torch.optim'):
            warnings.warn(
                """
    Use of uninitialized lazy parameter in Optimizer has been detected.
    Maybe you forgot to run forward before passing `module.parameters()` to the optimizer?"""
                )
        return True


class LazyInitializationMixin:
    """A mixin for modules that lazily initialize buffers and parameters.

    Unlike regular modules, subclasses of this module can initialize
    buffers and parameters outside of the constructor (``__init__``).
    This allows you to, for example, initialize parameters in ``forward``
    method to determine the shape of the weight based on the initial input.

    Be sure to run "dummy" forward once to initialize all parameters that
    should be trained, before passing ``module.parameters()`` to an optimizer;
    otherwise weights initialized after ``module.parameters()`` (e.g., in
    ``forward`` function) will never be trained.

    Note that lazy modules cannot validate if the shape is correct during
    deserialization.  Also note that the initial weights may become different
    from the original (non-lazy) module even if the random seed is manually
    configured, as the order of initialization is different from the original
    one; especially, ``module.cuda()`` may cause the initialization to run on
    a GPU.

    The default value of lazy buffers and parameters are ``torch.Tensor([])``
    and ``UninitializedParameter()``, respectively.
    """
    lazy_buffer_names = ()
    lazy_parameter_names = ()

    def __init__(self, *args, **kwargs):
        self._lazy_ready = False
        super().__init__(*args, **kwargs)
        for name in self.lazy_buffer_names:
            self.register_buffer(name, torch.Tensor([]))
        for name in self.lazy_parameter_names:
            self.register_parameter(name, UninitializedParameter())
        self._register_load_state_dict_pre_hook(self._lazy_load_hook)
        self._lazy_ready = True

    @property
    def lazy_parmeters_determined(self):
        """Returns if all lazy parameters are determined.

        Subclasses can perform parameters initialization after all lazy
        parameters are determined.  Note that this may be called during
        ``__init__``.
        """
        return self._lazy_ready and all([(not isinstance(getattr(self, x),
            UninitializedParameter)) for x in self.lazy_parameter_names])

    def state_dict(self, *args, **kwargs):
        """Returns a dictionary containing a whole state of the module.

        This function overrides the default behavior to exclude uninitialized
        parameter from serialization.  This is needed because we need to
        discriminate lazy parameters (``UninitializedParameter()`) and
        initialized empty parameters (``torch.nn.Parameter(torch.Tensor())``)
        during deserialization.

        See comments of ``_lazy_load_hook`` for details.
        """
        destination = super().state_dict(*args, **kwargs)
        for name in self.lazy_parameter_names:
            if isinstance(getattr(self, name), UninitializedParameter):
                del destination[name]
        return destination

    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs):
        """load_state_dict pre-hook function for lazy buffers and parameters.

        The purpose of this hook is to adjust the current state and/or
        ``state_dict`` being loaded so that a module instance serialized in
        both un/initialized state can be deserialized onto both un/initialized
        module instance.

        See comment in ``torch.nn.Module._register_load_state_dict_pre_hook``
        for the details of the hook specification.
        """
        for name in self.lazy_buffer_names:
            self.register_buffer(name, state_dict[prefix + name])
        for name in self.lazy_parameter_names:
            key = prefix + name
            if key in state_dict:
                self.register_parameter(name, torch.nn.Parameter(state_dict
                    [key]))
            else:
                param = UninitializedParameter()
                self.register_parameter(name, param)
                state_dict[key] = param


class _LazyConvNd(LazyInitializationMixin):
    lazy_parameter_names = 'weight',

    def __init__(self, in_channels, *args, **kwargs):
        super().__init__(in_channels or 0, *args, **kwargs)
        if in_channels is None:
            self.in_channels = None
            self.weight = UninitializedParameter()

    def forward(self, input):
        if isinstance(self.weight, UninitializedParameter):
            self.in_channels = input.shape[1]
            if self.transposed:
                shape = (self.in_channels, self.out_channels // self.groups,
                    *self.kernel_size)
            else:
                shape = (self.out_channels, self.in_channels // self.groups,
                    *self.kernel_size)
            self.weight = torch.nn.Parameter(self.weight.new_empty(*shape))
            self.reset_parameters()
        return super().forward(input)

    def reset_parameters(self):
        if self.lazy_parmeters_determined:
            super().reset_parameters()


class LazyConv1d(_LazyConvNd, torch.nn.Conv1d):
    """Conv1d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyConv2d(_LazyConvNd, torch.nn.Conv2d):
    """Conv2d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyConv3d(_LazyConvNd, torch.nn.Conv3d):
    """Conv3d module with lazy weight initialization.

    When ``in_channels`` is ``None``, it is determined at the first time of
    the forward step.
    """
    pass


class LazyLinear(LazyInitializationMixin, torch.nn.Linear):
    """Linear module with lazy weight initialization.

    When ``in_features`` is ``None``, it is determined at the first time of
    the forward step.
    """
    lazy_parameter_names = 'weight',

    def __init__(self, in_features, *args, **kwargs):
        super().__init__(in_features or 0, *args, **kwargs)
        if in_features is None:
            self.in_features = None
            self.weight = UninitializedParameter()

    def forward(self, input):
        if isinstance(self.weight, UninitializedParameter):
            self.in_features = input.shape[-1]
            self.weight = torch.nn.Parameter(self.weight.new_empty(self.
                out_features, self.in_features))
            self.reset_parameters()
        return super().forward(input)

    def reset_parameters(self):
        if self.lazy_parmeters_determined:
            super().reset_parameters()


class _MyFunc(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('const', torch.full((in_features,), 1))
        self._reset_params()

    def forward(self, input):
        return F.linear(input + self.const, self.weight)

    def _reset_params(self):
        self.weight.data.uniform_(-0.1, 0.1)


class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.args = []

    def forward(self, x):
        self.args.append(x)
        ppe.reporting.report({'loss': x.sum()}, self)


class DummyModelTwoArgs(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.args = []

    def forward(self, x, y):
        self.args.append((x, y))
        ppe.reporting.report({'loss': x.sum() + y.sum()}, self)


class IgniteDummyModel(torch.nn.Module):

    def __init__(self):
        super(IgniteDummyModel, self).__init__()
        self.count = 0.0

    def forward(self, *args):
        ppe.reporting.report({'x': self.count}, self)
        self.count += 1.0
        return 0.0


class _StateDictObj:

    def __init__(self, *, state_dict=None, state_dict_to_be_loaded=None):
        super().__init__()
        self.called_load_state_dict = 0
        self._state_dict = state_dict
        self._state_dict_to_be_loaded = state_dict_to_be_loaded

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self.called_load_state_dict += 1
        assert state_dict is self._state_dict_to_be_loaded


class _StateDictModel(_StateDictObj, torch.nn.Module):

    def forward(self, *args):
        pass


class Wrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self._wrapper_module = model
        self.accessed = False

    def wrapper_module(self):
        self.accessed = True
        return self._wrapper_module


class _StateDictModel(_StateDictObj, nn.Module):

    def forward(self, *args):
        pass


class Wrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self._wrapper_module = model
        self.accessed = False

    def wrapper_module(self):
        self.accessed = True
        return self._wrapper_module


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pfnet_pytorch_pfn_extras(_paritybench_base):
    pass
    def test_000(self):
        self._check(ExtendedSequential(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(LazyConv1d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(LazyConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(LazyConv3d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 64, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(LazyLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(_MyFunc(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(_StateDictModel(*[], **{}), [], {})

