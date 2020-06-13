import sys
_module = sys.modules[__name__]
del sys
main = _module
main = _module
amoebanet = _module
genotype = _module
operations = _module
resnet = _module
bottleneck = _module
flatten_sequential = _module
unet = _module
flatten_sequential = _module
main = _module
main = _module
main = _module
main = _module
gpu_utils = _module
main = _module
tuplify_skips = _module
conf = _module
setup = _module
tests = _module
conftest = _module
test_api = _module
test_gpipe = _module
test_inspect_skip_layout = _module
test_leak = _module
test_portal = _module
test_stash_pop = _module
test_tracker = _module
test_verify_skippables = _module
test_balance = _module
test_bugs = _module
test_checkpoint = _module
test_copy = _module
test_deferred_batch_norm = _module
test_dependency = _module
test_gpipe = _module
test_inplace = _module
test_microbatch = _module
test_phony = _module
test_pipeline = _module
test_stream = _module
test_transparency = _module
test_worker = _module
torchgpipe = _module
__version__ = _module
balance = _module
blockpartition = _module
profile = _module
batchnorm = _module
checkpoint = _module
copy = _module
dependency = _module
gpipe = _module
microbatch = _module
phony = _module
pipeline = _module
skip = _module
layout = _module
namespace = _module
portal = _module
skippable = _module
tracker = _module
stream = _module
worker = _module
torchgpipe_balancing = _module

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


from typing import Any


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import cast


import torch


from torch import Tensor


from torch import nn


import torch.nn.functional as F


from torch.optim import RMSprop


from torch.optim import SGD


import torch.utils.data


from collections import OrderedDict


from typing import TYPE_CHECKING


from typing import Iterator


from typing import Union


from typing import Generator


from torch.optim.lr_scheduler import LambdaLR


from torch.utils.data import DataLoader


from collections import deque


from typing import Deque


import copy


from functools import partial


import torch.cuda


from copy import deepcopy


import torch.nn as nn


from typing import TypeVar


from torch.nn.modules.batchnorm import _BatchNorm


from torch import ByteTensor


import torch.autograd


from typing import Iterable


from typing import ClassVar


from typing import FrozenSet


from typing import Set


from typing import Type


class Classify(nn.Module):

    def __init__(self, channels_prev: int, num_classes: int):
        super().__init__()
        self.pool = nn.AvgPool2d(7)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(channels_prev, num_classes)

    def forward(self, states: Tuple[Tensor, Tensor]) ->Tensor:
        x, _ = states
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc(x)
        return x


class Stem(nn.Module):

    def __init__(self, channels: int) ->None:
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(3, channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, input: Tensor) ->Tensor:
        x = input
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def relu_conv_bn(in_channels: int, out_channels: int, kernel_size: int=1,
    stride: int=1, padding: int=0) ->nn.Module:
    return nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(in_channels,
        out_channels, kernel_size, stride, padding, bias=False), nn.
        BatchNorm2d(out_channels))


NORMAL_CONCAT = [0, 3, 4, 6]


REDUCTION_CONCAT = [4, 5, 6]


class Operation(nn.Module):
    """Includes the operation name into the representation string for
    debugging.
    """

    def __init__(self, name: str, module: nn.Module):
        super().__init__()
        self.name = name
        self.module = module

    def __repr__(self) ->str:
        return f'{self.__class__.__name__}[{self.name}]'

    def forward(self, *args: Any) ->Any:
        return self.module(*args)


class FactorizedReduce(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.pad = nn.ZeroPad2d((0, 1, 0, 1))
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=
            1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=
            1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) ->Tensor:
        x = input
        x = self.relu(x)
        x = torch.cat([self.conv1(x), self.conv2(self.pad(x[:, :, 1:, 1:]))
            ], dim=1)
        x = self.bn(x)
        return x


class stash:
    """The command to stash a skip tensor.

    ::

        def forward(self, input):
            yield stash('name', input)
            return f(input)

    Args:
        name (str): name of skip tensor
        input (torch.Tensor or None): tensor to pass to the skip connection

    """
    __slots__ = 'name', 'tensor'

    def __init__(self, name: str, tensor: Optional[Tensor]) ->None:
        self.name = name
        self.tensor = tensor


class pop:
    """The command to pop a skip tensor.

    ::

        def forward(self, input):
            skip = yield pop('name')
            return f(input) + skip

    Args:
        name (str): name of skip tensor

    Returns:
        the skip tensor previously stashed by another layer under the same name

    """
    __slots__ = 'name',

    def __init__(self, name: str) ->None:
        self.name = name


Tensors = Tuple[Tensor, ...]


TensorOrTensors = Union[Tensor, Tensors]


class Pass(nn.Module):

    def forward(self, input):
        return input


def is_recomputing() ->bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.

    .. seealso:: :ref:`Detecting Recomputation`

    """
    return thread_local.is_recomputing


TModule = TypeVar('TModule', bound=nn.Module)


class DeferredBatchNorm(_BatchNorm):
    """A BatchNorm layer tracks multiple micro-batches to update running
    statistics per mini-batch.
    """
    sum: Tensor
    sum_squares: Tensor

    def __init__(self, num_features: int, eps: float=1e-05, momentum:
        Optional[float]=0.1, affine: bool=True, chunks: int=1) ->None:
        super().__init__(num_features, eps, momentum, affine,
            track_running_stats=True)
        self.register_buffer('sum', torch.zeros_like(self.running_mean))
        self.register_buffer('sum_squares', torch.zeros_like(self.running_var))
        self.counter = 0
        self.tracked = 0
        self.chunks = chunks

    def _check_input_dim(self, input: Tensor) ->None:
        if input.dim() <= 2:
            raise ValueError('expected at least 3D input (got %dD input)' %
                input.dim())

    def _track(self, input: Tensor) ->bool:
        """Tracks statistics of a micro-batch."""
        dim = [0]
        dim.extend(range(2, input.dim()))
        with torch.no_grad():
            self.sum += input.sum(dim)
            self.sum_squares += (input ** 2).sum(dim)
        size = input.size().numel() // input.size(1)
        self.counter += size
        self.tracked += 1
        return self.tracked == self.chunks

    def _commit(self) ->None:
        """Updates the running statistics of a mini-batch."""
        exponential_average_factor = 0.0
        self.num_batches_tracked += 1
        if self.momentum is None:
            exponential_average_factor = 1.0 / float(self.num_batches_tracked)
        else:
            exponential_average_factor = self.momentum
        mean = self.sum / self.counter
        var = self.sum_squares / self.counter - mean ** 2
        m = exponential_average_factor
        self.running_mean *= 1 - m
        self.running_mean += mean * m
        self.running_var *= 1 - m
        self.running_var += var * m
        self.sum.zero_()
        self.sum_squares.zero_()
        self.counter = 0
        self.tracked = 0

    def forward(self, input: Tensor) ->Tensor:
        if not self.training:
            return F.batch_norm(input, running_mean=self.running_mean,
                running_var=self.running_var, weight=self.weight, bias=self
                .bias, training=False, momentum=0.0, eps=self.eps)
        if not is_recomputing():
            tracked_enough = self._track(input)
            if tracked_enough:
                self._commit()
        return F.batch_norm(input, running_mean=None, running_var=None,
            weight=self.weight, bias=self.bias, training=True, momentum=0.0,
            eps=self.eps)

    @classmethod
    def convert_deferred_batch_norm(cls, module: TModule, chunks: int=1
        ) ->TModule:
        """Converts a :class:`nn.BatchNorm` or underlying
        :class:`nn.BatchNorm`s into :class:`DeferredBatchNorm`::

            from torchvision.models.resnet import resnet101
            from torchgpipe.batchnorm import DeferredBatchNorm
            model = resnet101()
            model = DeferredBatchNorm.convert_deferred_batch_norm(model)

        """
        if isinstance(module, DeferredBatchNorm) and module.chunks is chunks:
            return cast(TModule, module)
        module_output: nn.Module = module
        if isinstance(module, _BatchNorm) and module.track_running_stats:
            module_output = DeferredBatchNorm(module.num_features, module.
                eps, module.momentum, module.affine, chunks)
            if module.affine:
                module_output.register_parameter('weight', module.weight)
                module_output.register_parameter('bias', module.bias)
            module_output.register_buffer('running_mean', module.running_mean)
            module_output.register_buffer('running_var', module.running_var)
            module_output.register_buffer('num_batches_tracked', module.
                num_batches_tracked)
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_deferred_batch_norm(
                child, chunks))
        return cast(TModule, module_output)


def recommend_auto_balance(message: str) ->str:
    """Expands a message with recommendation to :mod:`torchgpipe.balance`."""
    return f"""{message}

If your model is still under development, its optimal balance would change
frequently. In this case, we highly recommend 'torchgpipe.balance' for naive
automatic balancing:

  from torchgpipe import GPipe
  from torchgpipe.balance import balance_by_time

  partitions = torch.cuda.device_count()
  sample = torch.empty(...)
  balance = balance_by_time(partitions, model, sample)

  model = GPipe(model, balance, ...)
"""


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_kakaobrain_torchgpipe(_paritybench_base):
    pass
    def test_000(self):
        self._check(Stem(*[], **{'channels': 4}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(FactorizedReduce(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Pass(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(DeferredBatchNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

