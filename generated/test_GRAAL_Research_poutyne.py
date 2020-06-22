import sys
_module = sys.modules[__name__]
del sys
conf = _module
poutyne = _module
framework = _module
callbacks = _module
_utils = _module
best_model_restore = _module
checkpoint = _module
clip_grad = _module
delay = _module
earlystopping = _module
logger = _module
lr_scheduler = _module
periodic = _module
policies = _module
progress = _module
terminate_on_nan = _module
exceptions = _module
experiment = _module
iterators = _module
metrics = _module
epoch_metrics = _module
base = _module
fscores = _module
metrics = _module
utils = _module
model = _module
optimizers = _module
warning_manager = _module
layers = _module
utils = _module
version = _module
setup = _module
tests = _module
context = _module
test_best_model_restore = _module
test_checkpoint = _module
test_delay = _module
test_earlystopping = _module
test_logger = _module
test_lr_scheduler = _module
test_lr_scheduler_checkpoint = _module
test_optimizer_checkpoint = _module
test_policies = _module
test_fscores = _module
test_model = _module
test_utils = _module

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


from torch.nn.utils import clip_grad_norm_


from torch.nn.utils import clip_grad_value_


import warnings


import numpy as np


import torch


from abc import ABC


from abc import abstractmethod


import torch.nn as nn


import torch.nn.functional as F


from collections import defaultdict


from torch.utils.data import DataLoader


from math import ceil


class EpochMetric(ABC, nn.Module):
    """
    The abstract class representing a epoch metric which can be accumulated at each batch and calculated at the end
    of the epoch.
    """

    @abstractmethod
    def forward(self, y_pred, y_true):
        """
        To define the behavior of the metric when called.

        Args:
            y_pred: The prediction of the model.
            y_true: Target to evaluate the model.
        """
        pass

    @abstractmethod
    def get_metric(self):
        """
        Compute and return the metric.
        """
        pass


class Lambda(nn.Module):
    """
    Applies a function to the input tensor.

    Args:
        func (Callable[[~torch.Tensor], ~torch.Tensor]): The function to apply.

    Example:

        .. code-block:: python

            # Alternate version to the ``nn.Flatten`` module.
            my_flatten = Lambda(lambda x: x.flatten(1))

    """

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class MultiIOModel(nn.Module):
    """Model to test multiple inputs/outputs"""

    def __init__(self, num_input=2, num_output=2):
        super(MultiIOModel, self).__init__()
        inputs = []
        for _ in range(num_input):
            inputs.append(nn.Linear(1, 1))
        self.inputs = nn.ModuleList(inputs)
        outputs = []
        for _ in range(num_output):
            outputs.append(nn.Linear(num_input, 1))
        self.outputs = nn.ModuleList(outputs)

    def forward(self, *x):
        inp_to_cat = []
        for i, inp in enumerate(self.inputs):
            inp_to_cat.append(inp(x[i]))
        inp_cat = torch.cat(inp_to_cat, dim=1)
        outputs = []
        for out in self.outputs:
            outputs.append(out(inp_cat))
        outputs = outputs if len(outputs) > 1 else outputs[0]
        return outputs


class DictOutputModel(nn.Module):
    """Model to test multiple dictionnary output"""

    def __init__(self):
        super(DictOutputModel, self).__init__()
        self.input = nn.Linear(1, 1)
        self.output1 = nn.Linear(1, 1)
        self.output2 = nn.Linear(1, 1)

    def forward(self, x):
        out1 = self.output1(self.input(x))
        out2 = self.output2(self.input(x))
        return {'out1': out1, 'out2': out2}


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_GRAAL_Research_poutyne(_paritybench_base):
    pass
    def test_000(self):
        self._check(DictOutputModel(*[], **{}), [torch.rand([1, 1])], {})

    def test_001(self):
        self._check(Lambda(*[], **{'func': ReLU()}), [torch.rand([4, 4, 4, 4])], {})

