import sys
_module = sys.modules[__name__]
del sys
distributed_data_parallel = _module
tensorboard = _module
visdom_note = _module
conf = _module
setup = _module
tests = _module
callbacks = _module
imaging = _module
test_imaging = _module
test_inside_cnns = _module
test_aggregate_predictions = _module
test_between_class = _module
test_callbacks = _module
test_checkpointers = _module
test_csv_logger = _module
test_cutout = _module
test_decorators = _module
test_early_stopping = _module
test_gradient_clipping = _module
test_init = _module
test_label_smoothing = _module
test_live_loss_plot = _module
test_manifold_mixup = _module
test_mixup = _module
test_printer = _module
test_pycm = _module
test_sample_pairing = _module
test_tensor_board = _module
test_terminate_on_nan = _module
test_torch_scheduler = _module
test_unpack_state = _module
test_weight_decay = _module
metrics = _module
test_aggregators = _module
test_default = _module
test_lr = _module
test_metrics = _module
test_primitives = _module
test_roc_auc_score = _module
test_timer = _module
test_wrappers = _module
test_bases = _module
test_cv_utils = _module
test_end_to_end = _module
test_magics = _module
test_state = _module
test_trial = _module
torchbearer = _module
bases = _module
aggregate_predictions = _module
between_class = _module
checkpointers = _module
csv_logger = _module
cutout = _module
decorators = _module
early_stopping = _module
gradient_clipping = _module
inside_cnns = _module
init = _module
label_smoothing = _module
live_loss_plot = _module
lsuv = _module
manifold_mixup = _module
mixup = _module
printer = _module
pycm = _module
sample_pairing = _module
tensor_board = _module
terminate_on_nan = _module
torch_scheduler = _module
unpack_state = _module
weight_decay = _module
cv_utils = _module
magics = _module
aggregators = _module
default = _module
lr = _module
primitives = _module
roc_auc_score = _module
timer = _module
wrappers = _module
state = _module
trial = _module
version = _module

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


import torch.distributed as dist


import torch.nn as nn


import torch.optim as optim


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel as DDP


from torch import nn


import warnings


import torch.nn.functional as F


from torch.nn import Module


from torch.nn import Linear


import torch.nn.init as init


from torch.utils.data import DataLoader


from torch.distributions import Beta


import numpy as np


import torch.nn.init


from torch.distributions.beta import Beta


from collections import OrderedDict


from functools import partial


import copy


import math


import functools


import itertools


import torch.nn


from torch.utils.data import TensorDataset


from torch.optim import Optimizer


old_super = super


def super(_, obj):
    return old_super(obj.__class__, obj)


class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(100, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(3, 16, stride=2, kernel_size=3
            ), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 32, stride=2,
            kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 64,
            stride=2, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU())
        self.classifier = nn.Linear(576, 10)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 576)
        return self.classifier(x)


class SimpleModel(nn.Module):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(3, 16, stride=2, kernel_size=3
            ), nn.BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 32, stride=2,
            kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 64,
            stride=2, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU())
        self.classifier = nn.Linear(576, 10)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, 576)
        return self.classifier(x)


class TestModule(nn.Module):

    def __init__(self):
        super(TestModule, self).__init__()
        self.conv = nn.Conv1d(1, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.conv(x.view(-1, 1, 1))
        x = self.relu(x)
        x = self.bn(x)
        return x


class TestModule2(nn.Module):

    def __init__(self):
        super(TestModule2, self).__init__()
        self.layer1 = TestModule()

    def forward(self, x):
        return self.layer1(x)


class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.conv1 = nn.Conv1d(1, 1, 1)
        self.relu = nn.ReLU()
        self.layer1 = TestModule()
        self.layer2 = TestModule2()

    def forward(self, x):
        x = self.fc1(x)
        x = self.conv1(x.view(-1, 1, 1))
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Net(Module):

    def __init__(self, x):
        super(Net, self).__init__()
        self.pars = torch.nn.Parameter(x)

    def f(self):
        """
        function to be minimised:
        f(x) = (x[0]-5)^2 + x[1]^2 + (x[2]-1)^2
        Solution:
        x = [5,0,1]
        """
        out = torch.zeros_like(self.pars)
        out[0] = self.pars[0] - 5
        out[1] = self.pars[1]
        out[2] = self.pars[2] - 1
        return torch.sum(out ** 2)

    def forward(self, _):
        return self.f()


class _CAMWrapper(nn.Module):

    def __init__(self, input_size, base_model, transform=None):
        super(_CAMWrapper, self).__init__()
        self.base_model = base_model
        input_image = torch.zeros(input_size)
        self.input_image = nn.Parameter(input_image, requires_grad=True)
        self.transform = (lambda x: x) if transform is None else transform

    def forward(self, _, state):
        try:
            return self.base_model(self.transform(self.input_image.sigmoid(
                )).unsqueeze(0), state)
        except TypeError:
            return self.base_model(self.transform(self.input_image.sigmoid(
                )).unsqueeze(0))


class MockModel(torch.nn.Module):

    def forward(self, x, state=None):
        return None


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pytorchbearer_torchbearer(_paritybench_base):
    pass

    def test_000(self):
        self._check(ToyModel(*[], **{}), [torch.rand([784, 784])], {})

    def test_001(self):
        self._check(TestModule(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(TestModule2(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(TestModel(*[], **{}), [torch.rand([1, 1])], {})
    @_fails_compile()

    def test_004(self):
        self._check(MockModel(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
