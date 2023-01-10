import sys
_module = sys.modules[__name__]
del sys
FKD_train = _module
extensions = _module
data_parallel = _module
kd_loss = _module
teacher_wrapper = _module
hubconf = _module
imagenet = _module
inference = _module
loss = _module
models = _module
blocks = _module
discriminator = _module
model_factory = _module
opts = _module
test = _module
train = _module
utils = _module
utils_FKD = _module

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


import logging


import time


import math


import numpy as np


import torch


from torch import nn


from torch.nn.modules import loss


from torch.nn import functional as F


import random


from torch.utils import data as data_utils


from torchvision import datasets as torch_datasets


from torchvision import transforms


from torchvision.transforms import InterpolationMode


import warnings


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.distributed as dist


import torch.optim


import torch.multiprocessing as mp


import torch.utils.data


import torch.utils.data.distributed


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import torchvision.models as models


import torch.nn.functional as F


import collections


import torch.distributed


import torchvision


from torchvision.ops import roi_align


from torchvision.transforms import functional as t_F


from torchvision.datasets.folder import ImageFolder


class DataParallel(nn.DataParallel):
    """An extension of nn.DataParallel.

    The only extensions are:
        1) If an attribute is missing in an object of this class, it will look
            for it in the wrapped module. This is useful for getting `LR_REGIME`
            of the wrapped module for example.
        2) state_dict() of this class calls the wrapped module's state_dict(),
            hence the weights can be transferred from a data parallel wrapped
            module to a single gpu module.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            underlying_module = super().__getattr__('module')
            return getattr(underlying_module, name)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)


class KLLoss(loss._Loss):
    """The KL-Divergence loss for the model and soft labels output.

    output must be a pair of (model_output, soft_labels), both NxC tensors.
    The rows of soft_labels must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, output, target):
        if not self.training:
            return F.cross_entropy(output, target)
        assert type(output) == tuple and len(output) == 2 and output[0].size() == output[1].size(), 'output must a pair of tensors of same size.'
        model_output, soft_labels = output
        if soft_labels.requires_grad:
            raise ValueError('soft labels should not require gradients.')
        model_output_log_prob = F.log_softmax(model_output, dim=1)
        del model_output
        soft_labels = soft_labels.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(soft_labels, model_output_log_prob)
        cross_entropy_loss = cross_entropy_loss.mean()
        model_output_log_prob = model_output_log_prob.squeeze(2)
        return cross_entropy_loss, model_output_log_prob


class ModelDistillationWrapper(nn.Module):
    """Convenient wrapper class to train a model with soft label ."""

    def __init__(self, model, teacher):
        super().__init__()
        self.model = model
        self.teachers_0 = teacher
        self.combine = True
        for model in self.teachers_0:
            for param in model.parameters():
                param.requires_grad = False
        self.false = False

    @property
    def LR_REGIME(self):
        return self.model.LR_REGIME

    def state_dict(self):
        return self.model.state_dict()

    def forward(self, input, before=False):
        if self.training:
            if len(self.teachers_0) == 3 and self.combine == False:
                index = [0, 1, 1, 2, 2]
                idx = random.randint(0, 4)
                soft_labels_ = self.teachers_0[index[idx]](input)
                soft_labels = F.softmax(soft_labels_, dim=1)
            elif self.combine:
                soft_labels_ = [torch.unsqueeze(self.teachers_0[idx](input), dim=2) for idx in range(len(self.teachers_0))]
                soft_labels_softmax = [F.softmax(i, dim=1) for i in soft_labels_]
                soft_labels_ = torch.cat(soft_labels_, dim=2).mean(dim=2)
                soft_labels = torch.cat(soft_labels_softmax, dim=2).mean(dim=2)
            else:
                idx = random.randint(0, len(self.teachers_0) - 1)
                soft_labels_ = self.teachers_0[idx](input)
                soft_labels = F.softmax(soft_labels_, dim=1)
            model_output = self.model(input)
            if before:
                return model_output, soft_labels, soft_labels_
            return model_output, soft_labels
        else:
            return self.model(input)


class betweenLoss(nn.Module):

    def __init__(self, gamma=[1, 1, 1, 1, 1, 1], loss=nn.L1Loss()):
        super(betweenLoss, self).__init__()
        self.gamma = gamma
        self.loss = loss

    def forward(self, outputs, targets):
        assert len(outputs)
        assert len(outputs) == len(targets)
        length = len(outputs)
        res = sum([(self.gamma[i] * self.loss(outputs[i], targets[i])) for i in range(length)])
        return res


class discriminatorLoss(nn.Module):

    def __init__(self, models, loss=nn.BCEWithLogitsLoss()):
        super(discriminatorLoss, self).__init__()
        self.models = models
        self.loss = loss

    def forward(self, outputs, targets):
        inputs = [torch.cat((i, j), 0) for i, j in zip(outputs, targets)]
        inputs = torch.cat(inputs, 1)
        batch_size = inputs.size(0)
        target = torch.FloatTensor([[1, 0] for _ in range(batch_size // 2)] + [[0, 1] for _ in range(batch_size // 2)])
        target = target
        output = self.models(inputs)
        res = self.loss(output, target)
        return res


class discriminatorFakeLoss(nn.Module):

    def forward(self, outputs, targets):
        res = (0 * outputs[0]).sum()
        return res


class Conv2dBnRelu(nn.Module):
    """A commonly used building block: Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, pooling=None, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pooling = pooling
        self.activation = activation

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.pooling is not None:
            x = self.pooling(x)
        return self.activation(x)


class LinearBnRelu(nn.Module):
    """A commonly used building block: FC -> BN -> ReLU"""

    def __init__(self, in_features, out_features, bias=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.bn(self.linear(x)))


class Discriminator(nn.Module):

    def __init__(self, outputs_size, K=2):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size // K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv2 = nn.Conv2d(in_channels=outputs_size, out_channels=outputs_size // K, kernel_size=1, stride=1, bias=True)
        outputs_size = outputs_size // K
        self.conv3 = nn.Conv2d(in_channels=outputs_size, out_channels=2, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = x[:, :, None, None]
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = out.view(out.size(0), -1)
        return out


class RandomHorizontalFlip_FKD(torch.nn.Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img, coords, status):
        if status == True:
            return t_F.hflip(img)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv2dBnRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Discriminator,
     lambda: ([], {'outputs_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (KLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearBnRelu,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (RandomHorizontalFlip_FKD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), 0], {}),
     False),
    (betweenLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (discriminatorFakeLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_szq0214_MEAL_V2(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

