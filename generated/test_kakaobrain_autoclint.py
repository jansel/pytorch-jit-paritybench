import sys
_module = sys.modules[__name__]
del sys
architectures = _module
resnet = _module
model = _module
skeleton = _module
data = _module
augmentations = _module
dataloader = _module
dataset = _module
stratified_sampler = _module
transforms = _module
nn = _module
modules = _module
hooks = _module
loss = _module
profile = _module
wrappers = _module
optim = _module
optimizers = _module
scheduler = _module
sgdw = _module
projects = _module
api = _module
logic = _module
others = _module
utils = _module
timer = _module

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
xrange = range
wraps = functools.wraps


import logging


from collections import OrderedDict


import torch


import torchvision.models as models


from torch.utils import model_zoo


from torchvision.models.resnet import BasicBlock


from torchvision.models.resnet import model_urls


from torchvision.models.resnet import Bottleneck


import random


import tensorflow as tf


import torchvision as tv


import numpy as np


from torch.utils.data import Dataset


from abc import ABC


from collections import defaultdict


from torch.utils.data import Sampler


from torch import nn


from functools import wraps


from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import StratifiedShuffleSplit


class MoveToHook(nn.Module):

    @staticmethod
    def to(tensors, device, half=False):
        for t in tensors:
            if isinstance(t, (tuple, list)):
                MoveToHook
            if not isinstance(t, torch.Tensor):
                continue
            t.data = t.data
            if half:
                if t.is_floating_point():
                    t.data = t.data.half()

    @staticmethod
    def get_forward_pre_hook(device, half=False):

        def hook(module, inputs):
            _ = module
            MoveToHook
        return hook


class CrossEntropyLabelSmooth(torch.nn.Module):

    def __init__(self, num_classes, epsilon=0.1, sparse_target=True, reduction='avg'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.sparse_target = sparse_target
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, input, target):
        log_probs = self.logsoftmax(input)
        if self.sparse_target:
            targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        else:
            targets = target
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = -targets * log_probs
        if self.reduction == 'avg':
            loss = loss.mean(0).sum()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BinaryCrossEntropyLabelSmooth(torch.nn.BCEWithLogitsLoss):

    def __init__(self, num_classes, epsilon=0.1, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BinaryCrossEntropyLabelSmooth, self).__init__(weight, size_average, reduce, reduction, pos_weight)
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, input, target):
        target = (1 - self.epsilon) * target + self.epsilon
        return super(BinaryCrossEntropyLabelSmooth, self).forward(input, target)


class ToDevice(torch.nn.Module):

    def __init__(self):
        super(ToDevice, self).__init__()
        self.register_buffer('buf', torch.zeros(1, dtype=torch.float32))

    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], (tuple, list)):
            xs = xs[0]
        device = self.buf.device
        out = []
        for x in xs:
            if x is not None and x.device != device:
                out.append(x)
            else:
                out.append(x)
        return out[0] if len(xs) == 1 else tuple(out)


class CopyChannels(torch.nn.Module):

    def __init__(self, multiple=3, dim=1):
        super(CopyChannels, self).__init__()
        self.multiple = multiple
        self.dim = dim

    def forward(self, x):
        return torch.cat([x for _ in range(self.multiple)], dim=self.dim)


class Normalize(torch.nn.Module):

    def __init__(self, mean, std, inplace=False):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.tensor([mean], dtype=torch.float32)[(None), :, (None), (None)])
        self.register_buffer('std', torch.tensor([std], dtype=torch.float32)[(None), :, (None), (None)])
        self.inplace = inplace

    def forward(self, x):
        if not self.inplace:
            x = x.clone()
        x.sub_(self.mean).div_(self.std)
        return x


class Reshape(torch.nn.Module):

    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Flatten(torch.nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class SplitTime(torch.nn.Module):

    def __init__(self, times):
        super(SplitTime, self).__init__()
        self.times = times

    def forward(self, x):
        batch, channels, height, width = x.shape
        return x.view(-1, self.times, channels, height, width)


class Permute(torch.nn.Module):

    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class Cutout(torch.nn.Module):

    def __init__(self, ratio=0.0):
        super(Cutout, self).__init__()
        self.ratio = ratio

    def forward(self, input):
        batch, channel, height, width = input.shape
        w = int(width * self.ratio)
        h = int(height * self.ratio)
        if self.training and w > 0 and h > 0:
            x = np.random.randint(width, size=(batch,))
            y = np.random.randint(height, size=(batch,))
            x1s = np.clip(x - w // 2, 0, width)
            x2s = np.clip(x + w // 2, 0, width)
            y1s = np.clip(y - h // 2, 0, height)
            y2s = np.clip(y + h // 2, 0, height)
            mask = torch.ones_like(input)
            for idx, (x1, x2, y1, y2) in enumerate(zip(x1s, x2s, y1s, y2s)):
                mask[(idx), :, y1:y2, x1:x2] = 0.0
            input = input * mask
        return input


class Mul(torch.nn.Module):

    def __init__(self, weight):
        super(Mul, self).__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


def decorator_tuple_to_args(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        if len(args) == 2 and isinstance(args[1], (tuple, list)):
            args[1:] = list(args[1])
        return func(*args, **kwargs)
    return wrapper


class Concat(torch.nn.Module):

    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    @decorator_tuple_to_args
    def forward(self, *xs):
        return torch.cat(xs, dim=self.dim)


class MergeSum(torch.nn.Module):

    @decorator_tuple_to_args
    def forward(self, *xs):
        return torch.sum(torch.stack(xs), dim=0)


class MergeProd(torch.nn.Module):

    @decorator_tuple_to_args
    def forward(self, *xs):
        return xs[0] * xs[1]


class Choice(torch.nn.Module):

    def __init__(self, idx=0):
        super(Choice, self).__init__()
        self.idx = idx

    @decorator_tuple_to_args
    def forward(self, *xs):
        return xs[self.idx]


class Toggle(torch.nn.Module):

    def __init__(self, module):
        super(Toggle, self).__init__()
        self.module = module
        self.on = True

    def forward(self, x):
        return self.module(x) if self.on else x


class Split(torch.nn.Module):

    def __init__(self, *modules):
        super(Split, self).__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            for key, module in modules[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(modules):
                self.add_module(str(idx), module)

    def forward(self, x):
        return tuple([m(x) for m in self._modules.values()])


class DropPath(torch.nn.Module):

    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self._half = False

    def forward(self, x):
        if self.training and self.drop_prob > 0.0:
            shape = list(x.shape[:1]) + [(1) for _ in x.shape[1:]]
            keep_prob = 1.0 - self.drop_prob
            mask = torch.FloatTensor(*shape).bernoulli_(keep_prob)
            if self._half:
                mask = mask.half()
            x.div_(keep_prob)
            x.mul_(mask)
        return x

    def half(self):
        self._half = True

    def float(self):
        self._half = False


class DelayedPass(torch.nn.Module):

    def __init__(self):
        super(DelayedPass, self).__init__()
        self.register_buffer('keep', None)

    def forward(self, x):
        rv = self.keep
        self.keep = x
        return rv


class Reader(torch.nn.Module):

    def __init__(self, x=None):
        super(Reader, self).__init__()
        self.x = x

    def forward(self, x):
        return self.x


class KeepByPass(torch.nn.Module):

    def __init__(self):
        super(KeepByPass, self).__init__()
        self._reader = Reader()
        self.info = {}

    @property
    def x(self):
        return self._reader.x

    def forward(self, x):
        self._reader.x = x
        return x

    def reader(self):
        return self._reader


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BinaryCrossEntropyLabelSmooth,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CopyChannels,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Cutout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DelayedPass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DropPath,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KeepByPass,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mul,
     lambda: ([], {'weight': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Normalize,
     lambda: ([], {'mean': 4, 'std': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Reader,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Split,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SplitTime,
     lambda: ([], {'times': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ToDevice,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (Toggle,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kakaobrain_autoclint(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

