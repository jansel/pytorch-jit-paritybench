import sys
_module = sys.modules[__name__]
del sys
kekas = _module
callbacks = _module
data = _module
keker = _module
loss = _module
metrics = _module
modules = _module
parallel = _module
transformations = _module
utils = _module
setup = _module

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


import warnings


from collections import defaultdict


from typing import Callable


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


from typing import Type


from typing import Union


import numpy as np


import torch


from torch.optim import SGD


from torch.optim import Optimizer


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch import nn


import torch.cuda.comm as comm


from torch.autograd import Function


from torch.nn.modules import Module


from torch.nn.parallel._functions import Broadcast


from torch.nn.parallel.parallel_apply import parallel_apply


from torch.nn.parallel.replicate import replicate


from torch.nn.parallel.scatter_gather import scatter_kwargs


import logging


from functools import reduce


from typing import Any


from typing import Hashable


class FocalLoss(nn.Module):

    def __init__(self, alpha=Union[Tuple[float, int], List], gamma: int=0,
        size_average: bool=True) ->None:
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input: torch.Tensor, target: torch.Tensor
        ) ->torch.Tensor:
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size()[0], -1)


class AdaptiveConcatPool2d(nn.Module):
    """Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."""

    def __init__(self, size: Optional[int]=None):
        """Output will be 2*size or 2 if size is None"""
        super().__init__()
        size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x: torch.Tensor) ->torch.Tensor:
        return torch.cat([self.mp(x), self.ap(x)], 1)


class DataParallelModel(Module):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelModel, self).__init__()
        if not torch.is_available():
            self.module = module
            self.device_ids = []
            return
        if device_ids is None:
            device_ids = list(range(torch.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return outputs

    @staticmethod
    def replicate(module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[:
            len(replicas)])


class Reduce(Function):

    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, *gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput[0])


def get_a_var(obj):
    if isinstance(obj, torch.tensor):
        return obj
    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.tensor):
                return result
    return None


def criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None,
    devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(*input, *target, **kwargs)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e
    if len(modules) > 1:
        threads = [threading.Thread(target=_worker, args=(i, module, input,
            target, kwargs, device)) for i, (module, input, target, kwargs,
            device) in enumerate(zip(modules, inputs, targets, kwargs_tup,
            devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], targets[0], kwargs_tup[0], devices[0]
            )
    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


class DataParallelCriterion(Module):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallelCriterion, self).__init__()
        if not torch.is_available():
            self.module = module
            self.device_ids = []
            return
        if device_ids is None:
            device_ids = list(range(torch.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module

    def forward(self, inputs, *targets, **kwargs):
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0], **kwargs[0])
        inputs = [(input,) for input in inputs]
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, targets, kwargs)
        return self.gather(outputs, self.output_device)

    @staticmethod
    def replicate(module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(self, replicas, inputs, targets, kwargs):
        return criterion_parallel_apply(replicas, inputs, targets, kwargs,
            self.device_ids[:len(replicas)])

    @staticmethod
    def gather(outputs, output_device):
        if not len(outputs[0].shape):
            outputs = [output.unsqueeze(0) for output in outputs]
        return Reduce.apply(*outputs) / len(outputs)


class ParameterModule(nn.Module):
    """Register a lone parameter `p` in a module."""

    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_belskikh_kekas(_paritybench_base):
    pass
    def test_000(self):
        self._check(AdaptiveConcatPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ParameterModule(*[], **{'p': 4}), [torch.rand([4, 4, 4, 4])], {})

