import sys
_module = sys.modules[__name__]
del sys
conf = _module
examples = _module
custom_callback_event = _module
train_mnist = _module
train_with_metrics_in_callback = _module
train_with_metrics_in_loop = _module
hf_bert_glue_mrpc = _module
vision = _module
faster_rcnn = _module
calculate_map_callback = _module
coco_evaluator = _module
dataset = _module
model = _module
trainer = _module
train_cars = _module
transfer_learning = _module
pets_finetune = _module
progressive_resizing = _module
pytorch_tutorial_finetune = _module
using_timm_components = _module
all_timm_components = _module
train_mixup_ema = _module
pytorch_accelerated = _module
_version = _module
callbacks = _module
finetuning = _module
run_config = _module
schedulers = _module
cosine_scheduler = _module
scheduler_base = _module
tracking = _module
trainer = _module
utils = _module
setup = _module
test = _module
scheduler = _module
test_cosine_scheduler = _module
test_scheduler_base = _module
test_finetuning = _module
test_placeholders = _module
test_run_history = _module
test_trainer = _module
test_utils = _module
versioneer = _module

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


from torch import nn


from torch import optim


from torch.utils.data import random_split


from torchvision import transforms


from torchvision.datasets import MNIST


from functools import partial


import torch


import pandas as pd


import numpy as np


from torch.utils.data import Dataset


from torchvision.models.detection.anchor_utils import AnchorGenerator


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn


from collections import defaultdict


import random


import re


from torch.optim.lr_scheduler import OneCycleLR


from torchvision.transforms import Compose


from torchvision.transforms import Normalize


from torchvision.transforms import RandomResizedCrop


from torchvision.transforms import Resize


from torchvision.transforms import ToTensor


from collections import namedtuple


from torchvision import datasets


from torch.optim import lr_scheduler


from torchvision import models


from torchvision.models import ResNet18_Weights


import inspect


import itertools


import logging


import time


from abc import ABC


from typing import List


from numbers import Number


from typing import Union


import math


from typing import Callable


from abc import abstractmethod


from typing import Iterable


from enum import Enum


from torch.utils.data import DataLoader


from copy import deepcopy


from functools import wraps


from torch import Tensor


class MNISTModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(nn.Linear(in_features=784, out_features=128), nn.ReLU(), nn.Linear(in_features=128, out_features=64), nn.ReLU(), nn.Linear(in_features=64, out_features=10))

    def forward(self, x):
        return self.main(x.view(x.shape[0], -1))


class ModelEma(nn.Module):
    """
    Maintains a moving average of everything in the model state_dict (parameters and buffers), based on the ideas
    from https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage.

    This class maintains a copy of the model that we are training. However,
    rather than updating all of the parameters of this model after every update step,
    we set these parameters using a linear combination of the existing parameter values and the updated values

    .. Note:: It is important to note that this class is sensitive to where it is initialised.
        During distributed training, it should be applied before before the conversion to :class:`~torch.nn.SyncBatchNorm`
        takes place and before the :class:`torch.nn.parallel.DistributedDataParallel` wrapper is used!
    """

    def __init__(self, model, decay=0.9999):
        """
        Create an instance of :class:`torch.nn.Module` to maintain an exponential moving average of the weights of
        the given model.

        This is done using the following formula:

        `updated_EMA_model_weights = decay * EMA_model_weights + (1. — decay) * updated_model_weights`

        where the decay is a parameter that we set.

        :param model: the subclass of :class: `torch.nn.Module` that we are training. This is the model that will be updated in our training loop as normal.
        :param decay: the amount of decay to use, which determines how much of the previous state will be maintained. The TensorFlow documentation suggests that reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999

        """
        super().__init__()
        self.module = deepcopy(model)
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.module.eval()
        self.decay = decay

    def update_fn(self, ema_model_weights, updated_model_weights):
        return self.decay * ema_model_weights + (1.0 - self.decay) * updated_model_weights

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                updated_v = update_fn(ema_v, model_v)
                ema_v.copy_(updated_v)

    def update(self, model):
        self._update(model, update_fn=self.update_fn)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class TestModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(100, 100)
        self.block_1 = nn.Sequential(nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU())
        self.block_2 = nn.Sequential(nn.Linear(100, 100), nn.BatchNorm1d(100), nn.Sequential(nn.Linear(100, 100), nn.BatchNorm1d(100), nn.ReLU()))
        self.output_1 = nn.Linear(100, 10)
        self.output_2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.input(x)
        x = self.block_1(x)
        x = self.block_2(x)
        out_1 = self.output_1(x)
        out_2 = self.output_2(x)
        return out_1, out_2


class SimpleModel(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SimpleModel,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Chris_hughes10_pytorch_accelerated(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

