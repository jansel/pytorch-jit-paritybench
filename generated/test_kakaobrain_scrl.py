import sys
_module = sys.modules[__name__]
del sys
augment = _module
gaussian_blur = _module
getters = _module
normalize = _module
transforms = _module
data = _module
dataloader = _module
datasets = _module
getters = _module
distributed = _module
comm = _module
launch = _module
main = _module
models = _module
backbone = _module
box_generator = _module
heads = _module
optim = _module
getters = _module
optim = _module
scheduler = _module
trainer = _module
base = _module
helper = _module
linear_eval = _module
result_pack = _module
trainer = _module
utils = _module
color = _module
config = _module
exception_logger = _module
logger = _module
parser = _module
progress_disp = _module
stderr_redirector = _module
utils = _module
watch = _module

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


import random


import torch


from torchvision import transforms


import torch.nn as nn


import numpy as np


from typing import NamedTuple


from typing import List


from typing import Tuple


from functools import wraps


import torch.nn.functional as F


from torchvision.transforms import Resize


from torchvision.transforms import CenterCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import ColorJitter


from torchvision.transforms import RandomGrayscale


from torchvision.transforms import ToTensor


import torchvision.transforms.functional as F


import torch.utils.data


import logging


from torchvision import datasets


from torch.utils.data._utils.collate import default_collate


from torch.utils.data.distributed import DistributedSampler


import functools


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel


import torch.multiprocessing as mp


import torch.nn


import torchvision.models as models


import torchvision.ops as ops


from torch import nn


from copy import deepcopy


from itertools import chain


from torch.optim.lr_scheduler import CosineAnnealingLR


import warnings


from torch.optim import Optimizer


from torch.optim.lr_scheduler import ReduceLROnPlateau


import copy


import math


from collections import OrderedDict


from enum import Enum


from typing import Dict


from torch.utils.tensorboard import SummaryWriter


from torch.autograd import enable_grad


from torch.cuda.amp import autocast


from torch.utils.data import DataLoader


import pandas as pd


import collections


from types import SimpleNamespace


from typing import Any


from collections import defaultdict


class TwoLayerLinearHead(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerLinearHead, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU(inplace=True), nn.Linear(hidden_size, output_size, bias=False))

    @classmethod
    def init_projector_from_config(cls, cfg):
        return cls(input_size=cfg.network.proj_head.input_size, hidden_size=cfg.network.proj_head.hidden_size, output_size=cfg.network.proj_head.output_size)

    @classmethod
    def init_predictor_from_config(cls, cfg):
        return cls(input_size=cfg.network.proj_head.output_size, hidden_size=cfg.network.proj_head.hidden_size, output_size=cfg.network.proj_head.output_size)

    def forward(self, x):
        return self.net(x)


class Backbone(torch.nn.Module):

    def __init__(self, name, proj_head_kwargs, scrl_kwargs, trainable=True):
        super(Backbone, self).__init__()
        assert name in ['resnet50', 'resnet101'], 'only supports resnet50 and resnet101 for now.'
        self.scrl_enabled = scrl_kwargs.enabled
        self.trainable = trainable
        network = eval(f'models.{name}')()
        self.encoder = torch.nn.Sequential(*list(network.children())[:-1])
        if self.trainable and self.scrl_enabled:
            roi_out_size = (scrl_kwargs.pool_size,) * 2
            self.roi_align = ops.RoIAlign(output_size=roi_out_size, sampling_ratio=scrl_kwargs.sampling_ratio, spatial_scale=scrl_kwargs.spatial_scale, aligned=scrl_kwargs.detectron_aligned)
        if self.trainable:
            self.projector = TwoLayerLinearHead(**proj_head_kwargs)

    @classmethod
    def init_from_config(cls, cfg):
        return cls(name=cfg.network.name, proj_head_kwargs=cfg.network.proj_head, scrl_kwargs=cfg.network.scrl, trainable=cfg.train.enabled)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, boxes=None, no_projection=False):
        """
        Args:
            x: augmented(randomly cropped)images.
            boxes: boxes coordinates to be pooled.
            no_projection: ignore the projection layer (for evaluation)
        Returns:
            p: after projection / roi_p: RoI-aligned feature after projection
            h: before projection
        """
        for n, layer in enumerate(self.encoder):
            x = layer(x)
            if n == len(self.encoder) - 2:
                h_pre_gap = x
        h = x.squeeze()
        if not self.trainable or no_projection:
            return h
        if self.scrl_enabled:
            assert boxes is not None
            roi_h = self.roi_align(h_pre_gap, boxes).squeeze()
            roi_p = self.projector(roi_h)
            return roi_p, h
        else:
            p = self.projector(h)
            return p, h


class SingleLayerLinearHead(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(SingleLayerLinearHead, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    @classmethod
    def init_evaluator_from_config(cls, cfg, num_classes):
        return cls(input_size=cfg.network.proj_head.input_size, output_size=num_classes)

    def forward(self, x):
        x = self.linear(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SingleLayerLinearHead,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TwoLayerLinearHead,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_kakaobrain_scrl(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

