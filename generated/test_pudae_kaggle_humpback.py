import sys
_module = sys.modules[__name__]
del sys
datasets = _module
dataset_factory = _module
identification = _module
landmark = _module
ensemble_landmarks = _module
inference_landmark = _module
inference_similarity = _module
losses = _module
loss_factory = _module
make_submission = _module
models = _module
model_factory = _module
optimizers = _module
optimizer_factory = _module
post_processing = _module
schedulers = _module
scheduler_factory = _module
swa = _module
tasks = _module
identifier = _module
landmark_detector = _module
task_factory = _module
train = _module
transforms = _module
align_transform = _module
landmark_transform = _module
transform_factory = _module
utils = _module
checkpoint = _module
config = _module
swa = _module

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


from torch.utils.data import DataLoader


import torch.utils.data.sampler


import random


import numpy as np


import scipy.misc as misc


from torch.utils.data.dataset import Dataset


import math


from collections import defaultdict


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


import types


class ArcModule(nn.Module):

    def __init__(self, in_features, out_features, s=65, m=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)
        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]
        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size())
        onehot.scatter_(1, labels, 1)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs


def get_pretrainedmodels(model_name='resnet18', num_outputs=None, pretrained=True, **_):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=pretrained)
    if 'dpn' in model_name:
        in_channels = model.last_linear.in_channels
        model.last_linear = nn.Conv2d(in_channels, num_outputs, kernel_size=1, bias=True)
    else:
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features, num_outputs)
    return model


def get_model(config, **kwargs):
    None
    f = lambda **kwargs: get_pretrainedmodels(config.model.name, **kwargs)
    if config.model.params is None:
        return f(**kwargs)
    else:
        return f(**config.model.params, **kwargs)


class ArcNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        num_outputs = config.model.params.num_outputs
        feature_size = config.model.params.feature_size
        if 'channel_size' in config.model.params:
            channel_size = config.model.params.channel_size
        else:
            channel_size = 512
        self.model = get_model(config)
        if isinstance(self.model.last_linear, nn.Conv2d):
            in_features = self.model.last_linear.in_channels
        else:
            in_features = self.model.last_linear.in_features
        self.bn1 = nn.BatchNorm2d(in_features)
        self.dropout = nn.Dropout2d(config.model.params.drop_rate, inplace=True)
        self.fc1 = nn.Linear(in_features * feature_size * feature_size, channel_size)
        self.bn2 = nn.BatchNorm1d(channel_size)
        s = config.model.params.s if 's' in config.model.params else 65
        m = config.model.params.m if 'm' in config.model.params else 0.5
        self.arc = ArcModule(channel_size, num_outputs, s=s, m=m)
        if config.model.params.pretrained:
            self.mean = self.model.mean
            self.std = self.model.std
            self.input_range = self.model.input_range
            self.input_space = self.model.input_space
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
            self.input_range = [0, 1]
            self.input_space = 'RGB'

    def forward(self, images, labels=None):
        features = self.model.features(images)
        features = self.bn1(features)
        features = self.dropout(features)
        features = features.view(features.size(0), -1)
        features = self.fc1(features)
        features = self.bn2(features)
        features = F.normalize(features)
        if labels is not None:
            assert self.training
            return self.arc(features, labels)
        return features

    def logits(self, features, labels):
        return self.arc(features, labels)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ArcModule,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.zeros([4, 4, 4, 4], dtype=torch.int64)], {}),
     True),
]

class Test_pudae_kaggle_humpback(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

