import sys
_module = sys.modules[__name__]
del sys
dataset = _module
device = _module
imageaug = _module
loss = _module
main = _module
metrics = _module
models = _module
base = _module
resnet = _module
tests = _module
center_loss_test = _module
trainer = _module
utils = _module

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


from math import ceil


from math import floor


from torch.utils import data


import numpy as np


import torch


from torch.utils.data import DataLoader


from torchvision import transforms


from sklearn.model_selection import KFold


from torch import nn


from torchvision.models import resnet18


from torchvision.models import resnet50


import numpy


device = torch.device('cuda')


class FaceModel(nn.Module):

    def __init__(self, num_classes, feature_dim):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        if num_classes:
            self.register_buffer('centers', (torch.rand(num_classes, feature_dim) - 0.5) * 2)
            self.classifier = nn.Linear(self.feature_dim, num_classes)


class ResnetFaceModel(FaceModel):
    IMAGE_SHAPE = 96, 128

    def __init__(self, num_classes, feature_dim):
        super().__init__(num_classes, feature_dim)
        self.extract_feature = nn.Linear(self.feature_dim * 4 * 3, self.feature_dim)
        self.num_classes = num_classes
        if self.num_classes:
            self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = x.view(x.size(0), -1)
        feature = self.extract_feature(x)
        logits = self.classifier(feature) if self.num_classes else None
        feature_normed = feature.div(torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))
        return logits, feature_normed


class Resnet18FaceModel(ResnetFaceModel):
    FEATURE_DIM = 512

    def __init__(self, num_classes):
        super().__init__(num_classes, self.FEATURE_DIM)
        self.base = resnet18(pretrained=True)


class Resnet50FaceModel(ResnetFaceModel):
    FEATURE_DIM = 2048

    def __init__(self, num_classes):
        super().__init__(num_classes, self.FEATURE_DIM)
        self.base = resnet50(pretrained=True)

