import sys
_module = sys.modules[__name__]
del sys
augmentation_search = _module
datasets = _module
dataset_factory = _module
default = _module
small = _module
test = _module
inference = _module
losses = _module
loss_factory = _module
make_submission = _module
models = _module
model_factory = _module
optimizers = _module
optimizer_factory = _module
schedulers = _module
scheduler_factory = _module
swa = _module
download = _module
find_data_leak = _module
find_duplicate_images = _module
make_external_csv = _module
make_rgby = _module
stratified_split = _module
train = _module
transforms = _module
policy_transform = _module
transform_factory = _module
tta_transform = _module
utils = _module
checkpoint = _module
config = _module
metrics = _module
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
xrange = range
wraps = functools.wraps


import random


import numpy as np


import torch


import torch.nn.functional as F


import itertools


import torch.utils.data


import torch.utils.data.sampler


from torch.utils.data import DataLoader


import scipy.misc as misc


from torch.utils.data.dataset import Dataset


import math


import torch.nn as nn


import types


import torchvision.models


import torch.optim as optim


import torch.optim.lr_scheduler as lr_scheduler


class Attention(nn.Module):

    def __init__(self, num_classes=28, cnn='resnet34', attention_size=8):
        super().__init__()
        self.num_classes = num_classes
        self.cnn = globals().get('get_' + cnn)()
        self.attention_size = attention_size
        self.avgpool = nn.AdaptiveAvgPool2d(self.attention_size)
        in_features = self.cnn.last_linear.in_features
        self.last_linear = nn.Conv2d(in_features, self.num_classes, kernel_size=1, padding=0)
        self.attention = nn.Conv2d(in_features, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        features = self.cnn.features(x)
        if self.attention_size != features.size(-1):
            features = self.avgpool(features)
        logits = self.last_linear(features)
        assert logits.size(1) == self.num_classes and logits.size(2) == self.attention_size and logits.size(3) == self.attention_size
        logits_attention = self.attention(features)
        assert logits_attention.size(1) == self.num_classes and logits_attention.size(2) == self.attention_size and logits_attention.size(3) == self.attention_size
        logits_attention = logits_attention.view(-1, self.num_classes, self.attention_size * self.attention_size)
        attention = F.softmax(logits_attention, dim=2)
        attention = attention.view(-1, self.num_classes, self.attention_size, self.attention_size)
        logits = logits * attention
        return logits.view(-1, self.num_classes, self.attention_size * self.attention_size).sum(2).view(-1, self.num_classes)


class AttentionInceptionV3(nn.Module):

    def __init__(self, num_classes=28, attention_size=8, aux_attention_size=8):
        super().__init__()
        self.num_classes = num_classes
        self.cnn = torchvision.models.inception_v3(pretrained=True)
        self.attention_size = attention_size
        self.aux_attention_size = aux_attention_size
        conv = self.cnn.Conv2d_1a_3x3.conv
        self.cnn.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels=4, out_channels=conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, bias=conv.bias)
        self.cnn.Conv2d_1a_3x3.conv.weight.data[:, :3, :, :] = conv.weight.data
        self.cnn.Conv2d_1a_3x3.conv.weight.data[:, 3:, :, :] = conv.weight.data[:, :1, :, :]
        self.features_a = nn.Sequential(self.cnn.Conv2d_1a_3x3, self.cnn.Conv2d_2a_3x3, self.cnn.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2), self.cnn.Conv2d_3b_1x1, self.cnn.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2), self.cnn.Mixed_5b, self.cnn.Mixed_5c, self.cnn.Mixed_5d, self.cnn.Mixed_6a, self.cnn.Mixed_6b, self.cnn.Mixed_6c, self.cnn.Mixed_6d, self.cnn.Mixed_6e)
        self.features_b = nn.Sequential(self.cnn.Mixed_7a, self.cnn.Mixed_7b, self.cnn.Mixed_7c)
        self.aux_avgpool = nn.AdaptiveAvgPool2d(self.aux_attention_size)
        aux_in_features = self.cnn.AuxLogits.fc.in_features
        self.aux_linear = nn.Conv2d(aux_in_features, self.num_classes, kernel_size=1, padding=0)
        self.aux_attention = nn.Conv2d(aux_in_features, self.num_classes, kernel_size=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(self.attention_size)
        in_features = self.cnn.fc.in_features
        self.last_linear = nn.Conv2d(in_features, self.num_classes, kernel_size=1, padding=0)
        self.attention = nn.Conv2d(in_features, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        features_a = self.features_a(x)
        if self.training:
            if self.aux_attention_size != features_a.size(-1):
                aux_features = self.aux_avgpool(features_a)
            else:
                aux_features = features_a
            aux_logits = self.aux_linear(aux_features)
            assert aux_logits.size(1) == self.num_classes and aux_logits.size(2) == self.aux_attention_size and aux_logits.size(3) == self.aux_attention_size
            aux_logits_attention = self.aux_attention(aux_features)
            assert aux_logits_attention.size(1) == self.num_classes and aux_logits_attention.size(2) == self.aux_attention_size and aux_logits_attention.size(3) == self.aux_attention_size
            aux_logits_attention = aux_logits_attention.view(-1, self.num_classes, self.aux_attention_size * self.aux_attention_size)
            aux_attention = F.softmax(aux_logits_attention, dim=2)
            aux_attention = aux_attention.view(-1, self.num_classes, self.aux_attention_size, self.aux_attention_size)
            aux_logits = aux_logits * aux_attention
            aux_logits = aux_logits.view(-1, self.num_classes, self.aux_attention_size * self.aux_attention_size).sum(2).view(-1, self.num_classes)
        features_b = self.features_b(features_a)
        if self.aux_attention_size != features_b.size(-1):
            features_b = self.avgpool(features_b)
        logits = self.last_linear(features_b)
        assert logits.size(1) == self.num_classes and logits.size(2) == self.attention_size and logits.size(3) == self.attention_size
        logits_attention = self.attention(features_b)
        assert logits_attention.size(1) == self.num_classes and logits_attention.size(2) == self.attention_size and logits_attention.size(3) == self.attention_size
        logits_attention = logits_attention.view(-1, self.num_classes, self.attention_size * self.attention_size)
        attention = F.softmax(logits_attention, dim=2)
        attention = attention.view(-1, self.num_classes, self.attention_size, self.attention_size)
        logits = logits * attention
        logits = logits.view(-1, self.num_classes, self.attention_size * self.attention_size).sum(2).view(-1, self.num_classes)
        if self.training:
            return logits, aux_logits
        return logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionInceptionV3,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 128, 128])], {}),
     False),
]

class Test_pudae_kaggle_hpa(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

