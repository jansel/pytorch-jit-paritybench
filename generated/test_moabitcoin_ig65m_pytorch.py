import sys
_module = sys.modules[__name__]
del sys
hubconf = _module
ig65m = _module
cli = _module
client = _module
convert = _module
dreamer = _module
extract = _module
index = _module
semcode = _module
server = _module
datasets = _module
models = _module
samplers = _module
transforms = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


from torchvision.models.video.resnet import BasicBlock


from torchvision.models.video.resnet import R2Plus1dStem


from torchvision.models.video.resnet import Conv2Plus1D


from torchvision.transforms import Compose


import numpy as np


from torch.utils.data import DataLoader


import math


from torch.utils.data import IterableDataset


from torch.utils.data import get_worker_info


import torch.hub


from torchvision.models.video.resnet import VideoResNet


model_urls = {'r2plus1d_34_8_ig65m': 'https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth', 'r2plus1d_34_32_ig65m': 'https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth', 'r2plus1d_34_8_kinetics': 'https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth', 'r2plus1d_34_32_kinetics': 'https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth'}


def r2plus1d_34(num_classes, pretrained=False, progress=False, arch=None):
    model = VideoResNet(block=BasicBlock, conv_makers=[Conv2Plus1D] * 4, layers=[3, 4, 6, 3], stem=R2Plus1dStem)
    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 0.001
            m.momentum = 0.9
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def r2plus1d_34_32_ig65m(num_classes, pretrained=False, progress=False):
    """R(2+1)D 34-layer IG65M model for clips of length 32 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads weights pretrained on 65 million Instagram videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 359, 'pretrained on 359 classes'
    return r2plus1d_34(num_classes=num_classes, arch='r2plus1d_34_32_ig65m', pretrained=pretrained, progress=progress)


class VideoModel(nn.Module):

    def __init__(self, pool_spatial='mean', pool_temporal='mean'):
        super().__init__()
        self.model = r2plus1d_34_32_ig65m(num_classes=359, pretrained=True, progress=True)
        self.pool_spatial = Reduce('n c t h w -> n c t', reduction=pool_spatial)
        self.pool_temporal = Reduce('n c t -> n c', reduction=pool_temporal)

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.pool_spatial(x)
        x = self.pool_temporal(x)
        return x


class TotalVariationLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        loss = 0.0
        loss += (inputs[:, :, :-1, :, :] - inputs[:, :, 1:, :, :]).abs().sum()
        loss += (inputs[:, :, :, :-1, :] - inputs[:, :, :, 1:, :]).abs().sum()
        loss += (inputs[:, :, :, :, :-1] - inputs[:, :, :, :, 1:]).abs().sum()
        return loss

