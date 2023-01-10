import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
load_data = _module
main = _module
model = _module
train_model = _module

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


from torch.utils.data.dataset import Dataset


from scipy.io import loadmat


from scipy.io import savemat


from torch.utils.data import DataLoader


import numpy as np


import torch


import torch.optim as optim


import matplotlib.pyplot as plt


import torch.nn as nn


import torch.nn.functional as F


from torchvision import models


import torchvision


import time


import copy


class VGGNet(nn.Module):

    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = F.relu(self.denseL1(x))
        return out


class IDCM_NN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, img_input_dim=4096, img_output_dim=2048, text_input_dim=1024, text_output_dim=2048, minus_one_dim=1024, output_dim=10):
        super(IDCM_NN, self).__init__()
        self.img_net = ImgNN(img_input_dim, img_output_dim)
        self.text_net = TextNN(text_input_dim, text_output_dim)
        self.linearLayer = nn.Linear(img_output_dim, minus_one_dim)
        self.linearLayer2 = nn.Linear(minus_one_dim, output_dim)

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)
        view1_feature = self.linearLayer(view1_feature)
        view2_feature = self.linearLayer(view2_feature)
        view1_predict = self.linearLayer2(view1_feature)
        view2_predict = self.linearLayer2(view2_feature)
        return view1_feature, view2_feature, view1_predict, view2_predict

