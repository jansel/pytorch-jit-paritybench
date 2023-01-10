import sys
_module = sys.modules[__name__]
del sys
config = _module
data_loader = _module
finetune = _module
eval_ijba_1_1 = _module
eval_lfw = _module
models = _module
run_resnet_demo = _module
test_colorizer = _module
test_data_reader = _module
test_fcn32s = _module
vis_prediction = _module
train = _module
run_crop_face = _module
train_resnet_umdface = _module
utils = _module
calculate_image_mean = _module
train_resnet101_vggface = _module
train_resnet50_vggface_scratch = _module

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


import collections


import numpy as np


import scipy.io


import torch


from torch.utils import data


import math


import scipy.misc


from torch.autograd import Variable


import torch.nn.functional as F


import torchvision


from torchvision import models


import torch.nn as nn


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.datasets as datasets


import sklearn.metrics


from sklearn import metrics


from scipy import interpolate


import scipy.io as sio


import matplotlib


import matplotlib.pyplot as plt


from scipy.optimize import brentq


import torch.optim


class NormFeat(nn.Module):
    """ L2 normalization of features """

    def __init__(self, scale_factor=1.0):
        super(NormFeat, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        return self.scale_factor * F.normalize(input, p=2, dim=1)


class ScaleFeat(nn.Module):

    def __init__(self, scale_factor=50.0):
        super().__init__()
        self.scale = scale_factor

    def forward(self, input):
        return input * self.scale


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN32sColor(nn.Module):

    def __init__(self, n_class=32, bin_type='one-hot', batch_norm=True):
        super(FCN32sColor, self).__init__()
        self.n_class = n_class
        self.bin_type = bin_type
        self.batch_norm = batch_norm
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_2_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_2_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_3_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_3_bn = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_3_bn = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        if batch_norm:
            self.fc6_bn = nn.BatchNorm2d(4096)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc7_bn = nn.BatchNorm2d(4096)
        self.drop7 = nn.Dropout2d()
        if bin_type == 'one-hot':
            self.score_fr_hue = nn.Conv2d(4096, n_class, 1)
            self.upscore_hue = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
            self.score_fr_chroma = nn.Conv2d(4096, n_class, 1)
            self.upscore_chroma = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
            self.upscore_hue.weight.requires_grad = False
            self.upscore_chroma.weight.requires_grad = False
        elif bin_type == 'soft':
            self.score_fr = nn.Conv2d(4096, n_class, 1)
            self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, stride=32, bias=False)
            self.upscore.weight.requires_grad = False
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.conv1_1(h)
        if self.batch_norm:
            h = self.conv1_1_bn(h)
        h = self.relu1_1(h)
        h = self.conv1_2(h)
        if self.batch_norm:
            h = self.conv1_2_bn(h)
        h = self.relu1_2(h)
        h = self.pool1(h)
        if self.batch_norm:
            h = self.relu2_1(self.conv2_1_bn(self.conv2_1(h)))
        else:
            h = self.relu2_1(self.conv2_1(h))
        if self.batch_norm:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        else:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        h = self.pool2(h)
        if self.batch_norm:
            h = self.relu3_1(self.conv3_1_bn(self.conv3_1(h)))
        else:
            h = self.relu3_1(self.conv3_1(h))
        if self.batch_norm:
            h = self.relu3_2(self.conv3_2_bn(self.conv3_2(h)))
        else:
            h = self.relu3_2(self.conv3_2(h))
        if self.batch_norm:
            h = self.relu3_3(self.conv3_3_bn(self.conv3_3(h)))
        else:
            h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        if self.batch_norm:
            h = self.relu4_1(self.conv4_1_bn(self.conv4_1(h)))
        else:
            h = self.relu4_1(self.conv4_1(h))
        if self.batch_norm:
            h = self.relu4_2(self.conv4_2_bn(self.conv4_2(h)))
        else:
            h = self.relu4_2(self.conv4_2(h))
        if self.batch_norm:
            h = self.relu4_3(self.conv4_3_bn(self.conv4_3(h)))
        else:
            h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        if self.batch_norm:
            h = self.relu5_1(self.conv5_1_bn(self.conv5_1(h)))
        else:
            h = self.relu5_1(self.conv5_1(h))
        if self.batch_norm:
            h = self.relu5_2(self.conv5_2_bn(self.conv5_2(h)))
        else:
            h = self.relu5_2(self.conv5_2(h))
        if self.batch_norm:
            h = self.relu5_3(self.conv5_3_bn(self.conv5_3(h)))
        else:
            h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        if self.batch_norm:
            h = self.relu6(self.fc6_bn(self.fc6(h)))
        else:
            h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        if self.batch_norm:
            h = self.relu7(self.fc7_bn(self.fc7(h)))
        else:
            h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        if self.bin_type == 'one-hot':
            h_hue = self.score_fr_hue(h)
            h_hue = self.upscore_hue(h_hue)
            h_hue = h_hue[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
            h_chroma = self.score_fr_chroma(h)
            h_chroma = self.upscore_chroma(h_chroma)
            h_chroma = h_chroma[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
            h = h_hue, h_chroma
        elif self.bin_type == 'soft':
            h = self.score_fr(h)
            h = self.upscore(h)
            h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        return h


class FCN16sColor(nn.Module):

    def __init__(self, n_class=32, bin_type='one-hot', batch_norm=True):
        super(FCN16sColor, self).__init__()
        self.n_class = n_class
        self.bin_type = bin_type
        self.batch_norm = batch_norm
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_2_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_2_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_3_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_3_bn = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_3_bn = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        if batch_norm:
            self.fc6_bn = nn.BatchNorm2d(4096)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc7_bn = nn.BatchNorm2d(4096)
        self.drop7 = nn.Dropout2d()
        if bin_type == 'one-hot':
            raise NotImplementedError('TODO - FCN 16s for separate hue-chroma')
        elif bin_type == 'soft':
            self.score_fr = nn.Conv2d(4096, n_class, 1)
            self.score_pool4 = nn.Conv2d(512, n_class, 1)
            self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
            self.upscore16 = nn.ConvTranspose2d(n_class, n_class, 32, stride=16, bias=False)
            self.upscore2.weight.requires_grad = False
            self.upscore16.weight.requires_grad = False
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.conv1_1(h)
        if self.batch_norm:
            h = self.conv1_1_bn(h)
        h = self.relu1_1(h)
        h = self.conv1_2(h)
        if self.batch_norm:
            h = self.conv1_2_bn(h)
        h = self.relu1_2(h)
        h = self.pool1(h)
        if self.batch_norm:
            h = self.relu2_1(self.conv2_1_bn(self.conv2_1(h)))
        else:
            h = self.relu2_1(self.conv2_1(h))
        if self.batch_norm:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        else:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        h = self.pool2(h)
        if self.batch_norm:
            h = self.relu3_1(self.conv3_1_bn(self.conv3_1(h)))
        else:
            h = self.relu3_1(self.conv3_1(h))
        if self.batch_norm:
            h = self.relu3_2(self.conv3_2_bn(self.conv3_2(h)))
        else:
            h = self.relu3_2(self.conv3_2(h))
        if self.batch_norm:
            h = self.relu3_3(self.conv3_3_bn(self.conv3_3(h)))
        else:
            h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        if self.batch_norm:
            h = self.relu4_1(self.conv4_1_bn(self.conv4_1(h)))
        else:
            h = self.relu4_1(self.conv4_1(h))
        if self.batch_norm:
            h = self.relu4_2(self.conv4_2_bn(self.conv4_2(h)))
        else:
            h = self.relu4_2(self.conv4_2(h))
        if self.batch_norm:
            h = self.relu4_3(self.conv4_3_bn(self.conv4_3(h)))
        else:
            h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h
        if self.batch_norm:
            h = self.relu5_1(self.conv5_1_bn(self.conv5_1(h)))
        else:
            h = self.relu5_1(self.conv5_1(h))
        if self.batch_norm:
            h = self.relu5_2(self.conv5_2_bn(self.conv5_2(h)))
        else:
            h = self.relu5_2(self.conv5_2(h))
        if self.batch_norm:
            h = self.relu5_3(self.conv5_3_bn(self.conv5_3(h)))
        else:
            h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        if self.batch_norm:
            h = self.relu6(self.fc6_bn(self.fc6(h)))
        else:
            h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        if self.batch_norm:
            h = self.relu7(self.fc7_bn(self.fc7(h)))
        else:
            h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        if self.bin_type == 'one-hot':
            raise NotImplementedError('TODO - FCN 16s for separate hue-chroma')
        elif self.bin_type == 'soft':
            h = self.score_fr(h)
            h = self.upscore2(h)
            upscore2 = h
            h = self.score_pool4(pool4)
            h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            score_pool4c = h
            h = upscore2 + score_pool4c
            h = self.upscore16(h)
            h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()
        return h

    def copy_params_from_fcn32s(self, fcn32s):
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)


class FCN8sColor(nn.Module):

    def __init__(self, n_class=32, bin_type='one-hot', batch_norm=True):
        super(FCN8sColor, self).__init__()
        self.n_class = n_class
        self.bin_type = bin_type
        self.batch_norm = batch_norm
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv1_2_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv2_2_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_1_bn = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_2_bn = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv3_3_bn = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_1_bn = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_2_bn = nn.BatchNorm2d(512)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv4_3_bn = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_1_bn = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_2_bn = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        if batch_norm:
            self.conv5_3_bn = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        if batch_norm:
            self.fc6_bn = nn.BatchNorm2d(4096)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc7_bn = nn.BatchNorm2d(4096)
        self.drop7 = nn.Dropout2d()
        if bin_type == 'one-hot':
            raise NotImplementedError('TODO - FCN 16s for separate hue-chroma')
        elif bin_type == 'soft':
            self.score_fr = nn.Conv2d(4096, n_class, 1)
            self.score_pool3 = nn.Conv2d(256, n_class, 1)
            self.score_pool4 = nn.Conv2d(512, n_class, 1)
            self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
            self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, bias=False)
            self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, bias=False)
            self.upscore2.weight.requires_grad = False
            self.upscore8.weight.requires_grad = False
            self.upscore_pool4.weight.requires_grad = False
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                pass
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x
        h = self.conv1_1(h)
        if self.batch_norm:
            h = self.conv1_1_bn(h)
        h = self.relu1_1(h)
        h = self.conv1_2(h)
        if self.batch_norm:
            h = self.conv1_2_bn(h)
        h = self.relu1_2(h)
        h = self.pool1(h)
        if self.batch_norm:
            h = self.relu2_1(self.conv2_1_bn(self.conv2_1(h)))
        else:
            h = self.relu2_1(self.conv2_1(h))
        if self.batch_norm:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        else:
            h = self.relu2_2(self.conv2_2_bn(self.conv2_2(h)))
        h = self.pool2(h)
        if self.batch_norm:
            h = self.relu3_1(self.conv3_1_bn(self.conv3_1(h)))
        else:
            h = self.relu3_1(self.conv3_1(h))
        if self.batch_norm:
            h = self.relu3_2(self.conv3_2_bn(self.conv3_2(h)))
        else:
            h = self.relu3_2(self.conv3_2(h))
        if self.batch_norm:
            h = self.relu3_3(self.conv3_3_bn(self.conv3_3(h)))
        else:
            h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h
        if self.batch_norm:
            h = self.relu4_1(self.conv4_1_bn(self.conv4_1(h)))
        else:
            h = self.relu4_1(self.conv4_1(h))
        if self.batch_norm:
            h = self.relu4_2(self.conv4_2_bn(self.conv4_2(h)))
        else:
            h = self.relu4_2(self.conv4_2(h))
        if self.batch_norm:
            h = self.relu4_3(self.conv4_3_bn(self.conv4_3(h)))
        else:
            h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h
        if self.batch_norm:
            h = self.relu5_1(self.conv5_1_bn(self.conv5_1(h)))
        else:
            h = self.relu5_1(self.conv5_1(h))
        if self.batch_norm:
            h = self.relu5_2(self.conv5_2_bn(self.conv5_2(h)))
        else:
            h = self.relu5_2(self.conv5_2(h))
        if self.batch_norm:
            h = self.relu5_3(self.conv5_3_bn(self.conv5_3(h)))
        else:
            h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        if self.batch_norm:
            h = self.relu6(self.fc6_bn(self.fc6(h)))
        else:
            h = self.relu6(self.fc6(h))
        h = self.drop6(h)
        if self.batch_norm:
            h = self.relu7(self.fc7_bn(self.fc7(h)))
        else:
            h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        if self.bin_type == 'one-hot':
            raise NotImplementedError('TODO - FCN 16s for separate hue-chroma')
        elif self.bin_type == 'soft':
            h = self.score_fr(h)
            h = self.upscore2(h)
            upscore2 = h
            h = self.score_pool4(pool4)
            h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
            score_pool4c = h
            h = upscore2 + score_pool4c
            h = self.upscore_pool4(h)
            upscore_pool4 = h
            h = self.score_pool3(pool3)
            h = h[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
            score_pool3c = h
            h = upscore_pool4 + score_pool3c
            h = self.upscore8(h)
            h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        return h

    def copy_params_from_fcn16s(self, fcn16s):
        for name, l1 in fcn16s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            l2.weight.data.copy_(l1.weight.data)
            if l1.bias is not None:
                assert l1.bias.size() == l2.bias.size()
                l2.bias.data.copy_(l1.bias.data)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FCN32sColor,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (NormFeat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScaleFeat,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_AruniRC_resnet_face_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

