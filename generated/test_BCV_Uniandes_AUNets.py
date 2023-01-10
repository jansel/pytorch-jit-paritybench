import sys
_module = sys.modules[__name__]
del sys
config = _module
data_loader = _module
OF_BP4D = _module
OF_resizeBP4D = _module
parallelCallOFPython = _module
get_augmentation = _module
get_face_mean = _module
get_oneAU = _module
get_resize_aligned = _module
get_resize_aligned_allBP4D = _module
get_show = _module
jittering = _module
logger = _module
main = _module
models = _module
vgg16 = _module
vgg_pytorch = _module
solver = _module
split_train_val_test = _module
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


import torch


import random


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torchvision import transforms


import numpy as np


from torch.backends import cudnn


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import math


import torch.nn.functional as F


import time


from torch.autograd import Variable


from torchvision.utils import save_image


import warnings


class Classifier(nn.Module):

    def __init__(self, pretrained='/npy/weights', OF_option='None', model_save_path='', test_model=''):
        super(Classifier, self).__init__()
        self.finetuning = pretrained
        self.OF_option = OF_option
        self.model_save_path = model_save_path
        if test_model == '':
            self._initialize_weights()
        else:
            self.model = model_vgg16(OF_option=self.OF_option, model_save_path=self.model_save_path, num_classes=1)

    def _initialize_weights(self):
        if 'emotionnet' in self.finetuning:
            mode = 'emotionnet'
            self.model = model_vgg16(pretrained=mode, OF_option=self.OF_option, model_save_path=self.model_save_path, num_classes=22)
            modules = self.model.modules()
            for m in modules:
                if isinstance(m, nn.Linear) and m.weight.data.size()[0] == 22:
                    w1 = m.weight.data[1:2].view(1, -1)
                    b1 = torch.FloatTensor(np.array(m.bias.data[1]).reshape(-1))
            mod = list(self.model.classifier)
            mod.pop()
            if self.OF_option == 'FC7':
                dim_fc7 = 4096 * 2
            else:
                dim_fc7 = 4096
            mod.append(torch.nn.Linear(dim_fc7, 1))
            new_classifier = torch.nn.Sequential(*mod)
            self.model.classifier = new_classifier
            modules = self.model.modules()
            flag = False
            for m in modules:
                if isinstance(m, nn.Linear) and m.weight.data.size()[0] == 1:
                    m.weight.data = w1
                    m.bias.data = b1
                    flag = True
            assert flag
        elif 'imagenet' in self.finetuning:
            mode = 'ImageNet'
            self.model = model_vgg16(pretrained=mode, OF_option=self.OF_option, num_classes=1000)
            modules = self.model.modules()
            for m in modules:
                if isinstance(m, nn.Linear) and m.weight.data.size()[0] == 1000:
                    w1 = m.weight.data[1:2].view(1, -1)
                    b1 = torch.FloatTensor(np.array(m.bias.data[1]).reshape(-1))
            mod = list(self.model.classifier)
            mod.pop()
            if self.OF_option == 'FC7':
                dim_fc7 = 4096 * 2
            else:
                dim_fc7 = 4096
            mod.append(torch.nn.Linear(dim_fc7, 1))
            new_classifier = torch.nn.Sequential(*mod)
            self.model.classifier = new_classifier
            flag = False
            modules = self.model.modules()
            for m in modules:
                if isinstance(m, nn.Linear) and m.weight.data.size()[0] == 1:
                    m.weight.data = w1
                    m.bias.data = b1
                    flag = True
            assert flag
        elif self.finetuning == 'random':
            mode = 'RANDOM'
            self.model = model_vgg16(pretrained='', OF_option=self.OF_option, num_classes=1)
        None

    def forward(self, image, OF=None):
        if OF is not None:
            x = self.model(image, OF=OF)
        else:
            x = self.model(image)
        return x


class VGG(nn.Module):

    def __init__(self, features, num_classes=2):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_ALONE(nn.Module):

    def __init__(self, features, num_classes=2):
        super(VGG_ALONE, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        self._initialize_weights()

    def forward(self, x, OF=None):
        OF = self.features(OF)
        OF = OF.view(OF.size(0), -1)
        OF = self.classifier(OF)
        return OF

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_IMAGE(nn.Module):

    def __init__(self, features, num_classes=2, OF_option='horizontal'):
        super(VGG_IMAGE, self).__init__()
        self.OF_option = OF_option
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 14 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        self._initialize_weights()

    def forward(self, x, OF=None):
        if self.OF_option.lower() == 'horizontal':
            dim = 3
        else:
            dim = 2
        img_of = torch.cat([x, OF], dim=dim)
        x = self.features(img_of)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_CHANNELS(nn.Module):

    def __init__(self, features, num_classes=2):
        super(VGG_CHANNELS, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        self._initialize_weights()

    def forward(self, x, OF=None):
        img_of = torch.cat([x, OF], dim=1)
        out = self.features(img_of)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_CONV(nn.Module):

    def __init__(self, features_rgb, features_of, num_classes=2):
        super(VGG_CONV, self).__init__()
        self.features_rgb = features_rgb
        self.features_of = features_of
        self.classifier = nn.Sequential(nn.Linear(1024 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        self._initialize_weights()

    def forward(self, x, OF=None):
        conv_rgb = self.features_rgb(x)
        conv_of = self.features_of(OF)
        conv_out = torch.cat([conv_rgb, conv_of], dim=1)
        conv_out = conv_out.view(conv_out.size(0), -1)
        out = self.classifier(conv_out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_FC6(nn.Module):

    def __init__(self, features_rgb, features_of, num_classes=2):
        super(VGG_FC6, self).__init__()
        self.features_rgb = features_rgb
        self.features_of = features_of
        self.classifier_rgb = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout())
        self.classifier_of = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout())
        self.classifier = nn.Sequential(nn.Linear(8192, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, num_classes))
        self._initialize_weights()

    def forward(self, x, OF=None):
        conv_rgb = self.features_rgb(x)
        conv_rgb = conv_rgb.view(conv_rgb.size(0), -1)
        fc6_rgb = self.classifier_rgb(conv_rgb)
        conv_of = self.features_of(OF)
        conv_of = conv_of.view(conv_of.size(0), -1)
        fc6_of = self.classifier_of(conv_of)
        fc_cat = torch.cat([fc6_rgb, fc6_of], dim=1)
        out = self.classifier(fc_cat)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG_FC7(nn.Module):

    def __init__(self, features_rgb, features_of, num_classes=2):
        super(VGG_FC7, self).__init__()
        self.features_rgb = features_rgb
        self.features_of = features_of
        self.classifier_rgb = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout())
        self.classifier_of = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout())
        self.classifier = nn.Sequential(nn.Linear(8192, num_classes))
        self._initialize_weights()

    def forward(self, x, OF=None):
        conv_rgb = self.features_rgb(x)
        conv_rgb = conv_rgb.view(conv_rgb.size(0), -1)
        fc7_rgb = self.classifier_rgb(conv_rgb)
        conv_of = self.features_of(OF)
        conv_of = conv_of.view(conv_of.size(0), -1)
        fc7_of = self.classifier_rgb(conv_of)
        fc_cat = torch.cat([fc7_rgb, fc7_of], dim=1)
        out = self.classifier(fc_cat)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

