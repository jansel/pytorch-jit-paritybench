import sys
_module = sys.modules[__name__]
del sys
folder = _module
reid_dataset = _module
cuhk03_to_image = _module
gdrive_downloader = _module
import_CUHK01 = _module
import_CUHK03 = _module
import_DukeMTMC = _module
import_DukeMTMCAttribute = _module
import_Market1501 = _module
import_Market1501Attribute = _module
import_MarketDuke = _module
import_MarketDuke_nodistractors = _module
import_VIPeR = _module
marketduke_to_hdf5 = _module
pytorch_prepare = _module
reiddataset_downloader = _module
save_json = _module
inference = _module
DenseNet121_nFC = _module
ResNet18_nFC = _module
ResNet34_nFC = _module
ResNet50_nFC = _module
ResNet50_nFC_softmax = _module
net = _module
test = _module
train = _module

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


from torch.utils import data


import numpy as np


from torchvision import transforms as T


from torch import nn


from torch.nn import init


from torchvision import models


from torch.nn import functional as F


import scipy.io


from sklearn.metrics import accuracy_score


from sklearn.metrics import precision_score


from sklearn.metrics import recall_score


from sklearn.metrics import f1_score


import warnings


import time


import matplotlib


import matplotlib.pyplot as plt


import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):

    def __init__(self, input_dim, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]
        add_block += [nn.Linear(num_bottleneck, 1)]
        add_block += [nn.Sigmoid()]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.classifier = add_block

    def forward(self, x):
        x = self.classifier(x)
        return x


class DenseNet121_nFC(nn.Module):

    def __init__(self, class_num):
        super(DenseNet121_nFC, self).__init__()
        self.model_name = 'densenet121_nfc'
        self.class_num = class_num
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft.features
        self.num_ftrs = 1024
        num_bottleneck = 512
        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, num_bottleneck))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        for c in range(self.class_num):
            if c == 0:
                pred = self.__getattr__('class_%d' % c)(x)
            else:
                pred = torch.cat((pred, self.__getattr__('class_%d' % c)(x)), dim=1)
        return pred


class ResNet18_nFC(nn.Module):

    def __init__(self, class_num):
        super(ResNet18_nFC, self).__init__()
        self.model_name = 'resnet18_nfc'
        self.class_num = class_num
        model_ft = models.resnet18(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 512
        num_bottleneck = 512
        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, num_bottleneck))

    def forward(self, x):
        x = self.features(x)
        for c in range(self.class_num):
            if c == 0:
                pred = self.__getattr__('class_%d' % c)(x)
            else:
                pred = torch.cat((pred, self.__getattr__('class_%d' % c)(x)), dim=1)
        return pred


class ResNet34_nFC(nn.Module):

    def __init__(self, class_num):
        super(ResNet34_nFC, self).__init__()
        self.model_name = 'resnet34_nfc'
        self.class_num = class_num
        model_ft = models.resnet34(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 512
        num_bottleneck = 512
        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, num_bottleneck))

    def forward(self, x):
        x = self.features(x)
        for c in range(self.class_num):
            if c == 0:
                pred = self.__getattr__('class_%d' % c)(x)
            else:
                pred = torch.cat((pred, self.__getattr__('class_%d' % c)(x)), dim=1)
        return pred


class ResNet50_nFC(nn.Module):

    def __init__(self, class_num):
        super(ResNet50_nFC, self).__init__()
        self.model_name = 'resnet50_nfc'
        self.class_num = class_num
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 2048
        num_bottleneck = 512
        for c in range(self.class_num):
            self.__setattr__('class_%d' % c, ClassBlock(self.num_ftrs, num_bottleneck))

    def forward(self, x):
        x = self.features(x)
        for c in range(self.class_num):
            if c == 0:
                pred = self.__getattr__('class_%d' % c)(x)
            else:
                pred = torch.cat((pred, self.__getattr__('class_%d' % c)(x)), dim=1)
        return pred


class ResNet50_nFC_softmax(nn.Module):

    def __init__(self, class_num, id_num, **kwargs):
        super(ResNet50_nFC_softmax, self).__init__()
        self.model_name = 'resnet50_nfc_softmax'
        self.class_num = class_num
        self.id_num = id_num
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.features = model_ft
        self.num_ftrs = 2048
        num_bottleneck = 512
        for c in range(self.class_num + 1):
            if c == self.class_num:
                self.__setattr__('class_%d' % c, nn.Sequential(nn.Linear(self.num_ftrs, num_bottleneck), nn.BatchNorm1d(num_bottleneck), nn.LeakyReLU(0.1), nn.Dropout(p=0.5), nn.Linear(num_bottleneck, self.id_num)))
            else:
                self.__setattr__('class_%d' % c, nn.Sequential(nn.Linear(self.num_ftrs, num_bottleneck), nn.BatchNorm1d(num_bottleneck), nn.LeakyReLU(0.1), nn.Dropout(p=0.5), nn.Linear(num_bottleneck, 2)))

    def forward(self, x):
        x = self.features(x)
        return (self.__getattr__('class_%d' % c)(x) for c in range(self.class_num + 1)), x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClassBlock,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (DenseNet121_nFC,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet18_nFC,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet34_nFC,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet50_nFC,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResNet50_nFC_softmax,
     lambda: ([], {'class_num': 4, 'id_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_hyk1996_Person_Attribute_Recognition_MarketDuke(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

