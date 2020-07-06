import sys
_module = sys.modules[__name__]
del sys
download = _module
main = _module
model = _module
ops = _module
resizeImage = _module
utils = _module
changeIndex = _module
evaluate = _module
evaluate_rerank = _module
model = _module
prepare = _module
random_erasing = _module
re_ranking = _module
test = _module
train_baseline = _module

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


import scipy.io


import torch


import numpy as np


import time


import torch.nn as nn


from torch.nn import init


from torchvision import models


from torch.autograd import Variable


from torchvision.transforms import *


import random


import math


import torch.optim as optim


from torch.optim import lr_scheduler


import torchvision


from torchvision import datasets


from torchvision import transforms


from torchvision.datasets.folder import default_loader


import torch.nn.functional as F


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class ft_net(nn.Module):

    def __init__(self, class_num):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = model_ft.fc.in_features
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(num_ftrs, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


class ft_net_dense(nn.Module):

    def __init__(self, class_num):
        super(ft_net_dense, self).__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(1024, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        x = self.classifier(x)
        return x


class LSROloss(nn.Module):

    def __init__(self):
        super(LSROloss, self).__init__()

    def forward(self, input, target, flg):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        maxRow, _ = torch.max(input.data, 1)
        maxRow = maxRow.unsqueeze(1)
        input.data = input.data - maxRow
        target = target.view(-1, 1)
        flg = flg.view(-1, 1)
        flos = F.log_softmax(input)
        flos = torch.sum(flos, 1) / flos.size(1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        flg = flg.view(-1)
        flg = flg.type(torch.FloatTensor)
        loss = -1 * logpt * (1 - flg) - flos * flg
        return loss.mean()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ft_net,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (ft_net_dense,
     lambda: ([], {'class_num': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_qiaoguan_Person_reid_GAN_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

