import sys
_module = sys.modules[__name__]
del sys
AttrEvaRes18_256V0 = _module
AttrEvaRes34_256V0 = _module
AttrEvaRes34_256V0CE = _module
AttrListGen = _module
AttrPreModelRes18_256V0 = _module
AttrPreModelRes34_256V0 = _module
FocalLoss = _module
TrainAttrPreRes18V0 = _module
TrainAttrPreV0 = _module
TrainAttrPreV0FocalLoss = _module
TrainAttrPreV0OHEM = _module
dataloadercelebA = _module
detMTCNN_celebA = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import numpy as np


import torchvision


from torch.autograd import Variable


from torchvision import datasets


from torchvision import models


from torchvision import transforms


import torch.optim as optim


import torch.nn.functional as F


from torch.optim import lr_scheduler


import re


import torchvision.models as models


import math


class FeatureExtraction(torch.nn.Module):

    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet

    def forward(self, image_batch):
        return self.resnet(image_batch)


class Classifier(nn.Module):

    def __init__(self, output_dim=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc1
        self.fc2 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc2
        self.fc3 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc3
        self.fc4 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc4
        self.fc5 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc5
        self.fc6 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc6

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        x6 = self.fc6(x)
        return x1, x2, x3, x4, x5, x6


class AttrPre(nn.Module):

    def __init__(self):
        super(AttrPre, self).__init__()
        self.FeatureExtraction = FeatureExtraction()
        output_dim = 1
        self.classifier = Classifier(output_dim)

    def forward(self, img):
        feature = self.FeatureExtraction(img)
        Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young = (self.
            classifier(feature))
        return Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young


class FeatureExtraction(torch.nn.Module):

    def __init__(self):
        super(FeatureExtraction, self).__init__()
        self.resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.resnet

    def forward(self, image_batch):
        return self.resnet(image_batch)


class Classifier(nn.Module):

    def __init__(self, output_dim=1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc1
        self.fc2 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc2
        self.fc3 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc3
        self.fc4 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc4
        self.fc5 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc5
        self.fc6 = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(True), nn.
            Dropout(p=0.5), nn.Linear(512, 128), nn.ReLU(True), nn.Dropout(
            p=0.5), nn.Linear(128, output_dim))
        self.fc6

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        x6 = self.fc6(x)
        return x1, x2, x3, x4, x5, x6


class AttrPre(nn.Module):

    def __init__(self):
        super(AttrPre, self).__init__()
        self.FeatureExtraction = FeatureExtraction()
        output_dim = 1
        self.classifier = Classifier(output_dim)

    def forward(self, img):
        feature = self.FeatureExtraction(img)
        Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young = (self.
            classifier(feature))
        return Attractive, EyeGlasses, Male, MouthOpen, Smiling, Young


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class OESM_CrossEntropy(nn.Module):

    def __init__(self, down_k=0.9, top_k=0.7):
        super(OESM_CrossEntropy, self).__init__()
        self.loss = nn.NLLLoss()
        self.down_k = down_k
        self.top_k = top_k
        self.softmax = nn.LogSoftmax()
        return

    def forward(self, input, target):
        softmax_result = self.softmax(input)
        loss = Variable(torch.Tensor(1).zero_())
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            cost = self.loss(pred, gt)
            loss = torch.cat((loss, cost.cpu()), 0)
        loss = loss[1:]
        loss_m = -loss
        if self.top_k == 1:
            valid_loss = loss
        index = torch.topk(loss_m, int(self.down_k * loss.size()[0]))
        loss = loss[index[1]]
        index = torch.topk(loss, int(self.top_k * loss.size()[0]))
        valid_loss = loss[index[1]]
        return torch.mean(valid_loss)


class OESM_CrossEntropy(nn.Module):

    def __init__(self, down_k=1, top_k=0.6):
        super(OESM_CrossEntropy, self).__init__()
        self.loss = nn.NLLLoss()
        self.down_k = down_k
        self.top_k = top_k
        self.softmax = nn.LogSoftmax()
        return

    def forward(self, input, target):
        softmax_result = self.softmax(input)
        loss = Variable(torch.Tensor(1).zero_())
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            pred = torch.unsqueeze(row, 0)
            cost = self.loss(pred, gt)
            loss = torch.cat((loss, cost.cpu()), 0)
        loss = loss[1:]
        loss_m = -loss
        if self.top_k == 1:
            valid_loss = loss
        index = torch.topk(loss_m, int(self.down_k * loss.size()[0]))
        loss = loss[index[1]]
        index = torch.topk(loss, int(self.top_k * loss.size()[0]))
        valid_loss = loss[index[1]]
        return torch.mean(valid_loss)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_WynMew_FaceAttribute(_paritybench_base):
    pass
    def test_000(self):
        self._check(Classifier(*[], **{}), [torch.rand([2048, 2048])], {})

    def test_001(self):
        self._check(FeatureExtraction(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

