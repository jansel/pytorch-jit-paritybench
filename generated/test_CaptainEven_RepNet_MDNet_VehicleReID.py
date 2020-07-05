import sys
_module = sys.modules[__name__]
del sys
InitRepNet = _module
ProcessVehicleID = _module
RepNet = _module
dataset = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch.nn import Parameter


import math


import torch.nn.functional as F


import copy


import torch.nn as nn


import random


import torchvision


from torchvision import transforms as T


import numpy as np


from torch.utils import data


from collections import defaultdict


device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')


class ArcFC(torch.nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output_layer sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        """
        ArcMargin
        :param in_features:
        :param out_features:
        :param s:
        :param m:
        :param easy_margin:
        """
        super(ArcFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        None
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input, p=2), F.normalize(self.weight, p=2))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return output


class InitRepNet(torch.nn.Module):

    def __init__(self, vgg_orig, out_ids, out_attribs):
        """
        网络结构定义与初始化
        :param vgg_orig: pre-trained VggNet
        :param out_ids:
        :param out_attribs:
        """
        super(InitRepNet, self).__init__()
        self.out_ids, self.out_attribs = out_ids, out_attribs
        None
        feats = vgg_orig.features._modules
        classifier = vgg_orig.classifier._modules
        self.conv1_1 = copy.deepcopy(feats['0'])
        self.conv1_2 = copy.deepcopy(feats['1'])
        self.conv1_3 = copy.deepcopy(feats['2'])
        self.conv1_4 = copy.deepcopy(feats['3'])
        self.conv1_5 = copy.deepcopy(feats['4'])
        self.conv1 = torch.nn.Sequential(self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4, self.conv1_5)
        self.conv2_1 = copy.deepcopy(feats['5'])
        self.conv2_2 = copy.deepcopy(feats['6'])
        self.conv2_3 = copy.deepcopy(feats['7'])
        self.conv2_4 = copy.deepcopy(feats['8'])
        self.conv2_5 = copy.deepcopy(feats['9'])
        self.conv2 = torch.nn.Sequential(self.conv2_1, self.conv2_2, self.conv2_3, self.conv2_4, self.conv2_5)
        self.conv3_1 = copy.deepcopy(feats['10'])
        self.conv3_2 = copy.deepcopy(feats['11'])
        self.conv3_3 = copy.deepcopy(feats['12'])
        self.conv3_4 = copy.deepcopy(feats['13'])
        self.conv3_5 = copy.deepcopy(feats['14'])
        self.conv3_6 = copy.deepcopy(feats['15'])
        self.conv3_7 = copy.deepcopy(feats['16'])
        self.conv3 = torch.nn.Sequential(self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4, self.conv3_5, self.conv3_6, self.conv3_7)
        self.conv4_1_1 = copy.deepcopy(feats['17'])
        self.conv4_1_2 = copy.deepcopy(feats['18'])
        self.conv4_1_3 = copy.deepcopy(feats['19'])
        self.conv4_1_4 = copy.deepcopy(feats['20'])
        self.conv4_1_5 = copy.deepcopy(feats['21'])
        self.conv4_1_6 = copy.deepcopy(feats['22'])
        self.conv4_1_7 = copy.deepcopy(feats['23'])
        self.conv4_1 = torch.nn.Sequential(self.conv4_1_1, self.conv4_1_2, self.conv4_1_3, self.conv4_1_4, self.conv4_1_5, self.conv4_1_6, self.conv4_1_7)
        self.conv4_2_1 = copy.deepcopy(self.conv4_1_1)
        self.conv4_2_2 = copy.deepcopy(self.conv4_1_2)
        self.conv4_2_3 = copy.deepcopy(self.conv4_1_3)
        self.conv4_2_4 = copy.deepcopy(self.conv4_1_4)
        self.conv4_2_5 = copy.deepcopy(self.conv4_1_5)
        self.conv4_2_6 = copy.deepcopy(self.conv4_1_6)
        self.conv4_2_7 = copy.deepcopy(self.conv4_1_7)
        self.conv4_2 = torch.nn.Sequential(self.conv4_2_1, self.conv4_2_2, self.conv4_2_3, self.conv4_2_4, self.conv4_2_5, self.conv4_2_6, self.conv4_2_7)
        self.conv5_1_1 = copy.deepcopy(feats['24'])
        self.conv5_1_2 = copy.deepcopy(feats['25'])
        self.conv5_1_3 = copy.deepcopy(feats['26'])
        self.conv5_1_4 = copy.deepcopy(feats['27'])
        self.conv5_1_5 = copy.deepcopy(feats['28'])
        self.conv5_1_6 = copy.deepcopy(feats['29'])
        self.conv5_1_7 = copy.deepcopy(feats['30'])
        self.conv5_1 = torch.nn.Sequential(self.conv5_1_1, self.conv5_1_2, self.conv5_1_3, self.conv5_1_4, self.conv5_1_5, self.conv5_1_6, self.conv5_1_7)
        self.conv5_2_1 = copy.deepcopy(self.conv5_1_1)
        self.conv5_2_2 = copy.deepcopy(self.conv5_1_2)
        self.conv5_2_3 = copy.deepcopy(self.conv5_1_3)
        self.conv5_2_4 = copy.deepcopy(self.conv5_1_4)
        self.conv5_2_5 = copy.deepcopy(self.conv5_1_5)
        self.conv5_2_6 = copy.deepcopy(self.conv5_1_6)
        self.conv5_2_7 = copy.deepcopy(self.conv5_1_7)
        self.conv5_2 = torch.nn.Sequential(self.conv5_2_1, self.conv5_2_2, self.conv5_2_3, self.conv5_2_4, self.conv5_2_5, self.conv5_2_6, self.conv5_2_7)
        self.FC6_1_1 = copy.deepcopy(classifier['0'])
        self.FC6_1_2 = copy.deepcopy(classifier['1'])
        self.FC6_1_3 = copy.deepcopy(classifier['2'])
        self.FC6_1_4 = copy.deepcopy(classifier['3'])
        self.FC6_1_5 = copy.deepcopy(classifier['4'])
        self.FC6_1_6 = copy.deepcopy(classifier['5'])
        self.FC6_1 = torch.nn.Sequential(self.FC6_1_1, self.FC6_1_2, self.FC6_1_3, self.FC6_1_4, self.FC6_1_5, self.FC6_1_6)
        self.FC6_2_1 = copy.deepcopy(self.FC6_1_1)
        self.FC6_2_2 = copy.deepcopy(self.FC6_1_2)
        self.FC6_2_3 = copy.deepcopy(self.FC6_1_3)
        self.FC6_2_4 = copy.deepcopy(self.FC6_1_4)
        self.FC6_2_5 = copy.deepcopy(self.FC6_1_5)
        self.FC6_2_6 = copy.deepcopy(self.FC6_1_6)
        self.FC6_2 = torch.nn.Sequential(self.FC6_2_1, self.FC6_2_2, self.FC6_2_3, self.FC6_2_4, self.FC6_2_5, self.FC6_2_6)
        self.FC7_1 = copy.deepcopy(classifier['6'])
        self.FC7_2 = copy.deepcopy(self.FC7_1)
        self.FC_8 = torch.nn.Linear(in_features=2000, out_features=1024)
        self.attrib_classifier = torch.nn.Linear(in_features=1000, out_features=out_attribs)
        self.arc_fc_br2 = ArcFC(in_features=1000, out_features=out_ids, s=30.0, m=0.5, easy_margin=False)
        self.arc_fc_br3 = ArcFC(in_features=1024, out_features=out_ids, s=30.0, m=0.5, easy_margin=False)
        self.shared_layers = torch.nn.Sequential(self.conv1, self.conv2, self.conv3)
        self.branch_1_feats = torch.nn.Sequential(self.shared_layers, self.conv4_1, self.conv5_1)
        self.branch_1_fc = torch.nn.Sequential(self.FC6_1, self.FC7_1)
        self.branch_1 = torch.nn.Sequential(self.branch_1_feats, self.branch_1_fc)
        self.branch_2_feats = torch.nn.Sequential(self.shared_layers, self.conv4_2, self.conv5_2)
        self.branch_2_fc = torch.nn.Sequential(self.FC6_2, self.FC7_2)
        self.branch_2 = torch.nn.Sequential(self.branch_2_feats, self.branch_2_fc)

    def forward(self, X, branch, label=None):
        """
        先单独训练branch_1, 然后brach_1, branch_2, branch_3联合训练
        :param X:
        :param branch:
        :param label:
        :return:
        """
        N = X.size(0)
        if branch == 1:
            X = self.branch_1_feats(X)
            X = X.view(N, -1)
            X = self.branch_1_fc(X)
            assert X.size() == (N, 1000)
            X = self.attrib_classifier(X)
            assert X.size() == (N, self.out_attribs)
            return X
        elif branch == 2:
            if label is None:
                None
                return None
            X = self.branch_2_feats(X)
            X = X.view(N, -1)
            X = self.branch_2_fc(X)
            assert X.size() == (N, 1000)
            X = self.arc_fc_br2.forward(input=X, label=label)
            assert X.size() == (N, self.out_ids)
            return X
        elif branch == 3:
            if label is None:
                None
                return None
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)
            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)
            assert fusion_feats.size() == (N, 2000)
            X = self.FC_8(fusion_feats)
            X = self.arc_fc_br3.forward(input=X, label=label)
            assert X.size() == (N, self.out_ids)
            return X
        elif branch == 4:
            X = self.branch_1_feats(X)
            X = X.view(N, -1)
            X = self.branch_1_fc(X)
            assert X.size() == (N, 1000)
            return X
        elif branch == 5:
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)
            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)
            assert fusion_feats.size() == (N, 2000)
            X = self.FC_8(fusion_feats)
            assert X.size() == (N, 1024)
            return X
        else:
            None
            return None


class FocalLoss(nn.Module):
    """
    Focal loss: focus more on hard samples
    """

    def __init__(self, gamma=0, eps=1e-07):
        """
        :param gamma:
        :param eps:
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        """
        :param input:
        :param target:
        :return:
        """
        log_p = self.ce(input, target)
        p = torch.exp(-log_p)
        loss = (1.0 - p) ** self.gamma * log_p
        return loss.mean()


class ArcFC(nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output_layer sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        """
        ArcMargin
        :param in_features:
        :param out_features:
        :param s:
        :param m:
        :param easy_margin:
        """
        super(ArcFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        None
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input, p=2), F.normalize(self.weight, p=2))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        output *= self.s
        return output


class RepNet(torch.nn.Module):

    def __init__(self, out_ids, out_attribs):
        """
        Network definition
        :param out_ids:
        :param out_attribs:
        """
        super(RepNet, self).__init__()
        self.out_ids, self.out_attribs = out_ids, out_attribs
        None
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_2 = torch.nn.ReLU(inplace=True)
        self.conv1_3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1_4 = torch.nn.ReLU(inplace=True)
        self.conv1_5 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv1 = torch.nn.Sequential(self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4, self.conv1_5)
        self.conv2_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_2 = torch.nn.ReLU(inplace=True)
        self.conv2_3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2_4 = torch.nn.ReLU(inplace=True)
        self.conv2_5 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv2 = torch.nn.Sequential(self.conv2_1, self.conv2_2, self.conv2_3, self.conv2_4, self.conv2_5)
        self.conv3_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_2 = torch.nn.ReLU(inplace=True)
        self.conv3_3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_4 = torch.nn.ReLU(inplace=True)
        self.conv3_5 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3_6 = torch.nn.ReLU(inplace=True)
        self.conv3_7 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv3 = torch.nn.Sequential(self.conv3_1, self.conv3_2, self.conv3_3, self.conv3_4, self.conv3_5, self.conv3_6, self.conv3_7)
        self.conv4_1_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_1_2 = torch.nn.ReLU(inplace=True)
        self.conv4_1_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_1_4 = torch.nn.ReLU(inplace=True)
        self.conv4_1_5 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_1_6 = torch.nn.ReLU(inplace=True)
        self.conv4_1_7 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv4_1 = torch.nn.Sequential(self.conv4_1_1, self.conv4_1_2, self.conv4_1_3, self.conv4_1_4, self.conv4_1_5, self.conv4_1_6, self.conv4_1_7)
        self.conv4_2_1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_2_2 = torch.nn.ReLU(inplace=True)
        self.conv4_2_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_2_4 = torch.nn.ReLU(inplace=True)
        self.conv4_2_5 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4_2_6 = torch.nn.ReLU(inplace=True)
        self.conv4_2_7 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv4_2 = torch.nn.Sequential(self.conv4_2_1, self.conv4_2_2, self.conv4_2_3, self.conv4_2_4, self.conv4_2_5, self.conv4_2_6, self.conv4_2_7)
        self.conv5_1_1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_1_2 = torch.nn.ReLU(inplace=True)
        self.conv5_1_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_1_4 = torch.nn.ReLU(inplace=True)
        self.conv5_1_5 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_1_6 = torch.nn.ReLU(inplace=True)
        self.conv5_1_7 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv5_1 = torch.nn.Sequential(self.conv5_1_1, self.conv5_1_2, self.conv5_1_3, self.conv5_1_4, self.conv5_1_5, self.conv5_1_6, self.conv5_1_7)
        self.conv5_2_1 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_2_2 = torch.nn.ReLU(inplace=True)
        self.conv5_2_3 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_2_4 = torch.nn.ReLU(inplace=True)
        self.conv5_2_5 = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5_2_6 = torch.nn.ReLU(inplace=True)
        self.conv5_2_7 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.conv5_2 = torch.nn.Sequential(self.conv5_2_1, self.conv5_2_2, self.conv5_2_3, self.conv5_2_4, self.conv5_2_5, self.conv5_2_6, self.conv5_2_7)
        self.FC6_1_1 = torch.nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.FC6_1_2 = torch.nn.ReLU(inplace=True)
        self.FC6_1_3 = torch.nn.Dropout(p=0.5)
        self.FC6_1_4 = torch.nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.FC6_1_5 = torch.nn.ReLU(inplace=True)
        self.FC6_1_6 = torch.nn.Dropout(p=0.5)
        self.FC6_1 = torch.nn.Sequential(self.FC6_1_1, self.FC6_1_2, self.FC6_1_3, self.FC6_1_4, self.FC6_1_5, self.FC6_1_6)
        self.FC6_2_1 = copy.deepcopy(self.FC6_1_1)
        self.FC6_2_2 = copy.deepcopy(self.FC6_1_2)
        self.FC6_2_3 = copy.deepcopy(self.FC6_1_3)
        self.FC6_2_4 = copy.deepcopy(self.FC6_1_4)
        self.FC6_2_5 = copy.deepcopy(self.FC6_1_5)
        self.FC6_2_6 = copy.deepcopy(self.FC6_1_6)
        self.FC6_2 = torch.nn.Sequential(self.FC6_2_1, self.FC6_2_2, self.FC6_2_3, self.FC6_2_4, self.FC6_2_5, self.FC6_2_6)
        self.FC7_1 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)
        self.FC7_2 = torch.nn.Linear(in_features=4096, out_features=1000, bias=True)
        self.FC_8 = torch.nn.Linear(in_features=2000, out_features=1024)
        self.attrib_classifier = torch.nn.Linear(in_features=1000, out_features=out_attribs)
        self.arc_fc_br2 = ArcFC(in_features=1000, out_features=out_ids, s=30.0, m=0.5, easy_margin=False)
        self.arc_fc_br3 = ArcFC(in_features=1024, out_features=out_ids, s=30.0, m=0.5, easy_margin=False)
        self.shared_layers = torch.nn.Sequential(self.conv1, self.conv2, self.conv3)
        self.branch_1_feats = torch.nn.Sequential(self.shared_layers, self.conv4_1, self.conv5_1)
        self.branch_1_fc = torch.nn.Sequential(self.FC6_1, self.FC7_1)
        self.branch_1 = torch.nn.Sequential(self.branch_1_feats, self.branch_1_fc)
        self.branch_2_feats = torch.nn.Sequential(self.shared_layers, self.conv4_2, self.conv5_2)
        self.branch_2_fc = torch.nn.Sequential(self.FC6_2, self.FC7_2)
        self.branch_2 = torch.nn.Sequential(self.branch_2_feats, self.branch_2_fc)

    def forward(self, X, branch, label=None):
        """
        :param X:
        :param branch:
        :param label:
        :return:
        """
        N = X.size(0)
        if branch == 1:
            X = self.branch_1_feats(X)
            X = X.view(N, -1)
            X = self.branch_1_fc(X)
            assert X.size() == (N, 1000)
            X = self.attrib_classifier(X)
            assert X.size() == (N, self.out_attribs)
            return X
        elif branch == 2:
            if label is None:
                None
                return None
            X = self.branch_2_feats(X)
            X = X.view(N, -1)
            X = self.branch_2_fc(X)
            assert X.size() == (N, 1000)
            X = self.arc_fc_br2.forward(input=X, label=label)
            assert X.size() == (N, self.out_ids)
            return X
        elif branch == 3:
            if label is None:
                None
                return None
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)
            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)
            assert fusion_feats.size() == (N, 2000)
            X = self.FC_8(fusion_feats)
            X = self.arc_fc_br3.forward(input=X, label=label)
            assert X.size() == (N, self.out_ids)
            return X
        elif branch == 4:
            X = self.branch_1_feats(X)
            X = X.view(N, -1)
            X = self.branch_1_fc(X)
            assert X.size() == (N, 1000)
            return X
        elif branch == 5:
            branch_1 = self.branch_1_feats(X)
            branch_2 = self.branch_2_feats(X)
            branch_1 = branch_1.view(N, -1)
            branch_2 = branch_2.view(N, -1)
            branch_1 = self.branch_1_fc(branch_1)
            branch_2 = self.branch_2_fc(branch_2)
            assert branch_1.size() == (N, 1000) and branch_2.size() == (N, 1000)
            fusion_feats = torch.cat((branch_1, branch_2), dim=1)
            assert fusion_feats.size() == (N, 2000)
            X = self.FC_8(fusion_feats)
            assert X.size() == (N, 1024)
            return X
        else:
            None
            return None

