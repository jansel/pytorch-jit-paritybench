import sys
_module = sys.modules[__name__]
del sys
SRPN = _module
axis = _module
data_otb = _module
download = _module
test_otb = _module
train = _module
video2pic = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch


from torch.nn import Module


from torch.nn import functional as F


import math


from torch.autograd import Variable as V


import numpy as np


model_urls = {'alexnet':
    'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}


class SiameseRPN(nn.Module):

    def __init__(self):
        super(SiameseRPN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,
            stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2), nn.Conv2d(64, 192, kernel_size=5), nn.ReLU(inplace=
            True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 
            384, kernel_size=3), nn.ReLU(inplace=True), nn.Conv2d(384, 256,
            kernel_size=3), nn.ReLU(inplace=True), nn.Conv2d(256, 256,
            kernel_size=3))
        self.k = 5
        self.conv1 = nn.Conv2d(256, 2 * self.k * 256, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4 * self.k * 256, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu4 = nn.ReLU(inplace=True)
        self.cconv = nn.Conv2d(256, 2 * self.k, kernel_size=4, bias=False)
        self.rconv = nn.Conv2d(256, 4 * self.k, kernel_size=4, bias=False)
        self.reset_params()

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in
            model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, template, detection):
        template = self.features(template)
        detection = self.features(detection)
        ckernal = self.conv1(template)
        ckernal = ckernal.view(2 * self.k, 256, 4, 4)
        self.cconv.weight = nn.Parameter(ckernal)
        cinput = self.conv3(detection)
        coutput = self.cconv(cinput)
        rkernal = self.conv2(template)
        rkernal = rkernal.view(4 * self.k, 256, 4, 4)
        self.rconv.weight = nn.Parameter(rkernal)
        rinput = self.conv4(detection)
        routput = self.rconv(rinput)
        return coutput, routput


class SmoothL1Loss(Module):

    def __init__(self, use_gpu):
        super(SmoothL1Loss, self).__init__()
        self.use_gpu = use_gpu
        return

    def forward(self, clabel, target, routput, rlabel):
        rloss = F.smooth_l1_loss(routput, rlabel, size_average=False,
            reduce=False)
        e = torch.eq(clabel.float(), target)
        e = e.squeeze()
        e0, e1, e2, e3, e4 = e[0].unsqueeze(0), e[1].unsqueeze(0), e[2
            ].unsqueeze(0), e[3].unsqueeze(0), e[4].unsqueeze(0)
        eq = torch.cat([e0, e0, e0, e0, e1, e1, e1, e1, e2, e2, e2, e2, e3,
            e3, e3, e3, e4, e4, e4, e4], dim=0).float()
        rloss = rloss.squeeze()
        rloss = torch.mul(eq, rloss)
        rloss = torch.sum(rloss)
        rloss = torch.div(rloss, eq.nonzero().shape[0] + 0.0001)
        return rloss


class Myloss(Module):

    def __init__(self):
        super(Myloss, self).__init__()
        return

    def forward(self, coutput, clabel, target, routput, rlabel, lmbda):
        closs = F.cross_entropy(coutput, clabel)
        rloss = F.smooth_l1_loss(routput, rlabel, size_average=False,
            reduce=False)
        e = torch.eq(clabel.float(), target)
        e = e.squeeze()
        e0, e1, e2, e3, e4 = e[0].unsqueeze(0), e[1].unsqueeze(0), e[2
            ].unsqueeze(0), e[3].unsqueeze(0), e[4].unsqueeze(0)
        eq = torch.cat([e0, e0, e0, e0, e1, e1, e1, e1, e2, e2, e2, e2, e3,
            e3, e3, e3, e4, e4, e4, e4], dim=0).float()
        rloss = rloss.squeeze()
        rloss = torch.mul(eq, rloss)
        rloss = torch.sum(rloss)
        rloss = torch.div(rloss, eq.nonzero().shape[0] + 0.0001)
        loss = torch.add(closs, lmbda, rloss)
        return loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zkisthebest_Siamese_RPN(_paritybench_base):
    pass
