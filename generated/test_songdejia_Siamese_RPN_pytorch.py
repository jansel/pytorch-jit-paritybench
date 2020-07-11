import sys
_module = sys.modules[__name__]
del sys
code = _module
data_loader = _module
net = _module
otb_SiamRPN = _module
run_SiamRPN = _module
test_siamrpn = _module
train_siamrpn = _module
utils = _module
video2image = _module
vot = _module
vot_SiamRPN = _module
data_loader = _module
net = _module
train_siamrpn = _module
compute_max_sequence_length = _module
process_otb15_gt = _module
process_vid = _module
process_vot15_gt = _module
process_vot15_img = _module
show_img = _module
unzip_otb15 = _module
vis_gt_box = _module

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


import time


import torch


import random


from torchvision import datasets


from torchvision import transforms


from torchvision import utils


import numpy as np


import torch.utils.model_zoo as model_zoo


import torch.nn as nn


import torch.nn.functional as F


import scipy.io as scio


from torch.autograd import Variable


import torch.nn.parallel


import torch.backends.cudnn as cudnn


from torch.nn import init


model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}


class SiameseRPN(nn.Module):

    def __init__(self, test_video=False):
        super(SiameseRPN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3))
        self.k = 5
        self.s = 4
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

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        None

    def forward(self, template, detection):
        template = self.features(template)
        detection = self.features(detection)
        ckernal = self.conv1(template)
        ckernal = ckernal.view(2 * self.k, 256, 4, 4)
        cinput = self.conv3(detection)
        rkernal = self.conv2(template)
        rkernal = rkernal.view(4 * self.k, 256, 4, 4)
        rinput = self.conv4(detection)
        coutput = F.conv2d(cinput, ckernal)
        routput = F.conv2d(rinput, rkernal)
        coutput = coutput.squeeze().permute(1, 2, 0).reshape(-1, 2)
        routput = routput.squeeze().permute(1, 2, 0).reshape(-1, 4)
        return coutput, routput

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        None


class SiameseRPN_bn(nn.Module):

    def __init__(self):
        super(SiameseRPN_bn, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11, stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5), nn.BatchNorm2d(192), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(192, 384, kernel_size=3), nn.BatchNorm2d(384), nn.ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3), nn.BatchNorm2d(256))
        self.k = 5
        self.s = 4
        self.conv1 = nn.Conv2d(256, 2 * self.k * 256, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(2 * self.k * 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4 * self.k * 256, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(4 * self.k * 256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.cconv = nn.Conv2d(256, 2 * self.k, kernel_size=4, bias=False)
        self.rconv = nn.Conv2d(256, 4 * self.k, kernel_size=4, bias=False)

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        None

    def forward(self, template, detection):
        """
        把template的类别,坐标的特征作为检测cconv和rconv的检测器
        把ckernel, rkernel转换到cconv, rconv
        """
        template = self.features(template)
        detection = self.features(detection)
        ckernal = self.bn1(self.conv1(template))
        ckernal = ckernal.view(2 * self.k, 256, 4, 4)
        cinput = self.bn3(self.conv3(detection))
        rkernal = self.bn2(self.conv2(template))
        rkernal = rkernal.view(4 * self.k, 256, 4, 4)
        rinput = self.bn4(self.conv4(detection))
        coutput = F.conv2d(cinput, ckernal)
        routput = F.conv2d(rinput, rkernal)
        """
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('c branch conv1 template  weight', self.conv1.weight[0,0,0,0])
        print('c branch conv3 detection weight', self.conv3.weight[0,0,0,0])
        """
        return coutput, routput

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        None


class MultiBoxLoss(nn.Module):

    def __init__(self):
        super(MultiBoxLoss, self).__init__()

    def forward(self, predictions, targets):
        None
        cout, rout = predictions
        """ class """
        class_pred, class_target = cout, targets[:, (0)].long()
        pos_index, neg_index = list(np.where(class_target == 1)[0]), list(np.where(class_target == 0)[0])
        pos_num, neg_num = len(pos_index), len(neg_index)
        class_pred, class_target = class_pred[pos_index + neg_index], class_target[pos_index + neg_index]
        closs = F.cross_entropy(class_pred, class_target, size_average=False, reduce=False)
        closs = torch.div(torch.sum(closs), 64)
        """ regression """
        reg_pred = rout
        reg_target = targets[:, 1:]
        rloss = F.smooth_l1_loss(reg_pred, reg_target, size_average=False, reduce=False)
        rloss = torch.div(torch.sum(rloss, dim=1), 4)
        rloss = torch.div(torch.sum(rloss[pos_index]), 16)
        loss = closs + rloss
        return closs, rloss, loss, reg_pred, reg_target, pos_index, neg_index

