import sys
_module = sys.modules[__name__]
del sys
demo = _module
config = _module
ctpn_model = _module
ctpn_predict = _module
ctpn_utils = _module
ocr = _module
crnn = _module
crnn_recognizer = _module
keys = _module
test_one = _module
crnn = _module
crnn_recognizer = _module
mydataset = _module
online_test = _module
recognizer = _module
split_train_test = _module
train_pytorch_ctc = _module
train_warp_ctc = _module
train_warp_ctc_v2 = _module
trans = _module
trans_utils = _module
utils = _module
ctpn_model = _module
ctpn_model_v2 = _module
ctpn_predict = _module
ctpn_train = _module
data = _module
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


import torch.nn as nn


import torch.nn.functional as F


import torchvision.models as models


import numpy as np


from collections import OrderedDict


import torchvision.transforms as transforms


from torch.autograd import Variable


import random


import torch.optim as optim


import torch.utils.data


from torch.nn import CTCLoss


import collections


class RPN_REGR_Loss(nn.Module):

    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        """
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        """
        try:
            cls = target[(0), :, (0)]
            regr = target[(0), :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, (0)]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            None
            loss = torch.tensor(0.0)
        return loss


class RPN_CLS_Loss(nn.Module):

    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device

    def forward(self, input, target):
        y_true = target[0][0]
        cls_keep = (y_true != -1).nonzero()[:, (0)]
        cls_true = y_true[cls_keep].long()
        cls_pred = input[0][cls_keep]
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)
        loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
        return loss


class basic_conv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(basic_conv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):

    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.rpn(x)
        x1 = x.permute(0, 2, 3, 1).contiguous()
        b = x1.size()
        x1 = x1.view(b[0] * b[1], b[2], b[3])
        x2, _ = self.brnn(x1)
        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x3 = self.lstm_fc(x3)
        x = x3
        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        regr = regr.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)
        return cls, regr


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.conv1 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        x = self.pool1(self.relu1(self.conv1(input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3_2(self.conv3_2(self.relu3_1(self.bn3(self.conv3_1(x))))))
        x = self.pool4(self.relu4_2(self.conv4_2(self.relu4_1(self.bn4(self.conv4_1(x))))))
        conv = self.relu5(self.bn5(self.conv5(x)))
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


class CRNN_v2(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN_v2, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.conv1_1 = nn.Conv2d(nc, 32, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU(True)
        self.conv2_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(128, 96, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(96)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(96, 192, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(192)
        self.relu3_2 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv4_1 = nn.Conv2d(192, 128, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.bn5 = nn.BatchNorm2d(256)
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        x = self.pool1(self.relu1_2(self.bn1_2(self.conv1_2(self.relu1_1(self.bn1_1(self.conv1_1(input)))))))
        x = self.pool2(self.relu2_2(self.bn2_2(self.conv2_2(self.relu2_1(self.bn2_1(self.conv2_1(x)))))))
        x = self.pool3(self.relu3_2(self.bn3_2(self.conv3_2(self.relu3_1(self.bn3_1(self.conv3_1(x)))))))
        x = self.pool4(self.relu4_2(self.bn4_2(self.conv4_2(self.relu4_1(self.bn4_1(self.conv4_1(x)))))))
        conv = self.bn5(x)
        b, c, h, w = conv.size()
        assert h == 2, 'the height of conv must be 2'
        conv = conv.reshape([b, c * h, w])
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


def conv3x3(nIn, nOut, stride=1):
    return nn.Conv2d(nIn, nOut, kernel_size=3, stride=stride, padding=1, bias=False)


class basic_res_block(nn.Module):

    def __init__(self, nIn, nOut, stride=1, downsample=None):
        super(basic_res_block, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(nIn, nOut, stride)
        m['bn1'] = nn.BatchNorm2d(nOut)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(nOut, nOut)
        m['bn2'] = nn.BatchNorm2d(nOut)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out


class CRNN_res(nn.Module):

    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN_res, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.conv1 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.res1 = basic_res_block(64, 64)
        down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(128))
        self.res2_1 = basic_res_block(64, 128, 2, down1)
        self.res2_2 = basic_res_block(128, 128)
        down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(256))
        self.res3_1 = basic_res_block(128, 256, 2, down2)
        self.res3_2 = basic_res_block(256, 256)
        self.res3_3 = basic_res_block(256, 256)
        down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=(2, 1), bias=False), nn.BatchNorm2d(512))
        self.res4_1 = basic_res_block(256, 512, (2, 1), down3)
        self.res4_2 = basic_res_block(512, 512)
        self.res4_3 = basic_res_block(512, 512)
        self.pool = nn.AvgPool2d((2, 2), (2, 1), (0, 1))
        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        x = self.res1(self.relu1(self.conv1(input)))
        x = self.res2_2(self.res2_1(x))
        x = self.res3_3(self.res3_2(self.res3_1(x)))
        x = self.res4_3(self.res4_2(self.res4_1(x)))
        x = self.pool(x)
        conv = self.relu5(self.bn5(self.conv5(x)))
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.conv1 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        x = self.pool1(self.relu1(self.conv1(input)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3_2(self.conv3_2(self.relu3_1(self.bn3(self.conv3_1(x))))))
        x = self.pool4(self.relu4_2(self.conv4_2(self.relu4_1(self.bn4(self.conv4_1(x))))))
        conv = self.relu5(self.bn5(self.conv5(x)))
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


class CRNN_v2(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, leakyRelu=False):
        super(CRNN_v2, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.conv1_1 = nn.Conv2d(nc, 32, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU(True)
        self.conv1_2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU(True)
        self.conv2_2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = nn.Conv2d(128, 96, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(96)
        self.relu3_1 = nn.ReLU(True)
        self.conv3_2 = nn.Conv2d(96, 192, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(192)
        self.relu3_2 = nn.ReLU(True)
        self.pool3 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.conv4_1 = nn.Conv2d(192, 128, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.relu4_1 = nn.ReLU(True)
        self.conv4_2 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.ReLU(True)
        self.pool4 = nn.MaxPool2d((2, 2), (2, 1), (0, 1))
        self.bn5 = nn.BatchNorm2d(256)
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        x = self.pool1(self.relu1_2(self.bn1_2(self.conv1_2(self.relu1_1(self.bn1_1(self.conv1_1(input)))))))
        x = self.pool2(self.relu2_2(self.bn2_2(self.conv2_2(self.relu2_1(self.bn2_1(self.conv2_1(x)))))))
        x = self.pool3(self.relu3_2(self.bn3_2(self.conv3_2(self.relu3_1(self.bn3_1(self.conv3_1(x)))))))
        x = self.pool4(self.relu4_2(self.bn4_2(self.conv4_2(self.relu4_1(self.bn4_1(self.conv4_1(x)))))))
        conv = self.bn5(x)
        b, c, h, w = conv.size()
        assert h == 2, 'the height of conv must be 2'
        conv = conv.reshape([b, c * h, w])
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


class basic_res_block(nn.Module):

    def __init__(self, nIn, nOut, stride=1, downsample=None):
        super(basic_res_block, self).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3(nIn, nOut, stride)
        m['bn1'] = nn.BatchNorm2d(nOut)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = conv3x3(nOut, nOut)
        m['bn2'] = nn.BatchNorm2d(nOut)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.group1(x) + residual
        out = self.relu(out)
        return out


class CRNN_res(nn.Module):

    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN_res, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        self.conv1 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.res1 = basic_res_block(64, 64)
        down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(128))
        self.res2_1 = basic_res_block(64, 128, 2, down1)
        self.res2_2 = basic_res_block(128, 128)
        down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False), nn.BatchNorm2d(256))
        self.res3_1 = basic_res_block(128, 256, 2, down2)
        self.res3_2 = basic_res_block(256, 256)
        self.res3_3 = basic_res_block(256, 256)
        down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=(2, 1), bias=False), nn.BatchNorm2d(512))
        self.res4_1 = basic_res_block(256, 512, (2, 1), down3)
        self.res4_2 = basic_res_block(512, 512)
        self.res4_3 = basic_res_block(512, 512)
        self.pool = nn.AvgPool2d((2, 2), (2, 1), (0, 1))
        self.conv5 = nn.Conv2d(512, 512, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(True)
        self.rnn = nn.Sequential(BidirectionalLSTM(512, nh, nh), BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        x = self.res1(self.relu1(self.conv1(input)))
        x = self.res2_2(self.res2_1(x))
        x = self.res3_3(self.res3_2(self.res3_1(x)))
        x = self.res4_3(self.res4_2(self.res4_1(x)))
        x = self.pool(x)
        conv = self.relu5(self.bn5(self.conv5(x)))
        b, c, h, w = conv.size()
        assert h == 1, 'the height of conv must be 1'
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


class RPN_REGR_Loss(nn.Module):

    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        """
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        """
        try:
            cls = target[(0), :, (0)]
            regr = target[(0), :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, (0)]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            None
            loss = torch.tensor(0.0)
        return loss


_global_config['RPN_TOTAL_NUM'] = 4


_global_config['OHEM'] = 4


class RPN_CLS_Loss(nn.Module):

    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none')
        self.pos_neg_ratio = 3

    def forward(self, input, target):
        if config.OHEM:
            cls_gt = target[0][0]
            num_pos = 0
            loss_pos_sum = 0
            if len((cls_gt == 1).nonzero()) != 0:
                cls_pos = (cls_gt == 1).nonzero()[:, (0)]
                gt_pos = cls_gt[cls_pos].long()
                cls_pred_pos = input[0][cls_pos]
                loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))
                loss_pos_sum = loss_pos.sum()
                num_pos = len(loss_pos)
            cls_neg = (cls_gt == 0).nonzero()[:, (0)]
            gt_neg = cls_gt[cls_neg].long()
            cls_pred_neg = input[0][cls_neg]
            loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
            loss_neg_topK, _ = torch.topk(loss_neg, min(len(loss_neg), config.RPN_TOTAL_NUM - num_pos))
            loss_cls = loss_pos_sum + loss_neg_topK.sum()
            loss_cls = loss_cls / config.RPN_TOTAL_NUM
            return loss_cls
        else:
            y_true = target[0][0]
            cls_keep = (y_true != -1).nonzero()[:, (0)]
            cls_true = y_true[cls_keep].long()
            cls_pred = input[0][cls_keep]
            loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)
            loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
            return loss


class basic_conv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(basic_conv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):

    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.rpn(x)
        x1 = x.permute(0, 2, 3, 1).contiguous()
        b = x1.size()
        x1 = x1.view(b[0] * b[1], b[2], b[3])
        x2, _ = self.brnn(x1)
        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x3 = self.lstm_fc(x3)
        x = x3
        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        regr = regr.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)
        return cls, regr


class RPN_REGR_Loss(nn.Module):

    def __init__(self, device, sigma=9.0):
        super(RPN_REGR_Loss, self).__init__()
        self.sigma = sigma
        self.device = device

    def forward(self, input, target):
        """
        smooth L1 loss
        :param input:y_preds
        :param target: y_true
        :return:
        """
        try:
            cls = target[(0), :, (0)]
            regr = target[(0), :, 1:3]
            regr_keep = (cls == 1).nonzero()[:, (0)]
            regr_true = regr[regr_keep]
            regr_pred = input[0][regr_keep]
            diff = torch.abs(regr_true - regr_pred)
            less_one = (diff < 1.0 / self.sigma).float()
            loss = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1 - less_one) * (diff - 0.5 / self.sigma)
            loss = torch.sum(loss, 1)
            loss = torch.mean(loss) if loss.numel() > 0 else torch.tensor(0.0)
        except Exception as e:
            None
            loss = torch.tensor(0.0)
        return loss


class RPN_CLS_Loss(nn.Module):

    def __init__(self, device):
        super(RPN_CLS_Loss, self).__init__()
        self.device = device

    def forward(self, input, target):
        y_true = target[0][0]
        cls_keep = (y_true != -1).nonzero()[:, (0)]
        cls_true = y_true[cls_keep].long()
        cls_pred = input[0][cls_keep]
        loss = F.nll_loss(F.log_softmax(cls_pred, dim=-1), cls_true)
        loss = torch.clamp(torch.mean(loss), 0, 10) if loss.numel() > 0 else torch.tensor(0.0)
        return loss


class RPN_Loss(nn.Module):

    def __init__(self, device):
        super(RPN_Loss, self).__init__()
        self.device = device
        self.L_cls = nn.CrossEntropyLoss(reduction='none')
        self.L_regr = nn.SmoothL1Loss()
        self.L_refi = nn.SmoothL1Loss()
        self.pos_neg_ratio = 3

    def forward(self, cls, regr, refi, target_cls, target_regr, target_refi):
        cls_gt = target_cls[0][0]
        cls_pos = (cls_gt == 1).nonzero()[:, (0)]
        gt_pos = cls_gt[cls_pos].long()
        cls_pred_pos = input[0][cls_pos]
        cls_neg = (cls_gt == 0).nonzero()[:, (0)]
        gt_neg = cls_gt[cls_neg].long()
        cls_pred_neg = input[0][cls_neg]
        loss_pos = self.L_cls(cls_pred_pos.view(-1, 2), gt_pos.view(-1))
        loss_neg = self.L_cls(cls_pred_neg.view(-1, 2), gt_neg.view(-1))
        loss_neg_topK, _ = torch.topk(loss_neg, min(len(loss_neg), len(loss_pos) * self.pos_neg_ratio))
        loss_cls = loss_pos.mean() + loss_neg_topK.mean()
        return loss_cls


class basic_conv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(basic_conv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CTPN_Model(nn.Module):

    def __init__(self):
        super().__init__()
        base_model = models.vgg16(pretrained=False)
        layers = list(base_model.features)[:-1]
        self.base_layers = nn.Sequential(*layers)
        self.rpn = basic_conv(512, 512, 3, 1, 1, bn=False)
        self.brnn = nn.GRU(512, 128, bidirectional=True, batch_first=True)
        self.lstm_fc = basic_conv(256, 512, 1, 1, relu=True, bn=False)
        self.rpn_class = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_regress = basic_conv(512, 10 * 2, 1, 1, relu=False, bn=False)
        self.rpn_refiment = basic_conv(512, 10, 1, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.rpn(x)
        x1 = x.permute(0, 2, 3, 1).contiguous()
        b = x1.size()
        x1 = x1.view(b[0] * b[1], b[2], b[3])
        x2, _ = self.brnn(x1)
        xsz = x.size()
        x3 = x2.view(xsz[0], xsz[2], xsz[3], 256)
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x3 = self.lstm_fc(x3)
        x = x3
        cls = self.rpn_class(x)
        regr = self.rpn_regress(x)
        refi = self.rpn_refiment(x)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        regr = regr.permute(0, 2, 3, 1).contiguous()
        refi = refi.permute(0, 2, 3, 1).contiguous()
        cls = cls.view(cls.size(0), cls.size(1) * cls.size(2) * 10, 2)
        regr = regr.view(regr.size(0), regr.size(1) * regr.size(2) * 10, 2)
        refi = refi.view(refi.size(0), refi.size(1) * refi.size(2) * 10, 1)
        return cls, regr, refi


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BidirectionalLSTM,
     lambda: ([], {'nIn': 4, 'nHidden': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (CTPN_Model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (RPN_CLS_Loss,
     lambda: ([], {'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RPN_REGR_Loss,
     lambda: ([], {'device': 0}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (basic_conv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (basic_res_block,
     lambda: ([], {'nIn': 4, 'nOut': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_courao_ocr_pytorch(_paritybench_base):
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

