import sys
_module = sys.modules[__name__]
del sys
camera = _module
SetPreparation = _module
datasets = _module
pfld = _module
detector = _module
loss = _module
utils = _module
pytorch2onnx = _module
test = _module

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


import torch


import torch.nn as nn


import math


import numpy as np


from torch.autograd import Variable


import torch.nn.functional as F


from collections import OrderedDict


from torch import nn


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(nn.Conv2d(inp, inp * expand_ratio, 1, 1, 
            0, bias=False), nn.BatchNorm2d(inp * expand_ratio), nn.ReLU(
            inplace=True), nn.Conv2d(inp * expand_ratio, inp * expand_ratio,
            3, stride, 1, groups=inp * expand_ratio, bias=False), nn.
            BatchNorm2d(inp * expand_ratio), nn.ReLU(inplace=True), nn.
            Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False), nn.
            BatchNorm2d(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(nn.Conv2d(inp, oup, kernel, stride, padding, bias=
        False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))


class PFLDInference(nn.Module):

    def __init__(self):
        super(PFLDInference, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv3_1 = InvertedResidual(64, 64, 2, False, 2)
        self.block3_2 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_3 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_4 = InvertedResidual(64, 64, 1, True, 2)
        self.block3_5 = InvertedResidual(64, 64, 1, True, 2)
        self.conv4_1 = InvertedResidual(64, 128, 2, False, 2)
        self.conv5_1 = InvertedResidual(128, 128, 1, False, 4)
        self.block5_2 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_3 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_4 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_5 = InvertedResidual(128, 128, 1, True, 4)
        self.block5_6 = InvertedResidual(128, 128, 1, True, 4)
        self.conv6_1 = InvertedResidual(128, 16, 1, False, 2)
        self.conv7 = conv_bn(16, 32, 3, 2)
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)
        self.bn8 = nn.BatchNorm2d(128)
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 196)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)
        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)
        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.relu(self.conv8(x))
        x3 = x3.view(x1.size(0), -1)
        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
        return out1, landmarks


class AuxiliaryNet(nn.Module):

    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(64, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 7, 1)
        self.max_pool1 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 
            10, 3, 1)), ('prelu1', nn.PReLU(10)), ('pool1', nn.MaxPool2d(2,
            2, ceil_mode=True)), ('conv2', nn.Conv2d(10, 16, 3, 1)), (
            'prelu2', nn.PReLU(16)), ('conv3', nn.Conv2d(16, 32, 3, 1)), (
            'prelu3', nn.PReLU(32))]))
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)
        weights = np.load(os.path.join(os.path.dirname(__file__),
            'pnet.npy'), allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a, dim=1)
        return b, a


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 
            28, 3, 1)), ('prelu1', nn.PReLU(28)), ('pool1', nn.MaxPool2d(3,
            2, ceil_mode=True)), ('conv2', nn.Conv2d(28, 48, 3, 1)), (
            'prelu2', nn.PReLU(48)), ('pool2', nn.MaxPool2d(3, 2, ceil_mode
            =True)), ('conv3', nn.Conv2d(48, 64, 2, 1)), ('prelu3', nn.
            PReLU(64)), ('flatten', Flatten()), ('conv4', nn.Linear(576, 
            128)), ('prelu4', nn.PReLU(128))]))
        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)
        weights = np.load(os.path.join(os.path.dirname(__file__),
            'rnet.npy'), allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a, 1)
        return b, a


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 
            32, 3, 1)), ('prelu1', nn.PReLU(32)), ('pool1', nn.MaxPool2d(3,
            2, ceil_mode=True)), ('conv2', nn.Conv2d(32, 64, 3, 1)), (
            'prelu2', nn.PReLU(64)), ('pool2', nn.MaxPool2d(3, 2, ceil_mode
            =True)), ('conv3', nn.Conv2d(64, 64, 3, 1)), ('prelu3', nn.
            PReLU(64)), ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)), (
            'conv4', nn.Conv2d(64, 128, 2, 1)), ('prelu4', nn.PReLU(128)),
            ('flatten', Flatten()), ('conv5', nn.Linear(1152, 256)), (
            'drop5', nn.Dropout(0.25)), ('prelu5', nn.PReLU(256))]))
        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)
        weights = np.load(os.path.join(os.path.dirname(__file__),
            'onet.npy'), allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a, 1)
        return c, b, a


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PFLDLoss(nn.Module):

    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle,
        landmarks, train_batchsize):
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        attributes_w_n = attribute_gt[:, 1:6].float()
        mat_ratio = torch.mean(attributes_w_n, axis=0)
        mat_ratio = torch.Tensor([(1.0 / x if x > 0 else train_batchsize) for
            x in mat_ratio]).to(device)
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1)
        l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt -
            landmarks), axis=1)
        return torch.mean(weight_angle * weight_attribute * l2_distant
            ), torch.mean(l2_distant)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_polarisZhao_PFLD_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'use_res_connect': 4}), [torch.rand([4, 4, 4, 4])], {})

