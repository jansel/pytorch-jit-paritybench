import sys
_module = sys.modules[__name__]
del sys
config = _module
dataloader = _module
data_augment = _module
load_data = _module
loader = _module
eval_city = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
setup = _module
eval_script = _module
eval_MR_multisetup = _module
eval_demo = _module
net = _module
l2norm = _module
loss = _module
network = _module
resnet = _module
trainval_caffestyle = _module
trainval_torchstyle = _module
util = _module
functions = _module
nms = _module
py_cpu_nms = _module
nms_wrapper = _module

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


import random


import torch


import numpy as np


from torch.utils.data import Dataset


import torch.nn as nn


from torch.autograd import Function


from torch.autograd import Variable


import torch.nn.init as init


import math


import torch.utils.model_zoo as model_zoo


import time


import torch.optim as optim


from copy import deepcopy


from torch.utils.data import DataLoader


from torchvision.transforms import ToTensor


from torchvision.transforms import Normalize


from torchvision.transforms import Compose


from torchvision.transforms import ColorJitter


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class cls_pos(nn.Module):

    def __init__(self):
        super(cls_pos, self).__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pos_pred, pos_label):
        log_loss = self.bce(pos_pred[:, 0, :, :], pos_label[:, 2, :, :])
        positives = pos_label[:, 2, :, :]
        negatives = pos_label[:, 1, :, :] - pos_label[:, 2, :, :]
        fore_weight = positives * (1.0 - pos_pred[:, 0, :, :]) ** 2
        back_weight = negatives * (1.0 - pos_label[:, 0, :, :]) ** 4.0 * pos_pred[:, 0, :, :] ** 2.0
        focal_weight = fore_weight + back_weight
        assigned_box = torch.sum(pos_label[:, 2, :, :])
        cls_loss = 0.01 * torch.sum(focal_weight * log_loss) / max(1.0, assigned_box)
        return cls_loss


class reg_pos(nn.Module):

    def __init__(self):
        super(reg_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, h_pred, h_label):
        l1_loss = h_label[:, 1, :, :] * self.smoothl1(h_pred[:, 0, :, :] / (h_label[:, 0, :, :] + 1e-10), h_label[:, 0, :, :] / (h_label[:, 0, :, :] + 1e-10))
        reg_loss = torch.sum(l1_loss) / max(1.0, torch.sum(h_label[:, 1, :, :]))
        return reg_loss


class offset_pos(nn.Module):

    def __init__(self):
        super(offset_pos, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, offset_pred, offset_label):
        l1_loss = offset_label[:, 2, :, :].unsqueeze(dim=1) * self.smoothl1(offset_pred, offset_label[:, :2, :, :])
        off_loss = 0.1 * torch.sum(l1_loss) / max(1.0, torch.sum(offset_label[:, 2, :, :]))
        return off_loss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilate, padding=dilate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, receptive_keep=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, dilate=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=1)
        if receptive_keep:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=2)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1000)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion, momentum=0.01))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, dilate))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth', 'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth', 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', 'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth', 'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class CSPNet(nn.Module):

    def __init__(self):
        super(CSPNet, self).__init__()
        resnet = resnet50(pretrained=True, receptive_keep=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)
        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 0)
        nn.init.constant_(self.p4.bias, 0)
        nn.init.constant_(self.p5.bias, 0)
        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)
        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_bn = nn.BatchNorm2d(256, momentum=0.01)
        self.feat_act = nn.ReLU(inplace=True)
        self.pos_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.reg_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.off_conv = nn.Conv2d(256, 2, kernel_size=1)
        nn.init.xavier_normal_(self.feat.weight)
        nn.init.xavier_normal_(self.pos_conv.weight)
        nn.init.xavier_normal_(self.reg_conv.weight)
        nn.init.xavier_normal_(self.off_conv.weight)
        nn.init.constant_(self.pos_conv.bias, -math.log(0.99 / 0.01))
        nn.init.constant_(self.reg_conv.bias, 0)
        nn.init.constant_(self.off_conv.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        p3 = self.p3(x)
        p3 = self.p3_l2(p3)
        x = self.layer3(x)
        p4 = self.p4(x)
        p4 = self.p4_l2(p4)
        x = self.layer4(x)
        p5 = self.p5(x)
        p5 = self.p5_l2(p5)
        cat = torch.cat([p3, p4, p5], dim=1)
        feat = self.feat(cat)
        feat = self.feat_bn(feat)
        feat = self.feat_act(feat)
        x_cls = self.pos_conv(feat)
        x_cls = torch.sigmoid(x_cls)
        x_reg = self.reg_conv(feat)
        x_off = self.off_conv(feat)
        return x_cls, x_reg, x_off


class CSPNet_mod(nn.Module):

    def __init__(self):
        super(CSPNet_mod, self).__init__()
        resnet = resnet50(pretrained=True, receptive_keep=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=0)
        self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=4, padding=0)
        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 0)
        nn.init.constant_(self.p4.bias, 0)
        nn.init.constant_(self.p5.bias, 0)
        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)
        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.feat_act = nn.ReLU(inplace=True)
        self.pos_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.reg_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.off_conv = nn.Conv2d(256, 2, kernel_size=1)
        nn.init.xavier_normal_(self.feat.weight)
        nn.init.xavier_normal_(self.pos_conv.weight)
        nn.init.xavier_normal_(self.reg_conv.weight)
        nn.init.xavier_normal_(self.off_conv.weight)
        nn.init.constant_(self.feat.bias, 0)
        nn.init.constant_(self.reg_conv.bias, -math.log(0.99 / 0.01))
        nn.init.constant_(self.pos_conv.bias, 0)
        nn.init.constant_(self.off_conv.bias, 0)
        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.bn1.parameters():
            p.requires_grad = False
        for p in self.layer1.parameters():
            p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters():
                    p.requires_grad = False
        self.layer2.apply(set_bn_fix)
        self.layer3.apply(set_bn_fix)
        self.layer4.apply(set_bn_fix)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        p3 = self.p3(x)
        p3 = self.p3_l2(p3)
        x = self.layer3(x)
        p4 = self.p4(x)
        p4 = self.p4_l2(p4)
        x = self.layer4(x)
        p5 = self.p5(x)
        p5 = self.p5_l2(p5)
        cat = torch.cat([p3, p4, p5], dim=1)
        feat = self.feat(cat)
        feat = self.feat_act(feat)
        x_cls = self.pos_conv(feat)
        x_cls = torch.sigmoid(x_cls)
        x_reg = self.reg_conv(feat)
        x_off = self.off_conv(feat)
        return x_cls, x_reg, x_off

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            self.conv1.eval()
            self.bn1.eval()
            self.layer1.eval()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.layer2.apply(set_bn_eval)
            self.layer3.apply(set_bn_eval)
            self.layer4.apply(set_bn_eval)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CSPNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (CSPNet_mod,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (L2Norm,
     lambda: ([], {'n_channels': 4, 'scale': 1.0}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (cls_pos,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (offset_pos,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 2, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (reg_pos,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_lwpyr_CSP_pedestrian_detection_in_pytorch(_paritybench_base):
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

    def test_006(self):
        self._check(*TESTCASES[6])

