import sys
_module = sys.modules[__name__]
del sys
data_preprosess = _module
anchor = _module
hands2017 = _module
icvl = _module
itop_side = _module
itop_top = _module
k2hpd = _module
model = _module
nyu = _module
resnet = _module
anchor = _module
model = _module
random_erasing = _module
resnet = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.utils.data


import scipy.io as scio


from torch.nn import init


import torch.utils.model_zoo as model_zoo


def shift(shape, stride, anchors):
    shift_h = np.arange(0, shape[0]) * stride
    shift_w = np.arange(0, shape[1]) * stride
    shift_h, shift_w = np.meshgrid(shift_h, shift_w)
    shifts = np.vstack((shift_h.ravel(), shift_w.ravel())).transpose()
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)
        ).transpose((1, 0, 2))
    all_anchors = all_anchors.reshape((K * A, 2))
    return all_anchors


def generate_anchors(P_h=None, P_w=None):
    if P_h is None:
        P_h = np.array([2, 6, 10, 14])
    if P_w is None:
        P_w = np.array([2, 6, 10, 14])
    num_anchors = len(P_h) * len(P_h)
    anchors = np.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k, 1] = P_w[j]
            anchors[k, 0] = P_h[i]
            k += 1
    return anchors


class post_process(nn.Module):

    def __init__(self, P_h=[2, 6], P_w=[2, 6], shape=[48, 26], stride=8,
        thres=8, is_3D=True):
        super(post_process, self).__init__()
        anchors = generate_anchors(P_h=P_h, P_w=P_w)
        self.all_anchors = torch.from_numpy(shift(shape, stride, anchors)
            ).float()
        self.thres = torch.from_numpy(np.array(thres)).float()
        self.is_3D = is_3D

    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0], b.shape[0])
        for i in range(a.shape[1]):
            dis += torch.pow(torch.unsqueeze(a[:, (i)], dim=1) - b[:, (i)], 0.5
                )
        return dis

    def forward(self, heads, voting=False):
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
        batch_size = classifications.shape[0]
        anchor = self.all_anchors
        P_keys = []
        for j in range(batch_size):
            classification = classifications[(j), :, :]
            regression = regressions[(j), :, :, :]
            if self.is_3D:
                depthregression = depthregressions[(j), :, :]
            reg = torch.unsqueeze(anchor, 1) + regression
            reg_weight = F.softmax(classifications[(j), :, :], dim=0)
            reg_weight_xy = torch.unsqueeze(reg_weight, 2).expand(reg_weight
                .shape[0], reg_weight.shape[1], 2)
            P_xy = (reg_weight_xy * reg).sum(0)
            if self.is_3D:
                P_depth = (reg_weight * depthregression).sum(0)
                P_depth = torch.unsqueeze(P_depth, 1)
                P_key = torch.cat((P_xy, P_depth), 1)
                P_keys.append(P_key)
            else:
                P_keys.append(P_xy)
        return torch.stack(P_keys)


class DepthRegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=16, num_classes=15,
        feature_size=256):
        super(DepthRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes,
            kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.
            num_classes)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes)


class RegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=16, num_classes=15,
        feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes * 2,
            kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.
            num_classes, 2)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes, 2)


class ClassificationModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=16, num_classes=15,
        prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes,
            kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.
            num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNetBackBone(nn.Module):

    def __init__(self):
        super(ResNetBackBone, self).__init__()
        modelPreTrain50 = resnet.resnet50(pretrained=True)
        self.model = modelPreTrain50

    def forward(self, x):
        n, c, h, w = x.size()
        x = x[:, 0:1, :, :]
        x = x.expand(n, 3, h, w)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        return x3, x4


class A2J_model(nn.Module):

    def __init__(self, num_classes, is_3D=True):
        super(A2J_model, self).__init__()
        self.is_3D = is_3D
        self.Backbone = ResNetBackBone()
        self.regressionModel = RegressionModel(2048, num_classes=num_classes)
        self.classificationModel = ClassificationModel(1024, num_classes=
            num_classes)
        if is_3D:
            self.DepthRegressionModel = DepthRegressionModel(2048,
                num_classes=num_classes)

    def forward(self, x):
        x3, x4 = self.Backbone(x)
        classification = self.classificationModel(x3)
        regression = self.regressionModel(x4)
        if self.is_3D:
            DepthRegressionModel = self.DepthRegressionModel(x4)
            return classification, regression, DepthRegressionModel
        return classification, regression


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        dilation=dilation, padding=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), nn.BatchNorm2d(planes * block.
                expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class post_process(nn.Module):

    def __init__(self, P_h=[2, 6], P_w=[2, 6], shape=[48, 26], stride=8,
        thres=8, is_3D=True):
        super(post_process, self).__init__()
        anchors = generate_anchors(P_h=P_h, P_w=P_w)
        self.all_anchors = torch.from_numpy(shift(shape, stride, anchors)
            ).float()
        self.thres = torch.from_numpy(np.array(thres)).float()
        self.is_3D = is_3D

    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0], b.shape[0])
        for i in range(a.shape[1]):
            dis += torch.pow(torch.unsqueeze(a[:, (i)], dim=1) - b[:, (i)], 0.5
                )
        return dis

    def forward(self, heads, voting=False):
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
        batch_size = classifications.shape[0]
        anchor = self.all_anchors
        P_keys = []
        for j in range(batch_size):
            classification = classifications[(j), :, :]
            regression = regressions[(j), :, :, :]
            if self.is_3D:
                depthregression = depthregressions[(j), :, :]
            reg = torch.unsqueeze(anchor, 1) + regression
            reg_weight = F.softmax(classifications[(j), :, :], dim=0)
            reg_weight_xy = torch.unsqueeze(reg_weight, 2).expand(reg_weight
                .shape[0], reg_weight.shape[1], 2)
            P_xy = (reg_weight_xy * reg).sum(0)
            if self.is_3D:
                P_depth = (reg_weight * depthregression).sum(0)
                P_depth = torch.unsqueeze(P_depth, 1)
                P_key = torch.cat((P_xy, P_depth), 1)
                P_keys.append(P_key)
            else:
                P_keys.append(P_xy)
        return torch.stack(P_keys)


class A2J_loss(nn.Module):

    def __init__(self, P_h=[2, 6], P_w=[2, 6], shape=[8, 4], stride=8,
        thres=[10.0, 20.0], spatialFactor=0.1, img_shape=[0, 0], is_3D=True):
        super(A2J_loss, self).__init__()
        anchors = generate_anchors(P_h=P_h, P_w=P_w)
        self.all_anchors = torch.from_numpy(shift(shape, stride, anchors)
            ).float()
        self.thres = torch.from_numpy(np.array(thres)).float()
        self.spatialFactor = spatialFactor
        self.img_shape = img_shape
        self.is_3D = is_3D

    def calc_distance(self, a, b):
        dis = torch.zeros(a.shape[0], b.shape[0])
        for i in range(a.shape[1]):
            dis += torch.abs(torch.unsqueeze(a[:, (i)], dim=1) - b[:, (i)])
        return dis

    def forward(self, heads, annotations):
        alpha = 0.25
        gamma = 2.0
        if self.is_3D:
            classifications, regressions, depthregressions = heads
        else:
            classifications, regressions = heads
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
        anchor = self.all_anchors
        anchor_regression_loss_tuple = []
        for j in range(batch_size):
            classification = classifications[(j), :, :]
            regression = regressions[(j), :, :, :]
            if self.is_3D:
                depthregression = depthregressions[(j), :, :]
            bbox_annotation = annotations[(j), :, :]
            reg_weight = F.softmax(classification, dim=0)
            reg_weight_xy = torch.unsqueeze(reg_weight, 2).expand(reg_weight
                .shape[0], reg_weight.shape[1], 2)
            gt_xy = bbox_annotation[:, :2]
            anchor_diff = torch.abs(gt_xy - (reg_weight_xy * torch.
                unsqueeze(anchor, 1)).sum(0))
            anchor_loss = torch.where(torch.le(anchor_diff, 1), 0.5 * 1 *
                torch.pow(anchor_diff, 2), anchor_diff - 0.5 / 1)
            anchor_regression_loss = anchor_loss.mean()
            anchor_regression_loss_tuple.append(anchor_regression_loss)
            reg = torch.unsqueeze(anchor, 1) + regression
            regression_diff = torch.abs(gt_xy - (reg_weight_xy * reg).sum(0))
            regression_loss = torch.where(torch.le(regression_diff, 1), 0.5 *
                1 * torch.pow(regression_diff, 2), regression_diff - 0.5 / 1)
            regression_loss = regression_loss.mean() * self.spatialFactor
            if self.is_3D:
                gt_depth = bbox_annotation[:, (2)]
                regression_diff_depth = torch.abs(gt_depth - (reg_weight *
                    depthregression).sum(0))
                regression_loss_depth = torch.where(torch.le(
                    regression_diff_depth, 3), 0.5 * (1 / 3) * torch.pow(
                    regression_diff_depth, 2), regression_diff_depth - 0.5 /
                    (1 / 3))
                regression_loss += regression_diff_depth.mean()
            regression_losses.append(regression_loss)
        return torch.stack(anchor_regression_loss_tuple).mean(dim=0,
            keepdim=True), torch.stack(regression_losses).mean(dim=0,
            keepdim=True)


class DepthRegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=16, num_classes=15,
        feature_size=256):
        super(DepthRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes,
            kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.
            num_classes)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes)


class RegressionModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=16, num_classes=15,
        feature_size=256):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes * 2,
            kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.
            num_classes, 2)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes, 2)


class ClassificationModel(nn.Module):

    def __init__(self, num_features_in, num_anchors=16, num_classes=15,
        prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3,
            padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3,
            padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes,
            kernel_size=3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.act4(out)
        out = self.output(out)
        out1 = out.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.
            num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNetBackBone(nn.Module):

    def __init__(self):
        super(ResNetBackBone, self).__init__()
        modelPreTrain50 = resnet.resnet50(pretrained=True)
        self.model = modelPreTrain50

    def forward(self, x):
        n, c, h, w = x.size()
        x = x[:, 0:1, :, :]
        x = x.expand(n, 3, h, w)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)
        return x3, x4


class A2J_model(nn.Module):

    def __init__(self, num_classes, is_3D=True):
        super(A2J_model, self).__init__()
        self.is_3D = is_3D
        self.Backbone = ResNetBackBone()
        self.regressionModel = RegressionModel(2048, num_classes=num_classes)
        self.classificationModel = ClassificationModel(1024, num_classes=
            num_classes)
        if is_3D:
            self.DepthRegressionModel = DepthRegressionModel(2048,
                num_classes=num_classes)

    def forward(self, x):
        x3, x4 = self.Backbone(x)
        classification = self.classificationModel(x3)
        regression = self.regressionModel(x4)
        if self.is_3D:
            DepthRegressionModel = self.DepthRegressionModel(x4)
            return classification, regression, DepthRegressionModel
        return classification, regression


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), nn.BatchNorm2d(planes * block.
                expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_zhangboshen_A2J(_paritybench_base):
    pass
    def test_000(self):
        self._check(DepthRegressionModel(*[], **{'num_features_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(RegressionModel(*[], **{'num_features_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ClassificationModel(*[], **{'num_features_in': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

