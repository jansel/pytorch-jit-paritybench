import sys
_module = sys.modules[__name__]
del sys
imutils = _module
indexing = _module
pyutils = _module
torchutils = _module
resnet50 = _module
resnet50_cam = _module
resnet50_irn = _module
run_sample = _module
cam_to_ir_label = _module
eval_cam = _module
eval_ins_seg = _module
eval_sem_seg = _module
make_cam = _module
make_cocoann = _module
make_ins_seg_labels = _module
make_sem_seg_labels = _module
train_cam = _module
train_irn = _module
dataloader = _module
make_cls_labels = _module

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


import torch.nn.functional as F


import numpy as np


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


from torch import multiprocessing


from torch import cuda


from torch.utils.data import DataLoader


from torch.backends import cudnn


class FixedBatchNorm(nn.BatchNorm2d):

    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var,
            self.weight, self.bias, training=False, eps=self.eps)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = FixedBatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation, bias=False, dilation=dilation)
        self.bn2 = FixedBatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = FixedBatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

    def __init__(self, block, layers, strides=(2, 2, 2, 2), dilations=(1, 1,
        1, 1)):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0],
            padding=3, bias=False)
        self.bn1 = FixedBatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
            dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=
            strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=
            strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=
            strides[3], dilation=dilations[3])
        self.inplanes = 1024

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                FixedBatchNorm(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, dilation=1)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
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


model_urls = {'resnet50':
    'https://download.pytorch.org/models/resnet50-19c8e357.pth'}


def resnet50(pretrained=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1)
            )
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1,
            self.resnet50.relu, self.resnet50.maxpool, self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)
        self.classifier = nn.Conv2d(2048, 20, 1, bias=False)
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.
            stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier])

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x).detach()
        x = self.stage3(x)
        x = self.stage4(x)
        x = torchutils.gap2d(x, keepdims=True)
        x = self.classifier(x)
        x = x.view(-1, 20)
        return x

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):
        return list(self.backbone.parameters()), list(self.newly_added.
            parameters())


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.resnet50 = resnet50.resnet50(pretrained=True, strides=[2, 2, 2, 1]
            )
        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1,
            self.resnet50.relu, self.resnet50.maxpool)
        self.stage2 = nn.Sequential(self.resnet50.layer1)
        self.stage3 = nn.Sequential(self.resnet50.layer2)
        self.stage4 = nn.Sequential(self.resnet50.layer3)
        self.stage5 = nn.Sequential(self.resnet50.layer4)
        self.mean_shift = Net.MeanShift(2)
        self.fc_edge1 = nn.Sequential(nn.Conv2d(64, 32, 1, bias=False), nn.
            GroupNorm(4, 32), nn.ReLU(inplace=True))
        self.fc_edge2 = nn.Sequential(nn.Conv2d(256, 32, 1, bias=False), nn
            .GroupNorm(4, 32), nn.ReLU(inplace=True))
        self.fc_edge3 = nn.Sequential(nn.Conv2d(512, 32, 1, bias=False), nn
            .GroupNorm(4, 32), nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False), nn.ReLU(inplace=True))
        self.fc_edge4 = nn.Sequential(nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32), nn.Upsample(scale_factor=4, mode=
            'bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_edge5 = nn.Sequential(nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32), nn.Upsample(scale_factor=4, mode=
            'bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)
        self.fc_dp1 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.
            GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.fc_dp2 = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False), nn.
            GroupNorm(16, 128), nn.ReLU(inplace=True))
        self.fc_dp3 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False), nn.
            GroupNorm(16, 256), nn.ReLU(inplace=True))
        self.fc_dp4 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False), nn
            .GroupNorm(16, 256), nn.Upsample(scale_factor=2, mode=
            'bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_dp5 = nn.Sequential(nn.Conv2d(2048, 256, 1, bias=False), nn
            .GroupNorm(16, 256), nn.Upsample(scale_factor=2, mode=
            'bilinear', align_corners=False), nn.ReLU(inplace=True))
        self.fc_dp6 = nn.Sequential(nn.Conv2d(768, 256, 1, bias=False), nn.
            GroupNorm(16, 256), nn.Upsample(scale_factor=2, mode='bilinear',
            align_corners=False), nn.ReLU(inplace=True))
        self.fc_dp7 = nn.Sequential(nn.Conv2d(448, 256, 1, bias=False), nn.
            GroupNorm(16, 256), nn.ReLU(inplace=True), nn.Conv2d(256, 2, 1,
            bias=False), self.mean_shift)
        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.
            stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2,
            self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])
        self.dp_layers = nn.ModuleList([self.fc_dp1, self.fc_dp2, self.
            fc_dp3, self.fc_dp4, self.fc_dp5, self.fc_dp6, self.fc_dp7])


    class MeanShift(nn.Module):

        def __init__(self, num_features):
            super(Net.MeanShift, self).__init__()
            self.register_buffer('running_mean', torch.zeros(num_features))

        def forward(self, input):
            if self.training:
                return input
            return input - self.running_mean.view(1, 2, 1, 1)

    def forward(self, x):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[(...), :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[(...), :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[(...), :edge2.size(2), :edge2.size(3)]
        edge_out = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4,
            edge5], dim=1))
        dp1 = self.fc_dp1(x1)
        dp2 = self.fc_dp2(x2)
        dp3 = self.fc_dp3(x3)
        dp4 = self.fc_dp4(x4)[(...), :dp3.size(2), :dp3.size(3)]
        dp5 = self.fc_dp5(x5)[(...), :dp3.size(2), :dp3.size(3)]
        dp_up3 = self.fc_dp6(torch.cat([dp3, dp4, dp5], dim=1))[(...), :dp2
            .size(2), :dp2.size(3)]
        dp_out = self.fc_dp7(torch.cat([dp1, dp2, dp_up3], dim=1))
        return edge_out, dp_out

    def trainable_parameters(self):
        return tuple(self.edge_layers.parameters()), tuple(self.dp_layers.
            parameters())

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jiwoon_ahn_irn(_paritybench_base):
    pass
    def test_000(self):
        self._check(FixedBatchNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

