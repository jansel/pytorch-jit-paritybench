import sys
_module = sys.modules[__name__]
del sys
dorn = _module
models = _module
dorn = _module
modules = _module
backbones = _module
resnet = _module
OrdinalRegression = _module
decoders = _module
SceneUnderstandingModule = _module
encoders = _module
layers = _module
basic_layers = _module
losses = _module
ordinal_regression_loss = _module

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
xrange = range
wraps = functools.wraps


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import BatchNorm2d


import math


from torch.nn.modules.utils import _pair


from torch.nn.modules.utils import _triple


class OrdinalRegressionLayer(nn.Module):

    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: NxCxHxW, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1 x H x W
                 ord prob is the probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size()
        ord_num = C // 2
        x = x.view(-1, 2, ord_num, H, W)
        if self.training:
            prob = F.log_softmax(x, dim=1).view(N, C, H, W)
            return prob
        ord_prob = F.softmax(x, dim=1)[:, (0), :, :, :]
        ord_label = torch.sum(ord_prob > 0.5, dim=1)
        return ord_prob, ord_label


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, beta, discretization='SID'):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_label(self, gt):
        N, H, W = gt.shape
        ord_c0 = torch.ones(N, self.ord_num, H, W)
        if self.discretization == 'SID':
            label = self.ord_num * torch.log(gt) / np.log(self.beta)
        else:
            label = self.ord_num * (gt - 1.0) / (self.beta - 1.0)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False).view(1, self.ord_num, 1, 1)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = mask > label
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        valid_mask = gt > 0.0
        ord_label, mask = self._create_ord_label(gt)
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask]
        return loss.mean()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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
        out = out + residual
        out = self.relu_inplace(out)
        return out


affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNetBackbone(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3])
        if pretrained:
            saved_state_dict = torch.load('./dp/modules/backbones/pretrained_models/resnet101-imagenet.pth', map_location='cpu')
            new_params = self.backbone.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[0] == 'fc':
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]
            self.backbone.load_state_dict(new_params)

    def forward(self, input):
        return self.backbone(input)


class FullImageEncoder(nn.Module):

    def __init__(self, h, w, kernel_size):
        super(FullImageEncoder, self).__init__()
        self.global_pooling = nn.AvgPool2d(kernel_size, stride=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout2d(p=0.5)
        self.h = h // kernel_size + 1
        self.w = w // kernel_size + 1
        self.global_fc = nn.Linear(2048 * self.h * self.w, 512)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(512, 512, 1)

    def forward(self, x):
        x1 = self.global_pooling(x)
        x2 = self.dropout(x1)
        x3 = x2.view(-1, 2048 * self.h * self.w)
        x4 = self.relu(self.global_fc(x3))
        x4 = x4.view(-1, 512, 1, 1)
        x5 = self.conv1(x4)
        return x5


def consistent_padding_with_dilation(padding, dilation, dim=2):
    assert dim == 2 or dim == 3, 'Convolution layer only support 2D and 3D'
    if dim == 2:
        padding = _pair(padding)
        dilation = _pair(dilation)
    else:
        padding = _triple(padding)
        dilation = _triple(dilation)
    padding = list(padding)
    for d in range(dim):
        padding[d] = dilation[d] if dilation[d] > 1 else padding[d]
    padding = tuple(padding)
    return padding, dilation


def conv_bn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True), nn.ReLU(inplace=True))


class SceneUnderstandingModule(nn.Module):

    def __init__(self, ord_num, size, kernel_size, pyramid=[6, 12, 18], batch_norm=False):
        super(SceneUnderstandingModule, self).__init__()
        assert len(size) == 2
        assert len(pyramid) == 3
        self.size = size
        h, w = self.size
        self.encoder = FullImageEncoder(h // 8, w // 8, kernel_size)
        self.aspp1 = nn.Sequential(conv_bn_relu(batch_norm, 2048, 512, kernel_size=1, padding=0), conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0))
        self.aspp2 = nn.Sequential(conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[0], dilation=pyramid[0]), conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0))
        self.aspp3 = nn.Sequential(conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[1], dilation=pyramid[1]), conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0))
        self.aspp4 = nn.Sequential(conv_bn_relu(batch_norm, 2048, 512, kernel_size=3, padding=pyramid[2], dilation=pyramid[2]), conv_bn_relu(batch_norm, 512, 512, kernel_size=1, padding=0))
        self.concat_process = nn.Sequential(nn.Dropout2d(p=0.5), conv_bn_relu(batch_norm, 512 * 5, 2048, kernel_size=1, padding=0), nn.Dropout2d(p=0.5), nn.Conv2d(2048, ord_num * 2, 1))

    def forward(self, x):
        N, C, H, W = x.shape
        x1 = self.encoder(x)
        x1 = F.interpolate(x1, size=(H, W), mode='bilinear', align_corners=True)
        x2 = self.aspp1(x)
        x3 = self.aspp2(x)
        x4 = self.aspp3(x)
        x5 = self.aspp4(x)
        x6 = torch.cat((x1, x2, x3, x4, x5), dim=1)
        out = self.concat_process(x6)
        out = F.interpolate(out, size=self.size, mode='bilinear', align_corners=True)
        return out


class DepthPredModel(nn.Module):

    def __init__(self, ord_num=90, gamma=1.0, beta=80.0, input_size=(385, 513), kernel_size=16, pyramid=[8, 12, 16], batch_norm=False, discretization='SID', pretrained=True):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.ord_num = ord_num
        self.gamma = gamma
        self.beta = beta
        self.discretization = discretization
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.SceneUnderstandingModule = SceneUnderstandingModule(ord_num, size=input_size, kernel_size=kernel_size, pyramid=pyramid, batch_norm=batch_norm)
        self.regression_layer = OrdinalRegressionLayer()
        self.criterion = OrdinalRegressionLoss(ord_num, beta, discretization)

    def forward(self, image, target=None):
        """
        :param image: RGB image, torch.Tensor, Nx3xHxW
        :param target: ground truth depth, torch.Tensor, NxHxW
        :return: output: if training, return loss, torch.Float,
                         else return {"target": depth, "prob": prob, "label": label},
                         depth: predicted depth, torch.Tensor, NxHxW
                         prob: probability of each label, torch.Tensor, NxCxHxW, C is number of label
                         label: predicted label, torch.Tensor, NxHxW
        """
        N, C, H, W = image.shape
        feat = self.backbone(image)
        feat = self.SceneUnderstandingModule(feat)
        if self.training:
            prob = self.regression_layer(feat)
            loss = self.criterion(prob, target)
            return loss
        prob, label = self.regression_layer(feat)
        if self.discretization == 'SID':
            t0 = torch.exp(np.log(self.beta) * label.float() / self.ord_num)
            t1 = torch.exp(np.log(self.beta) * (label.float() + 1) / self.ord_num)
        else:
            t0 = 1.0 + (self.beta - 1.0) * label.float() / self.ord_num
            t1 = 1.0 + (self.beta - 1.0) * (label.float() + 1) / self.ord_num
        depth = (t0 + t1) / 2 - self.gamma
        return {'target': [depth], 'prob': [prob], 'label': [label]}


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (OrdinalRegressionLayer,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_dontLoveBugs_DORN_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

