import sys
_module = sys.modules[__name__]
del sys
common = _module
utils = _module
models = _module
basenet = _module
src = _module
box_utils = _module
detector = _module
first_stage = _module
get_nets = _module
visualization_utils = _module
test_batch_mtcnn = _module
test_camera_light_onnx = _module
test_camera_mtcnn_onnx = _module
test_camera_pfld_onnx = _module
eval = _module
test = _module
logger = _module
misc = _module
osutils = _module
progress = _module
bar = _module
counter = _module
helpers = _module
spinner = _module
setup = _module
test_progress = _module
transforms = _module
visualize = _module
vision = _module
datasets = _module
voc_dataset = _module
nn = _module
mb_tiny = _module
mb_tiny_RFB = _module
multibox_loss = _module
ssd = _module
config = _module
fd_config = _module
data_preprocessing = _module
mb_tiny_RFB_fd = _module
mb_tiny_fd = _module
predictor = _module
ssd = _module
transforms = _module
box_utils = _module
box_utils_numpy = _module
misc = _module

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


import torch.nn as nn


import torch.nn.functional as F


import torchvision.models as models


import numpy as np


import torch


from torch.autograd import Variable


import math


from collections import OrderedDict


import matplotlib.pyplot as plt


import time


import torch.onnx


import torchvision.transforms as transforms


import torch.nn.init as init


import torchvision


from torch.nn import Conv2d


from torch.nn import Sequential


from torch.nn import ModuleList


from torch.nn import ReLU


from collections import namedtuple


from typing import List


from typing import Tuple


import types


from numpy import random


from torchvision import transforms


class ConvBlock(nn.Module):

    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class SEModule(nn.Module):
    """Squeeze and Excitation Module"""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class MobileNet_GDConv(nn.Module):

    def __init__(self, num_classes):
        super(MobileNet_GDConv, self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)

    def forward(self, x):
        x = self.base_net(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


class MobileNet_GDConv_56(nn.Module):

    def __init__(self, num_classes):
        super(MobileNet_GDConv_56, self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=False)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (2, 2), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)

    def forward(self, x):
        x = self.base_net(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


class MobileNet_GDConv_SE(nn.Module):

    def __init__(self, num_classes):
        super(MobileNet_GDConv_SE, self).__init__()
        self.pretrain_net = models.mobilenet_v2(pretrained=True)
        self.base_net = nn.Sequential(*list(self.pretrain_net.children())[:-1])
        self.linear7 = ConvBlock(1280, 1280, (7, 7), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(1280, num_classes, 1, 1, 0, linear=True)
        self.attention = SEModule(1280, 8)

    def forward(self, x):
        x = self.base_net(x)
        x = self.attention(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 10, 3, 1)), ('prelu1', nn.PReLU(10)), ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)), ('conv2', nn.Conv2d(10, 16, 3, 1)), ('prelu2', nn.PReLU(16)), ('conv3', nn.Conv2d(16, 32, 3, 1)), ('prelu3', nn.PReLU(32))]))
        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)
        weights = np.load('src/weights/pnet.npy', allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a)
        return b, a


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 28, 3, 1)), ('prelu1', nn.PReLU(28)), ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)), ('conv2', nn.Conv2d(28, 48, 3, 1)), ('prelu2', nn.PReLU(48)), ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)), ('conv3', nn.Conv2d(48, 64, 2, 1)), ('prelu3', nn.PReLU(64)), ('flatten', Flatten()), ('conv4', nn.Linear(576, 128)), ('prelu4', nn.PReLU(128))]))
        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)
        weights = np.load('src/weights/rnet.npy', allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a)
        return b, a


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()
        self.features = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 32, 3, 1)), ('prelu1', nn.PReLU(32)), ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)), ('conv2', nn.Conv2d(32, 64, 3, 1)), ('prelu2', nn.PReLU(64)), ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)), ('conv3', nn.Conv2d(64, 64, 3, 1)), ('prelu3', nn.PReLU(64)), ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)), ('conv4', nn.Conv2d(64, 128, 2, 1)), ('prelu4', nn.PReLU(128)), ('flatten', Flatten()), ('conv5', nn.Linear(1152, 256)), ('drop5', nn.Dropout(0.25)), ('prelu5', nn.PReLU(256))]))
        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)
        weights = np.load('src/weights/onet.npy', allow_pickle=True)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a)
        return c, b, a


class Mb_Tiny(nn.Module):

    def __init__(self, num_classes=2):
        super(Mb_Tiny, self).__init__()
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, self.base_channel, 2), conv_dw(self.base_channel, self.base_channel * 2, 1), conv_dw(self.base_channel * 2, self.base_channel * 2, 2), conv_dw(self.base_channel * 2, self.base_channel * 2, 1), conv_dw(self.base_channel * 2, self.base_channel * 4, 2), conv_dw(self.base_channel * 4, self.base_channel * 4, 1), conv_dw(self.base_channel * 4, self.base_channel * 4, 1), conv_dw(self.base_channel * 4, self.base_channel * 4, 1), conv_dw(self.base_channel * 4, self.base_channel * 8, 2), conv_dw(self.base_channel * 8, self.base_channel * 8, 1), conv_dw(self.base_channel * 8, self.base_channel * 8, 1), conv_dw(self.base_channel * 8, self.base_channel * 16, 2), conv_dw(self.base_channel * 16, self.base_channel * 16, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False), BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups))
        self.branch1 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False), BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups))
        self.branch2 = nn.Sequential(BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False), BasicConv(inter_planes, inter_planes // 2 * 3, kernel_size=3, stride=1, padding=1, groups=groups), BasicConv(inter_planes // 2 * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups), BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups))
        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)
        return out


class Mb_Tiny_RFB(nn.Module):

    def __init__(self, num_classes=2):
        super(Mb_Tiny_RFB, self).__init__()
        self.base_channel = 8 * 2

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, self.base_channel, 2), conv_dw(self.base_channel, self.base_channel * 2, 1), conv_dw(self.base_channel * 2, self.base_channel * 2, 2), conv_dw(self.base_channel * 2, self.base_channel * 2, 1), conv_dw(self.base_channel * 2, self.base_channel * 4, 2), conv_dw(self.base_channel * 4, self.base_channel * 4, 1), conv_dw(self.base_channel * 4, self.base_channel * 4, 1), BasicRFB(self.base_channel * 4, self.base_channel * 4, stride=1, scale=1.0), conv_dw(self.base_channel * 4, self.base_channel * 8, 2), conv_dw(self.base_channel * 8, self.base_channel * 8, 1), conv_dw(self.base_channel * 8, self.base_channel * 8, 1), conv_dw(self.base_channel * 8, self.base_channel * 16, 2), conv_dw(self.base_channel * 16, self.base_channel * 16, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class MultiboxLoss(nn.Module):

    def __init__(self, priors, iou_threshold, neg_pos_ratio, center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, (0)]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        confidence = confidence[(mask), :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction='sum')
        pos_mask = labels > 0
        predicted_locations = predicted_locations[(pos_mask), :].reshape(-1, 4)
        gt_locations = gt_locations[(pos_mask), :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


GraphPath = namedtuple('GraphPath', ['s0', 'name', 's1'])


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class SSD(nn.Module):

    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int], extras: nn.ModuleList, classification_headers: nn.ModuleList, regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if is_test:
            self.config = config
            self.priors = config.priors

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        end_layer_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index:end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        for layer in self.base_net[end_layer_index:]:
            x = layer(x)
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(locations, self.priors, self.config.center_variance, self.config.size_variance)
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('classification_headers') or k.startswith('regression_headers'))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicConv,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicRFB,
     lambda: ([], {'in_planes': 18, 'out_planes': 4}),
     lambda: ([torch.rand([4, 18, 64, 64])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'inp': 4, 'oup': 4, 'k': 4, 's': 4, 'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mb_Tiny,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (Mb_Tiny_RFB,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
    (MobileNet_GDConv,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (MobileNet_GDConv_56,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (MobileNet_GDConv_SE,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     False),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_cunjian_pytorch_face_landmark(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

