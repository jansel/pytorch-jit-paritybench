import sys
_module = sys.modules[__name__]
del sys
FaceBoxes = _module
FaceBoxes_ONNX = _module
models = _module
faceboxes = _module
onnx = _module
utils = _module
box_utils = _module
build = _module
config = _module
functions = _module
nms = _module
py_cpu_nms = _module
nms_wrapper = _module
prior_box = _module
timer = _module
Sim3DR = _module
_init_paths = _module
lighting = _module
setup = _module
TDDFA = _module
TDDFA_ONNX = _module
bfm = _module
bfm_onnx = _module
demo = _module
demo_video = _module
demo_video_smooth = _module
demo_webcam_smooth = _module
gradiodemo = _module
latency = _module
mobilenet_v1 = _module
mobilenet_v3 = _module
resnet = _module
speed_cpu = _module
depth = _module
io = _module
onnx = _module
pncc = _module
pose = _module
render = _module
render_ctypes = _module
serialization = _module
tddfa_util = _module
uv = _module

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


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


from itertools import product as product


from math import ceil


import time


from torchvision.transforms import Compose


import torch.backends.cudnn as cudnn


import matplotlib.pyplot as plt


import math


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)
        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)
        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)
        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
        return torch.cat(outputs, 1)


class CRelu(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        x = F.relu(x, inplace=True)
        return x


class FaceBoxesNet(nn.Module):

    def __init__(self, phase, size, num_classes):
        super(FaceBoxesNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size
        self.conv1 = CRelu(3, 24, kernel_size=7, stride=4, padding=3)
        self.conv2 = CRelu(48, 64, kernel_size=5, stride=2, padding=2)
        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()
        self.conv3_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv4_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.loc, self.conf = self.multibox(self.num_classes)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def multibox(self, num_classes):
        loc_layers = []
        conf_layers = []
        loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=3, padding=1)]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    def forward(self, x):
        detection_sources = list()
        loc = list()
        conf = list()
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        detection_sources.append(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        detection_sources.append(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        detection_sources.append(x)
        for x, l, c in zip(detection_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == 'test':
            output = loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1, self.num_classes))
        else:
            output = loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes)
        return output


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


_numpy_to_tensor = lambda x: torch.from_numpy(x)


class BFMModel_ONNX(nn.Module):
    """BFM serves as a decoder"""

    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        super(BFMModel_ONNX, self).__init__()
        _to_tensor = _numpy_to_tensor
        bfm = _load(bfm_fp)
        u = _to_tensor(bfm.get('u').astype(np.float32))
        self.u = u.view(-1, 3).transpose(1, 0)
        w_shp = _to_tensor(bfm.get('w_shp').astype(np.float32)[..., :shape_dim])
        w_exp = _to_tensor(bfm.get('w_exp').astype(np.float32)[..., :exp_dim])
        w = torch.cat((w_shp, w_exp), dim=1)
        self.w = w.view(-1, 3, w.shape[-1]).contiguous().permute(1, 0, 2)

    def forward(self, *inps):
        R, offset, alpha_shp, alpha_exp = inps
        alpha = torch.cat((alpha_shp, alpha_exp))
        pts3d = R @ (self.u + self.w.matmul(alpha).squeeze()) + offset
        return pts3d


class DepthWiseBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, prelu=False):
        super(DepthWiseBlock, self).__init__()
        inplanes, planes = int(inplanes), int(planes)
        self.conv_dw = nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, stride=stride, groups=inplanes, bias=False)
        self.bn_dw = nn.BatchNorm2d(inplanes)
        self.conv_sep = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_sep = nn.BatchNorm2d(planes)
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv_dw(x)
        out = self.bn_dw(out)
        out = self.relu(out)
        out = self.conv_sep(out)
        out = self.bn_sep(out)
        out = self.relu(out)
        return out


class MobileNet(nn.Module):

    def __init__(self, widen_factor=1.0, num_classes=1000, prelu=False, input_channel=3):
        """ Constructor
        Args:
            widen_factor: config of widen_factor
            num_classes: number of classes
        """
        super(MobileNet, self).__init__()
        block = DepthWiseBlock
        self.conv1 = nn.Conv2d(input_channel, int(32 * widen_factor), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(32 * widen_factor))
        if prelu:
            self.relu = nn.PReLU()
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dw2_1 = block(32 * widen_factor, 64 * widen_factor, prelu=prelu)
        self.dw2_2 = block(64 * widen_factor, 128 * widen_factor, stride=2, prelu=prelu)
        self.dw3_1 = block(128 * widen_factor, 128 * widen_factor, prelu=prelu)
        self.dw3_2 = block(128 * widen_factor, 256 * widen_factor, stride=2, prelu=prelu)
        self.dw4_1 = block(256 * widen_factor, 256 * widen_factor, prelu=prelu)
        self.dw4_2 = block(256 * widen_factor, 512 * widen_factor, stride=2, prelu=prelu)
        self.dw5_1 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_2 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_3 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_4 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_5 = block(512 * widen_factor, 512 * widen_factor, prelu=prelu)
        self.dw5_6 = block(512 * widen_factor, 1024 * widen_factor, stride=2, prelu=prelu)
        self.dw6 = block(1024 * widen_factor, 1024 * widen_factor, prelu=prelu)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(int(1024 * widen_factor), num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dw2_1(x)
        x = self.dw2_2(x)
        x = self.dw3_1(x)
        x = self.dw3_2(x)
        x = self.dw4_1(x)
        x = self.dw4_2(x)
        x = self.dw5_1(x)
        x = self.dw5_2(x)
        x = self.dw5_3(x)
        x = self.dw5_4(x)
        x = self.dw5_5(x)
        x = self.dw5_6(x)
        x = self.dw6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Hswish(nn.Module):

    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class SEModule(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channel // reduction, channel, bias=False), Hsigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):

    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):

    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity
        self.conv = nn.Sequential(conv_layer(inp, exp, 1, 1, 0, bias=False), norm_layer(exp), nlin_layer(inplace=True), conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False), norm_layer(exp), SELayer(exp), nlin_layer(inplace=True), conv_layer(exp, oup, 1, 1, 0, bias=False), norm_layer(oup))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(conv_layer(inp, oup, 1, 1, 0, bias=False), norm_layer(oup), nlin_layer(inplace=True))


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(conv_layer(inp, oup, 3, stride, 1, bias=False), norm_layer(oup), nlin_layer(inplace=True))


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1.0 / divisible_by) * divisible_by)


class MobileNetV3(nn.Module):

    def __init__(self, widen_factor=1.0, num_classes=141, num_landmarks=136, input_size=120, mode='small'):
        super(MobileNetV3, self).__init__()
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            mobile_setting = [[3, 16, 16, False, 'RE', 1], [3, 64, 24, False, 'RE', 2], [3, 72, 24, False, 'RE', 1], [5, 72, 40, True, 'RE', 2], [5, 120, 40, True, 'RE', 1], [5, 120, 40, True, 'RE', 1], [3, 240, 80, False, 'HS', 2], [3, 200, 80, False, 'HS', 1], [3, 184, 80, False, 'HS', 1], [3, 184, 80, False, 'HS', 1], [3, 480, 112, True, 'HS', 1], [3, 672, 112, True, 'HS', 1], [5, 672, 160, True, 'HS', 2], [5, 960, 160, True, 'HS', 1], [5, 960, 160, True, 'HS', 1]]
        elif mode == 'small':
            mobile_setting = [[3, 16, 16, True, 'RE', 2], [3, 72, 24, False, 'RE', 2], [3, 88, 24, False, 'RE', 1], [5, 96, 40, True, 'HS', 2], [5, 240, 40, True, 'HS', 1], [5, 240, 40, True, 'HS', 1], [5, 120, 48, True, 'HS', 1], [5, 144, 48, True, 'HS', 1], [5, 288, 96, True, 'HS', 2], [5, 576, 96, True, 'HS', 1], [5, 576, 96, True, 'HS', 1]]
        else:
            raise NotImplementedError
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * widen_factor) if widen_factor > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        for k, exp, c, se, nl, s in mobile_setting:
            output_channel = make_divisible(c * widen_factor)
            exp_channel = make_divisible(exp * widen_factor)
            self.features.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
        if mode == 'large':
            last_conv = make_divisible(960 * widen_factor)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        elif mode == 'small':
            last_conv = make_divisible(576 * widen_factor)
            self.features.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
            self.features.append(nn.AdaptiveAvgPool2d(1))
            self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
            self.features.append(Hswish(inplace=True))
        else:
            raise NotImplementedError
        self.features = nn.Sequential(*self.features)
        self.fc = nn.Linear(int(last_channel), num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x_share = x.mean(3).mean(2)
        xp = self.fc(x_share)
        return xp

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


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


class ResNet(nn.Module):
    """Another Strucutre used in caffe-resnet25"""

    def __init__(self, block, layers, num_classes=62, num_landmarks=136, input_channel=3, fc_flg=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 128, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.conv_param = nn.Conv2d(512, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc_flg = fc_flg
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        xp = self.conv_param(x)
        xp = self.avgpool(xp)
        xp = xp.view(xp.size(0), -1)
        return xp


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CRelu,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DepthWiseBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FaceBoxesNet,
     lambda: ([], {'phase': 4, 'size': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Hsigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Inception,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 128, 64, 64])], {}),
     True),
    (MobileBottleneck,
     lambda: ([], {'inp': 4, 'oup': 4, 'kernel': 3, 'stride': 1, 'exp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (SEModule,
     lambda: ([], {'channel': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_cleardusk_3DDFA_V2(_paritybench_base):
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

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

