import sys
_module = sys.modules[__name__]
del sys
RecordReader = _module
layers = _module
modules = _module
planenet = _module
train_planenet = _module
utils = _module
RecordReaderAll = _module
CopyTexture = _module
LayeredSceneDecomposition = _module
PlaneStatisticsGlobal = _module
RecordChecker = _module
RecordConverter = _module
RecordConverter3D = _module
RecordConverterRGBD = _module
RecordFilter = _module
RecordReader3D = _module
RecordReaderRGBD = _module
RecordReaderWithoutPlane = _module
RecordSampler = _module
RecordWriter = _module
RecordWriter3D = _module
RecordWriterRGBD = _module
RecordWriterRGBD_backup = _module
RecordWriterWithoutPlane = _module
SegmentationRefinement = _module
cluster = _module
compare = _module
crfasrnn_layer = _module
evaluate = _module
evaluate_depth = _module
evaluate_separate = _module
high_dim_filter_grad = _module
html = _module
kaffe = _module
caffe = _module
caffepb = _module
resolver = _module
errors = _module
graph = _module
shapes = _module
tensorflow = _module
network = _module
transformer = _module
transformers = _module
plane_utils = _module
planenet_group = _module
planenet_layer = _module
main = _module
move_to_origin = _module
obj2egg = _module
plane_scene = _module
polls = _module
predict = _module
predict_custom = _module
room_layout = _module
script = _module
test_bernoulli = _module
test_sampling = _module
train_finetuning = _module
train_hybrid = _module
train_pixelwise = _module
train_planenet_backup = _module
train_planenet_confidence = _module
train_planenet_layer = _module
train_planenet_separate = _module
train_sample = _module
crfasrnn = _module
parse = _module
nndistance = _module
planenet_inference = _module
add_texture = _module
writeDisk = _module
parts_scene = _module
pool = _module
augmentation = _module
datasets = _module
plane_dataset = _module
plane_dataset_scannet = _module
scannet_scene = _module
models = _module
drn = _module
modules = _module
planenet = _module
options = _module
data_converter = _module
train_planenet = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.utils.data import Dataset


import numpy as np


import time


import torch.nn as nn


import math


import torch.utils.model_zoo as model_zoo


from torch.nn import functional as F


import torch


from torch import nn


from torch.utils.data import DataLoader


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation[1], bias=False, dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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


class DRN(nn.Module):

    def __init__(self, block, layers, num_classes=1000, channels=(16, 32, 64, 128, 256, 512, 512, 512), out_map=-1, out_middle=False, pool_size=28, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch
        if arch == 'C':
            self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(channels[0])
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], stride=1)
            self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], stride=2)
        elif arch == 'D':
            self.layer0 = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
            self.layer1 = self._make_conv_layers(channels[0], layers[0], stride=1)
            self.layer2 = self._make_conv_layers(channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.layer5 = self._make_layer(block, channels[4], layers[4], dilation=2, new_level=False)
        self.layer6 = None if layers[5] == 0 else self._make_layer(block, channels[5], layers[5], dilation=4, new_level=False)
        if arch == 'C':
            self.layer7 = None if layers[6] == 0 else self._make_layer(BasicBlock, channels[6], layers[6], dilation=2, new_level=False, residual=False)
            self.layer8 = None if layers[7] == 0 else self._make_layer(BasicBlock, channels[7], layers[7], dilation=1, new_level=False, residual=False)
        elif arch == 'D':
            self.layer7 = None if layers[6] == 0 else self._make_conv_layers(channels[6], layers[6], dilation=2)
            self.layer8 = None if layers[7] == 0 else self._make_conv_layers(channels[7], layers[7], dilation=1)
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.avgpool = nn.AvgPool2d(pool_size)
            self.pred = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.out_map < 32:
            self.out_pool = nn.MaxPool2d(32 // self.out_map)
            pass

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=(1, 1) if dilation == 1 else (dilation // 2 if new_level else dilation, dilation), residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual, dilation=(dilation, dilation)))
        return nn.Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([nn.Conv2d(self.inplanes, channels, kernel_size=3, stride=stride if i == 0 else 1, padding=dilation, bias=False, dilation=dilation), nn.BatchNorm2d(channels), nn.ReLU(inplace=True)])
            self.inplanes = channels
        return nn.Sequential(*modules)

    def forward(self, x):
        y = list()
        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)
        x = self.layer1(x)
        y.append(x)
        x = self.layer2(x)
        y.append(x)
        x = self.layer3(x)
        y.append(x)
        x = self.layer4(x)
        y.append(x)
        x = self.layer5(x)
        y.append(x)
        if self.layer6 is not None:
            x = self.layer6(x)
            y.append(x)
        if self.layer7 is not None:
            x = self.layer7(x)
            y.append(x)
        if self.layer8 is not None:
            x = self.layer8(x)
            y.append(x)
        if self.out_map > 0:
            if self.num_classes > 0:
                if self.out_map == x.shape[2]:
                    x = self.pred(x)
                elif self.out_map > x.shape[2]:
                    x = self.pred(x)
                    x = F.upsample(input=x, size=(self.out_map, self.out_map), mode='bilinear')
                else:
                    x = self.out_pool(x)
                    y.append(x)
                    x = self.pred(x)
                    pass
            else:
                if self.out_map > x.shape[3]:
                    x = F.upsample(input=x, size=(self.out_map, self.out_map), mode='bilinear')
                    pass
                pass
        else:
            x = self.avgpool(x)
            x = self.pred(x)
            x = x.view(x.size(0), -1)
        if self.out_middle:
            return x, y
        else:
            return x


class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=None, mode='conv'):
        super(ConvBlock, self).__init__()
        if padding == None:
            padding = (kernel_size - 1) // 2
            pass
        if mode == 'conv':
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'deconv':
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'conv_3d':
            self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        elif mode == 'deconv_3d':
            self.conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        else:
            None
            exit(1)
            pass
        if '3d' not in mode:
            self.bn = nn.BatchNorm2d(out_planes)
        else:
            self.bn = nn.BatchNorm3d(out_planes)
            pass
        self.relu = nn.ReLU(inplace=True)
        return

    def forward(self, inp):
        return self.relu(self.bn(self.conv(inp)))


class PyramidModule(nn.Module):

    def __init__(self, options, in_planes, middle_planes, scales=[32, 16, 8, 4]):
        super(PyramidModule, self).__init__()
        self.pool_1 = torch.nn.AvgPool2d((scales[0] * options.height / options.width, scales[0]))
        self.pool_2 = torch.nn.AvgPool2d((scales[1] * options.height / options.width, scales[1]))
        self.pool_3 = torch.nn.AvgPool2d((scales[2] * options.height / options.width, scales[2]))
        self.pool_4 = torch.nn.AvgPool2d((scales[3] * options.height / options.width, scales[3]))
        self.conv_1 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_2 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_3 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.conv_4 = ConvBlock(in_planes, middle_planes, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(scales[0] * options.height / options.width, scales[0]), mode='bilinear')
        return

    def forward(self, inp):
        x_1 = self.upsample(self.conv_1(self.pool_1(inp)))
        x_2 = self.upsample(self.conv_2(self.pool_2(inp)))
        x_3 = self.upsample(self.conv_3(self.pool_3(inp)))
        x_4 = self.upsample(self.conv_4(self.pool_4(inp)))
        out = torch.cat([inp, x_1, x_2, x_3, x_4], dim=1)
        return out


webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'


model_urls = {'drn-c-26': webroot + 'drn_c_26-ddedf421.pth', 'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth', 'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth', 'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth', 'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth', 'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth', 'drn-d-105': webroot + 'drn_d_105-12b40979.pth'}


def drn_d_54(pretrained=False, out_map=256, num_classes=20, **kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', out_map=out_map, num_classes=num_classes, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['drn-d-54'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        state = model.state_dict()
        state.update(pretrained_dict)
        model.load_state_dict(state)
    return model


class PlaneNet(nn.Module):

    def __init__(self, options):
        super(PlaneNet, self).__init__()
        self.options = options
        self.drn = drn_d_54(pretrained=True, out_map=32, num_classes=-1, out_middle=False)
        self.pool = torch.nn.AvgPool2d((32 * options.height / options.width, 32))
        self.plane_pred = nn.Linear(512, options.numOutputPlanes * 3)
        self.pyramid = PyramidModule(options, 512, 128)
        self.feature_conv = ConvBlock(1024, 512)
        self.segmentation_pred = nn.Conv2d(512, options.numOutputPlanes + 1, kernel_size=1)
        self.depth_pred = nn.Conv2d(512, 1, kernel_size=1)
        self.upsample = torch.nn.Upsample(size=(options.outputHeight, options.outputWidth), mode='bilinear')
        return

    def forward(self, inp):
        features = self.drn(inp)
        planes = self.plane_pred(self.pool(features).view((-1, 512))).view((-1, self.options.numOutputPlanes, 3))
        features = self.pyramid(features)
        features = self.feature_conv(features)
        segmentation = self.upsample(self.segmentation_pred(features))
        depth = self.upsample(self.depth_pred(features))
        return planes, segmentation, depth


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_planes': 4, 'out_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_art_programmer_PlaneNet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

