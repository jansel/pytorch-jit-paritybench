import sys
_module = sys.modules[__name__]
del sys
benchmark = _module
yolo_config = _module
data = _module
coco = _module
transforms = _module
voc = _module
demo = _module
eval = _module
cocoapi_evaluator = _module
vocapi_evaluator = _module
models = _module
backbone = _module
cspdarknet53 = _module
cspdarknet_tiny = _module
darknet = _module
resnet = _module
shufflenetv2 = _module
vit = _module
yolox_backbone = _module
basic = _module
bottleneck_csp = _module
conv = _module
upsample = _module
head = _module
coupled_head = _module
decoupled_head = _module
neck = _module
dilated_encoder = _module
fpn = _module
spp = _module
yolo = _module
yolo_nano = _module
yolo_tiny = _module
yolov1 = _module
yolov2 = _module
yolov3 = _module
yolov4 = _module
test = _module
train = _module
utils = _module
box_ops = _module
com_flops_params = _module
create_labels = _module
criterion = _module
distributed_utils = _module
fuse_conv_bn = _module
kmeans_anchor = _module
misc = _module
vis = _module

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


import numpy as np


import time


import torch


import random


from torch.utils.data import Dataset


from numpy import random


import torch.utils.data as data


import torch.nn as nn


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


from functools import partial


import math


from copy import deepcopy


import torch.optim as optim


import torch.backends.cudnn as cudnn


import torch.distributed as dist


from torch.nn.parallel import DistributedDataParallel as DDP


class DarkBlock(nn.Module):

    def __init__(self, inplanes, planes, dilation=1, downsample=None):
        """Residual Block for DarkNet.
        This module has the dowsample layer (optional),
        1x1 conv layer and 3x3 conv layer.
        """
        super(DarkBlock, self).__init__()
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(inplanes, eps=0.0001, momentum=0.03)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.0001, momentum=0.03)
        self.conv1 = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.activation = nn.Mish(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        out += identity
        return out


def ConvNormActivation(inplanes, planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1):
    """
    A help function to build a 'conv-bn-activation' module
    """
    layers = []
    layers.append(nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    layers.append(nn.BatchNorm2d(planes, eps=0.0001, momentum=0.03))
    layers.append(nn.Mish(inplace=True))
    return nn.Sequential(*layers)


class CrossStagePartialBlock(nn.Module):
    """CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
    Refer to the paper for more details: https://arxiv.org/abs/1911.11929.
    In this module, the inputs go throuth the base conv layer at the first,
    and then pass the two partial transition layers.
    1. go throuth basic block (like DarkBlock)
        and one partial transition layer.
    2. go throuth the other partial transition layer.
    At last, They are concat into fuse transition layer.
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels
        stage_layers (nn.Module): the basic block which applying CSPNet.
        is_csp_first_stage (bool): Is the first stage or not.
            The number of input and output channels in the first stage of
            CSPNet is different from other stages.
        dilation (int): conv dilation
        stride (int): stride for the base layer
    """

    def __init__(self, inplanes, planes, stage_layers, is_csp_first_stage, dilation=1, stride=2):
        super(CrossStagePartialBlock, self).__init__()
        self.base_layer = ConvNormActivation(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
        self.partial_transition1 = ConvNormActivation(inplanes=planes, planes=inplanes if not is_csp_first_stage else planes, kernel_size=1, stride=1, padding=0)
        self.stage_layers = stage_layers
        self.partial_transition2 = ConvNormActivation(inplanes=inplanes if not is_csp_first_stage else planes, planes=inplanes if not is_csp_first_stage else planes, kernel_size=1, stride=1, padding=0)
        self.fuse_transition = ConvNormActivation(inplanes=planes if not is_csp_first_stage else planes * 2, planes=planes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.base_layer(x)
        out1 = self.partial_transition1(x)
        out2 = self.stage_layers(x)
        out2 = self.partial_transition2(out2)
        out = torch.cat([out2, out1], dim=1)
        out = self.fuse_transition(out)
        return out


def make_cspdark_layer(block, inplanes, planes, num_blocks, is_csp_first_stage, dilation=1):
    downsample = ConvNormActivation(inplanes=planes, planes=planes if is_csp_first_stage else inplanes, kernel_size=1, stride=1, padding=0)
    layers = []
    for i in range(0, num_blocks):
        layers.append(block(inplanes=inplanes, planes=planes if is_csp_first_stage else inplanes, downsample=downsample if i == 0 else None, dilation=dilation))
    return nn.Sequential(*layers)


class CSPDarkNet53(nn.Module):
    """CSPDarkNet backbone.
    Refer to the paper for more details: https://arxiv.org/pdf/1804.02767
    Args:
        depth (int): Depth of Darknet, from {53}.
        num_stages (int): Darknet stages, normally 5.
        with_csp (bool): Use cross stage partial connection or not.
        out_features (List[str]): Output features.
        norm_type (str): type of normalization layer.
        res5_dilation (int): dilation for the last stage
    """

    def __init__(self):
        super(CSPDarkNet53, self).__init__()
        self.block = DarkBlock
        self.stage_blocks = 1, 2, 8, 8, 4
        self.with_csp = True
        self.inplanes = 32
        self.backbone = nn.ModuleDict()
        self.layer_names = []
        self.backbone['conv1'] = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.backbone['bn1'] = nn.BatchNorm2d(self.inplanes, eps=0.0001, momentum=0.03)
        self.backbone['act1'] = nn.Mish(inplace=True)
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            dilation = 1
            stride = 2
            layer = make_cspdark_layer(block=self.block, inplanes=self.inplanes, planes=planes, num_blocks=num_blocks, is_csp_first_stage=True if i == 0 else False, dilation=dilation)
            layer = CrossStagePartialBlock(self.inplanes, planes, stage_layers=layer, is_csp_first_stage=True if i == 0 else False, dilation=dilation, stride=stride)
            self.inplanes = planes
            layer_name = 'layer{}'.format(i + 1)
            self.backbone[layer_name] = layer
            self.layer_names.append(layer_name)

    def forward(self, x):
        outputs = []
        x = self.backbone['conv1'](x)
        x = self.backbone['bn1'](x)
        x = self.backbone['act1'](x)
        for i, layer_name in enumerate(self.layer_names):
            layer = self.backbone[layer_name]
            x = layer(x)
            outputs.append(x)
        return outputs[-3:]


def get_activation(name='lrelu', inplace=True):
    if name == 'silu':
        module = nn.SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act='lrelu', depthwise=False, bias=False):
        super(Conv, self).__init__()
        if depthwise:
            assert c1 == c2
            self.convs = nn.Sequential(nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=c1, bias=bias), nn.BatchNorm2d(c2), get_activation(name=act), nn.Conv2d(c2, c2, kernel_size=1, bias=bias), nn.BatchNorm2d(c2), get_activation(name=act))
        else:
            self.convs = nn.Sequential(nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias), nn.BatchNorm2d(c2), get_activation(name=act))

    def forward(self, x):
        return self.convs(x)


class ResidualBlock(nn.Module):
    """
    basic residual block for CSP-Darknet
    """

    def __init__(self, in_ch):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_ch, in_ch, k=1)
        self.conv2 = Conv(in_ch, in_ch, k=3, p=1, act=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        h = self.conv2(self.conv1(x))
        out = self.act(x + h)
        return out


class CSPStage(nn.Module):

    def __init__(self, c1, n=1):
        super(CSPStage, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(in_ch=c_) for _ in range(n)])
        self.cv3 = Conv(2 * c_, c1, k=1)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.res_blocks(self.cv2(x))
        return self.cv3(torch.cat([y1, y2], dim=1))


class CSPDarknetTiny(nn.Module):
    """
    CSPDarknet_Tiny.
    """

    def __init__(self):
        super(CSPDarknetTiny, self).__init__()
        self.layer_1 = nn.Sequential(Conv(3, 16, k=3, p=1), Conv(16, 32, k=3, p=1, s=2), CSPStage(c1=32, n=1))
        self.layer_2 = nn.Sequential(Conv(32, 64, k=3, p=1, s=2), CSPStage(c1=64, n=1))
        self.layer_3 = nn.Sequential(Conv(64, 128, k=3, p=1, s=2), CSPStage(c1=128, n=1))
        self.layer_4 = nn.Sequential(Conv(128, 256, k=3, p=1, s=2), CSPStage(c1=256, n=1))
        self.layer_5 = nn.Sequential(Conv(256, 512, k=3, p=1, s=2), CSPStage(c1=512, n=1))

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        return c3, c4, c5


class Conv_BN_LeakyReLU(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation), nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        return self.convs(x)


class resblock(nn.Module):

    def __init__(self, ch, nblocks=1):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(Conv_BN_LeakyReLU(ch, ch // 2, 1), Conv_BN_LeakyReLU(ch // 2, ch, 3, padding=1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """

    def __init__(self, num_classes=1000):
        super(DarkNet_53, self).__init__()
        self.layer_1 = nn.Sequential(Conv_BN_LeakyReLU(3, 32, 3, padding=1), Conv_BN_LeakyReLU(32, 64, 3, padding=1, stride=2), resblock(64, nblocks=1))
        self.layer_2 = nn.Sequential(Conv_BN_LeakyReLU(64, 128, 3, padding=1, stride=2), resblock(128, nblocks=2))
        self.layer_3 = nn.Sequential(Conv_BN_LeakyReLU(128, 256, 3, padding=1, stride=2), resblock(256, nblocks=8))
        self.layer_4 = nn.Sequential(Conv_BN_LeakyReLU(256, 512, 3, padding=1, stride=2), resblock(512, nblocks=8))
        self.layer_5 = nn.Sequential(Conv_BN_LeakyReLU(512, 1024, 3, padding=1, stride=2), resblock(1024, nblocks=4))

    def forward(self, x, targets=None):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)
        return c3, c4, c5


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

    def __init__(self, c1, c2, shortcut=True, d=1, e=0.5, depthwise=False, act='lrelu'):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, act=act)
        self.cv2 = Conv(c_, c2, k=3, p=d, d=d, act=act, depthwise=depthwise)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ShuffleV2Block(nn.Module):

    def __init__(self, inp, oup, stride):
        super(ShuffleV2Block, self).__init__()
        if not 1 <= stride <= 3:
            raise ValueError('illegal stride value')
        self.stride = stride
        branch_features = oup // 2
        assert self.stride != 1 or inp == branch_features << 1
        if self.stride > 1:
            self.branch1 = nn.Sequential(self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(inp), nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True), self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1), nn.BatchNorm2d(branch_features), nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):

    def __init__(self, model_size='1.0x', out_stages=(2, 3, 4), with_last_conv=False, kernal_size=3):
        super(ShuffleNetV2, self).__init__()
        None
        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        if model_size == '0.5x':
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True))
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, self.stage_repeats, self._stage_out_channels[1:]):
            seq = [ShuffleV2Block(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        self._initialize_weights()

    def _initialize_weights(self, pretrain=True):
        None
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) ->str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.patch_shape = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def get_sinusoid_encoding_table(n_position, d_hid):
    """ Sinusoid position encoding table """

    def get_position_angle_vec(position):
        return [(position / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=None, use_learnable_pos_emb=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).clone().detach()
        B, _, C = x.shape
        x = x.reshape(B, -1, C)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, encoder_in_chans=3, encoder_num_classes=0, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=0.0, use_learnable_pos_emb=False, num_classes=0, in_chans=0):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(img_size=img_size, patch_size=patch_size, in_chans=encoder_in_chans, num_classes=encoder_num_classes, embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, use_learnable_pos_emb=use_learnable_pos_emb)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x):
        fmap_list = []
        x = self.encoder(x)
        fmp_h = self.encoder.patch_embed.img_size[0] // self.encoder.patch_embed.patch_size[0]
        fmp_w = self.encoder.patch_embed.img_size[1] // self.encoder.patch_embed.patch_size[1]
        x = x.permute(0, 2, 1).contiguous().view(x.size(0), x.size(-1), fmp_h, fmp_w)
        fmap_list.append(x)
        return fmap_list


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act='silu'):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act='silu'):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act='silu'):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)


class CSPDarknet(nn.Module):

    def __init__(self, dep_mul, wid_mul, out_features=('dark3', 'dark4', 'dark5'), depthwise=False, act='silu'):
        super().__init__()
        assert out_features, 'please provide output features of Darknet'
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)
        self.stem = Focus(3, base_channels, ksize=3, act=act)
        self.dark2 = nn.Sequential(Conv(base_channels, base_channels * 2, 3, 2, act=act), CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act))
        self.dark3 = nn.Sequential(Conv(base_channels * 2, base_channels * 4, 3, 2, act=act), CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act))
        self.dark4 = nn.Sequential(Conv(base_channels * 4, base_channels * 8, 3, 2, act=act), CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act))
        self.dark5 = nn.Sequential(Conv(base_channels * 8, base_channels * 16, 3, 2, act=act), SPPBottleneck(base_channels * 16, base_channels * 16, activation=act), CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act))

    def freeze_stage(self):
        None
        for m in self.parameters():
            m.requires_grad = False

    def forward(self, x):
        outputs = {}
        c1 = self.stem(x)
        c2 = self.dark2(c1)
        c3 = self.dark3(c2)
        c4 = self.dark4(c3)
        c5 = self.dark5(c4)
        return c3, c4, c5


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, depthwise=False, act='lrelu'):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k=1, act=act)
        self.cv2 = Conv(c1, c_, k=1, act=act)
        self.cv3 = Conv(2 * c_, c2, k=1, act=act)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, e=1.0, depthwise=depthwise, act=act) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class ConvBlocks(nn.Module):

    def __init__(self, c1, c2, act='lrelu'):
        super().__init__()
        c_ = c2 * 2
        self.convs = nn.Sequential(Conv(c1, c2, k=1, act=act), Conv(c2, c_, k=3, p=1, act=act), Conv(c_, c2, k=1, act=act), Conv(c2, c_, k=3, p=1, act=act), Conv(c_, c2, k=1, act=act))

    def forward(self, x):
        return self.convs(x)


class UpSample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(input=x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corner)


class CoupledHead(nn.Module):

    def __init__(self, in_dim=[256, 512, 1024], stride=[8, 16, 32], kernel_size=3, padding=1, width=1.0, num_classes=80, num_anchors=3, depthwise=False, act='silu', init_bias=True, center_sample=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.width = width
        self.stride = stride
        self.center_sample = center_sample
        self.head_feat = nn.ModuleList()
        self.head_pred = nn.ModuleList()
        for c in in_dim:
            head_dim = int(c * width)
            self.head_feat.append(nn.Sequential(Conv(head_dim, head_dim, k=kernel_size, p=padding, act=act, depthwise=depthwise), Conv(head_dim, head_dim, k=kernel_size, p=padding, act=act, depthwise=depthwise)))
            self.head_pred.append(nn.Conv2d(head_dim, num_anchors * (1 + num_classes + 4), kernel_size=1))
        if init_bias:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        for head_pred in self.head_pred:
            nn.init.constant_(head_pred.bias[..., :self.num_anchors], bias_value)

    def forward(self, features, grid_cell=None, anchors_wh=None):
        """
            features: (List of Tensor) of multiple feature maps
        """
        B = features[0].size(0)
        obj_preds = []
        cls_preds = []
        box_preds = []
        for i in range(len(features)):
            feat = features[i]
            head_feat = self.head_feat[i](feat)
            head_pred = self.head_pred[i](head_feat)
            obj_pred = head_pred[:, :self.num_anchors, :, :]
            cls_pred = head_pred[:, self.num_anchors:self.num_anchors * (1 + self.num_classes), :, :]
            reg_pred = head_pred[:, self.num_anchors * (1 + self.num_classes):, :, :]
            obj_preds.append(obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1))
            cls_preds.append(cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes))
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            if self.center_sample:
                xy_pred = (grid_cell[i] + reg_pred[..., :2].sigmoid() * 2.0 - 1.0) * self.stride[i]
            else:
                xy_pred = (grid_cell[i] + reg_pred[..., :2].sigmoid()) * self.stride[i]
            if anchors_wh is not None:
                wh_pred = reg_pred[..., 2:].exp() * anchors_wh[i]
            else:
                wh_pred = reg_pred[..., 2:].exp() * self.stride[i]
            x1y1_pred = xy_pred - wh_pred * 0.5
            x2y2_pred = xy_pred + wh_pred * 0.5
            box_preds.append(torch.cat([x1y1_pred, x2y2_pred], dim=-1).view(B, -1, 4))
        obj_preds = torch.cat(obj_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        box_preds = torch.cat(box_preds, dim=1)
        return obj_preds, cls_preds, box_preds


class DecoupledHead(nn.Module):

    def __init__(self, in_dim=[256, 512, 1024], stride=[8, 16, 32], head_dim=256, kernel_size=3, padding=1, width=1.0, num_classes=80, num_anchors=3, depthwise=False, act='silu', init_bias=True, center_sample=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.head_dim = int(head_dim * width)
        self.width = width
        self.stride = stride
        self.center_sample = center_sample
        self.input_proj = nn.ModuleList()
        self.cls_feat = nn.ModuleList()
        self.reg_feat = nn.ModuleList()
        self.obj_pred = nn.ModuleList()
        self.cls_pred = nn.ModuleList()
        self.reg_pred = nn.ModuleList()
        for c in in_dim:
            self.input_proj.append(Conv(c, self.head_dim, k=1, act=act))
            self.cls_feat.append(nn.Sequential(Conv(self.head_dim, self.head_dim, k=kernel_size, p=padding, act=act, depthwise=depthwise), Conv(self.head_dim, self.head_dim, k=kernel_size, p=padding, act=act, depthwise=depthwise)))
            self.reg_feat.append(nn.Sequential(Conv(self.head_dim, self.head_dim, k=kernel_size, p=padding, act=act, depthwise=depthwise), Conv(self.head_dim, self.head_dim, k=kernel_size, p=padding, act=act, depthwise=depthwise)))
            self.obj_pred.append(nn.Conv2d(self.head_dim, num_anchors * 1, kernel_size=1))
            self.cls_pred.append(nn.Conv2d(self.head_dim, num_anchors * num_classes, kernel_size=1))
            self.reg_pred.append(nn.Conv2d(self.head_dim, num_anchors * 4, kernel_size=1))
        if init_bias:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        for obj_pred in self.obj_pred:
            nn.init.constant_(obj_pred.bias, bias_value)

    def forward(self, features, grid_cell=None, anchors_wh=None):
        """
            features: (List of Tensor) of multiple feature maps
        """
        B = features[0].size(0)
        obj_preds = []
        cls_preds = []
        box_preds = []
        for i in range(len(features)):
            feat = features[i]
            feat = self.input_proj[i](feat)
            cls_feat = self.cls_feat[i](feat)
            reg_feat = self.reg_feat[i](feat)
            obj_pred = self.obj_pred[i](reg_feat)
            cls_pred = self.cls_pred[i](cls_feat)
            reg_pred = self.reg_pred[i](reg_feat)
            obj_preds.append(obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1))
            cls_preds.append(cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes))
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)
            if self.center_sample:
                xy_pred = (grid_cell[i] + reg_pred[..., :2].sigmoid() * 2.0 - 1.0) * self.stride[i]
            else:
                xy_pred = (grid_cell[i] + reg_pred[..., :2].sigmoid()) * self.stride[i]
            if anchors_wh is not None:
                wh_pred = reg_pred[..., 2:].exp() * anchors_wh[i]
            else:
                wh_pred = reg_pred[..., 2:].exp() * self.stride[i]
            x1y1_pred = xy_pred - wh_pred * 0.5
            x2y2_pred = xy_pred + wh_pred * 0.5
            box_preds.append(torch.cat([x1y1_pred, x2y2_pred], dim=-1).view(B, -1, 4))
        obj_preds = torch.cat(obj_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        box_preds = torch.cat(box_preds, dim=1)
        return obj_preds, cls_preds, box_preds


class DilatedBottleneck(nn.Module):

    def __init__(self, c, d=1, e=0.5, act='lrelu'):
        super(DilatedBottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(Conv(c, c_, k=1, act=act), Conv(c_, c_, k=3, p=d, d=d, act=act), Conv(c_, c, k=1, act=act))

    def forward(self, x):
        return x + self.branch(x)


class DilatedEncoder(nn.Module):
    """ DilateEncoder """

    def __init__(self, c1, c2, act='lrelu', dilation_list=[2, 4, 6, 8]):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(Conv(c1, c2, k=1, act=None), Conv(c2, c2, k=3, p=1, act=None))
        encoders = []
        for d in dilation_list:
            encoders.append(DilatedBottleneck(c=c2, d=d, act=act))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)
        return x


class YoloFPN(nn.Module):

    def __init__(self, in_dim=[512, 1024, 2048]):
        super(YoloFPN, self).__init__()
        c3, c4, c5 = in_dim
        self.head_convblock_0 = ConvBlocks(c5, c5 // 2)
        self.head_conv_0 = Conv(c5 // 2, c4 // 2, k=1)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_conv_1 = Conv(c5 // 2, c5, k=3, p=1)
        self.head_convblock_1 = ConvBlocks(c4 + c4 // 2, c4 // 2)
        self.head_conv_2 = Conv(c4 // 2, c3 // 2, k=1)
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_conv_3 = Conv(c4 // 2, c4, k=3, p=1)
        self.head_convblock_2 = ConvBlocks(c3 + c3 // 2, c3 // 2)
        self.head_conv_4 = Conv(c3 // 2, c3, k=3, p=1)

    def forward(self, features):
        c3, c4, c5 = features
        p5 = self.head_convblock_0(c5)
        p5_up = self.head_upsample_0(self.head_conv_0(p5))
        p5 = self.head_conv_1(p5)
        p4 = self.head_convblock_1(torch.cat([c4, p5_up], dim=1))
        p4_up = self.head_upsample_1(self.head_conv_2(p4))
        p4 = self.head_conv_3(p4)
        p3 = self.head_convblock_2(torch.cat([c3, p4_up], dim=1))
        p3 = self.head_conv_4(p3)
        return [p3, p4, p5]


class YoloPaFPN(nn.Module):

    def __init__(self, in_dim=[256, 512, 1024], depth=1.0, depthwise=False, act='silu'):
        super(YoloPaFPN, self).__init__()
        c3, c4, c5 = in_dim
        nblocks = int(3 * depth)
        self.head_conv_0 = Conv(c5, c5 // 2, k=1, act=act)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(c4 + c5 // 2, c4, n=nblocks, shortcut=False, depthwise=depthwise, act=act)
        self.head_conv_1 = Conv(c4, c4 // 2, k=1, act=act)
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(c3 + c4 // 2, c3, n=nblocks, shortcut=False, depthwise=depthwise, act=act)
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2, depthwise=depthwise, act=act)
        self.head_csp_2 = BottleneckCSP(c3 + c4 // 2, c4, n=nblocks, shortcut=False, depthwise=depthwise, act=act)
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2, depthwise=depthwise, act=act)
        self.head_csp_3 = BottleneckCSP(c4 + c5 // 2, c5, n=nblocks, shortcut=False, depthwise=depthwise)

    def forward(self, features):
        c3, c4, c5 = features
        c6 = self.head_conv_0(c5)
        c7 = self.head_upsample_0(c6)
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        c10 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)
        return [c13, c16, c19]


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """

    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act='lrelu'):
        super(SPP, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, k=1, act=act)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in kernel_sizes])
        self.cv2 = Conv(c_ * (len(kernel_sizes) + 1), c2, k=1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.cv2(x)
        return x


class SPPBlock(nn.Module):
    """
        Spatial Pyramid Pooling Block
    """

    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act='lrelu'):
        super(SPPBlock, self).__init__()
        self.m = nn.Sequential(Conv(c1, c1 // 2, k=1, act=act), Conv(c1 // 2, c1, k=3, p=1, act=act), SPP(c1, c1 // 2, e=e, kernel_sizes=kernel_sizes, act=act), Conv(c1 // 2, c1, k=3, p=1, act=act), Conv(c1, c2, k=1, act=act))

    def forward(self, x):
        x = self.m(x)
        return x


class SPPBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """

    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act='lrelu'):
        super(SPPBlockCSP, self).__init__()
        self.cv1 = Conv(c1, c1 // 2, k=1, act=act)
        self.cv2 = Conv(c1, c1 // 2, k=1, act=act)
        self.m = nn.Sequential(Conv(c1 // 2, c1 // 2, k=3, p=1, act=act), SPP(c1 // 2, c1 // 2, e=e, kernel_sizes=kernel_sizes, act=act), Conv(c1 // 2, c1 // 2, k=3, p=1, act=act))
        self.cv3 = Conv(c1, c2, k=1, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))
        return y


class SPPBlockDW(nn.Module):
    """
        Depth-wise Spatial Pyramid Pooling Block
    """

    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act='lrelu'):
        super(SPPBlockDW, self).__init__()
        self.m = nn.Sequential(Conv(c1, c1 // 2, k=1, act=act), Conv(c1 // 2, c1 // 2, k=3, p=1, g=c1 // 2, act=act), SPP(c1 // 2, c1 // 2, e=e, kernel_sizes=kernel_sizes, act=act), Conv(c1 // 2, c1 // 2, k=3, p=1, g=c1 // 2, act=act), Conv(c1 // 2, c2, k=1, act=act))

    def forward(self, x):
        return self.m(x)


def cspdarknet53(pretrained=False):
    """
    Create a CSPDarkNet.
    """
    model = CSPDarkNet53()
    if pretrained:
        None
        path_to_weight = os.path.dirname(os.path.abspath(__file__)) + '/weights/cspdarknet53/cspdarknet53.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        checkpoint_state_dict = checkpoint.pop('model')
        model_state_dict = model.state_dict()
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                None
        model.load_state_dict(checkpoint_state_dict, strict=False)
    return model


def cspdarknet_tiny(pretrained=False, **kwargs):
    """Constructs a CSPDarknet53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSPDarknetTiny()
    if pretrained:
        None
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint = torch.load(path_to_dir + '/weights/cspdarknet_tiny/cspdarknet_tiny.pth', map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    return model


def darknet53(pretrained=False, **kwargs):
    """Constructs a darknet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_53()
    if pretrained:
        try:
            None
            path_to_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint = torch.load(path_to_dir + '/weights/darknet53/darknet53.pth', map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
        except:
            None
            pass
    return model


model_urls = {'shufflenetv2_0.5x': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth', 'shufflenetv2_1.0x': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth', 'shufflenetv2_1.5x': None, 'shufflenetv2_2.0x': None}


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        None
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        None
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        None
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


def shufflenetv2(model_size='1.0x', pretrained=False, **kwargs):
    """Constructs a shufflenetv2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShuffleNetV2(model_size=model_size)
    if pretrained:
        None
        url = model_urls['shufflenetv2_{}'.format(model_size)]
        None
        model.load_state_dict(model_zoo.load_url(url), strict=False)
    return model


def _cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': 0.9, 'interpolation': 'bicubic', 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5), **kwargs}


def yolox_cspdarknet_l(pretrained=False, freeze=False):
    backbone = CSPDarknet(dep_mul=1.0, wid_mul=1.0, depthwise=False, act='silu')
    if pretrained:
        None
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_l.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)
    if freeze:
        backbone.freeze_stage()
    return backbone


def yolox_cspdarknet_m(pretrained=False, freeze=False):
    backbone = CSPDarknet(dep_mul=0.67, wid_mul=0.75, depthwise=False, act='silu')
    if pretrained:
        None
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_m.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)
    if freeze:
        backbone.freeze_stage()
    return backbone


def yolox_cspdarknet_nano(pretrained=False, freeze=False):
    backbone = CSPDarknet(dep_mul=0.33, wid_mul=0.25, depthwise=True, act='silu')
    if pretrained:
        None
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_nano.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)
    if freeze:
        backbone.freeze_stage()
    return backbone


def yolox_cspdarknet_s(pretrained=False, freeze=False):
    backbone = CSPDarknet(dep_mul=0.33, wid_mul=0.5, depthwise=False, act='silu')
    if pretrained:
        None
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_s.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)
    if freeze:
        backbone.freeze_stage()
    return backbone


def yolox_cspdarknet_tiny(pretrained=False, freeze=False):
    backbone = CSPDarknet(dep_mul=0.33, wid_mul=0.375, depthwise=False, act='silu')
    if pretrained:
        None
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_tiny.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)
    if freeze:
        backbone.freeze_stage()
    return backbone


def yolox_cspdarknet_x(pretrained=False, freeze=False):
    backbone = CSPDarknet(dep_mul=1.33, wid_mul=1.25, depthwise=False, act='silu')
    if pretrained:
        None
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_x.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)
    if freeze:
        backbone.freeze_stage()
    return backbone


def build_backbone(model_name='r18', pretrained=False, freeze=None, img_size=224):
    if model_name == 'r18':
        None
        model = resnet18(pretrained=pretrained)
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'r50':
        None
        model = resnet50(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]
    elif model_name == 'r101':
        None
        model = resnet101(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]
    elif model_name == 'd53':
        None
        model = darknet53(pretrained=pretrained)
        feature_channels = [256, 512, 1024]
        strides = [8, 16, 32]
    elif model_name == 'cspd53':
        None
        model = cspdarknet53(pretrained=pretrained)
        feature_channels = [256, 512, 1024]
        strides = [8, 16, 32]
    elif model_name == 'cspd_tiny':
        None
        model = cspdarknet_tiny(pretrained=pretrained)
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'sfnet_v2':
        None
        model = shufflenetv2(pretrained=pretrained)
        feature_channels = [116, 232, 464]
        strides = [8, 16, 32]
    elif model_name == 'vit_base_16':
        None
        model = vit_base_patch16_224(img_size=img_size, pretrained=pretrained)
        feature_channels = [None, None, 768]
        strides = [None, None, 16]
    elif model_name == 'csp_s':
        None
        model = yolox_cspdarknet_s(pretrained=pretrained, freeze=freeze)
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'csp_m':
        None
        model = yolox_cspdarknet_m(pretrained=pretrained, freeze=freeze)
        feature_channels = [192, 384, 768]
        strides = [8, 16, 32]
    elif model_name == 'csp_l':
        None
        model = yolox_cspdarknet_l(pretrained=pretrained, freeze=freeze)
        feature_channels = [256, 512, 1024]
        strides = [8, 16, 32]
    elif model_name == 'csp_x':
        None
        model = yolox_cspdarknet_x(pretrained=pretrained, freeze=freeze)
        feature_channels = [320, 640, 1280]
        strides = [8, 16, 32]
    elif model_name == 'csp_t':
        None
        model = yolox_cspdarknet_tiny(pretrained=pretrained, freeze=freeze)
        feature_channels = [96, 192, 384]
        strides = [8, 16, 32]
    elif model_name == 'csp_n':
        None
        model = yolox_cspdarknet_nano(pretrained=pretrained, freeze=freeze)
        feature_channels = [64, 128, 256]
        strides = [8, 16, 32]
    return model, feature_channels, strides


class YOLONano(nn.Module):

    def __init__(self, cfg=None, device=None, img_size=640, num_classes=80, trainable=False, conf_thresh=0.001, nms_thresh=0.6, center_sample=False):
        super(YOLONano, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], pretrained=trainable)
        self.stride = strides
        anchor_size = cfg['anchor_size']
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // 3, 2).float()
        self.num_anchors = self.anchor_size.size(1)
        c3, c4, c5 = feature_channels
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)
        self.neck = SPP(c5, c5)
        self.conv1x1_0 = Conv(c3, 96, k=1)
        self.conv1x1_1 = Conv(c4, 96, k=1)
        self.conv1x1_2 = Conv(c5, 96, k=1)
        self.smooth_0 = Conv(96, 96, k=3, p=1)
        self.smooth_1 = Conv(96, 96, k=3, p=1)
        self.smooth_2 = Conv(96, 96, k=3, p=1)
        self.smooth_3 = Conv(96, 96, k=3, p=1)
        self.head_conv_1 = nn.Sequential(Conv(96, 96, k=3, p=1, g=96), Conv(96, 96, k=1), Conv(96, 96, k=3, p=1, g=96), Conv(96, 96, k=1))
        self.head_conv_2 = nn.Sequential(Conv(96, 96, k=3, p=1, g=96), Conv(96, 96, k=1), Conv(96, 96, k=3, p=1, g=96), Conv(96, 96, k=1))
        self.head_conv_3 = nn.Sequential(Conv(96, 96, k=3, p=1, g=96), Conv(96, 96, k=1), Conv(96, 96, k=3, p=1, g=96), Conv(96, 96, k=1))
        self.head_det_1 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), 1)
        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        nn.init.constant_(self.head_det_1.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_2.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_3.bias[..., :self.num_anchors], bias_value)

    def create_grid(self, img_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = img_size, img_size
        for ind, s in enumerate(self.stride):
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            grid_xy = grid_xy[None, :, None, :]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h * fmp_w, 1, 1).unsqueeze(0)
            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)
        return total_grid_xy, total_anchor_wh

    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)

    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """
        cls_inds = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), cls_inds]
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        return bboxes, scores, cls_inds

    @torch.no_grad()
    def inference_single_image(self, x):
        KA = self.num_anchors
        C = self.num_classes
        c3, c4, c5 = self.backbone(x)
        c5 = self.neck(c5)
        p3 = self.conv1x1_0(c3)
        p4 = self.conv1x1_1(c4)
        p5 = self.conv1x1_2(c5)
        p4 = self.smooth_0(p4 + F.interpolate(p5, scale_factor=2.0))
        p3 = self.smooth_1(p3 + F.interpolate(p4, scale_factor=2.0))
        p4 = self.smooth_2(p4 + F.interpolate(p3, scale_factor=0.5))
        p5 = self.smooth_3(p5 + F.interpolate(p4, scale_factor=0.5))
        pred_s = self.head_det_1(self.head_conv_1(p3))[0]
        pred_m = self.head_det_2(self.head_conv_2(p4))[0]
        pred_l = self.head_det_3(self.head_conv_3(p5))[0]
        preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        box_pred_list = []
        for i, pred in enumerate(preds):
            obj_pred_i = pred[:KA, :, :].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred_i = pred[KA:KA * (1 + C), :, :].permute(1, 2, 0).contiguous().view(-1, C)
            reg_pred_i = pred[KA * (1 + C):, :, :].permute(1, 2, 0).contiguous().view(-1, KA, 4)
            if self.center_sample:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
            else:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
            wh_pred_i = reg_pred_i[None, ..., 2:].exp() * self.anchors_wh[i]
            x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
            x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)[0].view(-1, 4)
            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            box_pred_list.append(box_pred_i)
        obj_pred = torch.cat(obj_pred_list, dim=0)
        cls_pred = torch.cat(cls_pred_list, dim=0)
        box_pred = torch.cat(box_pred_list, dim=0)
        bboxes = torch.clamp(box_pred / self.img_size, 0.0, 1.0)
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)
        scores = scores.numpy()
        bboxes = bboxes.numpy()
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
        return bboxes, scores, cls_inds

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            B = x.size(0)
            KA = self.num_anchors
            C = self.num_classes
            c3, c4, c5 = self.backbone(x)
            c5 = self.neck(c5)
            p3 = self.conv1x1_0(c3)
            p4 = self.conv1x1_1(c4)
            p5 = self.conv1x1_2(c5)
            p4 = self.smooth_0(p4 + F.interpolate(p5, scale_factor=2.0))
            p3 = self.smooth_1(p3 + F.interpolate(p4, scale_factor=2.0))
            p4 = self.smooth_2(p4 + F.interpolate(p3, scale_factor=0.5))
            p5 = self.smooth_3(p5 + F.interpolate(p4, scale_factor=0.5))
            pred_s = self.head_det_1(self.head_conv_1(p3))
            pred_m = self.head_det_2(self.head_conv_2(p4))
            pred_l = self.head_det_3(self.head_conv_3(p5))
            preds = [pred_s, pred_m, pred_l]
            obj_pred_list = []
            cls_pred_list = []
            box_pred_list = []
            for i, pred in enumerate(preds):
                obj_pred_i = pred[:, :KA, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred_i = pred[:, KA:KA * (1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
                reg_pred_i = pred[:, KA * (1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
                if self.center_sample:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
                else:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
                wh_pred_i = reg_pred_i[..., 2:].exp() * self.anchors_wh[i]
                x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
                x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
                box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1).view(B, -1, 4)
                obj_pred_list.append(obj_pred_i)
                cls_pred_list.append(cls_pred_i)
                box_pred_list.append(box_pred_i)
            obj_pred = torch.cat(obj_pred_list, dim=1)
            cls_pred = torch.cat(cls_pred_list, dim=1)
            box_pred = torch.cat(box_pred_list, dim=1)
            box_pred = box_pred / self.img_size
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)
            giou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
            targets = torch.cat([0.5 * (giou_pred[..., None].clone().detach() + 1.0), targets], dim=-1)
            return obj_pred, cls_pred, giou_pred, targets


def build_neck(model, in_ch, out_ch, act='lrelu'):
    if model == 'conv_blocks':
        None
        neck = ConvBlocks(c1=in_ch, c2=out_ch, act=act)
    elif model == 'spp':
        None
        neck = SPPBlock(c1=in_ch, c2=out_ch, act=act)
    elif model == 'spp-csp':
        None
        neck = SPPBlockCSP(c1=in_ch, c2=out_ch, act=act)
    elif model == 'spp-dw':
        None
        neck = SPPBlockDW(c1=in_ch, c2=out_ch, act=act)
    elif model == 'dilated_encoder':
        None
        neck = DilatedEncoder(c1=in_ch, c2=out_ch, act=act)
    return neck


class YOLOTiny(nn.Module):

    def __init__(self, cfg=None, device=None, img_size=640, num_classes=80, trainable=False, conf_thresh=0.001, nms_thresh=0.6, center_sample=False):
        super(YOLOTiny, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], pretrained=trainable)
        self.stride = strides
        anchor_size = cfg['anchor_size']
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // 3, 2).float()
        self.num_anchors = self.anchor_size.size(1)
        c3, c4, c5 = feature_channels
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)
        self.head_conv_0 = build_neck(model=cfg['neck'], in_ch=c5, out_ch=c5 // 2)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(c4 + c5 // 2, c4, n=1, shortcut=False)
        self.head_conv_1 = Conv(c4, c4 // 2, k=1)
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(c3 + c4 // 2, c3, n=1, shortcut=False)
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2)
        self.head_csp_2 = BottleneckCSP(c3 + c4 // 2, c4, n=1, shortcut=False)
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2)
        self.head_csp_3 = BottleneckCSP(c4 + c5 // 2, c5, n=1, shortcut=False)
        self.head_det_1 = nn.Conv2d(c3, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(c4, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(c5, self.num_anchors * (1 + self.num_classes + 4), 1)
        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        nn.init.constant_(self.head_det_1.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_2.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_3.bias[..., :self.num_anchors], bias_value)

    def create_grid(self, img_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = img_size, img_size
        for ind, s in enumerate(self.stride):
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            grid_xy = grid_xy[None, :, None, :]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h * fmp_w, 1, 1).unsqueeze(0)
            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)
        return total_grid_xy, total_anchor_wh

    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)

    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """
        cls_inds = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), cls_inds]
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        return bboxes, scores, cls_inds

    @torch.no_grad()
    def inference_single_image(self, x):
        KA = self.num_anchors
        C = self.num_classes
        c3, c4, c5 = self.backbone(x)
        c6 = self.head_conv_0(c5)
        c7 = self.head_upsample_0(c6)
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        c10 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)
        pred_s = self.head_det_1(c13)[0]
        pred_m = self.head_det_2(c16)[0]
        pred_l = self.head_det_3(c19)[0]
        preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        box_pred_list = []
        for i, pred in enumerate(preds):
            obj_pred_i = pred[:KA, :, :].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred_i = pred[KA:KA * (1 + C), :, :].permute(1, 2, 0).contiguous().view(-1, C)
            reg_pred_i = pred[KA * (1 + C):, :, :].permute(1, 2, 0).contiguous().view(-1, KA, 4)
            if self.center_sample:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
            else:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
            wh_pred_i = reg_pred_i[None, ..., 2:].exp() * self.anchors_wh[i]
            x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
            x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)[0].view(-1, 4)
            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            box_pred_list.append(box_pred_i)
        obj_pred = torch.cat(obj_pred_list, dim=0)
        cls_pred = torch.cat(cls_pred_list, dim=0)
        box_pred = torch.cat(box_pred_list, dim=0)
        bboxes = torch.clamp(box_pred / self.img_size, 0.0, 1.0)
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)
        scores = scores.numpy()
        bboxes = bboxes.numpy()
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
        return bboxes, scores, cls_inds

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            B = x.size(0)
            KA = self.num_anchors
            C = self.num_classes
            c3, c4, c5 = self.backbone(x)
            c6 = self.head_conv_0(c5)
            c7 = self.head_upsample_0(c6)
            c8 = torch.cat([c7, c4], dim=1)
            c9 = self.head_csp_0(c8)
            c10 = self.head_conv_1(c9)
            c11 = self.head_upsample_1(c10)
            c12 = torch.cat([c11, c3], dim=1)
            c13 = self.head_csp_1(c12)
            c14 = self.head_conv_2(c13)
            c15 = torch.cat([c14, c10], dim=1)
            c16 = self.head_csp_2(c15)
            c17 = self.head_conv_3(c16)
            c18 = torch.cat([c17, c6], dim=1)
            c19 = self.head_csp_3(c18)
            pred_s = self.head_det_1(c13)
            pred_m = self.head_det_2(c16)
            pred_l = self.head_det_3(c19)
            preds = [pred_s, pred_m, pred_l]
            obj_pred_list = []
            cls_pred_list = []
            box_pred_list = []
            for i, pred in enumerate(preds):
                obj_pred_i = pred[:, :KA, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred_i = pred[:, KA:KA * (1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
                reg_pred_i = pred[:, KA * (1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
                if self.center_sample:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
                else:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
                wh_pred_i = reg_pred_i[..., 2:].exp() * self.anchors_wh[i]
                x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
                x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
                box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1).view(B, -1, 4)
                obj_pred_list.append(obj_pred_i)
                cls_pred_list.append(cls_pred_i)
                box_pred_list.append(box_pred_i)
            obj_pred = torch.cat(obj_pred_list, dim=1)
            cls_pred = torch.cat(cls_pred_list, dim=1)
            box_pred = torch.cat(box_pred_list, dim=1)
            box_pred = box_pred / self.img_size
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)
            giou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
            targets = torch.cat([0.5 * (giou_pred[..., None].clone().detach() + 1.0), targets], dim=-1)
            return obj_pred, cls_pred, giou_pred, targets


class YOLOv1(nn.Module):

    def __init__(self, cfg=None, device=None, img_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.6, center_sample=False):
        super(YOLOv1, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], pretrained=trainable)
        self.stride = [strides[-1]]
        feature_dim = feature_channels[-1]
        head_dim = 512
        self.grid_xy = self.create_grid(img_size)
        self.neck = build_neck(model=cfg['neck'], in_ch=feature_dim, out_ch=head_dim)
        self.cls_feat = nn.Sequential(Conv(head_dim, head_dim, k=3, p=1, s=1), Conv(head_dim, head_dim, k=3, p=1, s=1))
        self.reg_feat = nn.Sequential(Conv(head_dim, head_dim, k=3, p=1, s=1), Conv(head_dim, head_dim, k=3, p=1, s=1), Conv(head_dim, head_dim, k=3, p=1, s=1), Conv(head_dim, head_dim, k=3, p=1, s=1))
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)
        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        nn.init.constant_(self.obj_pred.bias, bias_value)

    def create_grid(self, img_size):
        """img_size: [H, W]"""
        img_h = img_w = img_size
        fmp_h, fmp_w = img_h // self.stride[0], img_w // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
        grid_xy = grid_xy.unsqueeze(0)
        return grid_xy

    def set_grid(self, img_size):
        self.grid_xy = self.create_grid(img_size)
        self.img_size = img_size

    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, 4]"""
        if self.center_sample:
            xy_pred = reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy
        else:
            xy_pred = reg_pred[..., :2].sigmoid() + self.grid_xy
        wh_pred = reg_pred[..., 2:].exp()
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        box_pred = box_pred * self.stride[0]
        return box_pred

    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """
        cls_inds = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), cls_inds]
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        return bboxes, scores, cls_inds

    @torch.no_grad()
    def inference_single_image(self, x):
        x = self.backbone(x)[-1]
        x = self.neck(x)
        cls_feat = self.cls_feat(x)
        reg_feat = self.reg_feat(x)
        obj_pred = self.obj_pred(reg_feat)[0]
        cls_pred = self.cls_pred(cls_feat)[0]
        reg_pred = self.reg_pred(reg_feat)[0]
        obj_pred = obj_pred.flatten(1).permute(1, 0).contiguous()
        cls_pred = cls_pred.flatten(1).permute(1, 0).contiguous()
        reg_pred = reg_pred.flatten(1).permute(1, 0).contiguous()
        box_pred = self.decode_bbox(reg_pred[None])[0]
        bboxes = torch.clamp(box_pred / self.img_size, 0.0, 1.0)
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)
        scores = scores.numpy()
        bboxes = bboxes.numpy()
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
        return bboxes, scores, cls_inds

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            B = x.size(0)
            C = self.num_classes
            x = self.backbone(x)[-1]
            x = self.neck(x)
            cls_feat = self.cls_feat(x)
            reg_feat = self.reg_feat(x)
            obj_pred = self.obj_pred(reg_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            obj_pred = obj_pred.flatten(2).permute(0, 2, 1).contiguous()
            cls_pred = cls_pred.flatten(2).permute(0, 2, 1).contiguous()
            reg_pred = reg_pred.flatten(2).permute(0, 2, 1).contiguous()
            box_pred = self.decode_bbox(reg_pred)
            box_pred = box_pred / self.img_size
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)
            giou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
            targets = torch.cat([0.5 * (giou_pred[..., None].clone().detach() + 1.0), targets], dim=-1)
            return obj_pred, cls_pred, giou_pred, targets


class YOLOv2(nn.Module):

    def __init__(self, cfg=None, device=None, img_size=None, num_classes=20, trainable=False, conf_thresh=0.001, nms_thresh=0.6, center_sample=False):
        super(YOLOv2, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(cfg['anchor_size'])
        self.num_anchors = len(cfg['anchor_size'])
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], pretrained=trainable)
        self.stride = [strides[-1]]
        feature_dim = feature_channels[-1]
        head_dim = 512
        self.grid_xy, self.anchor_wh = self.create_grid(img_size)
        self.neck = build_neck(model=cfg['neck'], in_ch=feature_dim, out_ch=head_dim)
        self.cls_feat = nn.Sequential(Conv(head_dim, head_dim, k=3, p=1, s=1), Conv(head_dim, head_dim, k=3, p=1, s=1))
        self.reg_feat = nn.Sequential(Conv(head_dim, head_dim, k=3, p=1, s=1), Conv(head_dim, head_dim, k=3, p=1, s=1), Conv(head_dim, head_dim, k=3, p=1, s=1), Conv(head_dim, head_dim, k=3, p=1, s=1))
        self.obj_pred = nn.Conv2d(head_dim, self.num_anchors * 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, self.num_anchors * self.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, self.num_anchors * 4, kernel_size=1)
        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        nn.init.constant_(self.obj_pred.bias, bias_value)

    def create_grid(self, img_size):
        """img_size: [H, W]"""
        img_h = img_w = img_size
        fmp_h, fmp_w = img_h // self.stride[0], img_w // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
        grid_xy = grid_xy[None, :, None, :]
        anchor_wh = self.anchor_size.repeat(fmp_h * fmp_w, 1, 1).unsqueeze(0)
        return grid_xy, anchor_wh

    def set_grid(self, img_size):
        self.grid_xy, self.anchor_wh = self.create_grid(img_size)
        self.img_size = img_size

    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """
        cls_inds = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), cls_inds]
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        return bboxes, scores, cls_inds

    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        return box_pred

    @torch.no_grad()
    def inference_single_image(self, x):
        KA = self.num_anchors
        C = self.num_classes
        x = self.backbone(x)[-1]
        x = self.neck(x)
        cls_feat = self.cls_feat(x)
        reg_feat = self.reg_feat(x)
        obj_pred = self.obj_pred(reg_feat)[0]
        cls_pred = self.cls_pred(cls_feat)[0]
        reg_pred = self.reg_pred(reg_feat)[0]
        obj_pred = obj_pred.permute(1, 2, 0).contiguous().view(-1, 1)
        cls_pred = cls_pred.permute(1, 2, 0).contiguous().view(-1, C)
        reg_pred = reg_pred.permute(1, 2, 0).contiguous().view(-1, KA, 4)
        box_pred = self.decode_bbox(reg_pred[None])[0]
        bboxes = torch.clamp(box_pred / self.img_size, 0.0, 1.0)
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)
        scores = scores.numpy()
        bboxes = bboxes.numpy()
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
        return bboxes, scores, cls_inds

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            B = x.size(0)
            KA = self.num_anchors
            C = self.num_classes
            x = self.backbone(x)[-1]
            x = self.neck(x)
            cls_feat = self.cls_feat(x)
            reg_feat = self.reg_feat(x)
            obj_pred = self.obj_pred(reg_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
            box_pred = self.decode_bbox(reg_pred)
            box_pred = box_pred / self.img_size
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)
            giou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
            targets = torch.cat([0.5 * (giou_pred[..., None].clone().detach() + 1.0), targets], dim=-1)
            return obj_pred, cls_pred, giou_pred, targets


class YOLOv3(nn.Module):

    def __init__(self, cfg=None, device=None, img_size=640, num_classes=80, trainable=False, conf_thresh=0.001, nms_thresh=0.6, center_sample=False):
        super(YOLOv3, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], pretrained=trainable)
        self.stride = strides
        anchor_size = cfg['anchor_size']
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // 3, 2).float()
        self.num_anchors = self.anchor_size.size(1)
        c3, c4, c5 = feature_channels
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)
        self.head_convblock_0 = build_neck(model=cfg['neck'], in_ch=c5, out_ch=c5 // 2)
        self.head_conv_0 = Conv(c5 // 2, c4 // 2, k=1)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_conv_1 = Conv(c5 // 2, c5, k=3, p=1)
        self.head_convblock_1 = ConvBlocks(c4 + c4 // 2, c4 // 2)
        self.head_conv_2 = Conv(c4 // 2, c3 // 2, k=1)
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_conv_3 = Conv(c4 // 2, c4, k=3, p=1)
        self.head_convblock_2 = ConvBlocks(c3 + c3 // 2, c3 // 2)
        self.head_conv_4 = Conv(c3 // 2, c3, k=3, p=1)
        self.head_det_1 = nn.Conv2d(c3, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(c4, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(c5, self.num_anchors * (1 + self.num_classes + 4), 1)
        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        nn.init.constant_(self.head_det_1.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_2.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_3.bias[..., :self.num_anchors], bias_value)

    def create_grid(self, img_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = img_size, img_size
        for ind, s in enumerate(self.stride):
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            grid_xy = grid_xy[None, :, None, :]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h * fmp_w, 1, 1).unsqueeze(0)
            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)
        return total_grid_xy, total_anchor_wh

    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)

    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """
        cls_inds = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), cls_inds]
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        return bboxes, scores, cls_inds

    @torch.no_grad()
    def inference_single_image(self, x):
        KA = self.num_anchors
        C = self.num_classes
        c3, c4, c5 = self.backbone(x)
        p5 = self.head_convblock_0(c5)
        p5_up = self.head_upsample_0(self.head_conv_0(p5))
        p5 = self.head_conv_1(p5)
        p4 = self.head_convblock_1(torch.cat([c4, p5_up], dim=1))
        p4_up = self.head_upsample_1(self.head_conv_2(p4))
        p4 = self.head_conv_3(p4)
        p3 = self.head_convblock_2(torch.cat([c3, p4_up], dim=1))
        p3 = self.head_conv_4(p3)
        pred_s = self.head_det_1(p3)[0]
        pred_m = self.head_det_2(p4)[0]
        pred_l = self.head_det_3(p5)[0]
        preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        box_pred_list = []
        for i, pred in enumerate(preds):
            obj_pred_i = pred[:KA, :, :].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred_i = pred[KA:KA * (1 + C), :, :].permute(1, 2, 0).contiguous().view(-1, C)
            reg_pred_i = pred[KA * (1 + C):, :, :].permute(1, 2, 0).contiguous().view(-1, KA, 4)
            if self.center_sample:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
            else:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
            wh_pred_i = reg_pred_i[None, ..., 2:].exp() * self.anchors_wh[i]
            x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
            x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)[0].view(-1, 4)
            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            box_pred_list.append(box_pred_i)
        obj_pred = torch.cat(obj_pred_list, dim=0)
        cls_pred = torch.cat(cls_pred_list, dim=0)
        box_pred = torch.cat(box_pred_list, dim=0)
        bboxes = torch.clamp(box_pred / self.img_size, 0.0, 1.0)
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)
        scores = scores.numpy()
        bboxes = bboxes.numpy()
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
        return bboxes, scores, cls_inds

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            B = x.size(0)
            KA = self.num_anchors
            C = self.num_classes
            c3, c4, c5 = self.backbone(x)
            p5 = self.head_convblock_0(c5)
            p5_up = self.head_upsample_0(self.head_conv_0(p5))
            p5 = self.head_conv_1(p5)
            p4 = self.head_convblock_1(torch.cat([c4, p5_up], dim=1))
            p4_up = self.head_upsample_1(self.head_conv_2(p4))
            p4 = self.head_conv_3(p4)
            p3 = self.head_convblock_2(torch.cat([c3, p4_up], dim=1))
            p3 = self.head_conv_4(p3)
            pred_s = self.head_det_1(p3)
            pred_m = self.head_det_2(p4)
            pred_l = self.head_det_3(p5)
            preds = [pred_s, pred_m, pred_l]
            obj_pred_list = []
            cls_pred_list = []
            box_pred_list = []
            for i, pred in enumerate(preds):
                obj_pred_i = pred[:, :KA, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred_i = pred[:, KA:KA * (1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
                reg_pred_i = pred[:, KA * (1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
                if self.center_sample:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
                else:
                    xy_pred_i = (reg_pred_i[..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
                wh_pred_i = reg_pred_i[..., 2:].exp() * self.anchors_wh[i]
                x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
                x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
                box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1).view(B, -1, 4)
                obj_pred_list.append(obj_pred_i)
                cls_pred_list.append(cls_pred_i)
                box_pred_list.append(box_pred_i)
            obj_pred = torch.cat(obj_pred_list, dim=1)
            cls_pred = torch.cat(cls_pred_list, dim=1)
            box_pred = torch.cat(box_pred_list, dim=1)
            box_pred = box_pred / self.img_size
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)
            giou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
            targets = torch.cat([0.5 * (giou_pred[..., None].clone().detach() + 1.0), targets], dim=-1)
            return obj_pred, cls_pred, giou_pred, targets


class YOLOv4(nn.Module):

    def __init__(self, cfg=None, device=None, img_size=640, num_classes=80, trainable=False, conf_thresh=0.001, nms_thresh=0.6, center_sample=False):
        super(YOLOv4, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], pretrained=trainable)
        self.stride = strides
        anchor_size = cfg['anchor_size']
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // 3, 2).float()
        self.num_anchors = self.anchor_size.size(1)
        c3, c4, c5 = feature_channels
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)
        self.head_conv_0 = build_neck(model=cfg['neck'], in_ch=c5, out_ch=c5 // 2)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(c4 + c5 // 2, c4, n=3, shortcut=False)
        self.head_conv_1 = Conv(c4, c4 // 2, k=1)
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(c3 + c4 // 2, c3, n=3, shortcut=False)
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2)
        self.head_csp_2 = BottleneckCSP(c3 + c4 // 2, c4, n=3, shortcut=False)
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2)
        self.head_csp_3 = BottleneckCSP(c4 + c5 // 2, c5, n=3, shortcut=False)
        self.head_det_1 = nn.Conv2d(c3, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(c4, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(c5, self.num_anchors * (1 + self.num_classes + 4), 1)
        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1.0 - init_prob) / init_prob))
        nn.init.constant_(self.head_det_1.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_2.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_3.bias[..., :self.num_anchors], bias_value)

    def create_grid(self, img_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = img_size, img_size
        for ind, s in enumerate(self.stride):
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            grid_xy = grid_xy[None, :, None, :]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h * fmp_w, 1, 1).unsqueeze(0)
            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)
        return total_grid_xy, total_anchor_wh

    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)

    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """
        cls_inds = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), cls_inds]
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]
        return bboxes, scores, cls_inds

    @torch.no_grad()
    def inference_single_image(self, x):
        KA = self.num_anchors
        C = self.num_classes
        c3, c4, c5 = self.backbone(x)
        c6 = self.head_conv_0(c5)
        c7 = self.head_upsample_0(c6)
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        c10 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)
        pred_s = self.head_det_1(c13)[0]
        pred_m = self.head_det_2(c16)[0]
        pred_l = self.head_det_3(c19)[0]
        preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        box_pred_list = []
        for i, pred in enumerate(preds):
            obj_pred_i = pred[:KA, :, :].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred_i = pred[KA:KA * (1 + C), :, :].permute(1, 2, 0).contiguous().view(-1, C)
            reg_pred_i = pred[KA * (1 + C):, :, :].permute(1, 2, 0).contiguous().view(-1, KA, 4)
            if self.center_sample:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() * 2.0 - 1.0 + self.grid_cell[i]) * self.stride[i]
            else:
                xy_pred_i = (reg_pred_i[None, ..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
            wh_pred_i = reg_pred_i[None, ..., 2:].exp() * self.anchors_wh[i]
            x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
            x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)[0].view(-1, 4)
            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            box_pred_list.append(box_pred_i)
        obj_pred = torch.cat(obj_pred_list, dim=0)
        cls_pred = torch.cat(cls_pred_list, dim=0)
        box_pred = torch.cat(box_pred_list, dim=0)
        bboxes = torch.clamp(box_pred / self.img_size, 0.0, 1.0)
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)
        scores = scores.numpy()
        bboxes = bboxes.numpy()
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)
        return bboxes, scores, cls_inds

    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            B = x.size(0)
            KA = self.num_anchors
            C = self.num_classes
            c3, c4, c5 = self.backbone(x)
            c6 = self.head_conv_0(c5)
            c7 = self.head_upsample_0(c6)
            c8 = torch.cat([c7, c4], dim=1)
            c9 = self.head_csp_0(c8)
            c10 = self.head_conv_1(c9)
            c11 = self.head_upsample_1(c10)
            c12 = torch.cat([c11, c3], dim=1)
            c13 = self.head_csp_1(c12)
            c14 = self.head_conv_2(c13)
            c15 = torch.cat([c14, c10], dim=1)
            c16 = self.head_csp_2(c15)
            c17 = self.head_conv_3(c16)
            c18 = torch.cat([c17, c6], dim=1)
            c19 = self.head_csp_3(c18)
            pred_s = self.head_det_1(c13)
            pred_m = self.head_det_2(c16)
            pred_l = self.head_det_3(c19)
            preds = [pred_s, pred_m, pred_l]
            obj_pred_list = []
            cls_pred_list = []
            box_pred_list = []
            for i, pred in enumerate(preds):
                obj_pred_i = pred[:, :KA, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred_i = pred[:, KA:KA * (1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
                reg_pred_i = pred[:, KA * (1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
                if self.center_sample:
                    xy_pred_i = (self.grid_cell[i] + reg_pred_i[..., :2].sigmoid() * 2.0 - 1.0) * self.stride[i]
                else:
                    xy_pred_i = (self.grid_cell[i] + reg_pred_i[..., :2].sigmoid()) * self.stride[i]
                wh_pred_i = reg_pred_i[..., 2:].exp() * self.anchors_wh[i]
                x1y1_pred_i = xy_pred_i - wh_pred_i * 0.5
                x2y2_pred_i = xy_pred_i + wh_pred_i * 0.5
                box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1).view(B, -1, 4)
                obj_pred_list.append(obj_pred_i)
                cls_pred_list.append(cls_pred_i)
                box_pred_list.append(box_pred_i)
            obj_pred = torch.cat(obj_pred_list, dim=1)
            cls_pred = torch.cat(cls_pred_list, dim=1)
            box_pred = torch.cat(box_pred_list, dim=1)
            box_pred = box_pred / self.img_size
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)
            if self.cfg['loss_box'] == 'iou':
                iou_pred = box_ops.iou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
                obj_tgt = iou_pred[..., None].clone().detach().clamp(0.0)
            elif self.cfg['loss_box'] == 'giou':
                iou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
                obj_tgt = 0.5 * (iou_pred[..., None].clone().detach() + 1.0)
            elif self.cfg['loss_box'] == 'ciou':
                iou_pred = box_ops.ciou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
                obj_tgt = iou_pred[..., None].clone().detach().clamp(0.0)
            targets = torch.cat([obj_tgt, targets], dim=-1)
            return obj_pred, cls_pred, iou_pred, targets


class MSEWithLogitsLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets, target_pos):
        inputs = logits.sigmoid()
        loss = F.mse_loss(input=inputs, target=targets, reduction='none')
        pos_loss = loss * target_pos * 5.0
        neg_loss = loss * (1.0 - target_pos) * 1.0
        loss = pos_loss + neg_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, pos_weight=1.0, neg_weight=0.25, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction

    def forward(self, logits, targets, target_pos):
        loss = F.binary_cross_entropy_with_logits(input=logits, target=targets, reduction='none')
        pos_loss = loss * target_pos * self.pos_weight
        neg_loss = loss * (1.0 - target_pos) * self.neg_weight
        loss = pos_loss + neg_loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


class Criterion(nn.Module):

    def __init__(self, args, cfg, loss_obj_weight=1.0, loss_cls_weight=1.0, loss_reg_weight=1.0, num_classes=80):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        try:
            if cfg['loss_obj'] == 'mse':
                self.obj_loss_f = MSEWithLogitsLoss(reduction='none')
            elif cfg['loss_obj'] == 'bce':
                self.obj_loss_f = BCEWithLogitsLoss(reduction='none')
        except:
            self.obj_loss_f = MSEWithLogitsLoss(reduction='none')
        self.cls_loss_f = nn.CrossEntropyLoss(reduction='none')

    def loss_objectness(self, pred_obj, target_obj, target_pos):
        """
            pred_obj: (FloatTensor) [B, HW, 1]
            target_obj: (FloatTensor) [B, HW,]
            target_pos: (FloatTensor) [B, HW,]
        """
        loss_obj = self.obj_loss_f(pred_obj[..., 0], target_obj, target_pos)
        if self.args.scale_loss == 'batch':
            batch_size = pred_obj.size(0)
            loss_obj = loss_obj.sum() / batch_size
        elif self.args.scale_loss == 'positive':
            num_pos = target_pos.sum().clamp(1.0)
            loss_obj = loss_obj.sum() / num_pos
        return loss_obj

    def loss_class(self, pred_cls, target_cls, target_pos):
        """
            pred_cls: (FloatTensor) [B, HW, C]
            target_cls: (LongTensor) [B, HW,]
            target_pos: (FloatTensor) [B, HW,]
        """
        pred_cls = pred_cls.permute(0, 2, 1)
        loss_cls = self.cls_loss_f(pred_cls, target_cls)
        loss_cls = loss_cls * target_pos
        if self.args.scale_loss == 'batch':
            batch_size = pred_cls.size(0)
            loss_cls = loss_cls.sum() / batch_size
        elif self.args.scale_loss == 'positive':
            num_pos = target_pos.sum().clamp(1.0)
            loss_cls = loss_cls.sum() / num_pos
        return loss_cls

    def loss_bbox(self, pred_iou, target_pos, target_scale):
        """
            pred_iou: (FloatTensor) [B, HW, ]
            target_pos: (FloatTensor) [B, HW,]
            target_scale: (FloatTensor) [B, HW,]
        """
        loss_reg = 1.0 - pred_iou
        loss_reg = loss_reg * target_scale
        loss_reg = loss_reg * target_pos
        if self.args.scale_loss == 'batch':
            batch_size = pred_iou.size(0)
            loss_reg = loss_reg.sum() / batch_size
        elif self.args.scale_loss == 'positive':
            num_pos = target_pos.sum().clamp(1.0)
            loss_reg = loss_reg.sum() / num_pos
        return loss_reg

    def forward(self, pred_obj, pred_cls, pred_iou, targets):
        """
            pred_obj: (Tensor) [B, HW, 1]
            pred_cls: (Tensor) [B, HW, C]
            pred_iou: (Tensor) [B, HW,]
            targets: (Tensor) [B, HW, 1+1+1+4]
        """
        target_obj = targets[..., 0].float()
        target_pos = targets[..., 1].float()
        target_cls = targets[..., 2].long()
        target_scale = targets[..., -1].float()
        loss_obj = self.loss_objectness(pred_obj, target_obj, target_pos)
        loss_cls = self.loss_class(pred_cls, target_cls, target_pos)
        loss_reg = self.loss_bbox(pred_iou, target_pos, target_scale)
        losses = self.loss_obj_weight * loss_obj + self.loss_cls_weight * loss_cls + self.loss_reg_weight * loss_reg
        return loss_obj, loss_cls, loss_reg, losses


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BCEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BaseConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CSPDarkNet53,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Conv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlocks,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv_BN_LeakyReLU,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CrossStagePartialBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'stage_layers': _mock_layer(), 'is_csp_first_stage': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DWConv,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'ksize': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarkBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DarkNet_53,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (DilatedBottleneck,
     lambda: ([], {'c': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DilatedEncoder,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Focus,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MSEWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Mlp,
     lambda: ([], {'in_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPBlock,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPBlockCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPBlockDW,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPPBottleneck,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ShuffleV2Block,
     lambda: ([], {'inp': 4, 'oup': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (resblock,
     lambda: ([], {'ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_yjh0410_PyTorch_YOLO_Family(_paritybench_base):
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

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

