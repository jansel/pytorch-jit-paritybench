import sys
_module = sys.modules[__name__]
del sys
cityscapes = _module
configs = _module
configurations = _module
evaluate = _module
logger = _module
loss = _module
models = _module
darknet = _module
deeplabv3plus = _module
resnet = _module
xception = _module
modules = _module
bn = _module
functions = _module
misc = _module
one_hot = _module
optimizer = _module
train = _module
transform = _module

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


from torch.utils.data import Dataset


import torchvision.transforms as transforms


import numpy as np


import torch.nn as nn


from torch.utils.data import DataLoader


import torch.nn.functional as F


import torch.distributed as dist


import logging


import time


import torch.utils.model_zoo as modelzoo


import torchvision


import torch.utils.checkpoint as ckpt


import torch.nn.functional as functional


import torch.autograd as autograd


import torch.cuda.comm as comm


from torch.autograd.function import once_differentiable


from torch.utils.cpp_extension import load


class OhemCELoss(nn.Module):

    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = thresh
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        n_pixs = N * H * W
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        with torch.no_grad():
            scores = F.softmax(logits, dim=1)
            labels_cpu = labels
            invalid_mask = labels_cpu == self.ignore_lb
            labels_cpu[invalid_mask] = 0
            picks = scores[torch.arange(n_pixs), labels_cpu]
            picks[invalid_mask] = 1
            sorteds, _ = torch.sort(picks)
            thresh = self.thresh if sorteds[self.n_min] < self.thresh else sorteds[self.n_min]
            labels[picks > thresh] = self.ignore_lb
        labels = labels.clone()
        loss = self.criteria(logits, labels)
        return loss


class ConvBNLeaky(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, pad=0, dilation=1, slope=0.1, *args, **kwargs):
        super(ConvBNLeaky, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, padding=pad, dilation=dilation, stride=stride, bias=False)
        self.bn = BatchNorm2d(out_chan, slope=slope)
        self._init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def _init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        if not self.conv.bias is None:
            nn.init.constant_(self.conv.bias, 0)


class ResidualBlock(nn.Module):

    def __init__(self, in_chan, out_chan, dilation=1, *args, **kwargs):
        super(ResidualBlock, self).__init__()
        inner_chan = int(out_chan / 2)
        self.conv_blk1 = ConvBNLeaky(in_chan, inner_chan, ks=1, slope=0.1)
        self.conv_blk2 = ConvBNLeaky(inner_chan, out_chan, ks=3, pad=dilation, dilation=dilation, slope=0.1)

    def forward(self, x):
        residual = self.conv_blk1(x)
        residual = self.conv_blk2(residual)
        out = x + residual
        return out


def _make_stage(in_chan, out_chan, n_block, dilation=1):
    assert dilation in (1, 2, 4)
    stride, dila_first = (2, 1) if dilation == 1 else (1, dilation // 2)
    downsample = ConvBNLeaky(in_chan, out_chan, ks=3, stride=stride, pad=dila_first, dilation=dila_first)
    layers = [downsample]
    for i in range(n_block):
        layers.append(ResidualBlock(in_chan=out_chan, out_chan=out_chan, dilation=dilation))
    stage = nn.Sequential(*layers)
    return stage


class Darknet53(nn.Module):

    def __init__(self, stride=32, *args, **kwargs):
        super(Darknet53, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride == 32 else [(el * (16 // stride)) for el in (1, 2)]
        self.conv1 = ConvBNLeaky(3, 32, ks=3, stride=1, pad=1)
        self.layer1 = _make_stage(32, 64, 1)
        self.layer2 = _make_stage(64, 128, 2)
        self.layer3 = _make_stage(128, 256, 8)
        self.layer4 = _make_stage(256, 512, 8, dils[0])
        self.layer5 = _make_stage(512, 1024, 4, dils[1])

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.layer1(feat)
        feat4 = self.layer2(feat)
        feat8 = self.layer3(feat4)
        feat16 = self.layer4(feat8)
        feat32 = self.layer5(feat16)
        return feat4, feat8, feat16, feat32

    def get_params(self):
        bn_params = []
        non_bn_params = list(self.parameters())
        return bn_params, non_bn_params


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, bias=False, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=bias)
        self.bn = BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class ASPP(nn.Module):

    def __init__(self, in_chan=2048, out_chan=256, with_gp=True, *args, **kwargs):
        super(ASPP, self).__init__()
        self.with_gp = with_gp
        self.conv1 = ConvBNReLU(in_chan, out_chan, ks=1, dilation=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=6, padding=6)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=12, padding=12)
        self.conv4 = ConvBNReLU(in_chan, out_chan, ks=3, dilation=18, padding=18)
        if self.with_gp:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.conv1x1 = ConvBNReLU(in_chan, out_chan, ks=1)
            self.conv_out = ConvBNReLU(out_chan * 5, out_chan, ks=1)
        else:
            self.conv_out = ConvBNReLU(out_chan * 4, out_chan, ks=1)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat1 = self.conv1(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        if self.with_gp:
            avg = self.avg(x)
            feat5 = self.conv1x1(avg)
            feat5 = F.interpolate(feat5, (H, W), mode='bilinear', align_corners=True)
            feat = torch.cat([feat1, feat2, feat3, feat4, feat5], 1)
        else:
            feat = torch.cat([feat1, feat2, feat3, feat4], 1)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Decoder(nn.Module):

    def __init__(self, n_classes, low_chan=256, *args, **kwargs):
        super(Decoder, self).__init__()
        self.conv_low = ConvBNReLU(low_chan, 48, ks=1, padding=0)
        self.conv_cat = nn.Sequential(ConvBNReLU(304, 256, ks=3, padding=1), ConvBNReLU(256, 256, ks=3, padding=1))
        self.conv_out = nn.Conv2d(256, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, feat_low, feat_aspp):
        H, W = feat_low.size()[2:]
        feat_low = self.conv_low(feat_low)
        feat_aspp_up = F.interpolate(feat_aspp, (H, W), mode='bilinear', align_corners=True)
        feat_cat = torch.cat([feat_low, feat_aspp_up], dim=1)
        feat_out = self.conv_cat(feat_cat)
        logits = self.conv_out(feat_out)
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Bottleneck(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1, stride_at_1x1=False, dilation=1, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)
        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        mid_chan = int(out_chan / 4)
        self.conv1 = nn.Conv2d(in_chan, mid_chan, kernel_size=1, stride=stride1x1, bias=False)
        self.bn1 = BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(mid_chan, mid_chan, kernel_size=3, stride=stride3x3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False), BatchNorm2d(out_chan, activation='none'))
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.conv3(residual)
        residual = self.bn3(residual)
        if self.downsample == None:
            inten = x
        else:
            inten = self.downsample(x)
        out = residual + inten
        out = self.relu(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


def create_stage(in_chan, out_chan, b_num, stride=1, dilation=1):
    assert out_chan % 4 == 0
    mid_chan = out_chan / 4
    blocks = [Bottleneck(in_chan, out_chan, stride=stride, dilation=dilation)]
    for i in range(1, b_num):
        blocks.append(Bottleneck(out_chan, out_chan, stride=1, dilation=dilation))
    return nn.Sequential(*blocks)


resnet101_url = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'


class Resnet101(nn.Module):

    def __init__(self, stride=32, *args, **kwargs):
        super(Resnet101, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride == 32 else [(el * (16 // stride)) for el in (1, 2)]
        strds = [(2 if el == 1 else 1) for el in dils]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = create_stage(64, 256, 3, stride=1, dilation=1)
        self.layer2 = create_stage(256, 512, 4, stride=2, dilation=1)
        self.layer3 = create_stage(512, 1024, 23, stride=strds[0], dilation=dils[0])
        self.layer4 = create_stage(1024, 2048, 3, stride=strds[1], dilation=dils[1])
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet101_url)
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k in state_dict.keys():
                self_state_dict.update({k: state_dict[k]})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        bn_params = []
        non_bn_params = []
        for name, param in self.named_parameters():
            if 'bn' in name or 'downsample.1' in name:
                bn_params.append(param)
            else:
                non_bn_params.append(param)
        return bn_params, non_bn_params


class Deeplab_v3plus(nn.Module):

    def __init__(self, cfg, *args, **kwargs):
        super(Deeplab_v3plus, self).__init__()
        self.backbone = Resnet101(stride=16)
        self.aspp = ASPP(in_chan=2048, out_chan=256, with_gp=cfg.aspp_global_feature)
        self.decoder = Decoder(cfg.n_classes, low_chan=256)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat4, _, _, feat32 = self.backbone(x)
        feat_aspp = self.aspp(feat32)
        logits = self.decoder(feat4, feat_aspp)
        logits = F.interpolate(logits, (H, W), mode='bilinear', align_corners=True)
        return logits

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

    def get_params(self):
        back_bn_params, back_no_bn_params = self.backbone.get_params()
        tune_wd_params = list(self.aspp.parameters()) + list(self.decoder.parameters()) + back_no_bn_params
        no_tune_wd_params = back_bn_params
        return tune_wd_params, no_tune_wd_params


class SepConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, dilation=1, bias=False, *args, **kwargs):
        super(SepConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, in_chan, kernel_size=ks, stride=stride, padding=dilation, dilation=dilation, groups=in_chan, bias=bias)
        self.bn = BatchNorm2d(in_chan)
        self.pairwise = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=bias)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pairwise(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class Block(nn.Module):

    def __init__(self, in_chan, out_chan, reps=3, stride=1, dilation=1, start_with_relu=True, grow_layer=0, bias=False, *args, **kwargs):
        super(Block, self).__init__()
        self.stride = stride
        self.in_chan = in_chan
        self.out_chan = out_chan
        inchans = [in_chan] * reps
        outchans = [out_chan] * reps
        for i in range(reps):
            if i < grow_layer:
                inchans[i] = in_chan
                outchans[i] = in_chan
            else:
                inchans[i] = out_chan
                outchans[i] = out_chan
        inchans[grow_layer] = in_chan
        outchans[grow_layer] = out_chan
        self.shortcut = nn.Sequential(*[nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=bias), nn.BatchNorm2d(out_chan)])
        layers = []
        if start_with_relu:
            layers.append(nn.ReLU(inplace=False))
        for i in range(reps - 1):
            layers.append(SepConvBNReLU(inchans[i], outchans[i], kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=bias))
            layers.append(nn.BatchNorm2d(outchans[i]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(SepConvBNReLU(inchans[-1], outchans[-1], kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=bias))
        layers.append(nn.BatchNorm2d(outchans[i]))
        self.residual = nn.Sequential(*layers)
        self.init_weight()

    def forward(self, x):
        resd = self.residual(x)
        if not self.stride == 1 or not self.in_chan == self.out_chan:
            sc = self.shortcut(x)
        else:
            sc = x
        out = resd + sc
        return out

    def init_weight(self):
        for ly in self.shortcut.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class EntryFlow(nn.Module):

    def __init__(self, *args, **kwargs):
        super(EntryFlow, self).__init__()
        self.conv1 = ConvBNReLU(3, 32, ks=3, stride=2, padding=1)
        self.conv2 = ConvBNReLU(32, 64, ks=3, stride=1, padding=1)
        self.block1 = Block(64, 128, reps=3, stride=2, dilation=1, start_with_relu=False, grow_layer=0)
        self.block2 = Block(128, 256, reps=3, stride=1, dilation=1, start_with_relu=True, grow_layer=0)
        self.block3 = Block(256, 256, reps=3, stride=2, dilation=1, start_with_relu=True, grow_layer=0)
        self.block4 = Block(256, 728, reps=3, stride=1, dilation=1, start_with_relu=True, grow_layer=0)
        self.block5 = Block(728, 728, reps=3, stride=2, dilation=1, start_with_relu=True, grow_layer=0)

    def forward(self, x):
        feat2 = self.conv1(x)
        feat2 = self.conv2(feat2)
        feat4 = self.block1(feat2)
        feat4 = self.block2(feat4)
        feat8 = self.block3(feat4)
        feat8 = self.block4(feat8)
        feat16 = self.block5(feat8)
        return feat4, feat16


class MiddleFlow(nn.Module):

    def __init__(self, dilation=1, *args, **kwargs):
        super(MiddleFlow, self).__init__()
        middle_layers = []
        for i in range(16):
            middle_layers.append(Block(728, 728, reps=3, stride=1, dilation=dilation, start_with_relu=True, grow_layer=0))
        self.middle_flow = nn.Sequential(*middle_layers)

    def forward(self, x):
        out = self.middle_flow(x)
        return out


class ExitEntry(nn.Module):

    def __init__(self, stride=1, dilation=2, *args, **kwargs):
        super(ExitEntry, self).__init__()
        self.block1 = Block(728, 1024, reps=3, stride=stride, dilation=dilation, start_with_relu=True, grow_layer=1)
        self.sepconv1 = SepConvBNReLU(1024, 1536, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.sepconv2 = SepConvBNReLU(1536, 1536, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.sepconv3 = SepConvBNReLU(1536, 2048, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        out = self.sepconv3(x)
        return out


class Xception71(nn.Module):

    def __init__(self, *args, **kwargs):
        super(Xception71, self).__init__()
        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow(dilation=1)
        self.exit_flow = ExitEntry(stride=1, dilation=2)

    def forward(self, x):
        feat4, feat16 = self.entry_flow(x)
        feat_mid = self.middle_flow(feat16)
        feat_exit = self.exit_flow(feat_mid)
        return feat4, feat_exit


ACT_ELU = 'elu'


ACT_LEAKY_RELU = 'leaky_relu'


ACT_RELU = 'relu'


class ABN(nn.Module):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, activation='leaky_relu', slope=0.01):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        eps : float
            Small constant to prevent numerical issues.
        momentum : float
            Momentum factor applied to compute running statistics as.
        affine : bool
            If `True` apply learned scale and shift transformation after normalization.
        activation : str
            Name of the activation functions, one of: `leaky_relu`, `elu` or `none`.
        slope : float
            Negative slope for the `leaky_relu` activation.
        """
        super(ABN, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.momentum = momentum
        self.activation = activation
        self.slope = slope
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.running_mean, 0)
        nn.init.constant_(self.running_var, 1)
        if self.affine:
            nn.init.constant_(self.weight, 1)
            nn.init.constant_(self.bias, 0)

    def forward(self, x):
        x = functional.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)
        if self.activation == ACT_RELU:
            return functional.relu(x, inplace=True)
        elif self.activation == ACT_LEAKY_RELU:
            return functional.leaky_relu(x, negative_slope=self.slope, inplace=True)
        elif self.activation == ACT_ELU:
            return functional.elu(x, inplace=True)
        else:
            return x

    def __repr__(self):
        rep = '{name}({num_features}, eps={eps}, momentum={momentum}, affine={affine}, activation={activation}'
        if self.activation == 'leaky_relu':
            rep += ', slope={slope})'
        else:
            rep += ')'
        return rep.format(name=self.__class__.__name__, **self.__dict__)


ACT_NONE = 'none'


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_backward(x, dx, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_backward(x, dx)
    elif ctx.activation == ACT_NONE:
        pass


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _backend.leaky_relu_forward(x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _backend.elu_forward(x)
    elif ctx.activation == ACT_NONE:
        pass


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class InPlaceABN(autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        count = _count_samples(x)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)
        if ctx.training:
            mean, var = _backend.mean_var(x)
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * count / (count - 1))
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
        else:
            edz = dz.new_zeros(dz.size(1))
            eydz = dz.new_zeros(dz.size(1))
        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz * weight.sign() if ctx.affine else None
        dbias = edz if ctx.affine else None
        return dx, dweight, dbias, None, None, None, None, None, None, None


class InPlaceABNSync(autograd.Function):

    @classmethod
    def forward(cls, ctx, x, weight, bias, running_mean, running_var, training=True, momentum=0.1, eps=1e-05, activation=ACT_LEAKY_RELU, slope=0.01, equal_batches=True):
        ctx.training = training
        ctx.momentum = momentum
        ctx.eps = eps
        ctx.activation = activation
        ctx.slope = slope
        ctx.affine = weight is not None and bias is not None
        ctx.world_size = dist.get_world_size() if dist.is_initialized() else 1
        batch_size = x.new_tensor([x.shape[0]], dtype=torch.long)
        x = x.contiguous()
        weight = weight.contiguous() if ctx.affine else x.new_empty(0)
        bias = bias.contiguous() if ctx.affine else x.new_empty(0)
        if ctx.training:
            mean, var = _backend.mean_var(x)
            if ctx.world_size > 1:
                if equal_batches:
                    batch_size *= ctx.world_size
                else:
                    dist.all_reduce(batch_size, dist.ReduceOp.SUM)
                ctx.factor = x.shape[0] / float(batch_size.item())
                mean_all = mean.clone() * ctx.factor
                dist.all_reduce(mean_all, dist.ReduceOp.SUM)
                var_all = (var + (mean - mean_all) ** 2) * ctx.factor
                dist.all_reduce(var_all, dist.ReduceOp.SUM)
                mean = mean_all
                var = var_all
            running_mean.mul_(1 - ctx.momentum).add_(ctx.momentum * mean)
            count = batch_size.item() * x.view(x.shape[0], x.shape[1], -1).shape[-1]
            running_var.mul_(1 - ctx.momentum).add_(ctx.momentum * var * (float(count) / (count - 1)))
            ctx.mark_dirty(x, running_mean, running_var)
        else:
            mean, var = running_mean.contiguous(), running_var.contiguous()
            ctx.mark_dirty(x)
        _backend.forward(x, mean, var, weight, bias, ctx.affine, ctx.eps)
        _act_forward(ctx, x)
        ctx.var = var
        ctx.save_for_backward(x, var, weight, bias)
        return x

    @staticmethod
    @once_differentiable
    def backward(ctx, dz):
        z, var, weight, bias = ctx.saved_tensors
        dz = dz.contiguous()
        _act_backward(ctx, z, dz)
        if ctx.training:
            edz, eydz = _backend.edz_eydz(z, dz, weight, bias, ctx.affine, ctx.eps)
            edz_local = edz.clone()
            eydz_local = eydz.clone()
            if ctx.world_size > 1:
                edz *= ctx.factor
                dist.all_reduce(edz, dist.ReduceOp.SUM)
                eydz *= ctx.factor
                dist.all_reduce(eydz, dist.ReduceOp.SUM)
        else:
            edz_local = edz = dz.new_zeros(dz.size(1))
            eydz_local = eydz = dz.new_zeros(dz.size(1))
        dx = _backend.backward(z, dz, var, weight, bias, edz, eydz, ctx.affine, ctx.eps)
        dweight = eydz_local * weight.sign() if ctx.affine else None
        dbias = edz_local if ctx.affine else None
        return dx, dweight, dbias, None, None, None, None, None, None, None


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class SingleGPU(nn.Module):

    def __init__(self, module):
        super(SingleGPU, self).__init__()
        self.module = module

    def forward(self, input):
        return self.module(input)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ABN,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SingleGPU,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_CoinCheung_DeepLab_v3_plus_cityscapes(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

