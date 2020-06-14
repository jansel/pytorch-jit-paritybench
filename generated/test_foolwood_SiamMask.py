import sys
_module = sys.modules[__name__]
del sys
gen_json = _module
par_crop = _module
pycocotools = _module
coco = _module
cocoeval = _module
mask = _module
setup = _module
visual = _module
create_json = _module
parse_vid = _module
download_from_gdrive = _module
parse_ytb_vos = _module
datasets = _module
siam_mask_dataset = _module
siam_rpn_dataset = _module
custom = _module
resnet = _module
custom = _module
resnet = _module
custom = _module
resnet = _module
models = _module
features = _module
mask = _module
rpn = _module
siammask = _module
siammask_sharp = _module
siamrpn = _module
tools = _module
demo = _module
eval = _module
test = _module
train_siammask = _module
train_siammask_refine = _module
train_siamrpn = _module
tune_vos = _module
tune_vot = _module
utils = _module
anchors = _module
average_meter_helper = _module
bbox_helper = _module
benchmark_helper = _module
config_helper = _module
load_helper = _module
log_helper = _module
lr_helper = _module
pysot = _module
dataset = _module
video = _module
vot = _module
evaluation = _module
ar_benchmark = _module
eao_benchmark = _module
misc = _module
statistics = _module
pyvotkit = _module
tracker_config = _module

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


from torch.autograd import Variable


import math


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


import logging


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.collect_env import get_pretty_env_info


from torch.optim.lr_scheduler import _LRScheduler


class ResDownS(nn.Module):

    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(inplane, outplane,
            kernel_size=1, bias=False), nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


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


class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=0, bias=False)
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
        s = residual.size(3)
        residual = residual[:, :, 1:s - 1, 1:s - 1]
        out += residual
        out = self.relu(out)
        return out


class ResDownS(nn.Module):

    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(inplane, outplane,
            kernel_size=1, bias=False), nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


class Refine(nn.Module):

    def __init__(self):
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 4, 3, padding=1), nn.ReLU())
        self.v1 = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
        self.v2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(
            ), nn.Conv2d(128, 32, 3, padding=1), nn.ReLU())
        self.h2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.h1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.h0 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())
        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)
        self.post0 = nn.Conv2d(32, 16, 3, padding=1)
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)
        for modules in [self.v0, self.v1, self.v2, self.h2, self.h1, self.
            h0, self.deconv, self.post0, self.post1, self.post2]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.kaiming_uniform_(l.weight, a=1)

    def forward(self, f, corr_feature, pos=None, test=False):
        if test:
            p0 = torch.nn.functional.pad(f[0], [16, 16, 16, 16])[:, :, 4 *
                pos[0]:4 * pos[0] + 61, 4 * pos[1]:4 * pos[1] + 61]
            p1 = torch.nn.functional.pad(f[1], [8, 8, 8, 8])[:, :, 2 * pos[
                0]:2 * pos[0] + 31, 2 * pos[1]:2 * pos[1] + 31]
            p2 = torch.nn.functional.pad(f[2], [4, 4, 4, 4])[:, :, pos[0]:
                pos[0] + 15, pos[1]:pos[1] + 15]
        else:
            p0 = F.unfold(f[0], (61, 61), padding=0, stride=4).permute(0, 2, 1
                ).contiguous().view(-1, 64, 61, 61)
            if not pos is None:
                p0 = torch.index_select(p0, 0, pos)
            p1 = F.unfold(f[1], (31, 31), padding=0, stride=2).permute(0, 2, 1
                ).contiguous().view(-1, 256, 31, 31)
            if not pos is None:
                p1 = torch.index_select(p1, 0, pos)
            p2 = F.unfold(f[2], (15, 15), padding=0, stride=1).permute(0, 2, 1
                ).contiguous().view(-1, 512, 15, 15)
            if not pos is None:
                p2 = torch.index_select(p2, 0, pos)
        if not pos is None:
            p3 = corr_feature[:, :, (pos[0]), (pos[1])].view(-1, 256, 1, 1)
        else:
            p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256,
                1, 1)
        out = self.deconv(p3)
        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))
        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))
        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 
            127)))
        out = out.view(-1, 127 * 127)
        return out

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


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


class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=0, bias=False)
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
        s = residual.size(3)
        residual = residual[:, :, 1:s - 1, 1:s - 1]
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.feature_size = 128 * block.expansion
        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2)
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x
        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4)
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                    block.expansion, kernel_size=1, stride=stride, bias=
                    False), nn.BatchNorm2d(planes * block.expansion))
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                    block.expansion, kernel_size=3, stride=stride, bias=
                    False, padding=padding, dilation=dd), nn.BatchNorm2d(
                    planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            dilation=dd))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        p0 = self.relu(x)
        x = self.maxpool(p0)
        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        return p0, p1, p2, p3


class ResDownS(nn.Module):

    def __init__(self, inplane, outplane):
        super(ResDownS, self).__init__()
        self.downsample = nn.Sequential(nn.Conv2d(inplane, outplane,
            kernel_size=1, bias=False), nn.BatchNorm2d(outplane))

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = -4
            x = x[:, :, l:r, l:r]
        return x


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


class Bottleneck_nop(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_nop, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=0, bias=False)
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
        s = residual.size(3)
        residual = residual[:, :, 1:s - 1, 1:s - 1]
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, layer4=False, layer3=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.feature_size = 128 * block.expansion
        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                dilation=2)
            self.feature_size = (256 + 128) * block.expansion
        else:
            self.layer3 = lambda x: x
        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                dilation=4)
            self.feature_size = 512 * block.expansion
        else:
            self.layer4 = lambda x: x
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        dd = dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                    block.expansion, kernel_size=1, stride=stride, bias=
                    False), nn.BatchNorm2d(planes * block.expansion))
            else:
                if dilation > 1:
                    dd = dilation // 2
                    padding = dd
                else:
                    dd = 1
                    padding = 0
                downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                    block.expansion, kernel_size=3, stride=stride, bias=
                    False, padding=padding, dilation=dd), nn.BatchNorm2d(
                    planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
            dilation=dd))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        p1 = self.layer1(x)
        p2 = self.layer2(p1)
        p3 = self.layer3(p2)
        log_once('p3 {}'.format(p3.size()))
        p4 = self.layer4(p3)
        return p2, p3, p4


class Features(nn.Module):

    def __init__(self):
        super(Features, self).__init__()
        self.feature_size = -1

    def forward(self, x):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params

    def load_model(self, f='pretrain.model'):
        with open(f) as f:
            pretrained_dict = torch.load(f)
            model_dict = self.state_dict()
            None
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 
                k in model_dict}
            None
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


class Mask(nn.Module):

    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1):
        params = filter(lambda x: x.requires_grad, self.parameters())
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

    def template(self, template):
        raise NotImplementedError

    def track(self, search):
        raise NotImplementedError

    def param_groups(self, start_lr, feature_mult=1, key=None):
        if key is None:
            params = filter(lambda x: x.requires_grad, self.parameters())
        else:
            params = [v for k, v in self.named_parameters() if key in k and
                v.requires_grad]
        params = [{'params': params, 'lr': start_lr * feature_mult}]
        return params


def conv2d_dw_group(x, kernel):
    batch, channel = kernel.shape[:2]
    x = x.view(1, batch * channel, x.size(2), x.size(3))
    kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


class DepthCorr(nn.Module):

    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        self.conv_kernel = nn.Sequential(nn.Conv2d(in_channels, hidden,
            kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True))
        self.conv_search = nn.Sequential(nn.Conv2d(in_channels, hidden,
            kernel_size=kernel_size, bias=False), nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(hidden, hidden, kernel_size=1,
            bias=False), nn.BatchNorm2d(hidden), nn.ReLU(inplace=True), nn.
            Conv2d(hidden, out_channels, kernel_size=1))

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        out = self.head(feature)
        return out


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0:
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
    neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def iou_measure(pred, label):
    pred = pred.ge(0)
    mask_sum = pred.eq(1).add(label.eq(1))
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn / union
    return torch.mean(iou), torch.sum(iou > 0.5).float() / iou.shape[0
        ], torch.sum(iou > 0.7).float() / iou.shape[0]


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127):
    weight = weight.view(-1)
    pos = Variable(weight.data.eq(1).nonzero().squeeze())
    if pos.nelement() == 0:
        return p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0, p_m.sum() * 0
    if len(p_m.shape) == 4:
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m = torch.index_select(p_m, 0, pos)
        p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)
        p_m = p_m.view(-1, g_sz * g_sz)
    else:
        p_m = torch.index_select(p_m, 0, pos)
    mask_uf = F.unfold(mask, (g_sz, g_sz), padding=0, stride=8)
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)
    mask_uf = torch.index_select(mask_uf, 0, pos)
    loss = F.soft_margin_loss(p_m, mask_uf)
    iou_m, iou_5, iou_7 = iou_measure(p_m, mask_uf)
    return loss, iou_m, iou_5, iou_7


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    """
    :param pred_loc: [b, 4k, h, w]
    :param label_loc: [b, 4k, h, w]
    :param loss_weight:  [b, k, h, w]
    :return: loc loss value
    """
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


class SiamMask(nn.Module):

    def __init__(self, anchors=None, o_sz=127, g_sz=127):
        super(SiamMask, self).__init__()
        self.anchors = anchors
        self.anchor_num = len(self.anchors['ratios']) * len(self.anchors[
            'scales'])
        self.anchor = Anchors(anchors)
        self.features = None
        self.rpn_model = None
        self.mask_model = None
        self.o_sz = o_sz
        self.g_sz = g_sz
        self.upSample = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])
        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]
        self.all_anchors = torch.from_numpy(all_anchors).float()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def mask(self, template, search):
        pred_mask = self.mask_model(template, search)
        return pred_mask

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight,
        label_mask, label_mask_weight, rpn_pred_cls, rpn_pred_loc,
        rpn_pred_mask):
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)
        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight
            )
        rpn_loss_mask, iou_m, iou_5, iou_7 = select_mask_logistic_loss(
            rpn_pred_mask, label_mask, label_mask_weight)
        return rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_m, iou_5, iou_7

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        feature, search_feature = self.features.forward_all(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        corr_feature = self.mask_model.mask.forward_corr(template_feature,
            search_feature)
        rpn_pred_mask = self.refine_model(feature, corr_feature)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return (rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature,
            search_feature)

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
            label_mask = input['label_mask']
            label_mask_weight = input['label_mask_weight']
        (rpn_pred_cls, rpn_pred_loc, rpn_pred_mask, template_feature,
            search_feature) = self.run(template, search, softmax=self.training)
        outputs = dict()
        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, rpn_pred_mask,
            template_feature, search_feature]
        if self.training:
            (rpn_loss_cls, rpn_loss_loc, rpn_loss_mask, iou_acc_mean,
                iou_acc_5, iou_acc_7) = (self._add_rpn_loss(label_cls,
                label_loc, lable_loc_weight, label_mask, label_mask_weight,
                rpn_pred_cls, rpn_pred_loc, rpn_pred_mask))
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc, rpn_loss_mask]
            outputs['accuracy'] = [iou_acc_mean, iou_acc_5, iou_acc_7]
        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel,
            loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc


class SiamRPN(nn.Module):

    def __init__(self, anchors=None):
        super(SiamRPN, self).__init__()
        self.anchors = anchors
        self.anchor = Anchors(anchors)
        self.anchor_num = self.anchor.anchor_num
        self.features = None
        self.rpn_model = None
        self.all_anchors = None

    def set_all_anchors(self, image_center, size):
        if not self.anchor.generate_all_anchors(image_center, size):
            return
        all_anchors = self.anchor.all_anchors[1]
        self.all_anchors = torch.from_numpy(all_anchors).float()
        self.all_anchors = [self.all_anchors[i] for i in range(4)]

    def feature_extractor(self, x):
        return self.features(x)

    def rpn(self, template, search):
        pred_cls, pred_loc = self.rpn_model(template, search)
        return pred_cls, pred_loc

    def _add_rpn_loss(self, label_cls, label_loc, lable_loc_weight,
        rpn_pred_cls, rpn_pred_loc):
        """
        :param compute_anchor_targets_fn: functions to produce anchors' learning targets.
        :param rpn_pred_cls: [B, num_anchors * 2, h, w], output of rpn for classification.
        :param rpn_pred_loc: [B, num_anchors * 4, h, w], output of rpn for localization.
        :return: loss of classification and localization, respectively.
        """
        rpn_loss_cls = select_cross_entropy_loss(rpn_pred_cls, label_cls)
        rpn_loss_loc = weight_l1_loss(rpn_pred_loc, label_loc, lable_loc_weight
            )
        acc = torch.zeros(1)
        return rpn_loss_cls, rpn_loss_loc, acc

    def run(self, template, search, softmax=False):
        """
        run network
        """
        template_feature = self.feature_extractor(template)
        search_feature = self.feature_extractor(search)
        rpn_pred_cls, rpn_pred_loc = self.rpn(template_feature, search_feature)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc, template_feature, search_feature

    def softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, input):
        """
        :param input: dict of input with keys of:
                'template': [b, 3, h1, w1], input template image.
                'search': [b, 3, h2, w2], input search image.
                'label_cls':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
        :return: dict of loss, predict, accuracy
        """
        template = input['template']
        search = input['search']
        if self.training:
            label_cls = input['label_cls']
            label_loc = input['label_loc']
            lable_loc_weight = input['label_loc_weight']
        rpn_pred_cls, rpn_pred_loc, template_feature, search_feature = (self
            .run(template, search, softmax=self.training))
        outputs = dict(predict=[], losses=[], accuracy=[])
        outputs['predict'] = [rpn_pred_loc, rpn_pred_cls, template_feature,
            search_feature]
        if self.training:
            rpn_loss_cls, rpn_loss_loc, rpn_acc = self._add_rpn_loss(label_cls,
                label_loc, lable_loc_weight, rpn_pred_cls, rpn_pred_loc)
            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc]
        return outputs

    def template(self, z):
        self.zf = self.feature_extractor(z)
        cls_kernel, loc_kernel = self.rpn_model.template(self.zf)
        return cls_kernel, loc_kernel

    def track(self, x, cls_kernel=None, loc_kernel=None, softmax=False):
        xf = self.feature_extractor(x)
        rpn_pred_cls, rpn_pred_loc = self.rpn_model.track(xf, cls_kernel,
            loc_kernel)
        if softmax:
            rpn_pred_cls = self.softmax(rpn_pred_cls)
        return rpn_pred_cls, rpn_pred_loc


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_foolwood_SiamMask(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(DepthCorr(*[], **{'in_channels': 4, 'hidden': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(ResDownS(*[], **{'inplane': 4, 'outplane': 4}), [torch.rand([4, 4, 4, 4])], {})

