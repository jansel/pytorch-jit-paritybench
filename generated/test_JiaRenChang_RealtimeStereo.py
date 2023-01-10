import sys
_module = sys.modules[__name__]
del sys
Test_img = _module
KITTILoader = _module
KITTI_submission_loader = _module
KITTI_submission_loader2012 = _module
KITTIloader2012 = _module
KITTIloader2015 = _module
SecenFlowLoader = _module
dataloader = _module
listflowfile = _module
preprocess = _module
readpfm = _module
finetune = _module
main = _module
RT_stereo = _module
models = _module
basic = _module
stackhourglass = _module
submodule = _module
submission = _module

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


import torch.nn as nn


import torchvision.transforms as transforms


import torch.nn.functional as F


import numpy as np


import time


import math


import torch.utils.data as data


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


import copy


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False), nn.BatchNorm2d(out_planes))


norm_layer2d = nn.BatchNorm2d


class featexchange(nn.Module):

    def __init__(self):
        super(featexchange, self).__init__()
        self.x2_fusion = nn.Sequential(nn.ReLU(), convbn(4, 4, 3, 1, 1, 1, 1), nn.ReLU(), nn.Conv2d(4, 4, 3, 1, 1, bias=False))
        self.upconv4 = nn.Sequential(nn.Conv2d(8, 4, 1, 1, 0, bias=False), norm_layer2d(4))
        self.upconv8 = nn.Sequential(nn.Conv2d(20, 4, 1, 1, 0, bias=False), norm_layer2d(4))
        self.x4_fusion = nn.Sequential(nn.ReLU(), convbn(8, 8, 3, 1, 1, 1, 1), nn.ReLU(), nn.Conv2d(8, 8, 3, 1, 1, bias=False))
        self.downconv4 = nn.Sequential(nn.Conv2d(4, 8, 3, 2, 1, bias=False), norm_layer2d(8))
        self.upconv8_2 = nn.Sequential(nn.Conv2d(20, 8, 1, 1, 0, bias=False), norm_layer2d(8))
        self.x8_fusion = nn.Sequential(nn.ReLU(), convbn(20, 20, 3, 1, 1, 1, 1), nn.ReLU(), nn.Conv2d(20, 20, 3, 1, 1, bias=False))
        self.downconv81 = nn.Sequential(nn.Conv2d(8, 20, 3, 2, 1, bias=False), norm_layer2d(20))
        self.downconv82 = nn.Sequential(nn.Conv2d(8, 20, 3, 2, 1, bias=False), norm_layer2d(20))

    def forward(self, x2, x4, x8, attention):
        A = torch.split(attention, [4, 8, 20], dim=1)
        x4tox2 = self.upconv4(F.upsample(x4, (x2.size()[2], x2.size()[3])))
        x8tox2 = self.upconv8(F.upsample(x8, (x2.size()[2], x2.size()[3])))
        fusx2 = x2 + x4tox2 + x8tox2
        fusx2 = self.x2_fusion(fusx2) * A[0].contiguous() + fusx2
        x2tox4 = self.downconv4(x2)
        x8tox4 = self.upconv8_2(F.upsample(x8, (x4.size()[2], x4.size()[3])))
        fusx4 = x4 + x2tox4 + x8tox4
        fusx4 = self.x4_fusion(fusx4) * A[1].contiguous() + fusx4
        x2tox8 = self.downconv81(x2tox4)
        x4tox8 = self.downconv82(x4)
        fusx8 = x8 + x2tox8 + x4tox8
        fusx8 = self.x8_fusion(fusx8) * A[2].contiguous() + fusx8
        return fusx2, fusx4, fusx8


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation), nn.ReLU(inplace=True))
        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class feature_extraction(nn.Module):

    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True), convbn(32, 32, 3, 1, 1, 1), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)), convbn(128, 32, 1, 1, 0, 1), nn.ReLU(inplace=True))
        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)
        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')
        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        return output_feature


class disparityregression2(nn.Module):

    def __init__(self, start, end, stride=1):
        super(disparityregression2, self).__init__()
        self.disp = torch.arange(start * stride, end * stride, stride).view(1, -1, 1, 1).type(torch.FloatTensor).requires_grad_(False)

    def forward(self, x):
        out = torch.sum(x * self.disp, 1, keepdim=True)
        return out


norm_layer3d = nn.BatchNorm3d


def batch_relu_conv3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1, bn3d=True):
    if bn3d:
        return nn.Sequential(norm_layer3d(in_planes), nn.ReLU(), nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))
    else:
        return nn.Sequential(nn.ReLU(), nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))


def post_3dconvs(layers, channels):
    net = [nn.Conv3d(1, channels, kernel_size=3, padding=1, stride=1, bias=False)]
    net += [batch_relu_conv3d(channels, channels) for _ in range(layers)]
    net += [batch_relu_conv3d(channels, 1)]
    return nn.Sequential(*net)


class RTStereoNet(nn.Module):

    def __init__(self, maxdisp):
        super(RTStereoNet, self).__init__()
        self.feature_extraction = feature_extraction()
        self.maxdisp = maxdisp
        self.volume_postprocess = []
        layer_setting = [8, 4, 4]
        for i in range(3):
            net3d = post_3dconvs(3, layer_setting[i])
            self.volume_postprocess.append(net3d)
        self.volume_postprocess = nn.ModuleList(self.volume_postprocess)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            vgrid = grid
        vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        return output

    def _build_volume_2d(self, feat_l, feat_r, maxdisp, stride=1):
        assert maxdisp % stride == 0
        b, c, h, w = feat_l.size()
        cost = torch.zeros(b, 1, maxdisp // stride, h, w).requires_grad_(False)
        for i in range(0, maxdisp, stride):
            if i > 0:
                cost[:, :, i // stride, :, i:] = torch.norm(feat_l[:, :, :, i:] - feat_r[:, :, :, :-i], p=1, dim=1, keepdim=True)
            else:
                cost[:, :, i // stride, :, i:] = torch.norm(feat_l[:, :, :, :] - feat_r[:, :, :, :], p=1, dim=1, keepdim=True)
        return cost.contiguous()

    def _build_volume_2d3(self, feat_l, feat_r, maxdisp, disp, stride=1):
        b, c, h, w = feat_l.size()
        batch_disp = disp[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, 1, h, w)
        temp_array = np.tile(np.array(range(-maxdisp + 1, maxdisp)), b) * stride
        batch_shift = torch.Tensor(np.reshape(temp_array, [len(temp_array), 1, 1, 1])).requires_grad_(False)
        batch_disp = batch_disp - batch_shift
        batch_feat_l = feat_l[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, c, h, w)
        batch_feat_r = feat_r[:, None, :, :, :].repeat(1, maxdisp * 2 - 1, 1, 1, 1).view(-1, c, h, w)
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1, keepdim=True)
        cost = cost.view(b, 1, -1, h, w).contiguous()
        return cost

    def forward(self, left, right):
        img_size = left.size()
        feats_l = self.feature_extraction(left)
        feats_r = self.feature_extraction(right)
        pred = []
        for scale in range(len(feats_l)):
            if scale > 0:
                wflow = F.upsample(pred[scale - 1], (feats_l[scale].size(2), feats_l[scale].size(3)), mode='bilinear') * feats_l[scale].size(2) / img_size[2]
                cost = self._build_volume_2d3(feats_l[scale], feats_r[scale], 3, wflow, stride=1)
            else:
                cost = self._build_volume_2d(feats_l[scale], feats_r[scale], 12, stride=1)
            cost = self.volume_postprocess[scale](cost)
            cost = cost.squeeze(1)
            if scale == 0:
                pred_low_res = disparityregression2(0, 12)(F.softmax(cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up)
            else:
                pred_low_res = disparityregression2(-2, 3, stride=1)(F.softmax(cost, dim=1))
                pred_low_res = pred_low_res * img_size[2] / pred_low_res.size(2)
                disp_up = F.upsample(pred_low_res, (img_size[2], img_size[3]), mode='bilinear')
                pred.append(disp_up + pred[scale - 1])
        if self.training:
            return pred[0], pred[1], pred[2]
        else:
            return pred[-1]


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False), nn.BatchNorm3d(out_planes))


class disparityregression(nn.Module):

    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1]))

    def forward(self, x):
        out = torch.sum(x * self.disp.data, 2, keepdim=True)
        return out


class hourglass(nn.Module):

    def __init__(self, inplanes):
        super(hourglass, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)
        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(inplanes * 2))
        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False), nn.BatchNorm3d(inplanes))

    def forward(self, x, presqu, postsqu):
        out = self.conv1(x)
        pre = self.conv2(out)
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)
        out = self.conv3(pre)
        out = self.conv4(out)
        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)
        else:
            post = F.relu(self.conv5(out) + pre, inplace=True)
        out = self.conv6(post)
        return out, pre, post


class PSMNet(nn.Module):

    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True))
        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), convbn_3d(32, 32, 3, 1, 1))
        self.dres2 = hourglass(32)
        self.dres3 = hourglass(32)
        self.dres4 = hourglass(32)
        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)
        cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp // 4, refimg_fea.size()[2], refimg_fea.size()[3]).zero_())
        for i in range(self.maxdisp // 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()
        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        out1, pre1, post1 = self.dres2(cost0, None, None)
        out1 = out1 + cost0
        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost0
        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost0
        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2
        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)
        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparityregression(self.maxdisp)(pred3)
        if self.training:
            return pred1, pred2, pred3
        else:
            return pred3


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (PSMNet,
     lambda: ([], {'maxdisp': 4}),
     lambda: ([torch.rand([4, 3, 256, 256]), torch.rand([4, 3, 256, 256])], {}),
     False),
    (disparityregression,
     lambda: ([], {'maxdisp': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (disparityregression2,
     lambda: ([], {'start': 4, 'end': 4}),
     lambda: ([torch.rand([4, 0, 4, 4])], {}),
     True),
    (feature_extraction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 256, 256])], {}),
     True),
]

class Test_JiaRenChang_RealtimeStereo(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

