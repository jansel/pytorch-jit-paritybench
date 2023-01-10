import sys
_module = sys.modules[__name__]
del sys
UPS_Custom_Dataset = _module
UPS_DiLiGenT_main = _module
UPS_Synth_Dataset = _module
datasets = _module
custom_data_loader = _module
pms_transforms = _module
util = _module
run_stage1 = _module
run_stage2 = _module
main_stage1 = _module
main_stage2 = _module
LCNet = _module
NENet = _module
models = _module
custom_model = _module
model_utils = _module
solver_utils = _module
options = _module
base_opts = _module
run_model_opts = _module
stage1_opts = _module
stage2_opts = _module
cropDiLiGenTData = _module
test_stage1 = _module
test_stage2 = _module
train_stage1 = _module
train_stage2 = _module
utils = _module
eval_utils = _module
logger = _module
recorders = _module
time_utils = _module

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


import scipy.io as sio


import torch


import torch.utils.data as data


import torch.utils.data


import random


import torch.nn as nn


from torch.nn.init import kaiming_normal_


import math


from matplotlib import cm


import time


import torchvision.utils as vutils


import matplotlib


import matplotlib.pyplot as plt


from matplotlib.font_manager import FontProperties


from collections import OrderedDict


class FeatExtractor(nn.Module):

    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class Classifier(nn.Module):

    def __init__(self, batchNorm, c_in, other):
        super(Classifier, self).__init__()
        self.conv1 = model_utils.conv(batchNorm, 512, 256, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 256, 256, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 256, 256, k=3, stride=2, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 256, 256, k=3, stride=2, pad=1)
        self.other = other
        self.dir_x_est = nn.Sequential(model_utils.conv(batchNorm, 256, 64, k=1, stride=1, pad=0), model_utils.outputConv(64, other['dirs_cls'], k=1, stride=1, pad=0))
        self.dir_y_est = nn.Sequential(model_utils.conv(batchNorm, 256, 64, k=1, stride=1, pad=0), model_utils.outputConv(64, other['dirs_cls'], k=1, stride=1, pad=0))
        self.int_est = nn.Sequential(model_utils.conv(batchNorm, 256, 64, k=1, stride=1, pad=0), model_utils.outputConv(64, other['ints_cls'], k=1, stride=1, pad=0))

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        outputs = {}
        if self.other['s1_est_d']:
            outputs['dir_x'] = self.dir_x_est(out)
            outputs['dir_y'] = self.dir_y_est(out)
        if self.other['s1_est_i']:
            outputs['ints'] = self.int_est(out)
        return outputs


class LCNet(nn.Module):

    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(LCNet, self).__init__()
        self.featExtractor = FeatExtractor(batchNorm, c_in, 128)
        self.classifier = Classifier(batchNorm, 256, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def prepareInputs(self, x):
        n, c, h, w = x[0].shape
        t_h, t_w = self.other['test_h'], self.other['test_w']
        if h == t_h and w == t_w:
            imgs = x[0]
        else:
            None
            imgs = torch.nn.functional.upsample(x[0], size=(t_h, t_w), mode='bilinear')
        inputs = list(torch.split(imgs, 3, 1))
        idx = 1
        if self.other['in_light']:
            light = torch.split(x[idx], 3, 1)
            for i in range(len(inputs)):
                inputs[i] = torch.cat([inputs[i], light[i]], 1)
            idx += 1
        if self.other['in_mask']:
            mask = x[idx]
            if mask.shape[2] != inputs[0].shape[2] or mask.shape[3] != inputs[0].shape[3]:
                mask = torch.nn.functional.upsample(mask, size=(t_h, t_w), mode='bilinear')
            for i in range(len(inputs)):
                inputs[i] = torch.cat([inputs[i], mask], 1)
            idx += 1
        return inputs

    def fuseFeatures(self, feats, fuse_type):
        if fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        return feat_fused

    def convertMidDirs(self, pred):
        _, x_idx = pred['dirs_x'].data.max(1)
        _, y_idx = pred['dirs_y'].data.max(1)
        dirs = eval_utils.SphericalClassToDirs(x_idx, y_idx, self.other['dirs_cls'])
        return dirs

    def convertMidIntens(self, pred, img_num):
        _, idx = pred['ints'].data.max(1)
        ints = eval_utils.ClassToLightInts(idx, self.other['ints_cls'])
        ints = ints.view(-1, 1).repeat(1, 3)
        ints = torch.cat(torch.split(ints, ints.shape[0] // img_num, 0), 1)
        return ints

    def forward(self, x):
        inputs = self.prepareInputs(x)
        feats = []
        for i in range(len(inputs)):
            out_feat = self.featExtractor(inputs[i])
            shape = out_feat.data.shape
            feats.append(out_feat)
        feat_fused = self.fuseFeatures(feats, self.fuse_type)
        l_dirs_x, l_dirs_y, l_ints = [], [], []
        for i in range(len(inputs)):
            net_input = torch.cat([feats[i], feat_fused], 1)
            outputs = self.classifier(net_input)
            if self.other['s1_est_d']:
                l_dirs_x.append(outputs['dir_x'])
                l_dirs_y.append(outputs['dir_y'])
            if self.other['s1_est_i']:
                l_ints.append(outputs['ints'])
        pred = {}
        if self.other['s1_est_d']:
            pred['dirs_x'] = torch.cat(l_dirs_x, 0).squeeze()
            pred['dirs_y'] = torch.cat(l_dirs_y, 0).squeeze()
            pred['dirs'] = self.convertMidDirs(pred)
        if self.other['s1_est_i']:
            pred['ints'] = torch.cat(l_ints, 0).squeeze()
            if pred['ints'].ndimension() == 1:
                pred['ints'] = pred['ints'].view(1, -1)
            pred['intens'] = self.convertMidIntens(pred, len(inputs))
        return pred


class Regressor(nn.Module):

    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class NENet(nn.Module):

    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(NENet, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def prepareInputs(self, x):
        imgs = torch.split(x[0], 3, 1)
        idx = 1
        if self.other['in_light']:
            idx += 1
        if self.other['in_mask']:
            idx += 1
        dirs = torch.split(x[idx]['dirs'], x[0].shape[0], 0)
        ints = torch.split(x[idx]['intens'], 3, 1)
        s2_inputs = []
        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            l_dir = dirs[i] if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1)
            l_int = torch.diag(1.0 / (ints[i].contiguous().view(-1) + 1e-08))
            img = imgs[i].contiguous().view(n * c, h * w)
            img = torch.mm(l_int, img).view(n, c, h, w)
            img_light = torch.cat([img, l_dir.expand_as(img)], 1)
            s2_inputs.append(img_light)
        return s2_inputs

    def forward(self, x):
        inputs = self.prepareInputs(x)
        feats = torch.Tensor()
        for i in range(len(inputs)):
            feat, shape = self.extractor(inputs[i])
            if i == 0:
                feats = feat
            elif self.fuse_type == 'mean':
                feats = torch.stack([feats, feat], 1).sum(1)
            elif self.fuse_type == 'max':
                feats, _ = torch.stack([feats, feat], 1).max(1)
        if self.fuse_type == 'mean':
            feats = feats / len(img_split)
        feat_fused = feats
        normal = self.regressor(feat_fused, shape)
        pred = {}
        pred['n'] = normal
        return pred

