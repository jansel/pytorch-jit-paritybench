import sys
_module = sys.modules[__name__]
del sys
Config = _module
Test = _module
Train = _module
augmentations = _module
detection = _module
l2norm = _module
loss_function = _module
model_file_test = _module
ssd_net_vgg = _module
utils = _module
voc0712 = _module

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


from torch.autograd import Variable


import torch.nn as nn


import numpy as np


import torch.utils.data as data


import torch.optim as optim


from torchvision import transforms


import types


from numpy import random


from torch.autograd import Function


import torch.nn.init as init


import torch.nn.functional as F


from itertools import product as product


from math import sqrt as sqrt


class L2Norm(nn.Module):

    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        if Config.use_cuda:
            self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        else:
            self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class LossFun(nn.Module):

    def __init__(self):
        super(LossFun, self).__init__()

    def forward(self, prediction, targets, priors_boxes):
        loc_data, conf_data = prediction
        loc_data = torch.cat([o.view(o.size(0), -1, 4) for o in loc_data], 1)
        conf_data = torch.cat([o.view(o.size(0), -1, Config.class_num) for o in conf_data], 1)
        priors_boxes = torch.cat([o.view(-1, 4) for o in priors_boxes], 0)
        if Config.use_cuda:
            loc_data = loc_data
            conf_data = conf_data
            priors_boxes = priors_boxes
        batch_num = loc_data.size(0)
        box_num = loc_data.size(1)
        target_loc = torch.Tensor(batch_num, box_num, 4)
        target_loc.requires_grad_(requires_grad=False)
        target_conf = torch.LongTensor(batch_num, box_num)
        target_conf.requires_grad_(requires_grad=False)
        if Config.use_cuda:
            target_loc = target_loc
            target_conf = target_conf
        for batch_id in range(batch_num):
            target_truths = targets[batch_id][:, :-1].data
            target_labels = targets[batch_id][:, -1].data
            if Config.use_cuda:
                target_truths = target_truths
                target_labels = target_labels
            utils.match(0.5, target_truths, priors_boxes, target_labels, target_loc, target_conf, batch_id)
        pos = target_conf > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        pre_loc_xij = loc_data[pos_idx].view(-1, 4)
        tar_loc_xij = target_loc[pos_idx].view(-1, 4)
        loss_loc = F.smooth_l1_loss(pre_loc_xij, tar_loc_xij, size_average=False)
        batch_conf = conf_data.view(-1, Config.class_num)
        loss_c = utils.log_sum_exp(batch_conf) - batch_conf.gather(1, target_conf.view(-1, 1))
        loss_c = loss_c.view(batch_num, -1)
        loss_c[pos] = 0
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(3 * num_pos, max=pos.size(1) - 1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, Config.class_num)
        targets_weighted = target_conf[(pos + neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        N = num_pos.data.sum().double()
        loss_l = loss_loc.double()
        loss_c = loss_c.double()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c


class SSD(nn.Module):

    def __init__(self):
        super(SSD, self).__init__()
        self.vgg = []
        self.vgg.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vgg.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vgg.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
        self.vgg.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        self.vgg.append(nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg.append(nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1))
        self.vgg.append(nn.ReLU(inplace=True))
        self.vgg = nn.ModuleList(self.vgg)
        self.conv8_1 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1), nn.ReLU(inplace=True))
        self.conv8_2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv9_1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1), nn.ReLU(inplace=True))
        self.conv9_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv10_1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1), nn.ReLU(inplace=True))
        self.conv10_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1), nn.ReLU(inplace=True))
        self.conv11_1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1), nn.ReLU(inplace=True))
        self.conv11_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1), nn.ReLU(inplace=True))
        self.feature_map_loc_1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, stride=1, padding=1))
        self.feature_map_loc_2 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, stride=1, padding=1))
        self.feature_map_loc_3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, stride=1, padding=1))
        self.feature_map_loc_4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, stride=1, padding=1))
        self.feature_map_loc_5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1))
        self.feature_map_loc_6 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1))
        self.feature_map_conf_1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=4 * config.class_num, kernel_size=3, stride=1, padding=1))
        self.feature_map_conf_2 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=6 * config.class_num, kernel_size=3, stride=1, padding=1))
        self.feature_map_conf_3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=6 * config.class_num, kernel_size=3, stride=1, padding=1))
        self.feature_map_conf_4 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=6 * config.class_num, kernel_size=3, stride=1, padding=1))
        self.feature_map_conf_5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=4 * config.class_num, kernel_size=3, stride=1, padding=1))
        self.feature_map_conf_6 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=4 * config.class_num, kernel_size=3, stride=1, padding=1))

    def forward(self, image):
        out = self.vgg[0](image)
        out = self.vgg[1](out)
        out = self.vgg[2](out)
        out = self.vgg[3](out)
        out = self.vgg[4](out)
        out = self.vgg[5](out)
        out = self.vgg[6](out)
        out = self.vgg[7](out)
        out = self.vgg[8](out)
        out = self.vgg[9](out)
        out = self.vgg[10](out)
        out = self.vgg[11](out)
        out = self.vgg[12](out)
        out = self.vgg[13](out)
        out = self.vgg[14](out)
        out = self.vgg[15](out)
        out = self.vgg[16](out)
        out = self.vgg[17](out)
        out = self.vgg[18](out)
        out = self.vgg[19](out)
        out = self.vgg[20](out)
        out = self.vgg[21](out)
        out = self.vgg[22](out)
        my_L2Norm = l2norm.L2Norm(512, 20)
        feature_map_1 = out
        feature_map_1 = my_L2Norm(feature_map_1)
        loc_1 = self.feature_map_loc_1(feature_map_1).permute((0, 2, 3, 1)).contiguous()
        conf_1 = self.feature_map_conf_1(feature_map_1).permute((0, 2, 3, 1)).contiguous()
        out = self.vgg[23](out)
        out = self.vgg[24](out)
        out = self.vgg[25](out)
        out = self.vgg[26](out)
        out = self.vgg[27](out)
        out = self.vgg[28](out)
        out = self.vgg[29](out)
        out = self.vgg[30](out)
        out = self.vgg[31](out)
        out = self.vgg[32](out)
        out = self.vgg[33](out)
        out = self.vgg[34](out)
        feature_map_2 = out
        loc_2 = self.feature_map_loc_2(feature_map_2).permute((0, 2, 3, 1)).contiguous()
        conf_2 = self.feature_map_conf_2(feature_map_2).permute((0, 2, 3, 1)).contiguous()
        out = self.conv8_1(out)
        out = self.conv8_2(out)
        feature_map_3 = out
        loc_3 = self.feature_map_loc_3(feature_map_3).permute((0, 2, 3, 1)).contiguous()
        conf_3 = self.feature_map_conf_3(feature_map_3).permute((0, 2, 3, 1)).contiguous()
        out = self.conv9_1(out)
        out = self.conv9_2(out)
        feature_map_4 = out
        loc_4 = self.feature_map_loc_4(feature_map_4).permute((0, 2, 3, 1)).contiguous()
        conf_4 = self.feature_map_conf_4(feature_map_4).permute((0, 2, 3, 1)).contiguous()
        out = self.conv10_1(out)
        out = self.conv10_2(out)
        feature_map_5 = out
        loc_5 = self.feature_map_loc_5(feature_map_5).permute((0, 2, 3, 1)).contiguous()
        conf_5 = self.feature_map_conf_5(feature_map_5).permute((0, 2, 3, 1)).contiguous()
        out = self.conv11_1(out)
        out = self.conv11_2(out)
        feature_map_6 = out
        loc_6 = self.feature_map_loc_6(feature_map_6).permute((0, 2, 3, 1)).contiguous()
        conf_6 = self.feature_map_conf_6(feature_map_6).permute((0, 2, 3, 1)).contiguous()
        loc_list = [loc_1, loc_2, loc_3, loc_4, loc_5, loc_6]
        conf_list = [conf_1, conf_2, conf_3, conf_4, conf_5, conf_6]
        return loc_list, conf_list

