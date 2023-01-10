import sys
_module = sys.modules[__name__]
del sys
faster_rcnn = _module
model = _module
train = _module
utils = _module

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


import torchvision


import numpy as np


import torch.nn as nn


class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.features = self._make_layers(cfg)
        self._rpn_model()
        size = 7, 7
        self.adaptive_max_pool = torch.nn.AdaptiveMaxPool2d(size[0], size[1])
        self.roi_classifier()

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _rpn_model(self, mid_channels=512, in_channels=512, n_anchor=9):
        self.rpn_conv = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)
        self.cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)
        self.rpn_conv.weight.data.normal_(0, 0.01)
        self.rpn_conv.bias.data.zero_()
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()

    def forward(self, data):
        out_map = self.features(data)
        x = self.rpn_conv(out_map)
        pred_anchor_locs = self.reg_layer(x)
        pred_cls_scores = self.cls_layer(x)
        return out_map, pred_anchor_locs, pred_cls_scores

    def roi_classifier(self, class_num=20):
        self.roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU()])
        self.cls_loc = nn.Linear(4096, (class_num + 1) * 4)
        self.cls_loc.weight.data.normal_(0, 0.01)
        self.cls_loc.bias.data.zero_()
        self.score = nn.Linear(4096, class_num + 1)

    def rpn_loss(self, rpn_loc, rpn_score, gt_rpn_loc, gt_rpn_label, weight=10.0):
        gt_rpn_label = torch.autograd.Variable(gt_rpn_label.long())
        rpn_cls_loss = torch.nn.functional.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        pos = gt_rpn_label.data > 0
        mask = pos.unsqueeze(1).expand_as(rpn_loc)
        mask_pred_loc = rpn_loc[mask].view(-1, 4)
        mask_target_loc = gt_rpn_loc[mask].view(-1, 4)
        x = np.abs(mask_target_loc.numpy() - mask_pred_loc.data.numpy())
        rpn_loc_loss = (x < 1) * 0.5 * x ** 2 + (x >= 1) * (x - 0.5)
        rpn_loc_loss = rpn_loc_loss.sum()
        N_reg = (gt_rpn_label > 0).float().sum()
        N_reg = np.squeeze(N_reg.data.numpy())
        rpn_loc_loss = rpn_loc_loss / N_reg
        rpn_loc_loss = np.float32(rpn_loc_loss)
        rpn_cls_loss = np.squeeze(rpn_cls_loss.data.numpy())
        rpn_loss = rpn_cls_loss + weight * rpn_loc_loss
        return rpn_loss

    def roi_loss(self, pre_loc, pre_conf, target_loc, target_conf, weight=10.0):
        target_conf = torch.autograd.Variable(target_conf.long())
        pred_conf_loss = torch.nn.functional.cross_entropy(pre_conf, target_conf, ignore_index=-1)
        pos = target_conf.data > 0
        mask = pos.unsqueeze(1).expand_as(pre_loc)
        mask_pred_loc = pre_loc[mask].view(-1, 4)
        mask_target_loc = target_loc[mask].view(-1, 4)
        x = np.abs(mask_target_loc.numpy() - mask_pred_loc.data.numpy())
        pre_loc_loss = (x < 1) * 0.5 * x ** 2 + (x >= 1) * (x - 0.5)
        N_reg = (target_conf > 0).float().sum()
        N_reg = np.squeeze(N_reg.data.numpy())
        pre_loc_loss = pre_loc_loss.sum() / N_reg
        pre_loc_loss = np.float32(pre_loc_loss)
        pred_conf_loss = np.squeeze(pred_conf_loss.data.numpy())
        total_loss = pred_conf_loss + weight * pre_loc_loss
        return total_loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (VGG,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_liuyuemaicha_simple_faster_rcnn(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

