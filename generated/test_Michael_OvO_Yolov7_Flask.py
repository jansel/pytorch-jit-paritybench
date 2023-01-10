import sys
_module = sys.modules[__name__]
del sys
app = _module
utils = _module
activations = _module
add_nms = _module
autoanchor = _module
aws = _module
resume = _module
datasets = _module
general = _module
google_utils = _module
loss = _module
metrics = _module
plots = _module
torch_utils = _module
wandb_logging = _module
log_dataset = _module
wandb_utils = _module

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


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


from scipy.cluster.vq import kmeans


import logging


import math


import random


import time


from itertools import repeat


from torch.utils.data import Dataset


from copy import deepcopy


from torchvision.utils import save_image


from torchvision.ops import roi_pool


from torchvision.ops import roi_align


from torchvision.ops import ps_roi_pool


from torchvision.ops import ps_roi_align


import re


import pandas as pd


import torchvision


import matplotlib.pyplot as plt


from copy import copy


import matplotlib


from scipy.signal import butter


from scipy.signal import filtfilt


import torch.backends.cudnn as cudnn


class SiLU(nn.Module):

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class MemoryEfficientSwish(nn.Module):


    class F(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            return grad_output * (sx * (1 + x * (1 - sx)))

    def forward(self, x):
        return self.F.apply(x)


class Mish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


class MemoryEfficientMish(nn.Module):


    class F(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)


class FReLU(nn.Module):

    def __init__(self, c1, k=3):
        super().__init__()
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))


class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 0.0001))
        loss *= alpha_factor
        return loss.mean()


class SigmoidBin(nn.Module):
    stride = None
    export = False

    def __init__(self, bin_count=10, min=0.0, max=1.0, reg_scale=2.0, use_loss_regression=True, use_fw_regression=True, BCE_weight=1.0, smooth_eps=0.0):
        super(SigmoidBin, self).__init__()
        self.bin_count = bin_count
        self.length = bin_count + 1
        self.min = min
        self.max = max
        self.scale = float(max - min)
        self.shift = self.scale / 2.0
        self.use_loss_regression = use_loss_regression
        self.use_fw_regression = use_fw_regression
        self.reg_scale = reg_scale
        self.BCE_weight = BCE_weight
        start = min + self.scale / 2.0 / self.bin_count
        end = max - self.scale / 2.0 / self.bin_count
        step = self.scale / self.bin_count
        self.step = step
        bins = torch.range(start, end + 0.0001, step).float()
        self.register_buffer('bins', bins)
        self.cp = 1.0 - 0.5 * smooth_eps
        self.cn = 0.5 * smooth_eps
        self.BCEbins = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([BCE_weight]))
        self.MSELoss = nn.MSELoss()

    def get_length(self):
        return self.length

    def forward(self, pred):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        pred_reg = (pred[..., 0] * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:1 + self.bin_count]
        _, bin_idx = torch.max(pred_bin, dim=-1)
        bin_bias = self.bins[bin_idx]
        if self.use_fw_regression:
            result = pred_reg + bin_bias
        else:
            result = bin_bias
        result = result.clamp(min=self.min, max=self.max)
        return result

    def training_loss(self, pred, target):
        assert pred.shape[-1] == self.length, 'pred.shape[-1]=%d is not equal to self.length=%d' % (pred.shape[-1], self.length)
        assert pred.shape[0] == target.shape[0], 'pred.shape=%d is not equal to the target.shape=%d' % (pred.shape[0], target.shape[0])
        device = pred.device
        pred_reg = (pred[..., 0].sigmoid() * self.reg_scale - self.reg_scale / 2.0) * self.step
        pred_bin = pred[..., 1:1 + self.bin_count]
        diff_bin_target = torch.abs(target[..., None] - self.bins)
        _, bin_idx = torch.min(diff_bin_target, dim=-1)
        bin_bias = self.bins[bin_idx]
        bin_bias.requires_grad = False
        result = pred_reg + bin_bias
        target_bins = torch.full_like(pred_bin, self.cn, device=device)
        n = pred.shape[0]
        target_bins[range(n), bin_idx] = self.cp
        loss_bin = self.BCEbins(pred_bin, target_bins)
        if self.use_loss_regression:
            loss_regression = self.MSELoss(result, target)
            loss = loss_bin + loss_regression
        else:
            loss = loss_bin
        out_result = result.clamp(min=self.min, max=self.max)
        return loss, out_result


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):

    def _check_input_dim(self, input):
        return


def revert_sync_batchnorm(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


class TracedModel(nn.Module):

    def __init__(self, model=None, device=None, img_size=(640, 640)):
        super(TracedModel, self).__init__()
        None
        self.stride = model.stride
        self.names = model.names
        self.model = model
        self.model = revert_sync_batchnorm(self.model)
        self.model
        self.model.eval()
        self.detect_layer = self.model.model[-1]
        self.model.traced = True
        rand_example = torch.rand(1, 3, img_size, img_size)
        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        traced_script_module.save('traced_model.pt')
        None
        self.model = traced_script_module
        self.model
        self.detect_layer
        None

    def forward(self, x, augment=False, profile=False):
        out = self.model(x)
        out = self.detect_layer(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BCEBlurWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BatchNormXd,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FReLU,
     lambda: ([], {'c1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Hardswish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MemoryEfficientMish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (QFocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Michael_OvO_Yolov7_Flask(_paritybench_base):
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

