import sys
_module = sys.modules[__name__]
del sys
detect = _module
hubconf = _module
models = _module
common = _module
experimental = _module
onnx_export = _module
yolo = _module
utils = _module
activations = _module
datasets = _module
google_utils = _module
torch_utils = _module
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


import torch.nn as nn


import torch.nn.functional as F


import math


import random


import time


import numpy as np


from torch.utils.data import Dataset


from copy import deepcopy


import torch.backends.cudnn as cudnn


import torchvision.models as models


from copy import copy


import matplotlib


import matplotlib.pyplot as plt


import torchvision


from scipy.signal import butter


from scipy.signal import filtfilt


import itertools


class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(c2, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class Focus(nn.Module):

    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x):
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Concat(nn.Module):

    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Sum(nn.Module):

    def __init__(self, n, weight=False):
        super(Sum, self).__init__()
        self.weight = weight
        self.iter = range(n - 1)
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)

    def forward(self, x):
        y = x[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y


class GhostConv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super(GhostConv, self).__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


def DWConv(c1, c2, k=1, s=1, act=True):
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class GhostBottleneck(nn.Module):

    def __init__(self, c1, c2, k, s):
        super(GhostBottleneck, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1), DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(), GhostConv(c_, c2, 1, 1, act=False))
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class ConvPlus(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, g=1, bias=True):
        super(ConvPlus, self).__init__()
        self.cv1 = nn.Conv2d(c1, c2, (k, 1), s, (k // 2, 0), groups=g, bias=bias)
        self.cv2 = nn.Conv2d(c1, c2, (1, k), s, (0, k // 2), groups=g, bias=bias)

    def forward(self, x):
        return self.cv1(x) + self.cv2(x)


class MixConv2d(nn.Module):

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        super(MixConv2d, self).__init__()
        groups = len(k)
        if equal_ch:
            i = torch.linspace(0, groups - 1e-06, c2).floor()
            c_ = [(i == g).sum() for g in range(groups)]
        else:
            b = [c2] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()
        self.m = nn.ModuleList([nn.Conv2d(c1, int(c_[g]), k[g], s, k[g] // 2, bias=False) for g in range(groups)])
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return x + self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


class Detect(nn.Module):

    def __init__(self, nc=80, anchors=()):
        super(Detect, self).__init__()
        self.stride = None
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.export = False

    def forward(self, x):
        z = []
        self.training |= self.export
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor


def parse_model(md, ch):
    None
    anchors, nc, gd, gw = md['anchors'], md['nc'], md['depth_multiple'], md['width_multiple']
    na = len(anchors[0]) // 2
    no = na * (nc + 5)
    layers, save, c2 = [], [], ch[-1]
    for i, (f, n, m, args) in enumerate(md['backbone'] + md['head']):
        m = eval(m) if isinstance(m, str) else m
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a
            except:
                pass
        n = max(round(n * gd), 1) if n > 1 else n
        if m in [nn.Conv2d, Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, ConvPlus, BottleneckCSP]:
            c1, c2 = ch[f], args[0]
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]
            if m is BottleneckCSP:
                args.insert(2, n)
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            f = f or list(reversed([(-1 if j == i else j - 1) for j, x in enumerate(ch) if x == no]))
        else:
            c2 = ch[f]
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')
        np = sum([x.numel() for x in m_.parameters()])
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        None
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


class Model(nn.Module):

    def __init__(self, model_cfg='yolov5s.yaml', ch=3, nc=None):
        super(Model, self).__init__()
        if type(model_cfg) is dict:
            self.md = model_cfg
        else:
            with open(model_cfg) as f:
                self.md = yaml.load(f, Loader=yaml.FullLoader)
        if nc:
            self.md['nc'] = nc
        self.model, self.save = parse_model(self.md, ch=[ch])
        m = self.model[-1]
        m.stride = torch.tensor([(64 / x.shape[-2]) for x in self.forward(torch.zeros(1, ch, 64, 64))])
        m.anchors /= m.stride.view(-1, 1, 1)
        self.stride = m.stride
        torch_utils.initialize_weights(self)
        self._initialize_biases()
        torch_utils.model_info(self)
        None

    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]
            s = [0.83, 0.67]
            y = []
            for i, xi in enumerate((x, torch_utils.scale_img(x.flip(3), s[0]), torch_utils.scale_img(x, s[1]))):
                y.append(self.forward_once(xi)[0])
            y[1][..., :4] /= s[0]
            y[1][..., 0] = img_size[1] - y[1][..., 0]
            y[2][..., :4] /= s[1]
            return torch.cat(y, 1), None
        else:
            return self.forward_once(x, profile)

    def forward_once(self, x, profile=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [(x if j == -1 else y[j]) for j in m.f]
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1000000000.0 * 2
                t = torch_utils.time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((torch_utils.time_synchronized() - t) * 100)
                None
            x = m(x)
            y.append(x if m.i in self.save else None)
        if profile:
            None
        return x

    def _initialize_biases(self, cf=None):
        m = self.model[-1]
        for f, s in zip(m.f, m.stride):
            mi = self.model[f % m.i]
            b = mi.bias.view(m.na, -1)
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]
        for f in sorted([(x % m.i) for x in m.f]):
            b = self.model[f].bias.detach().view(m.na, -1).T
            None

    def fuse(self):
        None
        for m in self.model.modules():
            if type(m) is Conv:
                m.conv = torch_utils.fuse_conv_and_bn(m.conv, m.bn)
                m.bn = None
                m.forward = m.fuseforward
        torch_utils.model_info(self)


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MemoryEfficientSwish(nn.Module):

    @staticmethod
    def forward(x):
        return SwishImplementation.apply(x)


class HardSwish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0.0, 6.0, True) / 6.0


class Swish(nn.Module):

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class MishImplementation(torch.autograd.Function):

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


class MemoryEfficientMish(nn.Module):

    @staticmethod
    def forward(x):
        return MishImplementation.apply(x)


class Mish(nn.Module):

    @staticmethod
    def forward(x):
        return x * F.softplus(x).tanh()


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BCEBlurWithLogitsLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bottleneck,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BottleneckCSP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvPlus,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (Focus,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostBottleneck,
     lambda: ([], {'c1': 4, 'c2': 4, 'k': 4, 's': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GhostConv,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HardSwish,
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
    (MixConv2d,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SPP,
     lambda: ([], {'c1': 4, 'c2': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Sum,
     lambda: ([], {'n': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_ChargedMonk_Social_Distancing_using_YOLOv5(_paritybench_base):
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

