import sys
_module = sys.modules[__name__]
del sys
deep_sort = _module
deep = _module
evaluate = _module
feature_extractor = _module
model = _module
original_model = _module
test = _module
train = _module
deep_sort = _module
sort = _module
detection = _module
iou_matching = _module
kalman_filter = _module
linear_assignment = _module
nn_matching = _module
preprocessing = _module
track = _module
tracker = _module
track = _module
models = _module
utils = _module
datasets = _module
google_utils = _module
parse_config = _module
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


import torchvision.transforms as transforms


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import torch.backends.cudnn as cudnn


import torchvision


import time


import matplotlib.pyplot as plt


import math


import random


from torch.utils.data import Dataset


from collections import defaultdict


from torch.optim import Optimizer


import matplotlib


class BasicBlock(nn.Module):

    def __init__(self, c_in, c_out, is_downsample=False):
        super(BasicBlock, self).__init__()
        self.is_downsample = is_downsample
        if is_downsample:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if is_downsample:
            self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=2, bias=False), nn.BatchNorm2d(c_out))
        elif c_in != c_out:
            self.downsample = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride=1, bias=False), nn.BatchNorm2d(c_out))
            self.is_downsample = True

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.is_downsample:
            x = self.downsample(x)
        return F.relu(x.add(y), True)


def make_layers(c_in, c_out, repeat_times, is_downsample=False):
    blocks = []
    for i in range(repeat_times):
        if i == 0:
            blocks += [BasicBlock(c_in, c_out, is_downsample=is_downsample)]
        else:
            blocks += [BasicBlock(c_out, c_out)]
    return nn.Sequential(*blocks)


class Net(nn.Module):

    def __init__(self, num_classes=625, reid=False):
        super(Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ELU(inplace=True), nn.Conv2d(32, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ELU(inplace=True), nn.MaxPool2d(3, 2, padding=1))
        self.layer1 = make_layers(32, 32, 2, False)
        self.layer2 = make_layers(32, 64, 2, True)
        self.layer3 = make_layers(64, 128, 2, True)
        self.dense = nn.Sequential(nn.Dropout(p=0.6), nn.Linear(128 * 16 * 8, 128), nn.BatchNorm1d(128), nn.ELU(inplace=True))
        self.reid = reid
        self.batch_norm = nn.BatchNorm1d(128)
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        if self.reid:
            x = self.dense[0](x)
            x = self.dense[1](x)
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.dense(x)
        x = self.classifier(x)
        return x


class SwishImplementation(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):

    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):

    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):

    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())


ONNX_EXPORT = False


def create_grids(self, img_size=416, ng=(13, 13), device='cpu', type=torch.float32):
    nx, ny = ng
    self.img_size = max(img_size)
    self.stride = self.img_size / max(ng)
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    self.grid_xy = torch.stack((xv, yv), 2).type(type).view((1, 1, ny, nx, 2))
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).type(type)
    self.ng = torch.Tensor(ng)
    self.nx = nx
    self.ny = ny


class YOLOLayer(nn.Module):

    def __init__(self, anchors, nc, img_size, yolo_index, arc):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.na = len(anchors)
        self.nc = nc
        self.no = nc + 5
        self.nx = 0
        self.ny = 0
        self.arc = arc
        if ONNX_EXPORT:
            stride = [32, 16, 8][yolo_index]
            nx = int(img_size[1] / stride)
            ny = int(img_size[0] / stride)
            create_grids(self, img_size, (nx, ny))

    def forward(self, p, img_size, var=None):
        if ONNX_EXPORT:
            bs = 1
        else:
            bs, _, ny, nx = p.shape
            if (self.nx, self.ny) != (nx, ny):
                create_grids(self, img_size, (nx, ny), p.device, p.dtype)
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()
        if self.training:
            return p
        elif ONNX_EXPORT:
            m = self.na * self.nx * self.ny
            grid_xy = self.grid_xy.repeat((1, self.na, 1, 1, 1)).view(m, 2)
            anchor_wh = self.anchor_wh.repeat((1, 1, self.nx, self.ny, 1)).view(m, 2) / self.ng
            p = p.view(m, self.no)
            xy = torch.sigmoid(p[:, 0:2]) + grid_xy
            wh = torch.exp(p[:, 2:4]) * anchor_wh
            p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])
            return p_cls, xy / self.ng, wh
        else:
            io = p.clone()
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid_xy
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh
            io[..., :4] *= self.stride
            if 'default' in self.arc:
                torch.sigmoid_(io[..., 4:])
            elif 'BCE' in self.arc:
                torch.sigmoid_(io[..., 5:])
                io[..., 4] = 1
            elif 'CE' in self.arc:
                io[..., 4:] = F.softmax(io[..., 4:], dim=4)
                io[..., 4] = 1
            if self.nc == 1:
                io[..., 5] = 1
            return io.view(bs, -1, self.no), p


def create_modules(module_defs, img_size, arc):
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1
    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()
        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            size = int(mdef['size'])
            stride = int(mdef['stride']) if 'stride' in mdef else (int(mdef['stride_y']), int(mdef['stride_x']))
            pad = (size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1], out_channels=filters, kernel_size=size, stride=stride, padding=pad, groups=int(mdef['groups']) if 'groups' in mdef else 1, bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
        elif mdef['type'] == 'maxpool':
            size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=int((size - 1) // 2))
            if size == 2 and stride == 1:
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool
        elif mdef['type'] == 'upsample':
            if ONNX_EXPORT:
                g = (yolo_index + 1) * 2
                modules = nn.Upsample(size=(10 * g, 6 * g), mode='nearest')
            else:
                modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')
        elif mdef['type'] == 'route':
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([(l if l > 0 else l + i) for l in layers])
        elif mdef['type'] == 'shortcut':
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])
        elif mdef['type'] == 'reorg3d':
            pass
        elif mdef['type'] == 'yolo':
            yolo_index += 1
            mask = [int(x) for x in mdef['mask'].split(',')]
            modules = YOLOLayer(anchors=mdef['anchors'][mask], nc=int(mdef['classes']), img_size=img_size, yolo_index=yolo_index, arc=arc)
            try:
                if arc == 'default' or arc == 'Fdefault':
                    b = [-5.0, -5.0]
                elif arc == 'uBCE':
                    b = [0, -9.0]
                elif arc == 'uCE':
                    b = [10, -0.1]
                elif arc == 'uFBCE':
                    b = [0, -6.5]
                elif arc == 'uFCE':
                    b = [7.7, -1.1]
                bias = module_list[-1][0].bias.view(len(mask), -1)
                bias[:, 4] += b[0] - bias[:, 4].mean()
                bias[:, 5:] += b[1] - bias[:, 5:].mean()
                module_list[-1][0].bias = torch.nn.Parameter(bias.view(-1))
            except:
                None
        else:
            None
        module_list.append(modules)
        output_filters.append(filters)
    return module_list, routs


def get_yolo_layers(model):
    bool_vec = [(x['type'] == 'yolo') for x in model.module_defs]
    return [i for i, x in enumerate(bool_vec) if x]


def parse_model_cfg(path):
    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path
    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0
        else:
            key, val = line.split('=')
            key = key.rstrip()
            if 'anchors' in key:
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            else:
                mdefs[-1][key] = val.strip()
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups', 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random', 'stride_x', 'stride_y']
    f = []
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]
    assert not any(u), 'Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631' % (u, path)
    return mdefs


class Darknet(nn.Module):

    def __init__(self, cfg, img_size=(416, 416), arc='default'):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, arc)
        self.yolo_layers = get_yolo_layers(self)
        self.version = np.array([0, 2, 5], dtype=np.int32)
        self.seen = np.array([0], dtype=np.int64)

    def forward(self, x, var=None):
        img_size = x.shape[-2:]
        output, layer_outputs = [], []
        verbose = False
        if verbose:
            None
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if verbose:
                    None
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                else:
                    try:
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
                    except:
                        layer_outputs[layers[1]] = F.interpolate(layer_outputs[layers[1]], scale_factor=[0.5, 0.5])
                        x = torch.cat([layer_outputs[i] for i in layers], 1)
            elif mtype == 'shortcut':
                j = int(mdef['from'])
                if verbose:
                    None
                x = x + layer_outputs[j]
            elif mtype == 'yolo':
                output.append(module(x, img_size))
            layer_outputs.append(x if i in self.routs else [])
            if verbose:
                None
        if self.training:
            return output
        elif ONNX_EXPORT:
            x = [torch.cat(x, 0) for x in zip(*output)]
            return x[0], torch.cat(x[1:3], 1)
        else:
            io, p = list(zip(*output))
            return torch.cat(io, 1), p

    def fuse(self):
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        conv = a[i - 1]
                        fused = torch_utils.fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1:])
                        break
            fused_list.append(a)
        self.module_list = fused_list


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=0.5, alpha=1):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'c_in': 4, 'c_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FocalLoss,
     lambda: ([], {'loss_fcn': MSELoss()}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (MemoryEfficientSwish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Mish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mikel_brostrom_Yolov3_DeepSort_Pytorch(_paritybench_base):
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

