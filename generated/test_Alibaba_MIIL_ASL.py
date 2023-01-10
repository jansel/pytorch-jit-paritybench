import sys
_module = sys.modules[__name__]
del sys
infer = _module
helper_functions = _module
losses = _module
models = _module
tresnet = _module
anti_aliasing = _module
avg_pool = _module
general_layers = _module
tresnet = _module
utils = _module
factory = _module
train = _module
validate = _module

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


import matplotlib


import matplotlib.pyplot as plt


import numpy as np


from copy import deepcopy


import random


import time


from torchvision import datasets as datasets


import torch.nn as nn


import torch.nn.parallel


import torch.nn.functional as F


from torch.nn import Module as Module


from collections import OrderedDict


import torch.optim


import torch.utils.data.distributed


import torchvision.transforms as transforms


from torch.optim import lr_scheduler


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


class ModelEma(torch.nn.Module):

    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class AsymmetricLoss(nn.Module):

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-08, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w
        return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    """ Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-08, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        self.targets = y
        self.anti_targets = 1 - y
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg, self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    """
    This loss is intended for single-label classification problems
    """

    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float=0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()
        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        """
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        """
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg, self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w
        if self.eps > 0:
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)
        loss = -self.targets_classes.mul(log_preds)
        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class Downsample(nn.Module):

    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels
        assert self.filt_size == 3
        a = torch.tensor([1.0, 2.0, 1.0])
        filt = (a[:, None] * a[None, :]).clone().detach()
        filt = filt / torch.sum(filt)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


class AntiAliasDownsampleLayer(nn.Module):

    def __init__(self, remove_model_jit: bool=False, filt_size: int=3, stride: int=2, channels: int=0):
        super(AntiAliasDownsampleLayer, self).__init__()
        if not remove_model_jit:
            self.op = DownsampleJIT(filt_size, stride, channels)
        else:
            self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)


class FastAvgPool2d(nn.Module):

    def __init__(self, flatten=False):
        super(FastAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // self.bs ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // self.bs ** 2, H * self.bs, W * self.bs)
        return x


class SpaceToDepth(nn.Module):

    def __init__(self, block_size=4):
        super().__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * self.bs ** 2, H // self.bs, W // self.bs)
        return x


class SpaceToDepthModule(nn.Module):

    def __init__(self, remove_model_jit=False):
        super().__init__()
        if not remove_model_jit:
            self.op = SpaceToDepthJit()
        else:
            self.op = SpaceToDepth()

    def forward(self, x):
        return self.op(x)


class hard_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(hard_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
        else:
            return F.relu6(x + 3.0) / 6.0


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastAvgPool2d()
        self.fc1 = nn.Conv2d(channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=inplace)
        self.fc2 = nn.Conv2d(reduction_channels, channels, kernel_size=1, padding=0, bias=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se


class bottleneck_head(nn.Module):

    def __init__(self, num_features, num_classes, bottleneck_features=200):
        super(bottleneck_head, self).__init__()
        self.embedding_generator = nn.ModuleList()
        self.embedding_generator.append(nn.Linear(num_features, bottleneck_features))
        self.embedding_generator = nn.Sequential(*self.embedding_generator)
        self.FC = nn.Linear(bottleneck_features, num_classes)

    def forward(self, x):
        self.embedding = self.embedding_generator(x)
        logits = self.FC(self.embedding)
        return logits


def conv2d_ABN(ni, nf, stride, activation='leaky_relu', kernel_size=3, activation_param=0.01, groups=1):
    return nn.Sequential(nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False), InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param))


class BasicBlock(Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=1, activation_param=0.001)
        elif anti_alias_layer is None:
            self.conv1 = conv2d_ABN(inplanes, planes, stride=2, activation_param=0.001)
        else:
            self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=0.001), anti_alias_layer(channels=planes, filt_size=3, stride=2))
        self.conv2 = conv2d_ABN(planes, planes, stride=1, activation='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation='leaky_relu', activation_param=0.001)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation='leaky_relu', activation_param=0.001)
        elif anti_alias_layer is None:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation='leaky_relu', activation_param=0.001)
        else:
            self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation='leaky_relu', activation_param=0.001), anti_alias_layer(channels=planes, filt_size=3, stride=2))
        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1, activation='identity')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        out = out + residual
        out = self.relu(out)
        return out


class TResNet(Module):

    def __init__(self, layers, in_chans=3, num_classes=1000, width_factor=1.0, do_bottleneck_head=False, bottleneck_features=512):
        super(TResNet, self).__init__()
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = AntiAliasDownsampleLayer
        global_pool_layer = FastAvgPool2d(flatten=True)
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True, anti_alias_layer=anti_alias_layer)
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True, anti_alias_layer=anti_alias_layer)
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True, anti_alias_layer=anti_alias_layer)
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False, anti_alias_layer=anti_alias_layer)
        self.body = nn.Sequential(OrderedDict([('SpaceToDepth', space_to_depth), ('conv1', conv1), ('layer1', layer1), ('layer2', layer2), ('layer3', layer3), ('layer4', layer4)]))
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = self.planes * 8 * Bottleneck.expansion
        if do_bottleneck_head:
            fc = bottleneck_head(self.num_features, num_classes, bottleneck_features=bottleneck_features)
        else:
            fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, activation='identity')]
            downsample = nn.Sequential(*layers)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se, anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AsymmetricLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (AsymmetricLossOptimized,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FastAvgPool2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SEModule,
     lambda: ([], {'channels': 4, 'reduction_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SpaceToDepth,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (bottleneck_head,
     lambda: ([], {'num_features': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (hard_sigmoid,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_Alibaba_MIIL_ASL(_paritybench_base):
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

