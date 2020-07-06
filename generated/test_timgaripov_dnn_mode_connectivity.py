import sys
_module = sys.modules[__name__]
del sys
connect = _module
curves = _module
data = _module
eval_curve = _module
eval_ensemble = _module
fge = _module
models = _module
convfc = _module
preresnet = _module
vgg = _module
wide_resnet = _module
plane = _module
plane_plot = _module
test_curve = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import torch


import torch.nn.functional as F


import math


from torch.nn import Module


from torch.nn import Parameter


from torch.nn.modules.utils import _pair


from scipy.special import binom


import torchvision


import torchvision.transforms as transforms


import time


import torch.nn as nn


class Bezier(Module):

    def __init__(self, num_bends):
        super(Bezier, self).__init__()
        self.register_buffer('binom', torch.Tensor(binom(num_bends - 1, np.arange(num_bends), dtype=np.float32)))
        self.register_buffer('range', torch.arange(0, float(num_bends)))
        self.register_buffer('rev_range', torch.arange(float(num_bends - 1), -1, -1))

    def forward(self, t):
        return self.binom * torch.pow(t, self.range) * torch.pow(1.0 - t, self.rev_range)


class PolyChain(Module):

    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class CurveModule(Module):

    def __init__(self, fix_points, parameter_names=()):
        super(CurveModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t):
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                parameter = getattr(self, '%s_%d' % (parameter_name, j))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += torch.sum(w_t[i] ** 2)
        return w_t


class Linear(CurveModule):

    def __init__(self, in_features, out_features, fix_points, bias=True):
        super(Linear, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features
        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter('weight_%d' % i, Parameter(torch.Tensor(out_features, in_features), requires_grad=not fixed))
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter('bias_%d' % i, Parameter(torch.Tensor(out_features), requires_grad=not fixed))
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.linear(input, weight_t, bias_t)


class Conv2d(CurveModule):

    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(fix_points, ('weight', 'bias'))
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter('weight_%d' % i, Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size), requires_grad=not fixed))
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter('bias_%d' % i, Parameter(torch.Tensor(out_channels), requires_grad=not fixed))
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.conv2d(input, weight_t, bias_t, self.stride, self.padding, self.dilation, self.groups)


class _BatchNorm(CurveModule):
    _version = 2

    def __init__(self, num_features, fix_points, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(_BatchNorm, self).__init__(fix_points, ('weight', 'bias'))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter('weight_%d' % i, Parameter(torch.Tensor(num_features), requires_grad=not fixed))
            else:
                self.register_parameter('weight_%d' % i, None)
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter('bias_%d' % i, Parameter(torch.Tensor(num_features), requires_grad=not fixed))
            else:
                self.register_parameter('bias_%d' % i, None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for i in range(self.num_bends):
                getattr(self, 'weight_%d' % i).data.uniform_()
                getattr(self, 'bias_%d' % i).data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, coeffs_t):
        self._check_input_dim(input)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.batch_norm(input, self.running_mean, self.running_var, weight_t, bias_t, self.training or not self.track_running_stats, exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)
        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
        super(_BatchNorm, self)._load_from_state_dict(state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)


class BatchNorm2d(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))


class CurveNet(Module):

    def __init__(self, num_classes, curve, architecture, num_bends, fix_start=True, fix_end=True, architecture_kwargs={}):
        super(CurveNet, self).__init__()
        self.num_classes = num_classes
        self.num_bends = num_bends
        self.fix_points = [fix_start] + [False] * (self.num_bends - 2) + [fix_end]
        self.curve = curve
        self.architecture = architecture
        self.l2 = 0.0
        self.coeff_layer = self.curve(self.num_bends)
        self.net = self.architecture(num_classes, fix_points=self.fix_points, **architecture_kwargs)
        self.curve_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, CurveModule):
                self.curve_modules.append(module)

    def import_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)

    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(self.net._all_buffers(), base_model._all_buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.num_bends]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        parameters = list(self.net.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i + self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j].data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    def weights(self, t):
        coeffs_t = self.coeff_layer(t)
        weights = []
        for module in self.curve_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def _compute_l2(self):
        self.l2 = sum(module.l2 for module in self.curve_modules)

    def forward(self, input, t=None):
        if t is None:
            t = input.data.new(1).uniform_()
        coeffs_t = self.coeff_layer(t)
        output = self.net(input, coeffs_t)
        self._compute_l2()
        return output


class ConvFCBase(nn.Module):

    def __init__(self, num_classes):
        super(ConvFCBase, self).__init__()
        self.conv_part = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, padding=2), nn.ReLU(True), nn.MaxPool2d(kernel_size=3, stride=2), nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(True), nn.MaxPool2d(3, 2), nn.Conv2d(64, 128, kernel_size=5, padding=2), nn.ReLU(True), nn.MaxPool2d(3, 2))
        self.fc_part = nn.Sequential(nn.Linear(1152, 1000), nn.ReLU(True), nn.Linear(1000, 1000), nn.ReLU(True), nn.Linear(1000, num_classes))
        for m in self.conv_part.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x


class ConvFCCurve(nn.Module):

    def __init__(self, num_classes, fix_points):
        super(ConvFCCurve, self).__init__()
        self.conv1 = curves.Conv2d(3, 32, kernel_size=5, padding=2, fix_points=fix_points)
        self.relu1 = nn.ReLU(True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = curves.Conv2d(32, 64, kernel_size=5, padding=2, fix_points=fix_points)
        self.relu2 = nn.ReLU(True)
        self.max_pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = curves.Conv2d(64, 128, kernel_size=5, padding=2, fix_points=fix_points)
        self.relu3 = nn.ReLU(True)
        self.max_pool3 = nn.MaxPool2d(3, 2)
        self.fc4 = curves.Linear(1152, 1000, fix_points=fix_points)
        self.relu4 = nn.ReLU(True)
        self.fc5 = curves.Linear(1000, 1000, fix_points=fix_points)
        self.relu5 = nn.ReLU(True)
        self.fc6 = curves.Linear(1000, num_classes, fix_points=fix_points)
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2.0 / n))
                    getattr(m, 'bias_%d' % i).data.zero_()

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x, coeffs_t)
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x, coeffs_t)
        x = self.relu3(x)
        x = self.max_pool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc4(x, coeffs_t)
        x = self.relu4(x)
        x = self.fc5(x, coeffs_t)
        x = self.relu5(x)
        x = self.fc6(x, coeffs_t)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


def conv3x3curve(in_planes, out_planes, fix_points, stride=1):
    return curves.Conv2d(in_planes, out_planes, kernel_size=3, fix_points=fix_points, stride=stride, padding=1, bias=True)


class BasicBlockCurve(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, fix_points, stride=1, downsample=None):
        super(BasicBlockCurve, self).__init__()
        self.bn1 = curves.BatchNorm2d(inplanes, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3curve(inplanes, planes, stride=stride, fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = conv3x3curve(planes, planes, fix_points=fix_points)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        residual = x
        out = self.bn1(x, coeffs_t)
        out = self.relu(out)
        out = self.conv1(out, coeffs_t)
        out = self.bn2(out, coeffs_t)
        out = self.relu(out)
        out = self.conv2(out, coeffs_t)
        if self.downsample is not None:
            residual = self.downsample(x, coeffs_t)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class BottleneckCurve(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, fix_points, stride=1, downsample=None):
        super(BottleneckCurve, self).__init__()
        self.bn1 = curves.BatchNorm2d(inplanes, fix_points=fix_points)
        self.conv1 = curves.Conv2d(inplanes, planes, kernel_size=1, bias=False, fix_points=fix_points)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, fix_points=fix_points)
        self.bn3 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv3 = curves.Conv2d(planes, planes * 4, kernel_size=1, bias=False, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, coeffs_t):
        residual = x
        out = self.bn1(x, coeffs_t)
        out = self.relu(out)
        out = self.conv1(out, coeffs_t)
        out = self.bn2(out, coeffs_t)
        out = self.relu(out)
        out = self.conv2(out, coeffs_t)
        out = self.bn3(out, coeffs_t)
        out = self.relu(out)
        out = self.conv3(out, coeffs_t)
        if self.downsample is not None:
            residual = self.downsample(x, coeffs_t)
        out += residual
        return out


class PreResNetBase(nn.Module):

    def __init__(self, num_classes, depth=110):
        super(PreResNetBase, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class PreResNetCurve(nn.Module):

    def __init__(self, num_classes, fix_points, depth=110):
        super(PreResNetCurve, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = BottleneckCurve
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlockCurve
        self.inplanes = 16
        self.conv1 = curves.Conv2d(3, 16, kernel_size=3, padding=1, bias=False, fix_points=fix_points)
        self.layer1 = self._make_layer(block, 16, n, fix_points=fix_points)
        self.layer2 = self._make_layer(block, 32, n, stride=2, fix_points=fix_points)
        self.layer3 = self._make_layer(block, 64, n, stride=2, fix_points=fix_points)
        self.bn = curves.BatchNorm2d(64 * block.expansion, fix_points=fix_points)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = curves.Linear(64 * block.expansion, num_classes, fix_points=fix_points)
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, curves.BatchNorm2d):
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.fill_(1)
                    getattr(m, 'bias_%d' % i).data.zero_()

    def _make_layer(self, block, planes, blocks, fix_points, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = curves.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, fix_points=fix_points)
        layers = list()
        layers.append(block(self.inplanes, planes, fix_points=fix_points, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, fix_points=fix_points))
        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        x = self.conv1(x, coeffs_t)
        for block in self.layer1:
            x = block(x, coeffs_t)
        for block in self.layer2:
            x = block(x, coeffs_t)
        for block in self.layer3:
            x = block(x, coeffs_t)
        x = self.bn(x, coeffs_t)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x, coeffs_t)
        return x


def make_layers(config, batch_norm=False, fix_points=None):
    layer_blocks = nn.ModuleList()
    activation_blocks = nn.ModuleList()
    poolings = nn.ModuleList()
    kwargs = dict()
    conv = nn.Conv2d
    bn = nn.BatchNorm2d
    if fix_points is not None:
        kwargs['fix_points'] = fix_points
        conv = curves.Conv2d
        bn = curves.BatchNorm2d
    in_channels = 3
    for sizes in config:
        layer_blocks.append(nn.ModuleList())
        activation_blocks.append(nn.ModuleList())
        for channels in sizes:
            layer_blocks[-1].append(conv(in_channels, channels, kernel_size=3, padding=1, **kwargs))
            if batch_norm:
                layer_blocks[-1].append(bn(channels, **kwargs))
            activation_blocks[-1].append(nn.ReLU(inplace=True))
            in_channels = channels
        poolings.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layer_blocks, activation_blocks, poolings


class VGGBase(nn.Module):

    def __init__(self, num_classes, depth=16, batch_norm=False):
        super(VGGBase, self).__init__()
        layer_blocks, activation_blocks, poolings = make_layers(config[depth], batch_norm)
        self.layer_blocks = layer_blocks
        self.activation_blocks = activation_blocks
        self.poolings = poolings
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        for layers, activations, pooling in zip(self.layer_blocks, self.activation_blocks, self.poolings):
            for layer, activation in zip(layers, activations):
                x = layer(x)
                x = activation(x)
            x = pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGGCurve(nn.Module):

    def __init__(self, num_classes, fix_points, depth=16, batch_norm=False):
        super(VGGCurve, self).__init__()
        layer_blocks, activation_blocks, poolings = make_layers(config[depth], batch_norm, fix_points=fix_points)
        self.layer_blocks = layer_blocks
        self.activation_blocks = activation_blocks
        self.poolings = poolings
        self.dropout1 = nn.Dropout()
        self.fc1 = curves.Linear(512, 512, fix_points=fix_points)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc2 = curves.Linear(512, 512, fix_points=fix_points)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = curves.Linear(512, num_classes, fix_points=fix_points)
        for m in self.modules():
            if isinstance(m, curves.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.num_bends):
                    getattr(m, 'weight_%d' % i).data.normal_(0, math.sqrt(2.0 / n))
                    getattr(m, 'bias_%d' % i).data.zero_()

    def forward(self, x, coeffs_t):
        for layers, activations, pooling in zip(self.layer_blocks, self.activation_blocks, self.poolings):
            for layer, activation in zip(layers, activations):
                x = layer(x, coeffs_t)
                x = activation(x)
            x = pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x, coeffs_t)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.fc2(x, coeffs_t)
        x = self.relu2(x)
        x = self.fc3(x, coeffs_t)
        return x


class WideBasic(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideBasicCurve(nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, fix_points, stride=1):
        super(WideBasicCurve, self).__init__()
        self.bn1 = curves.BatchNorm2d(in_planes, fix_points=fix_points)
        self.conv1 = curves.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True, fix_points=fix_points)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = curves.BatchNorm2d(planes, fix_points=fix_points)
        self.conv2 = curves.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, fix_points=fix_points)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = curves.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True, fix_points=fix_points)

    def forward(self, x, coeffs_t):
        out = self.dropout(self.conv1(F.relu(self.bn1(x, coeffs_t)), coeffs_t))
        out = self.conv2(F.relu(self.bn2(out, coeffs_t)), coeffs_t)
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x, coeffs_t)
        out += residual
        return out


class WideResNetBase(nn.Module):

    def __init__(self, num_classes, depth=28, widen_factor=10, dropout_rate=0.0):
        super(WideResNetBase, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor
        nstages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, nstages[0])
        self.layer1 = self._wide_layer(WideBasic, nstages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nstages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nstages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
        self.linear = nn.Linear(nstages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class WideResNetCurve(nn.Module):

    def __init__(self, num_classes, fix_points, depth=28, widen_factor=10, dropout_rate=0.0):
        super(WideResNetCurve, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor
        nstages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3curve(3, nstages[0], fix_points=fix_points)
        self.layer1 = self._wide_layer(WideBasicCurve, nstages[1], n, dropout_rate, stride=1, fix_points=fix_points)
        self.layer2 = self._wide_layer(WideBasicCurve, nstages[2], n, dropout_rate, stride=2, fix_points=fix_points)
        self.layer3 = self._wide_layer(WideBasicCurve, nstages[3], n, dropout_rate, stride=2, fix_points=fix_points)
        self.bn1 = curves.BatchNorm2d(nstages[3], momentum=0.9, fix_points=fix_points)
        self.linear = curves.Linear(nstages[3], num_classes, fix_points=fix_points)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, fix_points):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, fix_points=fix_points, stride=stride))
            self.in_planes = planes
        return nn.ModuleList(layers)

    def forward(self, x, coeffs_t):
        out = self.conv1(x, coeffs_t)
        for block in self.layer1:
            out = block(out, coeffs_t)
        for block in self.layer2:
            out = block(out, coeffs_t)
        for block in self.layer3:
            out = block(out, coeffs_t)
        out = F.relu(self.bn1(out, coeffs_t))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out, coeffs_t)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Bezier,
     lambda: ([], {'num_bends': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvFCBase,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
    (PolyChain,
     lambda: ([], {'num_bends': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PreResNetBase,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
    (WideBasic,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WideResNetBase,
     lambda: ([], {'num_classes': 4}),
     lambda: ([torch.rand([4, 3, 32, 32])], {}),
     True),
]

class Test_timgaripov_dnn_mode_connectivity(_paritybench_base):
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

