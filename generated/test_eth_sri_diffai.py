import sys
_module = sys.modules[__name__]
del sys
master = _module
ai = _module
components = _module
convert = _module
goals = _module
losses = _module
models = _module
scheduling = _module

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


import copy


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torchvision import datasets


from torch.utils.data import Dataset


import torch.onnx


import inspect


from inspect import getargspec


import random


import math


import warnings


from torch.serialization import SourceChangeWarning


import torch.autograd


from functools import reduce


from torch.distributions import multinomial


from torch.distributions import categorical


import abc


from torch.nn.modules.conv import _ConvNd


from enum import Enum


from torchvision import transforms


from torchvision import utils


from itertools import count


import numpy as np


class InferModule(nn.Module):

    def __init__(self, *args, normal=False, ibp_init=False, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.infered = False
        self.normal = normal
        self.ibp_init = ibp_init

    def infer(self, in_shape, global_args=None):
        """ this is really actually stateful. """
        if self.infered:
            return self
        self.infered = True
        super(InferModule, self).__init__()
        self.inShape = list(in_shape)
        self.outShape = list(self.init(list(in_shape), *self.args, global_args=global_args, **self.kwargs))
        if self.outShape is None:
            raise 'init should set the out_shape'
        self.reset_parameters()
        return self

    def reset_parameters(self):
        if not hasattr(self, 'weight') or self.weight is None:
            return
        n = h.product(self.weight.size()) / self.outShape[0]
        stdv = 1 / math.sqrt(n)
        if self.ibp_init:
            torch.nn.init.orthogonal_(self.weight.data)
        elif self.normal:
            self.weight.data.normal_(0, stdv)
            self.weight.data.clamp_(-1, 1)
        else:
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            if self.ibp_init:
                self.bias.data.zero_()
            elif self.normal:
                self.bias.data.normal_(0, stdv)
                self.bias.data.clamp_(-1, 1)
            else:
                self.bias.data.uniform_(-stdv, stdv)

    def clip_norm(self):
        if not hasattr(self, 'weight'):
            return
        if not hasattr(self, 'weight_g'):
            if torch.__version__[0] == '0':
                nn.utils.weight_norm(self, dim=None)
            else:
                nn.utils.weight_norm(self)
        self.weight_g.data.clamp_(-h.max_c_for_norm, h.max_c_for_norm)
        if torch.__version__[0] != '0':
            self.weight_v.data.clamp_(-h.max_c_for_norm * 10000, h.max_c_for_norm * 10000)
            if hasattr(self, 'bias'):
                self.bias.data.clamp_(-h.max_c_for_norm * 10000, h.max_c_for_norm * 10000)

    def regularize(self, p):
        reg = 0
        if torch.__version__[0] == '0':
            for param in self.parameters():
                reg += param.norm(p)
        else:
            if hasattr(self, 'weight_g'):
                reg += self.weight_g.norm().sum()
                reg += self.weight_v.norm().sum()
            elif hasattr(self, 'weight'):
                reg += self.weight.norm().sum()
            if hasattr(self, 'bias'):
                reg += self.bias.view(-1).norm(p=p).sum()
        return reg

    def remove_norm(self):
        if hasattr(self, 'weight_g'):
            torch.nn.utils.remove_weight_norm(self)

    def showNet(self, t=''):
        None

    def printNet(self, f):
        None

    @abc.abstractmethod
    def forward(self, *args, **kargs):
        pass

    def __call__(self, *args, onyx=False, **kargs):
        if onyx:
            return self.forward(*args, onyx=onyx, **kargs)
        else:
            return super(InferModule, self).__call__(*args, **kargs)

    @abc.abstractmethod
    def neuronCount(self):
        pass

    def depth(self):
        return 0


class Linear(InferModule):

    def init(self, in_shape, out_shape, **kargs):
        self.in_neurons = h.product(in_shape)
        if isinstance(out_shape, int):
            out_shape = [out_shape]
        self.out_neurons = h.product(out_shape)
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_neurons, self.out_neurons))
        self.bias = torch.nn.Parameter(torch.Tensor(self.out_neurons))
        return out_shape

    def forward(self, x, **kargs):
        s = x.size()
        x = x.view(s[0], h.product(s[1:]))
        return (x.matmul(self.weight) + self.bias).view(s[0], *self.outShape)

    def neuronCount(self):
        return 0

    def showNet(self, t=''):
        None

    def printNet(self, f):
        None
        None
        None


class Activation(InferModule):

    def init(self, in_shape, global_args=None, activation='ReLU', **kargs):
        self.activation = ['ReLU', 'Sigmoid', 'Tanh', 'Softplus', 'ELU', 'SELU'].index(activation)
        self.activation_name = activation
        return in_shape

    def regularize(self, p):
        return 0

    def forward(self, x, **kargs):
        return [lambda x: x.relu(), lambda x: x.sigmoid(), lambda x: x.tanh(), lambda x: x.softplus(), lambda x: x.elu(), lambda x: x.selu()][self.activation](x)

    def neuronCount(self):
        return h.product(self.outShape)

    def depth(self):
        return 1

    def showNet(self, t=''):
        None

    def printNet(self, f):
        pass


class ReLU(Activation):
    pass


class Identity(InferModule):

    def init(self, in_shape, global_args=None, **kargs):
        return in_shape

    def forward(self, x, **kargs):
        return x

    def neuronCount(self):
        return 0

    def printNet(self, f):
        pass

    def regularize(self, p):
        return 0

    def showNet(self, *args, **kargs):
        pass


class Dropout(InferModule):

    def init(self, in_shape, p=0.5, use_2d=False, alpha_dropout=False, **kargs):
        self.p = S.Const.initConst(p)
        self.use_2d = use_2d
        self.alpha_dropout = alpha_dropout
        return in_shape

    def forward(self, x, time=0, **kargs):
        if self.training:
            with torch.no_grad():
                p = self.p.getVal(time=time)
                mask = (F.dropout2d if self.use_2d else F.dropout)(h.ones(x.size()), p=p, training=True)
            if self.alpha_dropout:
                with torch.no_grad():
                    keep_prob = 1 - p
                    alpha = -1.7580993408473766
                    a = math.pow(keep_prob + alpha * alpha * keep_prob * (1 - keep_prob), -0.5)
                    b = -a * alpha * (1 - keep_prob)
                    mask = mask * a
                return x * mask + b
            else:
                return x * mask
        else:
            return x

    def neuronCount(self):
        return 0

    def showNet(self, t=''):
        None

    def printNet(self, f):
        None


class PrintActivation(Identity):

    def init(self, in_shape, global_args=None, activation='ReLU', **kargs):
        self.activation = activation
        return in_shape

    def printNet(self, f):
        None


class PrintReLU(PrintActivation):
    pass


def getShapeConv(in_shape, conv_shape, stride=1, padding=0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]
    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return outChan, outH, outW


class Conv2D(InferModule):

    def init(self, in_shape, out_channels, kernel_size, stride=1, global_args=None, bias=True, padding=0, activation='ReLU', **kargs):
        self.prev = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.use_softplus = h.default(global_args, 'use_softplus', False)
        weights_shape = self.out_channels, self.in_channels, kernel_size, kernel_size
        self.weight = torch.nn.Parameter(torch.Tensor(*weights_shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(weights_shape[0]))
        else:
            self.bias = None
        outshape = getShapeConv(in_shape, (out_channels, kernel_size, kernel_size), stride, padding)
        return outshape

    def forward(self, input, **kargs):
        return input.conv2d(self.weight, bias=self.bias, stride=self.stride, padding=self.padding)

    def printNet(self, f):
        None
        sz = list(self.prev)
        None
        None
        None

    def showNet(self, t=''):
        sz = list(self.prev)
        None

    def neuronCount(self):
        return 0


def getShapeConvTranspose(in_shape, conv_shape, stride=1, padding=0, out_padding=0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]
    outH = (inH - 1) * stride - 2 * padding + kH + out_padding
    outW = (inW - 1) * stride - 2 * padding + kW + out_padding
    return outChan, outH, outW


class ConvTranspose2D(InferModule):

    def init(self, in_shape, out_channels, kernel_size, stride=1, global_args=None, bias=True, padding=0, out_padding=0, activation='ReLU', **kargs):
        self.prev = in_shape
        self.in_channels = in_shape[0]
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_padding = out_padding
        self.activation = activation
        self.use_softplus = h.default(global_args, 'use_softplus', False)
        weights_shape = self.in_channels, self.out_channels, kernel_size, kernel_size
        self.weight = torch.nn.Parameter(torch.Tensor(*weights_shape))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(weights_shape[0]))
        else:
            self.bias = None
        outshape = getShapeConvTranspose(in_shape, (out_channels, kernel_size, kernel_size), stride, padding, out_padding)
        return outshape

    def forward(self, input, **kargs):
        return input.conv_transpose2d(self.weight, bias=self.bias, stride=self.stride, padding=self.padding, output_padding=self.out_padding)

    def printNet(self, f):
        None
        None
        None
        None

    def neuronCount(self):
        return 0


class MaxPool2D(InferModule):

    def init(self, in_shape, kernel_size, stride=None, **kargs):
        self.prev = in_shape
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        return getShapeConv(in_shape, (in_shape[0], kernel_size, kernel_size), stride)

    def forward(self, x, **kargs):
        return x.max_pool2d(self.kernel_size, self.stride)

    def printNet(self, f):
        None

    def neuronCount(self):
        return h.product(self.outShape)


class AvgPool2D(InferModule):

    def init(self, in_shape, kernel_size, stride=None, **kargs):
        self.prev = in_shape
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        out_size = getShapeConv(in_shape, (in_shape[0], kernel_size, kernel_size), self.stride, padding=1)
        return out_size

    def forward(self, x, **kargs):
        if h.product(x.size()[2:]) == 1:
            return x
        return x.avg_pool2d(kernel_size=self.kernel_size, stride=self.stride, padding=1)

    def printNet(self, f):
        None

    def neuronCount(self):
        return h.product(self.outShape)


class AdaptiveAvgPool2D(InferModule):

    def init(self, in_shape, out_shape, **kargs):
        self.prev = in_shape
        self.out_shape = list(out_shape)
        return [in_shape[0]] + self.out_shape

    def forward(self, x, **kargs):
        return x.adaptive_avg_pool2d(self.out_shape)

    def printNet(self, f):
        None

    def neuronCount(self):
        return h.product(self.outShape)


class Normalize(InferModule):

    def init(self, in_shape, mean, std, **kargs):
        self.mean_v = mean
        self.std_v = std
        self.mean = h.dten(mean)
        self.std = 1 / h.dten(std)
        return in_shape

    def forward(self, x, **kargs):
        mean_ex = self.mean.view(self.mean.shape[0], 1, 1).expand(*x.size()[1:])
        std_ex = self.std.view(self.std.shape[0], 1, 1).expand(*x.size()[1:])
        return (x - mean_ex) * std_ex

    def neuronCount(self):
        return 0

    def printNet(self, f):
        None

    def showNet(self, t=''):
        None


class Flatten(InferModule):

    def init(self, in_shape, **kargs):
        return h.product(in_shape)

    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], h.product(s[1:]))

    def neuronCount(self):
        return 0


class BatchNorm(InferModule):

    def init(self, in_shape, track_running_stats=True, momentum=0.1, eps=1e-05, **kargs):
        self.gamma = torch.nn.Parameter(torch.Tensor(*in_shape))
        self.beta = torch.nn.Parameter(torch.Tensor(*in_shape))
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None
        self.num_batches_tracked = 0
        return in_shape

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, x, **kargs):
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        new_mean = x.vanillaTensorPart().detach().mean(dim=0)
        new_var = x.vanillaTensorPart().detach().var(dim=0, unbiased=False)
        if torch.isnan(new_var * 0).any():
            return x
        if self.training:
            self.running_mean = (1 - exponential_average_factor) * self.running_mean + exponential_average_factor * new_mean if self.running_mean is not None else new_mean
            if self.running_var is None:
                self.running_var = new_var
            else:
                q = (1 - exponential_average_factor) * self.running_var
                r = exponential_average_factor * new_var
                self.running_var = q + r
        if self.track_running_stats and self.running_mean is not None and self.running_var is not None:
            new_mean = self.running_mean
            new_var = self.running_var
        diver = 1 / (new_var + self.eps).sqrt()
        if torch.isnan(diver).any():
            None
            return x
        else:
            out = (x - new_mean) * diver * self.gamma + self.beta
            return out

    def neuronCount(self):
        return 0


class Unflatten2d(InferModule):

    def init(self, in_shape, w, **kargs):
        self.w = w
        self.outChan = int(h.product(in_shape) / (w * w))
        return self.outChan, self.w, self.w

    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], self.outChan, self.w, self.w)

    def neuronCount(self):
        return 0


class View(InferModule):

    def init(self, in_shape, out_shape, **kargs):
        assert h.product(in_shape) == h.product(out_shape)
        return out_shape

    def forward(self, x, **kargs):
        s = x.size()
        return x.view(s[0], *self.outShape)

    def neuronCount(self):
        return 0


class Seq(InferModule):

    def init(self, in_shape, *layers, **kargs):
        self.layers = layers
        self.net = nn.Sequential(*layers)
        self.prev = in_shape
        for s in layers:
            in_shape = s.infer(in_shape, **kargs).outShape
        return in_shape

    def forward(self, x, **kargs):
        for l in self.layers:
            x = l(x, **kargs)
        return x

    def clip_norm(self):
        for l in self.layers:
            l.clip_norm()

    def regularize(self, p):
        return sum(n.regularize(p) for n in self.layers)

    def remove_norm(self):
        for l in self.layers:
            l.remove_norm()

    def printNet(self, f):
        for l in self.layers:
            l.printNet(f)

    def showNet(self, *args, **kargs):
        for l in self.layers:
            l.showNet(*args, **kargs)

    def neuronCount(self):
        return sum([l.neuronCount() for l in self.layers])

    def depth(self):
        return sum([l.depth() for l in self.layers])


class FromByteImg(InferModule):

    def init(self, in_shape, **kargs):
        return in_shape

    def forward(self, x, **kargs):
        return x.to_dtype() / 256.0

    def neuronCount(self):
        return 0


class Skip(InferModule):

    def init(self, in_shape, net1, net2, **kargs):
        self.net1 = net1.infer(in_shape, **kargs)
        self.net2 = net2.infer(in_shape, **kargs)
        assert net1.outShape[1:] == net2.outShape[1:]
        return [net1.outShape[0] + net2.outShape[0]] + net1.outShape[1:]

    def forward(self, x, **kargs):
        r1 = self.net1(x, **kargs)
        r2 = self.net2(x, **kargs)
        return r1.cat(r2, dim=1)

    def regularize(self, p):
        return self.net1.regularize(p) + self.net2.regularize(p)

    def clip_norm(self):
        self.net1.clip_norm()
        self.net2.clip_norm()

    def remove_norm(self):
        self.net1.remove_norm()
        self.net2.remove_norm()

    def neuronCount(self):
        return self.net1.neuronCount() + self.net2.neuronCount()

    def printNet(self, f):
        None
        self.net1.printNet(f)
        None
        self.net2.printNet(f)
        None

    def showNet(self, t=''):
        None
        self.net1.showNet('    ' + t)
        None
        self.net2.showNet('    ' + t)
        None


class ParSum(InferModule):

    def init(self, in_shape, net1, net2, **kargs):
        self.net1 = net1.infer(in_shape, **kargs)
        self.net2 = net2.infer(in_shape, **kargs)
        assert net1.outShape == net2.outShape
        return net1.outShape

    def forward(self, x, **kargs):
        r1 = self.net1(x, **kargs)
        r2 = self.net2(x, **kargs)
        return x.addPar(r1, r2)

    def clip_norm(self):
        self.net1.clip_norm()
        self.net2.clip_norm()

    def remove_norm(self):
        self.net1.remove_norm()
        self.net2.remove_norm()

    def neuronCount(self):
        return self.net1.neuronCount() + self.net2.neuronCount()

    def depth(self):
        return max(self.net1.depth(), self.net2.depth())

    def printNet(self, f):
        None
        self.net1.printNet(f)
        None
        self.net2.printNet(f)
        None

    def showNet(self, t=''):
        None
        self.net1.showNet('    ' + t)
        None
        self.net2.showNet('    ' + t)
        None


class ToZono(Identity):

    def init(self, in_shape, customRelu=None, only_train=False, **kargs):
        self.customRelu = customRelu
        self.only_train = only_train
        return in_shape

    def forward(self, x, **kargs):
        return self.abstract_forward(x, **kargs) if self.training or not self.only_train else x

    def abstract_forward(self, x, **kargs):
        return x.abstractApplyLeaf('hybrid_to_zono', customRelu=self.customRelu)

    def showNet(self, t=''):
        None


class CorrelateAll(ToZono):

    def abstract_forward(self, x, **kargs):
        return x.abstractApplyLeaf('hybrid_to_zono', correlate=True, customRelu=self.customRelu)


class ToHZono(ToZono):

    def abstract_forward(self, x, **kargs):
        return x.abstractApplyLeaf('zono_to_hybrid', customRelu=self.customRelu)


class Concretize(ToZono):

    def init(self, in_shape, only_train=True, **kargs):
        self.only_train = only_train
        return in_shape

    def abstract_forward(self, x, **kargs):
        return x.abstractApplyLeaf('concretize')


class CorrRand(Concretize):

    def init(self, in_shape, num_correlate, only_train=True, **kargs):
        self.only_train = only_train
        self.num_correlate = num_correlate
        return in_shape

    def abstract_forward(self, x):
        return x.abstractApplyLeaf('stochasticCorrelate', self.num_correlate)

    def showNet(self, t=''):
        None


class CorrMaxK(CorrRand):

    def abstract_forward(self, x):
        return x.abstractApplyLeaf('correlateMaxK', self.num_correlate)


class CorrFix(Concretize):

    def init(self, in_shape, k, only_train=True, **kargs):
        self.k = k
        self.only_train = only_train
        return in_shape

    def abstract_forward(self, x):
        sz = x.size()
        """
        # for more control in the future
        indxs_1 = torch.arange(start = 0, end = sz[1], step = math.ceil(sz[1] / self.dims[1]) )
        indxs_2 = torch.arange(start = 0, end = sz[2], step = math.ceil(sz[2] / self.dims[2]) )
        indxs_3 = torch.arange(start = 0, end = sz[3], step = math.ceil(sz[3] / self.dims[3]) )

        indxs = torch.stack(torch.meshgrid((indxs_1,indxs_2,indxs_3)), dim=3).view(-1,3)
        """
        szm = h.product(sz[1:])
        indxs = torch.arange(start=0, end=szm, step=math.ceil(szm / self.k))
        indxs = indxs.unsqueeze(0).expand(sz[0], indxs.size()[0])
        return x.abstractApplyLeaf('correlate', indxs)

    def showNet(self, t=''):
        None


class DecorrRand(Concretize):

    def init(self, in_shape, num_decorrelate, only_train=True, **kargs):
        self.only_train = only_train
        self.num_decorrelate = num_decorrelate
        return in_shape

    def abstract_forward(self, x):
        return x.abstractApplyLeaf('stochasticDecorrelate', self.num_decorrelate)


class DecorrMin(Concretize):

    def init(self, in_shape, num_decorrelate, only_train=True, num_to_keep=False, **kargs):
        self.only_train = only_train
        self.num_decorrelate = num_decorrelate
        self.num_to_keep = num_to_keep
        return in_shape

    def abstract_forward(self, x):
        return x.abstractApplyLeaf('decorrelateMin', self.num_decorrelate, num_to_keep=self.num_to_keep)

    def showNet(self, t=''):
        None


class DeepLoss(ToZono):

    def init(self, in_shape, bw=0.01, act=F.relu, **kargs):
        self.only_train = True
        self.bw = S.Const.initConst(bw)
        self.act = act
        return in_shape

    def abstract_forward(self, x, **kargs):
        if x.isPoint():
            return x
        return ai.TaggedDomain(x, self.MLoss(self, x))


    class MLoss:

        def __init__(self, obj, x):
            self.obj = obj
            self.x = x

        def loss(self, a, *args, lr=1, time=0, **kargs):
            bw = self.obj.bw.getVal(time=time)
            pre_loss = a.loss(*args, time=time, **kargs, lr=lr * (1 - bw))
            if bw <= 0.0:
                return pre_loss
            return (1 - bw) * pre_loss + bw * self.x.deep_loss(act=self.obj.act)

    def showNet(self, t=''):
        None


class IdentLoss(DeepLoss):

    def abstract_forward(self, x, **kargs):
        return x


class AbstractNet(nn.Module):

    def __init__(self, domain, net, abstractNet):
        super(AbstractNet, self).__init__()
        self.net = net
        self.abstractNet = abstractNet
        if hasattr(domain, 'net') and domain.net is not None:
            self.netDom = domain.net

    def forward(self, inpt):
        return self.abstractNet(inpt)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AbstractNet,
     lambda: ([], {'domain': 4, 'net': 4, 'abstractNet': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_eth_sri_diffai(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

