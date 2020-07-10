import sys
_module = sys.modules[__name__]
del sys
alrao = _module
alrao_model = _module
custom_layers = _module
earlystopping = _module
gen_hyper = _module
learningratesgen = _module
optim_spec = _module
switch = _module
utils = _module
data_text = _module
main_cnn = _module
main_reg = _module
main_rnn = _module
models = _module
googlenet = _module
mobilenetv2 = _module
rnn = _module
senet = _module
vgg = _module
setup = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


import torch.nn.functional as F


import numpy as np


from scipy.stats import ortho_group


import torch.optim as optim


from numbers import Number


from torch.utils.data.dataset import Dataset


from collections import OrderedDict


from torch.utils import data


import torchvision


import torchvision.transforms as transforms


import time


from torch.autograd import Variable


def log_sum_exp(tensor, dim=None):
    """
    Numerically stable implementation of the operation.

    tensor.exp().sum(dim, keepdim).log()
    From https://github.com/pytorch/pytorch/issues/2591
    """
    if dim is not None:
        m, _ = torch.max(tensor, dim=dim, keepdim=True)
        tensor0 = tensor - m
        return m.squeeze(dim=dim) + torch.log(torch.sum(torch.exp(tensor0), dim=dim))
    else:
        m = torch.max(tensor)
        sum_exp = torch.sum(torch.exp(tensor - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


class Switch(nn.Module):
    """
    Model averaging method 'switch'.
    See van Erven and Grünwald (2008):
     (A) https://arxiv.org/abs/0807.1005,
     (B) http://papers.nips.cc/paper/3277-catching-up-faster-in-bayesian-model-selection-and-model-averaging.pdf.

    This class manages model averaging and updates its parameters using the update
        rule given in algorithm 1 of (B).

    Parameters:
        loss: loss used in the model (subclass of pytorch's '_Loss')
            loss(output, target, size_average = False) returns the loss embedded into a 0-dim tensor
            the option 'size_average = True' returns the averaged loss
    """

    def __init__(self, nb_models, theta=0.9999, alpha=0.001, save_ll_perf=False, task='classification', loss=None):
        super(Switch, self).__init__()
        self.task = task
        if task == 'classification' and loss is None:
            self.loss = F.nll_loss
        elif loss is not None:
            self.loss = loss
        else:
            raise ValueError('Invalid combination task/loss: {} / {}.'.format(task, loss))
        self.nb_models = nb_models
        self.theta = theta
        self.alpha = alpha
        self.t = 1
        self.save_ll_perf = save_ll_perf
        self.register_buffer('logw', torch.zeros((2, nb_models), requires_grad=False))
        self.register_buffer('logposterior', torch.full((nb_models,), -np.log(nb_models), requires_grad=False))
        self.logw[0].fill_(np.log(theta))
        self.logw[1].fill_(np.log(1 - theta))
        self.logw -= np.log(nb_models)
        if self.save_ll_perf:
            self.reset_ll_perf()

    def reset_ll_perf(self):
        """
        Resets the performance record of last layers models
        """
        self.ll_loss = [(0) for _ in range(self.nb_models)]
        self.ll_correct = [(0) for _ in range(self.nb_models)]
        self.ll_total = 0

    def get_ll_perf(self):
        """
        Return the performance (loss and acc) of each last layer
        """
        if self.task == 'classification':
            return [(loss / self.ll_total, corr / self.ll_total) for loss, corr in zip(self.ll_loss, self.ll_correct)]
        elif self.task == 'regression':
            return [(loss / self.ll_total) for loss in self.ll_loss]

    def piT(self, t):
        """
        Prior  pi_T in algorithm 1 of (B).
        """
        return 1 / (t + 1)

    def Supdate(self, lst_ll_out, y):
        """
        Switch update rule given in algorithm 1 of (B).

        Arguments:
            lst_ll_out: list of the outputs of the models, which are supposed to be
                tensors of log-probabilities
            y: tensor of targets
        """
        if self.save_ll_perf:
            self.ll_total += 1
            for k, x in enumerate(lst_ll_out):
                self.ll_loss[k] += self.loss(x, y).item()
                if self.task == 'classification':
                    self.ll_correct[k] += torch.max(x, 1)[1].eq(y.data).sum().item() / y.size(0)
        logpx = torch.stack([(-self.loss(x, y)) for x in lst_ll_out], dim=0).detach()
        if any(math.isnan(p) for p in logpx):
            raise ValueError
        if self.nb_models == 1:
            return
        self.logw += logpx
        pit = self.piT(self.t)
        logpool = log_sum_exp(self.logw[0]) + np.log(pit)
        self.logw[0] += np.log(1 - pit)
        addtensor = torch.zeros_like(self.logw)
        addtensor[0].fill_(np.log(self.theta))
        addtensor[1].fill_(np.log(1 - self.theta))
        self.logw = log_sum_exp(torch.stack([self.logw, addtensor + logpool - np.log(self.nb_models)], dim=0), dim=0)
        self.logw -= log_sum_exp(self.logw)
        self.logposterior = log_sum_exp(self.logw, dim=0)
        self.t += 1

    def forward(self, lst_ll_out):
        """
        Computes the average of the outputs of the different models.

        Arguments:
            lst_ll_out: list of the outputs of the models, which are supposed to be
                tensors of log-probabilities
        """
        if self.task == 'classification':
            return log_sum_exp(torch.stack(lst_ll_out, -1) + self.logposterior, dim=-1)
        elif self.task == 'regression':
            return torch.stack(lst_ll_out, -1), self.logposterior.exp()


class AlraoModel(nn.Module):
    """
    AlraoModel is the class transforming a internal NN into a model learnable with Alrao.

    Arguments:
        internal_nn: part of the neural network preceding the last layer.
        n_last_layers: number of parallel last layers to use with the model averaging method
        n_classes: number of classes in the classification task
        last_layer_gen: python class to use to construct the last layers
        task: either 'classification' or 'regression'
        loss: loss used in the model (subclass of pytorch's '_Loss')
            loss(output, target, size_average = False) returns the loss embedded into a 0-dim tensor
            the option 'size_average = True' returns the averaged loss
        *args, **kwargs: arguments to be passed to the constructor of 'last_layer_gen'
    """

    def __init__(self, task, loss, internal_nn, n_last_layers, last_layer_gen, *args, **kwargs):
        super(AlraoModel, self).__init__()
        self.task = task
        self.loss = loss
        self.switch = Switch(n_last_layers, save_ll_perf=True, task=task, loss=loss)
        self.internal_nn = internal_nn
        self.n_last_layers = n_last_layers
        for i in range(n_last_layers):
            last_layer = last_layer_gen(*args, **kwargs)
            setattr(self, 'last_layer_' + str(i), last_layer)
        self.last_x, self.last_lst_logpx = None, None

    def method_fwd_internal_nn(self, method_name_src, method_name_dst=None):
        """
        Allows the user to call directly a method of the internal NN.

        Creates a new method for the called instance of AlraoModel named 'method_name_dst'.
        Calling this method is exactly equivalent to calling the method of
        'self.internal_nn' named 'method_name_src'.
        If 'method_name_dst' is left to 'None', 'method_name_dst' is set to 'method_name_src'.

        Example:
            am = AlraoModel(intnn, nll, ll_gen)
            am.method_fwd_internal_nn('some_method')
            # call 'some_method' by the usual way
            am.internal_nn.some_method(some_args)
            # call 'some_method' using forwarding
            am.some_method(som_args)

        Arguments:
            method_name_src: name of the method of the internal NN to bind
            method_name_dst: name of the method to call
        """
        if method_name_dst is None:
            method_name_dst = method_name_src
        assert getattr(self, method_name_dst, None) is None, 'The method {} cannot be forwarded: an attribute with the same name already exists.'.format(method_name_dst)
        method = getattr(self.internal_nn, method_name_src)

        def forwarded_method(*args, **kwargs):
            return method(*args, **kwargs)
        forwarded_method.__doc__ = method.__doc__
        forwarded_method.__name__ = method_name_dst
        setattr(self, forwarded_method.__name__, forwarded_method)

    def method_fwd_last_layers(self, method_name_src, method_name_dst=None):
        """
        Allows the user to call directly a method of the last layers.

        Creates a new method for the called instance of AlraoModel named 'method_name_dst'.
        Calling this method is exactly equivalent to calling the method of the last layers named 'method_name_src'
            on each of them.
        If 'method_name_dst' is left to 'None', 'method_name_dst' is set to 'method_name_src'.

        Example:
            am = AlraoModel(intnn, nll, ll_gen)
            am.method_fwd_last_layers('some_method')
            # call 'some_method' by the usual way
            for ll in am.last_layers():
                ll.some_method(some_args)
            # call 'some_method' using forwarding
            am.some_method(som_args)

        Arguments:
            method_name_src: name of the method of the last layers to bind
            method_name_dst: name of the method to call
        """
        if method_name_dst is None:
            method_name_dst = method_name_src
        assert getattr(self, method_name_src, None) is None, 'The method {} cannot be forwarded: an attribute with the same name already exists.'.format(method_name_dst)
        lst_methods = [getattr(ll, method_name_src) for ll in self.last_layers()]

        def forwarded_method(*args, **kwargs):
            return [method(*args, **kwargs) for method in lst_methods]
        forwarded_method.__doc__ = lst_methods[0].__doc__
        forwarded_method.__name__ = method_name_dst
        setattr(self, forwarded_method.__name__, forwarded_method)

    def reset_parameters(self):
        """
        Resets both the last layers and internal NN's parameters.
        """
        self.internal_nn.reset_parameters()
        for ll in self.last_layers():
            ll.reset_parameters()

    def forward(self, *args, **kwargs):
        """
        Gives an input to the internal NN, then gives its output to each last layer,
            averages their output with 'switch', a model averaging method.

        The output 'x' of the internal NN is either a scalar or a tuple:
            - 'x' is a scalar: 'x' is used as input of each last layer
            - 'x' is a tuple: 'x[0]' is used as input of each last layer

        Arguments:
            *args, **kwargs: arguments to be passed to the forward method of the internal NN
        """
        x = self.internal_nn(*args, **kwargs)
        z = x
        if isinstance(z, tuple):
            z = x[0]
        if not torch.isfinite(z).all():
            raise ValueError
        lst_ll_out = [ll(z) for ll in self.last_layers()]
        self.last_x, self.last_lst_ll_out = z, lst_ll_out
        out = self.switch.forward(lst_ll_out)
        if isinstance(x, tuple):
            out = (out,) + x[1:]
        return out

    def update_switch(self, y, x=None, catch_up=False):
        """
        Updates the model averaging weights

        Arguments: 
            y: tensor of targets
            x: tensor of outputs of the internal NN
                if x is None, the stored outputs of the last layers are used
        """
        if x is None:
            lst_ll_out = self.last_lst_ll_out
        else:
            lst_ll_out = [ll(x) for ll in self.last_layers()]
        self.switch.Supdate(lst_ll_out, y)
        if catch_up:
            self.hard_catch_up()

    def hard_catch_up(self, threshold=-20):
        """
        The hard catch up allows to reset all the last layers with low performance, and to
        set their weights to the best last layer ones. This can be done periodically during learning
        """
        logpost = self.switch.logposterior
        weak_ll = [ll for ll, lp in zip(self.last_layers(), logpost) if lp < threshold]
        if not weak_ll:
            return
        mean_weight = torch.stack([(ll.fc.weight * p) for ll, p in zip(self.last_layers(), logpost.exp())], dim=-1).sum(dim=-1).detach()
        mean_bias = torch.stack([(ll.fc.bias * p) for ll, p in zip(self.last_layers(), logpost.exp())], dim=-1).sum(dim=-1).detach()
        for ll in weak_ll:
            ll.fc.weight.data = mean_weight.clone()
            ll.fc.bias.data = mean_bias.clone()

    def parameters_internal_nn(self):
        """
        Iterator over the internal NN parameters
        """
        return self.internal_nn.parameters()

    def last_layers(self):
        """
        Iterator over the last layers
        """
        for i in range(self.n_last_layers):
            yield getattr(self, 'last_layer_' + str(i))

    def last_layers_parameters_list(self):
        """
        List of iterators, each one over a last layer parameters list.
        """
        return [ll.parameters() for ll in self.last_layers()]

    def posterior(self):
        """
        Return the switch posterior over the last_layers
        """
        return self.switch.logposterior.exp()

    def last_layers_predictions(self, x=None):
        """
        Return all the predictions, for each last layer.
        If x is None, the last predictions are returned.
        """
        if x is None:
            return self.last_lst_logpx
        x = self.internal_nn(x)
        lst_px = [ll(x) for ll in self.last_layers()]
        self.last_lst_logpx = lst_px
        return lst_px

    def repr_posterior(self):
        """
        Compact string representation of the posterior
        """
        post = self.posterior()
        bars = u' ▁▂▃▄▅▆▇█'
        res = '|' + ''.join(bars[int(px)] for px in post / post.max() * 8) + '|'
        return res


class LinearClassifier(nn.Module):
    """
    Linear classifier layer: a linear layer followed by a log_softmax activation
    """

    def __init__(self, in_features, n_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        """
        Forward pass method
        """
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


class LinearClassifierRNN(nn.Module):
    """
    Linear classifier layer for RNNs: a decoder (linear layer) followed by a log_softmax activation
    """

    def __init__(self, nhid, ntoken):
        super(LinearClassifierRNN, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.nhid = nhid
        self.ntoken = ntoken

    def init_weights(self):
        """
        Initialization
        """
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, output):
        """
        Forward pass method: flattening of the output (specific to RNNs), which is processed by a
            linear layer followed by a log_softmax activation
        """
        ret = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return F.log_softmax(ret, dim=1)


class LinearRegressor(nn.Module):
    """
    Linear final layer of a regressor
    """

    def __init__(self, dim_input, dim_output):
        super(LinearRegressor, self).__init__()
        self.layer = nn.Linear(dim_input, dim_output)

    def forward(self, output):
        y = self.layer(output)
        if not torch.isfinite(y).all():
            raise ValueError
        return y


class _Loss(nn.Module):

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class L2LossLog(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean', sigma2=1.0):
        super(L2LossLog, self).__init__(size_average, reduce, reduction)
        self.sigma2 = sigma2

    def forward(self, input, target):
        if not torch.isfinite(input).all():
            raise ValueError
        return ((input - target).pow(2).sum(1) / (2 * self.sigma2)).mean() + 0.5 * math.log(2 * math.pi * self.sigma2)


class L2LossAdditional(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction='mean', sigma2=1.0):
        super(L2LossAdditional, self).__init__(size_average, reduce, reduction)
        self.sigma2 = sigma2

    def forward(self, input, target):
        means, ps = input
        if not torch.isfinite(means).all():
            raise ValueError
        means = means.transpose(2, 1).transpose(1, 0)
        log_probas_per_cl = -(means - target).pow(2).sum(2) / (2 * self.sigma2) - 0.5 * math.log(2 * math.pi * self.sigma2)
        log_probas_per_cl = log_probas_per_cl.transpose(0, 1)
        log_probas_per_cl = log_probas_per_cl + ps.log()
        log_probas = log_sum_exp(log_probas_per_cl, dim=1)
        return -log_probas.mean()


class StandardModel(nn.Module):

    def __init__(self, internal_nn, classifier, *args, **kwargs):
        super(StandardModel, self).__init__()
        self.internal_nn = internal_nn
        self.classifier = classifier(*args, **kwargs)

    def forward(self, *args, **kwargs):
        x = self.internal_nn(*args, **kwargs)
        if isinstance(x, tuple):
            x_0 = self.classifier(x[0])
            return (x_0,) + x[1:]
        else:
            return self.classifier(x)


class RegModel(nn.Module):

    def __init__(self, input_dim, pre_output_dim):
        super(RegModel, self).__init__()
        self.layer = nn.Linear(input_dim, pre_output_dim)
        self.relu = nn.ReLU()
        self.linearinputdim = pre_output_dim

    def forward(self, x):
        x = self.layer(x)
        x = self.relu(x)
        return x


class StandardModelReg(nn.Module):

    def __init__(self, internal_nn):
        super(StandardModelReg, self).__init__()
        self.internal_nn = internal_nn
        self.regressor = LinearRegressor(self.internal_nn.linearinputdim, 1)

    def forward(self, x):
        x = self.internal_nn(x)
        x = self.regressor(x)
        return x.unsqueeze(2), x.new().resize_(1).fill_(1.0)


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, gamma=1):
        super(Inception, self).__init__()
        in_planes *= gamma
        n1x1 *= gamma
        n3x3red *= gamma
        n3x3 *= gamma
        n5x5red *= gamma
        n5x5 *= gamma
        pool_planes *= gamma
        self.b1 = nn.Sequential(nn.Conv2d(in_planes, n1x1, kernel_size=1), nn.BatchNorm2d(n1x1), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(in_planes, n3x3red, kernel_size=1), nn.BatchNorm2d(n3x3red), nn.ReLU(True), nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1), nn.BatchNorm2d(n3x3), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(in_planes, n5x5red, kernel_size=1), nn.BatchNorm2d(n5x5red), nn.ReLU(True), nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(True), nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1), nn.BatchNorm2d(n5x5), nn.ReLU(True))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(in_planes, pool_planes, kernel_size=1), nn.BatchNorm2d(pool_planes), nn.ReLU(True))

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):

    def __init__(self, gamma=1):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(nn.Conv2d(3, gamma * 192, kernel_size=3, padding=1), nn.BatchNorm2d(gamma * 192), nn.ReLU(True))
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32, gamma=gamma)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64, gamma=gamma)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64, gamma=gamma)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64, gamma=gamma)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64, gamma=gamma)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64, gamma=gamma)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128, gamma=gamma)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128, gamma=gamma)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128, gamma=gamma)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linearinputdim = gamma * 1024

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out


class Block(nn.Module):
    """expand + depthwise + pointwise"""

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=10, gamma=1):
        super(MobileNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, gamma * 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(gamma * 32)
        self.layers = self._make_layers(32, gamma)
        self.conv2 = nn.Conv2d(gamma * 320, gamma * 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(gamma * 1280)
        self.linearinputdim = gamma * 1280

    def _make_layers(self, in_planes, gamma):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(gamma * in_planes, gamma * out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.linearinputdim = nhid
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, current):
        emb = self.drop(self.encoder(input))
        output, (new_hidden, new_current) = self.rnn(emb, (hidden, current))
        output = self.drop(output)
        return output, new_hidden, new_current

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return weight.new_zeros(self.nlayers, bsz, self.nhid), weight.new_zeros(self.nlayers, bsz, self.nhid)
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        out = out * w
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = F.sigmoid(self.fc2(w))
        out = out * w
        out += shortcut
        return out


class SENet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, gamma=1):
        super(SENet, self).__init__()
        self.in_planes = gamma * 64
        self.conv1 = nn.Conv2d(3, gamma * 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(gamma * 64)
        self.layer1 = self._make_layer(block, gamma * 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, gamma * 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, gamma * 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, gamma * 512, num_blocks[3], stride=2)
        self.linearinputdim = gamma * 512

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.linearinputdim = 512

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AlraoModel,
     lambda: ([], {'task': 4, 'loss': MSELoss(), 'internal_nn': _mock_layer(), 'n_last_layers': 1, 'last_layer_gen': _mock_layer}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (BasicBlock,
     lambda: ([], {'in_planes': 4, 'planes': 18}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Block,
     lambda: ([], {'in_planes': 4, 'out_planes': 4, 'expansion': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GoogLeNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (Inception,
     lambda: ([], {'in_planes': 4, 'n1x1': 4, 'n3x3red': 4, 'n3x3': 4, 'n5x5red': 4, 'n5x5': 4, 'pool_planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (L2LossAdditional,
     lambda: ([], {}),
     lambda: ([(torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])), torch.rand([4, 4])], {}),
     False),
    (L2LossLog,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearClassifier,
     lambda: ([], {'in_features': 4, 'n_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearClassifierRNN,
     lambda: ([], {'nhid': 4, 'ntoken': 4}),
     lambda: ([torch.rand([16, 4, 4])], {}),
     True),
    (LinearRegressor,
     lambda: ([], {'dim_input': 4, 'dim_output': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MobileNetV2,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (PreActBlock,
     lambda: ([], {'in_planes': 4, 'planes': 18}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RegModel,
     lambda: ([], {'input_dim': 4, 'pre_output_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SENet,
     lambda: ([], {'block': _mock_layer, 'num_blocks': [4, 4, 4, 4]}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (StandardModel,
     lambda: ([], {'internal_nn': _mock_layer(), 'classifier': _mock_layer}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
]

class Test_leonardblier_alrao(_paritybench_base):
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

