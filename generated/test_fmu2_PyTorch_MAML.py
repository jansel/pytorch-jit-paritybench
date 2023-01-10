import sys
_module = sys.modules[__name__]
del sys
datasets = _module
cifar100 = _module
cub200 = _module
datasets = _module
inatural = _module
mini_imagenet = _module
tiered_imagenet = _module
transforms = _module
models = _module
classifiers = _module
classifiers = _module
logistic = _module
encoders = _module
convnet4 = _module
encoders = _module
resnet12 = _module
resnet18 = _module
maml = _module
modules = _module
test = _module
train = _module
utils = _module
optimizers = _module

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


from torch.utils.data import Dataset


import numpy as np


import torch.nn as nn


from collections import OrderedDict


import torch.nn.functional as F


import torch.autograd as autograd


import torch.utils.checkpoint as cp


import re


import random


from torch.utils.data import DataLoader


from torch.optim import SGD


from torch.optim import RMSprop


from torch.optim import Adam


from torch.optim.lr_scheduler import MultiStepLR


from torch.optim.lr_scheduler import CosineAnnealingLR


class Linear(nn.Linear, Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, x, params=None, episode=None):
        if params is None:
            x = super(Linear, self).forward(x)
        else:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
            x = F.linear(x, weight, bias)
        return x


def get_child_dict(params, key=None):
    """
  Constructs parameter dictionary for a network module.

  Args:
    params (dict): a parent dictionary of named parameters.
    key (str, optional): a key that specifies the root of the child dictionary.

  Returns:
    child_dict (dict): a child dictionary of model parameters.
  """
    if params is None:
        return None
    if key is None or isinstance(key, str) and key == '':
        return params
    key_re = re.compile('^{0}\\.(.+)'.format(re.escape(key)))
    if not any(filter(key_re.match, params.keys())):
        key_re = re.compile('^module\\.{0}\\.(.+)'.format(re.escape(key)))
    child_dict = OrderedDict((key_re.sub('\\1', k), value) for k, value in params.items() if key_re.match(k) is not None)
    return child_dict


models = {}


def register(name):

    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


class LogisticClassifier(Module):

    def __init__(self, in_dim, n_way, temp=1.0, learn_temp=False):
        super(LogisticClassifier, self).__init__()
        self.in_dim = in_dim
        self.n_way = n_way
        self.temp = temp
        self.learn_temp = learn_temp
        self.linear = Linear(in_dim, n_way)
        if self.learn_temp:
            self.temp = nn.Parameter(torch.tensor(temp))

    def reset_parameters(self):
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x_shot, params=None):
        assert x_shot.dim() == 2
        logits = self.linear(x_shot, get_child_dict(params, 'linear'))
        logits = logits * self.temp
        return logits


class BatchNorm2d(nn.BatchNorm2d, Module):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, episodic=False, n_episode=4, alpha=False):
        """
    Args:
      episodic (bool, optional): if True, maintains running statistics for 
        each episode separately. It is ignored if track_running_stats=False. 
        Default: True
      n_episode (int, optional): number of episodes per mini-batch. It is 
        ignored if episodic=False.
      alpha (bool, optional): if True, learns to interpolate between batch 
        statistics computed over the support set and instance statistics from 
        a query at validation time. Default: True
        (It is ignored if track_running_stats=False or meta_learn=False)
    """
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.episodic = episodic
        self.n_episode = n_episode
        self.alpha = alpha
        if self.track_running_stats:
            if self.episodic:
                for ep in range(n_episode):
                    self.register_buffer('running_mean_%d' % ep, torch.zeros(num_features))
                    self.register_buffer('running_var_%d' % ep, torch.ones(num_features))
                    self.register_buffer('num_batches_tracked_%d' % ep, torch.tensor(0, dtype=torch.int))
            if self.alpha:
                self.register_buffer('batch_size', torch.tensor(0, dtype=torch.int))
                self.alpha_scale = nn.Parameter(torch.tensor(0.0))
                self.alpha_offset = nn.Parameter(torch.tensor(0.0))

    def is_episodic(self):
        return self.episodic

    def _batch_norm(self, x, mean, var, weight=None, bias=None):
        if self.affine:
            assert weight is not None and bias is not None
            weight = weight.view(1, -1, 1, 1)
            bias = bias.view(1, -1, 1, 1)
            x = weight * (x - mean) / (var + self.eps) ** 0.5 + bias
        else:
            x = (x - mean) / (var + self.eps) ** 0.5
        return x

    def reset_episodic_running_stats(self, episode):
        if self.episodic:
            getattr(self, 'running_mean_%d' % episode).zero_()
            getattr(self, 'running_var_%d' % episode).fill_(1.0)
            getattr(self, 'num_batches_tracked_%d' % episode).zero_()

    def forward(self, x, params=None, episode=None):
        self._check_input_dim(x)
        if params is not None:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
        else:
            weight, bias = self.weight, self.bias
        if self.track_running_stats:
            if self.episodic:
                assert episode is not None and episode < self.n_episode
                running_mean = getattr(self, 'running_mean_%d' % episode)
                running_var = getattr(self, 'running_var_%d' % episode)
                num_batches_tracked = getattr(self, 'num_batches_tracked_%d' % episode)
            else:
                running_mean, running_var = self.running_mean, self.running_var
                num_batches_tracked = self.num_batches_tracked
            if self.training:
                exp_avg_factor = 0.0
                if self.first_pass:
                    if self.alpha:
                        self.batch_size = x.size(0)
                    num_batches_tracked += 1
                    if self.momentum is None:
                        exp_avg_factor = 1.0 / float(num_batches_tracked)
                    else:
                        exp_avg_factor = self.momentum
                return F.batch_norm(x, running_mean, running_var, weight, bias, True, exp_avg_factor, self.eps)
            elif self.alpha:
                assert self.batch_size > 0
                alpha = torch.sigmoid(self.alpha_scale * self.batch_size + self.alpha_offset)
                running_mean = running_mean.view(1, -1, 1, 1)
                running_var = running_var.view(1, -1, 1, 1)
                sample_mean = torch.mean(x, dim=(2, 3), keepdim=True)
                sample_var = torch.var(x, dim=(2, 3), unbiased=False, keepdim=True)
                mean = alpha * running_mean + (1 - alpha) * sample_mean
                var = alpha * running_var + (1 - alpha) * sample_var + alpha * (1 - alpha) * (sample_mean - running_mean) ** 2
                return self._batch_norm(x, mean, var, weight, bias)
            else:
                return F.batch_norm(x, running_mean, running_var, weight, bias, False, 0.0, self.eps)
        else:
            return F.batch_norm(x, None, None, weight, bias, True, 0.0, self.eps)


class Conv2d(nn.Conv2d, Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, x, params=None, episode=None):
        if params is None:
            x = super(Conv2d, self).forward(x)
        else:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
            x = F.conv2d(x, weight, bias, self.stride, self.padding)
        return x


class ConvBlock(Module):

    def __init__(self, in_channels, out_channels, bn_args):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.bn = BatchNorm2d(out_channels, **bn_args)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, params=None, episode=None):
        out = self.conv(x, get_child_dict(params, 'conv'))
        out = self.bn(out, get_child_dict(params, 'bn'), episode)
        out = self.pool(self.relu(out))
        return out


class Sequential(nn.Sequential, Module):

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, x, params=None, episode=None):
        if params is None:
            for module in self:
                x = module(x, None, episode)
        else:
            for name, module in self._modules.items():
                x = module(x, get_child_dict(params, name), episode)
        return x


class ConvNet4(Module):

    def __init__(self, hid_dim, out_dim, bn_args):
        super(ConvNet4, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        episodic = bn_args.get('episodic') or []
        bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
        bn_args_ep['episodic'] = True
        bn_args_no_ep['episodic'] = False
        bn_args_dict = dict()
        for i in [1, 2, 3, 4]:
            if 'conv%d' % i in episodic:
                bn_args_dict[i] = bn_args_ep
            else:
                bn_args_dict[i] = bn_args_no_ep
        self.encoder = Sequential(OrderedDict([('conv1', ConvBlock(3, hid_dim, bn_args_dict[1])), ('conv2', ConvBlock(hid_dim, hid_dim, bn_args_dict[2])), ('conv3', ConvBlock(hid_dim, hid_dim, bn_args_dict[3])), ('conv4', ConvBlock(hid_dim, out_dim, bn_args_dict[4]))]))

    def get_out_dim(self, scale=25):
        return self.out_dim * scale

    def forward(self, x, params=None, episode=None):
        out = self.encoder(x, get_child_dict(params, 'encoder'), episode)
        out = out.view(out.shape[0], -1)
        return out


def conv1x1(in_channels, out_channels, stride=1):
    return Conv2d(in_channels, out_channels, 1, stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)


class Block(Module):

    def __init__(self, in_planes, planes, stride, bn_args):
        super(Block, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = BatchNorm2d(planes, **bn_args)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, **bn_args)
        if stride > 1:
            self.res_conv = Sequential(OrderedDict([('conv', conv1x1(in_planes, planes)), ('bn', BatchNorm2d(planes, **bn_args))]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, params=None, episode=None):
        out = self.conv1(x, get_child_dict(params, 'conv1'))
        out = self.bn1(out, get_child_dict(params, 'bn1'), episode)
        out = self.relu(out)
        out = self.conv2(out, get_child_dict(params, 'conv2'))
        out = self.bn2(out, get_child_dict(params, 'bn2'), episode)
        if self.stride > 1:
            x = self.res_conv(x, get_child_dict(params, 'res_conv'), episode)
        out = self.relu(out + x)
        return out


class ResNet12(Module):

    def __init__(self, channels, bn_args):
        super(ResNet12, self).__init__()
        self.channels = channels
        episodic = bn_args.get('episodic') or []
        bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
        bn_args_ep['episodic'] = True
        bn_args_no_ep['episodic'] = False
        bn_args_dict = dict()
        for i in [1, 2, 3, 4]:
            if 'layer%d' % i in episodic:
                bn_args_dict[i] = bn_args_ep
            else:
                bn_args_dict[i] = bn_args_no_ep
        self.layer1 = Block(3, channels[0], bn_args_dict[1])
        self.layer2 = Block(channels[0], channels[1], bn_args_dict[2])
        self.layer3 = Block(channels[1], channels[2], bn_args_dict[3])
        self.layer4 = Block(channels[2], channels[3], bn_args_dict[4])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels[3]
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def get_out_dim(self):
        return self.out_dim

    def forward(self, x, params=None, episode=None):
        out = self.layer1(x, get_child_dict(params, 'layer1'), episode)
        out = self.layer2(out, get_child_dict(params, 'layer2'), episode)
        out = self.layer3(out, get_child_dict(params, 'layer3'), episode)
        out = self.layer4(out, get_child_dict(params, 'layer4'), episode)
        out = self.pool(out).flatten(1)
        return out


class ResNet18(Module):

    def __init__(self, channels, bn_args):
        super(ResNet18, self).__init__()
        self.channels = channels
        episodic = bn_args.get('episodic') or []
        bn_args_ep, bn_args_no_ep = bn_args.copy(), bn_args.copy()
        bn_args_ep['episodic'] = True
        bn_args_no_ep['episodic'] = False
        bn_args_dict = dict()
        for i in [0, 1, 2, 3, 4]:
            if 'layer%d' % i in episodic:
                bn_args_dict[i] = bn_args_ep
            else:
                bn_args_dict[i] = bn_args_no_ep
        self.layer0 = Sequential(OrderedDict([('conv', conv3x3(3, 64)), ('bn', BatchNorm2d(64, **bn_args_dict[0]))]))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = Block(64, channels[0], 1, bn_args_dict[1])
        self.layer2 = Block(channels[0], channels[1], 2, bn_args_dict[2])
        self.layer3 = Block(channels[1], channels[2], 2, bn_args_dict[3])
        self.layer4 = Block(channels[2], channels[3], 2, bn_args_dict[4])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = channels[3]
        for m in self.modules():
            if isinstance(m, Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def get_out_dim(self, scale=1):
        return self.out_dim * scale

    def forward(self, x, params=None, episode=None):
        out = self.layer0(x, get_child_dict(params, 'layer0'), episode)
        out = self.relu(out)
        out = self.layer1(out, get_child_dict(params, 'layer1'), episode)
        out = self.layer2(out, get_child_dict(params, 'layer2'), episode)
        out = self.layer3(out, get_child_dict(params, 'layer3'), episode)
        out = self.layer4(out, get_child_dict(params, 'layer4'), episode)
        out = self.pool(out).flatten(1)
        return out


class MAML(Module):

    def __init__(self, encoder, classifier):
        super(MAML, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def reset_classifier(self):
        self.classifier.reset_parameters()

    def _inner_forward(self, x, params, episode):
        """ Forward pass for the inner loop. """
        feat = self.encoder(x, get_child_dict(params, 'encoder'), episode)
        logits = self.classifier(feat, get_child_dict(params, 'classifier'))
        return logits

    def _inner_iter(self, x, y, params, mom_buffer, episode, inner_args, detach):
        """ 
    Performs one inner-loop iteration of MAML including the forward and 
    backward passes and the parameter update.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): the model parameters BEFORE the update.
      mom_buffer (dict): the momentum buffer BEFORE the update.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      detach (bool): if True, detachs the graph for the current iteration.

    Returns:
      updated_params (dict): the model parameters AFTER the update.
      mom_buffer (dict): the momentum buffer AFTER the update.
    """
        with torch.enable_grad():
            logits = self._inner_forward(x, params, episode)
            loss = F.cross_entropy(logits, y)
            grads = autograd.grad(loss, params.values(), create_graph=not detach and not inner_args['first_order'], only_inputs=True, allow_unused=True)
            updated_params = OrderedDict()
            for (name, param), grad in zip(params.items(), grads):
                if grad is None:
                    updated_param = param
                else:
                    if inner_args['weight_decay'] > 0:
                        grad = grad + inner_args['weight_decay'] * param
                    if inner_args['momentum'] > 0:
                        grad = grad + inner_args['momentum'] * mom_buffer[name]
                        mom_buffer[name] = grad
                    if 'encoder' in name:
                        lr = inner_args['encoder_lr']
                    elif 'classifier' in name:
                        lr = inner_args['classifier_lr']
                    else:
                        raise ValueError('invalid parameter name')
                    updated_param = param - lr * grad
                if detach:
                    updated_param = updated_param.detach().requires_grad_(True)
                updated_params[name] = updated_param
        return updated_params, mom_buffer

    def _adapt(self, x, y, params, episode, inner_args, meta_train):
        """
    Performs inner-loop adaptation in MAML.

    Args:
      x (float tensor, [n_way * n_shot, C, H, W]): per-episode support set.
        (T: transforms, C: channels, H: height, W: width)
      y (int tensor, [n_way * n_shot]): per-episode support set labels.
      params (dict): a dictionary of parameters at meta-initialization.
      episode (int): the current episode index.
      inner_args (dict): inner-loop optimization hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      params (dict): model paramters AFTER inner-loop adaptation.
    """
        assert x.dim() == 4 and y.dim() == 1
        assert x.size(0) == y.size(0)
        mom_buffer = OrderedDict()
        if inner_args['momentum'] > 0:
            for name, param in params.items():
                mom_buffer[name] = torch.zeros_like(param)
        params_keys = tuple(params.keys())
        mom_buffer_keys = tuple(mom_buffer.keys())
        for m in self.modules():
            if isinstance(m, BatchNorm2d) and m.is_episodic():
                m.reset_episodic_running_stats(episode)

        def _inner_iter_cp(episode, *state):
            """ 
      Performs one inner-loop iteration when checkpointing is enabled. 
      The code is executed twice:
        - 1st time with torch.no_grad() for creating checkpoints.
        - 2nd time with torch.enable_grad() for computing gradients.
      """
            params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
            mom_buffer = OrderedDict(zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
            detach = not torch.is_grad_enabled()
            self.is_first_pass(detach)
            params, mom_buffer = self._inner_iter(x, y, params, mom_buffer, int(episode), inner_args, detach)
            state = tuple(t if t.requires_grad else t.clone().requires_grad_(True) for t in tuple(params.values()) + tuple(mom_buffer.values()))
            return state
        for step in range(inner_args['n_step']):
            if self.efficient:
                state = tuple(params.values()) + tuple(mom_buffer.values())
                state = cp.checkpoint(_inner_iter_cp, torch.as_tensor(episode), *state)
                params = OrderedDict(zip(params_keys, state[:len(params_keys)]))
                mom_buffer = OrderedDict(zip(mom_buffer_keys, state[-len(mom_buffer_keys):]))
            else:
                params, mom_buffer = self._inner_iter(x, y, params, mom_buffer, episode, inner_args, not meta_train)
        return params

    def forward(self, x_shot, x_query, y_shot, inner_args, meta_train):
        """
    Args:
      x_shot (float tensor, [n_episode, n_way * n_shot, C, H, W]): support sets.
      x_query (float tensor, [n_episode, n_way * n_query, C, H, W]): query sets.
        (T: transforms, C: channels, H: height, W: width)
      y_shot (int tensor, [n_episode, n_way * n_shot]): support set labels.
      inner_args (dict, optional): inner-loop hyperparameters.
      meta_train (bool): if True, the model is in meta-training.
      
    Returns:
      logits (float tensor, [n_episode, n_way * n_shot, n_way]): predicted logits.
    """
        assert self.encoder is not None
        assert self.classifier is not None
        assert x_shot.dim() == 5 and x_query.dim() == 5
        assert x_shot.size(0) == x_query.size(0)
        params = OrderedDict(self.named_parameters())
        for name in list(params.keys()):
            if not params[name].requires_grad or any(s in name for s in inner_args['frozen'] + ['temp']):
                params.pop(name)
        logits = []
        for ep in range(x_shot.size(0)):
            self.train()
            if not meta_train:
                for m in self.modules():
                    if isinstance(m, BatchNorm2d) and not m.is_episodic():
                        m.eval()
            updated_params = self._adapt(x_shot[ep], y_shot[ep], params, ep, inner_args, meta_train)
            with torch.set_grad_enabled(meta_train):
                self.eval()
                logits_ep = self._inner_forward(x_query[ep], updated_params, ep)
            logits.append(logits_ep)
        self.train(meta_train)
        logits = torch.stack(logits)
        return logits


class Module(nn.Module):

    def __init__(self):
        super(Module, self).__init__()
        self.efficient = False
        self.first_pass = True

    def go_efficient(self, mode=True):
        """ Switches on / off gradient checkpointing. """
        self.efficient = mode
        for m in self.children():
            if isinstance(m, Module):
                m.go_efficient(mode)

    def is_first_pass(self, mode=True):
        """ Tracks the progress of forward and backward pass when gradient 
    checkpointing is enabled. """
        self.first_pass = mode
        for m in self.children():
            if isinstance(m, Module):
                m.is_first_pass(mode)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchNorm2d,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Conv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LogisticClassifier,
     lambda: ([], {'in_dim': 4, 'n_way': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (Sequential,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_fmu2_PyTorch_MAML(_paritybench_base):
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

