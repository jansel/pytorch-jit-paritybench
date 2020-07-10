import sys
_module = sys.modules[__name__]
del sys
maml_toy = _module
rl = _module
dist_promp = _module
maml_dice = _module
maml_trpo = _module
metasgd_a2c = _module
policies = _module
promp = _module
news_topic_classification = _module
anil_fc100 = _module
maml_miniimagenet = _module
maml_omniglot = _module
meta_mnist = _module
protonet_miniimagenet = _module
reptile_miniimagenet = _module
learn2learn = _module
_version = _module
algorithms = _module
base_learner = _module
maml = _module
meta_sgd = _module
data = _module
utils = _module
gym = _module
async_vec_env = _module
envs = _module
meta_env = _module
mujoco = _module
ant_direction = _module
ant_forward_backward = _module
dummy_mujoco_env = _module
halfcheetah_forward_backward = _module
humanoid_direction = _module
humanoid_forward_backward = _module
particles = _module
particles_2d = _module
subproc_vec_env = _module
text = _module
datasets = _module
news_classification = _module
utils = _module
vision = _module
cifarfs = _module
fc100 = _module
fgvc_aircraft = _module
full_omniglot = _module
mini_imagenet = _module
tiered_imagenet = _module
vgg_flowers = _module
models = _module
transforms = _module
setup = _module
tests = _module
integration = _module
maml_miniimagenet_test_notravis = _module
maml_omniglot_test = _module
protonets_miniimagenet_test_notravis = _module
unit = _module
maml_test = _module
metasgd_test = _module
metadataset_test = _module
task_dataset_test = _module
transforms_test = _module
util_datasets = _module
utils_test = _module
cifarfs_test = _module
fc100_test = _module
fgvc_aircraft_test = _module
tiered_imagenet_test_notravis = _module
vgg_flowers_test = _module

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


import torch as th


from torch import nn


from torch import optim


from torch import distributions as dist


import random


from copy import deepcopy


import numpy as np


import torch


from torch import distributed as dist


from torch.distributions.kl import kl_divergence


from torch import autograd


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import vector_to_parameters


import math


from torch.distributions import Normal


from torch.distributions import Categorical


from torch.nn import functional as F


import torchvision as tv


from torchvision import transforms


from torch.utils.data import DataLoader


from torchvision.datasets import MNIST


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import grad


from torch.utils.data import Dataset


import copy


import torch.utils.data as data


from torchvision.datasets import ImageFolder


from torch.utils.data import ConcatDataset


from torchvision.datasets.omniglot import Omniglot


import scipy.io


from scipy.stats import truncnorm


import re


from torch.utils.data import TensorDataset


import string


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2))

    def forward(self, x):
        return self.model(x)


EPSILON = 1e-06


def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class DiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='relu'):
        super(DiagNormalPolicy, self).__init__()
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        loc = self.mean(state)
        scale = torch.exp(torch.clamp(self.sigma, min=math.log(EPSILON)))
        return Normal(loc=loc, scale=scale)

    def log_prob(self, state, action):
        density = self.density(state)
        return density.log_prob(action).mean(dim=1, keepdim=True)

    def forward(self, state):
        density = self.density(state)
        action = density.sample()
        return action


class CategoricalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None):
        super(CategoricalPolicy, self).__init__()
        if hiddens is None:
            hiddens = [100, 100]
        layers = [linear_init(nn.Linear(input_size, hiddens[0])), nn.ReLU()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(nn.ReLU())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.mean = nn.Sequential(*layers)
        self.input_size = input_size

    def forward(self, state):
        state = ch.onehot(state, dim=self.input_size)
        loc = self.mean(state)
        density = Categorical(logits=loc)
        action = density.sample()
        log_prob = density.log_prob(action).mean().view(-1, 1).detach()
        return action, {'density': density, 'log_prob': log_prob}


class Net(nn.Module):

    def __init__(self, ways=3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, ways)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), bn, nn.ReLU(), nn.MaxPool2d(2))


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(conv_block(x_dim, hid_dim), conv_block(hid_dim, hid_dim), conv_block(hid_dim, hid_dim), conv_block(hid_dim, z_dim))
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class BaseLearner(nn.Module):

    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.module = module

    def __getattr__(self, attr):
        try:
            return super(BaseLearner, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def clone_module(module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Creates a copy of a module, whose parameters/buffers/submodules
    are created using PyTorch's torch.clone().

    This implies that the computational graph is kept, and you can compute
    the derivatives of the new modules' parameters w.r.t the original
    parameters.

    **Arguments**

    * **module** (Module) - Module to be cloned.

    **Return**

    * (Module) - The cloned module.

    **Example**

    ~~~python
    net = nn.Sequential(Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    clone = clone_module(net)
    error = loss(clone(X), y)
    error.backward()  # Gradients are back-propagate all the way to net.
    ~~~
    """
    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                cloned = module._parameters[param_key].clone()
                clone._parameters[param_key] = cloned
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and clone._buffers[buffer_key].requires_grad:
                clone._buffers[buffer_key] = module._buffers[buffer_key].clone()
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(module._modules[module_key])
    clone = clone._apply(lambda x: x)
    return clone


def maml_update(model, lr, grads=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

    **Description**

    Performs a MAML update on model using grads and lr.
    The function re-routes the Python object, thus avoiding in-place
    operations.

    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.

    **Arguments**

    * **model** (Module) - The model to update.
    * **lr** (float) - The learning rate used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.

    **Example**
    ~~~python
    maml = l2l.algorithms.MAML(Model(), lr=0.1)
    model = maml.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    maml_update(model, lr=0.1, grads)
    ~~~
    """
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            None
        for p, g in zip(params, grads):
            p.grad = g
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - lr * p.grad
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None:
            model._buffers[buffer_key] = buff - lr * buff.grad
    for module_key in model._modules:
        model._modules[module_key] = maml_update(model._modules[module_key], lr=lr, grads=None)
    model._apply(lambda x: x)
    return model


class MAML(BaseLearner):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)

    **Description**

    High-level implementation of *Model-Agnostic Meta-Learning*.

    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt()`
    methods.

    For the first-order version of MAML (i.e. FOMAML), set the `first_order` flag to `True`
    upon initialization.

    **Arguments**

    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Fast adaptation learning rate.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order
        approximation of MAML. (FOMAML)
    * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to `allow_nograd`.
    * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
        parameters that have `requires_grad = False`.

    **References**

    1. Finn et al. 2017. "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks."

    **Example**

    ~~~python
    linear = l2l.algorithms.MAML(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(self, model, lr, first_order=False, allow_unused=None, allow_nograd=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        **Arguments**

        * **loss** (Tensor) - Loss to minimize upon update.
        * **first_order** (bool, *optional*, default=None) - Whether to use first- or
            second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
            of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=None) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order
        if allow_nograd:
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(loss, diff_params, retain_graph=second_order, create_graph=second_order, allow_unused=allow_unused)
            gradients = []
            grad_counter = 0
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss, self.module.parameters(), retain_graph=second_order, create_graph=second_order, allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                None
        self.module = maml_update(self.module, self.lr, gradients)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        """
        **Description**

        Returns a `MAML`-wrapped copy of the module whose parameters and buffers
        are `torch.clone`d from the original module.

        This implies that back-propagating losses on the cloned module will
        populate the buffers of the original module.
        For more information, refer to learn2learn.clone_module().

        **Arguments**

        * **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
            or second-order updates. Defaults to self.first_order.
        * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to self.allow_unused.
        * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
            parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

        """
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module), lr=self.lr, first_order=first_order, allow_unused=allow_unused, allow_nograd=allow_nograd)


def clone_parameters(param_list):
    return [p.clone() for p in param_list]


def meta_sgd_update(model, lrs=None, grads=None):
    """

    **Description**

    Performs a MetaSGD update on model using grads and lrs.
    The function re-routes the Python object, thus avoiding in-place
    operations.

    NOTE: The model itself is updated in-place (no deepcopy), but the
          parameters' tensors are not.

    **Arguments**

    * **model** (Module) - The model to update.
    * **lrs** (list) - The meta-learned learning rates used to update the model.
    * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the gradients in .grad attributes.

    **Example**
    ~~~python
    meta = l2l.algorithms.MetaSGD(Model(), lr=1.0)
    lrs = [th.ones_like(p) for p in meta.model.parameters()]
    model = meta.clone() # The next two lines essentially implement model.adapt(loss)
    grads = autograd.grad(loss, model.parameters(), create_graph=True)
    meta_sgd_update(model, lrs=lrs, grads)
    ~~~
    """
    if grads is not None and lrs is not None:
        for p, lr, g in zip(model.parameters(), lrs, grads):
            p.grad = g
            p._lr = lr
    for param_key in model._parameters:
        p = model._parameters[param_key]
        if p is not None and p.grad is not None:
            model._parameters[param_key] = p - p._lr * p.grad
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad
    for module_key in model._modules:
        model._modules[module_key] = meta_sgd_update(model._modules[module_key])
    return model


class MetaSGD(BaseLearner):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/meta_sgd.py)

    **Description**

    High-level implementation of *Meta-SGD*.

    This class wraps an arbitrary nn.Module and augments it with `clone()` and `adapt`
    methods.
    It behaves similarly to `MAML`, but in addition a set of per-parameters learning rates
    are learned for fast-adaptation.

    **Arguments**

    * **model** (Module) - Module to be wrapped.
    * **lr** (float) - Initialization value of the per-parameter fast adaptation learning rates.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order version.
    * **lrs** (list of Parameters, *optional*, default=None) - If not None, overrides `lr`, and uses the list
        as learning rates for fast-adaptation.

    **References**

    1. Li et al. 2017. “Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.” arXiv.

    **Example**

    ~~~python
    linear = l2l.algorithms.MetaSGD(nn.Linear(20, 10), lr=0.01)
    clone = linear.clone()
    error = loss(clone(X), y)
    clone.adapt(error)
    error = loss(clone(X), y)
    error.backward()
    ~~~
    """

    def __init__(self, model, lr=1.0, first_order=False, lrs=None):
        super(MetaSGD, self).__init__()
        self.module = model
        if lrs is None:
            lrs = [(th.ones_like(p) * lr) for p in model.parameters()]
            lrs = nn.ParameterList([nn.Parameter(lr) for lr in lrs])
        self.lrs = lrs
        self.first_order = first_order

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self):
        """
        **Descritpion**

        Akin to `MAML.clone()` but for MetaSGD: it includes a set of learnable fast-adaptation
        learning rates.
        """
        return MetaSGD(clone_module(self.module), lrs=clone_parameters(self.lrs), first_order=self.first_order)

    def adapt(self, loss, first_order=None):
        """
        **Descritpion**

        Akin to `MAML.adapt()` but for MetaSGD: it updates the model with the learnable
        per-parameter learning rates.
        """
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order
        gradients = grad(loss, self.module.parameters(), retain_graph=second_order, create_graph=second_order)
        self.module = meta_sgd_update(self.module, self.lrs, gradients)


def truncated_normal_(tensor, mean=0.0, std=1.0):
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor


def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias.data, 0.0)
    return module


class LinearBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearBlock, self).__init__()
        self.relu = nn.ReLU()
        self.normalize = nn.BatchNorm1d(output_size, affine=True, momentum=0.999, eps=0.001, track_running_stats=False)
        self.linear = nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x


def maml_init_(module):
    nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    nn.init.constant_(module.bias.data, 0.0)
    return module


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, max_pool=True, max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = int(2 * max_pool_factor), int(2 * max_pool_factor)
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=False)
            stride = 1, 1
        else:
            self.max_pool = lambda x: x
        self.normalize = nn.BatchNorm2d(out_channels, affine=True)
        nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):

    def __init__(self, output_size, hidden=64, channels=1, max_pool=False, layers=4, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, (3, 3), max_pool=max_pool, max_pool_factor=max_pool_factor)]
        for l in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, kernel_size=(3, 3), max_pool=max_pool, max_pool_factor=max_pool_factor))
        super(ConvBase, self).__init__(*core)


class OmniglotFC(nn.Sequential):
    """

    [[Source]]()

    **Description**

    The fully-connected network used for Omniglot experiments, as described in Santoro et al, 2016.

    **References**

    1. Santoro et al. 2016. “Meta-Learning with Memory-Augmented Neural Networks.” ICML.

    **Arguments**

    * **input_size** (int) - The dimensionality of the input.
    * **output_size** (int) - The dimensionality of the output.
    * **sizes** (list, *optional*, default=None) - A list of hidden layer sizes.

    **Example**
    ~~~python
    net = OmniglotFC(input_size=28**2,
                     output_size=10,
                     sizes=[64, 64, 64])
    ~~~

    """

    def __init__(self, input_size, output_size, sizes=None):
        if sizes is None:
            sizes = [256, 128, 64, 64]
        layers = [LinearBlock(input_size, sizes[0])]
        for s_i, s_o in zip(sizes[:-1], sizes[1:]):
            layers.append(LinearBlock(s_i, s_o))
        layers.append(fc_init_(nn.Linear(sizes[-1], output_size)))
        super(OmniglotFC, self).__init__(*layers)
        self.input_size = input_size

    def forward(self, x):
        return super(OmniglotFC, self).forward(x.view(-1, self.input_size))


class OmniglotCNN(nn.Module):
    """

    [Source]()

    **Description**

    The convolutional network commonly used for Omniglot, as described by Finn et al, 2017.

    This network assumes inputs of shapes (1, 28, 28).

    **References**

    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = OmniglotCNN(output_size=20, hidden_size=128, layers=3)
    ~~~

    """

    def __init__(self, output_size=5, hidden_size=64, layers=4):
        super(OmniglotCNN, self).__init__()
        self.hidden_size = hidden_size
        self.base = ConvBase(output_size=hidden_size, hidden=hidden_size, channels=1, max_pool=False, layers=layers)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.linear.weight.data.normal_()
        self.linear.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.base(x.view(-1, 1, 28, 28))
        x = x.mean(dim=[2, 3])
        x = self.linear(x)
        return x


class MiniImagenetCNN(nn.Module):
    """

    [[Source]]()

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    This network assumes inputs of shapes (3, 84, 84).

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=32) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.

    **Example**
    ~~~python
    model = MiniImagenetCNN(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(self, output_size, hidden_size=32, layers=4):
        super(MiniImagenetCNN, self).__init__()
        self.base = ConvBase(output_size=hidden_size, hidden=hidden_size, channels=3, max_pool=True, layers=layers, max_pool_factor=4 // layers)
        self.linear = nn.Linear(25 * hidden_size, output_size, bias=True)
        maml_init_(self.linear)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x.view(-1, 25 * self.hidden_size))
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvBase,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Convnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (DiagNormalPolicy,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Lambda,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBlock,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MAML,
     lambda: ([], {'model': _mock_layer(), 'lr': 4}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (MiniImagenetCNN,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 3, 81, 81])], {}),
     True),
    (Model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (OmniglotCNN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 28, 28])], {}),
     False),
    (OmniglotFC,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_learnables_learn2learn(_paritybench_base):
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

