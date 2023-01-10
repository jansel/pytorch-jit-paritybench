import sys
_module = sys.modules[__name__]
del sys
maml_sine = _module
maml_toy = _module
hypergrad_mnist = _module
rl = _module
cavia_trpo = _module
dist_promp = _module
maml_dice = _module
maml_trpo = _module
maml_trpo_metaworld = _module
metasgd_a2c = _module
policies = _module
promp = _module
news_topic_classification = _module
anil_fc100 = _module
anilkfo_cifarfs = _module
distributed_maml = _module
main = _module
maml_miniimagenet = _module
maml_omniglot = _module
MAMLpp = _module
cnn4_bnrs = _module
meta_mnist = _module
metacurvature_fc100 = _module
protonet_miniimagenet = _module
reptile_miniimagenet = _module
supervised_pretraining = _module
learn2learn = _module
_version = _module
algorithms = _module
base_learner = _module
gbml = _module
lightning = _module
lightning_anil = _module
lightning_episodic_module = _module
lightning_maml = _module
lightning_metaoptnet = _module
lightning_protonet = _module
maml = _module
meta_sgd = _module
data = _module
utils = _module
gym = _module
async_vec_env = _module
envs = _module
meta_env = _module
metaworld = _module
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
nn = _module
kroneckers = _module
metaoptnet = _module
misc = _module
protonet = _module
optim = _module
learnable_optimizer = _module
parameter_update = _module
transforms = _module
kronecker_transform = _module
metacurvature_transform = _module
module_transform = _module
transform_dictionary = _module
update_rules = _module
differentiable_sgd = _module
text = _module
datasets = _module
news_classification = _module
utils = _module
vision = _module
benchmarks = _module
cifarfs_benchmark = _module
fc100_benchmark = _module
mini_imagenet_benchmark = _module
omniglot_benchmark = _module
tiered_imagenet_benchmark = _module
cifarfs = _module
cu_birds200 = _module
describable_textures = _module
fc100 = _module
fgvc_aircraft = _module
fgvc_fungi = _module
full_omniglot = _module
mini_imagenet = _module
quickdraw = _module
tiered_imagenet = _module
vgg_flowers = _module
models = _module
cnn4 = _module
resnet12 = _module
wrn28 = _module
bibtex_to_gsheets = _module
compile_paper_list = _module
setup = _module
tests = _module
integration = _module
maml_miniimagenet_test_notravis = _module
maml_omniglot_test = _module
protonets_miniimagenet_test_notravis = _module
unit = _module
gbml_test = _module
lightning_anil_test_notravis = _module
lightning_maml_test_notravis = _module
lightning_metaoptnet_test_notravis = _module
lightning_protonet_test_notravis = _module
maml_test = _module
metasgd_test = _module
metadataset_test = _module
task_dataset_test = _module
transforms_test = _module
util_datasets = _module
utils_test = _module
kroneckers_test = _module
metaoptnet_test_notravis = _module
misc = _module
protonet_test = _module
utils_test = _module
benchmarks_test_notravis = _module
cifarfs_test_notravis = _module
cu_birds200_test_notravis = _module
describable_textures_test_notravis = _module
fc100_test_notravis = _module
fgvc_aircraft_test_notravis = _module
pretrained_backbones_notravis = _module
quickdraw_test_no = _module
tiered_imagenet_test_notravis = _module
transform_test_notravis = _module
vgg_flowers_test_notravis = _module

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


import random


import numpy as np


import torch


from torch import nn


from torch import optim


import torch as th


from torch import distributions as dist


from torch.nn import functional as F


import torchvision as tv


from torch import autograd


from torch.distributions.kl import kl_divergence


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import vector_to_parameters


from copy import deepcopy


from torch import distributed as dist


import math


from torch.distributions import Normal


from torch.distributions import Categorical


from torch.autograd import grad


import torch.nn.functional as F


from collections import namedtuple


from typing import Tuple


from torch.utils.data import DataLoader


from torchvision import transforms


from torchvision.datasets import MNIST


import torch.nn as nn


import copy


import warnings


from torch.utils.data import Dataset


from torchvision.transforms import Compose


from torchvision.transforms import ToPILImage


from torchvision.transforms import ToTensor


from torchvision.transforms import RandomCrop


from torchvision.transforms import RandomHorizontalFlip


from torchvision.transforms import ColorJitter


from torchvision.transforms import Normalize


import torch.utils.data as data


from torchvision.datasets import ImageFolder


from torchvision.datasets.folder import default_loader


import scipy.io


from collections import defaultdict


from torch.utils.data import ConcatDataset


from torchvision.datasets.omniglot import Omniglot


from scipy.stats import truncnorm


import torch.nn.init as init


import re


from torch.utils.data import TensorDataset


import string


class SineModel(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.hidden1 = nn.Linear(1, dim)
        self.hidden2 = nn.Linear(dim, dim)
        self.hidden3 = nn.Linear(dim, 1)

    def forward(self, x):
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2))

    def forward(self, x):
        return self.model(x)


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


class HypergradTransform(torch.nn.Module):
    """Hypergradient-style per-parameter learning rates"""

    def __init__(self, param, lr=0.01):
        super(HypergradTransform, self).__init__()
        self.lr = lr * torch.ones_like(param, requires_grad=True)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


EPSILON = 1e-08


def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class CaviaDiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='relu', num_context_params=2, device='cpu'):
        super(CaviaDiagNormalPolicy, self).__init__()
        self.device = device
        if hiddens is None:
            hiddens = [100, 100]
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        layers = [linear_init(nn.Linear(input_size + num_context_params, hiddens[0])), activation()]
        for i, o in zip(hiddens[:-1], hiddens[1:]):
            layers.append(linear_init(nn.Linear(i, o)))
            layers.append(activation())
        layers.append(linear_init(nn.Linear(hiddens[-1], output_size)))
        self.num_context_params = num_context_params
        self.context_params = torch.zeros(self.num_context_params, requires_grad=True)
        self.mean = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.Tensor(output_size))
        self.sigma.data.fill_(math.log(1))

    def density(self, state):
        state = state
        state = torch.cat((state, self.context_params.expand(state.shape[:-1] + self.context_params.shape)), dim=len(state.shape) - 1)
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

    def reset_context(self):
        self.context_params[:] = 0


class DiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='relu', device='cpu'):
        super(DiagNormalPolicy, self).__init__()
        self.device = device
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
        state = state
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


class CifarCNN(torch.nn.Module):
    """
    Example of a 4-layer CNN network for FC100/CIFAR-FS.
    """

    def __init__(self, output_size=5, hidden_size=32, layers=4):
        super(CifarCNN, self).__init__()
        self.hidden_size = hidden_size
        features = l2l.vision.models.ConvBase(hidden=hidden_size, channels=3, max_pool=False, layers=layers, max_pool_factor=0.5)
        self.features = torch.nn.Sequential(features, l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])), l2l.nn.Flatten())
        self.linear = torch.nn.Linear(self.hidden_size, output_size, bias=True)
        l2l.vision.models.maml_init_(self.linear)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x


class MetaBatchNormLayer(torch.nn.Module):
    """
    An extension of Pytorch's BatchNorm layer, with the Per-Step Batch Normalisation Running
    Statistics and Per-Step Batch Normalisation Weights and Biases improvements proposed in
    MAML++ by Antoniou et al. It is adapted from the original Pytorch implementation at
    https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch,
    with heavy refactoring and a bug fix
    (https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/issues/42).
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, meta_batch_norm=True, adaptation_steps: int=1):
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.meta_batch_norm = meta_batch_norm
        self.num_features = num_features
        self.running_mean = torch.nn.Parameter(torch.zeros(adaptation_steps, num_features), requires_grad=False)
        self.running_var = torch.nn.Parameter(torch.ones(adaptation_steps, num_features), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.zeros(adaptation_steps, num_features), requires_grad=True)
        self.weight = torch.nn.Parameter(torch.ones(adaptation_steps, num_features), requires_grad=True)
        self.backup_running_mean = torch.zeros(self.running_mean.shape)
        self.backup_running_var = torch.ones(self.running_var.shape)
        self.momentum = momentum

    def forward(self, input, step):
        """
        :param input: input data batch, size either can be any.
        :param step: The current inner loop step being taken. This is used when to learn per step params and
         collecting per step batch statistics.
        :return: The result of the batch norm operation.
        """
        assert step < self.running_mean.shape[0], f'Running forward with step={step} when initialised with {self.running_mean.shape[0]} steps!'
        return F.batch_norm(input, self.running_mean[step], self.running_var[step], self.weight[step], self.bias[step], training=True, momentum=self.momentum, eps=self.eps)

    def backup_stats(self):
        self.backup_running_mean.data = deepcopy(self.running_mean.data)
        self.backup_running_var.data = deepcopy(self.running_var.data)

    def restore_backup_stats(self):
        """
        Resets batch statistics to their backup values which are collected after each forward pass.
        """
        self.running_mean = torch.nn.Parameter(self.backup_running_mean, requires_grad=False)
        self.running_var = torch.nn.Parameter(self.backup_running_var, requires_grad=False)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}'.format(**self.__dict__)


def truncated_normal_(tensor, mean=0.0, std=1.0):
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor


def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class LinearBlock_BNRS(torch.nn.Module):

    def __init__(self, input_size, output_size, adaptation_steps):
        super(LinearBlock_BNRS, self).__init__()
        self.relu = torch.nn.ReLU()
        self.normalize = MetaBatchNormLayer(output_size, affine=True, momentum=0.999, eps=0.001, adaptation_steps=adaptation_steps)
        self.linear = torch.nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x, step):
        x = self.linear(x)
        x = self.normalize(x, step)
        x = self.relu(x)
        return x


def maml_init_(module):
    torch.nn.init.xavier_uniform_(module.weight.data, gain=1.0)
    torch.nn.init.constant_(module.bias.data, 0.0)
    return module


class ConvBlock_BNRS(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, max_pool=True, max_pool_factor=1.0, adaptation_steps=1):
        super(ConvBlock_BNRS, self).__init__()
        stride = int(2 * max_pool_factor), int(2 * max_pool_factor)
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=False)
            stride = 1, 1
        else:
            self.max_pool = lambda x: x
        self.normalize = MetaBatchNormLayer(out_channels, affine=True, adaptation_steps=adaptation_steps)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=True)
        maml_init_(self.conv)

    def forward(self, x, step):
        x = self.conv(x)
        x = self.normalize(x, step)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase_BNRS(torch.nn.Sequential):

    def __init__(self, hidden=64, channels=1, max_pool=False, layers=4, max_pool_factor=1.0, adaptation_steps=1):
        core = [ConvBlock_BNRS(channels, hidden, (3, 3), max_pool=max_pool, max_pool_factor=max_pool_factor, adaptation_steps=adaptation_steps)]
        for _ in range(layers - 1):
            core.append(ConvBlock_BNRS(hidden, hidden, kernel_size=(3, 3), max_pool=max_pool, max_pool_factor=max_pool_factor, adaptation_steps=adaptation_steps))
        super(ConvBase_BNRS, self).__init__(*core)

    def forward(self, x, step):
        for module in self:
            x = module(x, step)
        return x


class CNN4Backbone_BNRS(ConvBase_BNRS):

    def __init__(self, hidden_size=64, layers=4, channels=3, max_pool=True, max_pool_factor=None, adaptation_steps=1):
        if max_pool_factor is None:
            max_pool_factor = 4 // layers
        super(CNN4Backbone_BNRS, self).__init__(hidden=hidden_size, layers=layers, channels=channels, max_pool=max_pool, max_pool_factor=max_pool_factor, adaptation_steps=adaptation_steps)

    def forward(self, x, step):
        x = super(CNN4Backbone_BNRS, self).forward(x, step)
        x = x.reshape(x.size(0), -1)
        return x


class CNN4_BNRS(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    This network assumes inputs of shapes (3, 84, 84).

    Instantiate `CNN4Backbone` if you only need the feature extractor.

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.
    * **channels** (int, *optional*, default=3) - The number of channels in input.
    * **max_pool** (bool, *optional*, default=True) - Whether ConvBlocks use max-pooling.
    * **embedding_size** (int, *optional*, default=None) - Size of feature embedding.
        Defaults to 25 * hidden_size (for mini-Imagenet).

    **Example**
    ~~~python
    model = CNN4(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(self, output_size, hidden_size=64, layers=4, channels=3, max_pool=True, embedding_size=None, adaptation_steps=1):
        super(CNN4_BNRS, self).__init__()
        if embedding_size is None:
            embedding_size = 25 * hidden_size
        self.features = CNN4Backbone_BNRS(hidden_size=hidden_size, channels=channels, max_pool=max_pool, layers=layers, max_pool_factor=4 // layers, adaptation_steps=adaptation_steps)
        self.classifier = torch.nn.Linear(embedding_size, output_size, bias=True)
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def backup_stats(self):
        """
        Backup stored batch statistics before running a validation epoch.
        """
        for layer in self.features.modules():
            if type(layer) is MetaBatchNormLayer:
                layer.backup_stats()

    def restore_backup_stats(self):
        """
        Reset stored batch statistics from the stored backup.
        """
        for layer in self.features.modules():
            if type(layer) is MetaBatchNormLayer:
                layer.restore_backup_stats()

    def forward(self, x, step):
        x = self.features(x, step)
        x = self.classifier(x)
        return x


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


class GBML(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/gbml.py)

    **Description**

    General wrapper for gradient-based meta-learning implementations.

    A variety of algorithms can simply be implemented by changing the kind
    of `transform` used during fast-adaptation.
    For example, if the transform is `Scale` we recover Meta-SGD [2] with `adapt_transform=False`
    and Alpha MAML [4] with `adapt_transform=True`.
    If the transform is a Kronecker-factored module (e.g. neural network, or linear), we recover
    KFO from [5].

    **Arguments**

    * **module** (Module) - Module to be wrapped.
    * **tranform** (Module) - Transform used to update the module.
    * **lr** (float) - Fast adaptation learning rate.
    * **adapt_transform** (bool, *optional*, default=False) - Whether to update the transform's
        parameters during fast-adaptation.
    * **first_order** (bool, *optional*, default=False) - Whether to use the first-order
        approximation.
    * **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
        of unused parameters. Defaults to `allow_nograd`.
    * **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
        parameters that have `requires_grad = False`.

    **References**

    1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.”
    2. Li et al. 2017. “Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.”
    3. Park & Oliva. 2019. “Meta-Curvature.”
    4. Behl et al. 2019. “Alpha MAML: Adaptive Model-Agnostic Meta-Learning.”
    5. Arnold et al. 2019. “When MAML Can Adapt Fast and How to Assist When It Cannot.”

    **Example**

    ~~~python
    model = SmallCNN()
    transform = l2l.optim.ModuleTransform(torch.nn.Linear)
    gbml = l2l.algorithms.GBML(
        module=model,
        transform=transform,
        lr=0.01,
        adapt_transform=True,
    )
    gbml.to(device)
    opt = torch.optim.SGD(gbml.parameters(), lr=0.001)

    # Training with 1 adaptation step
    for iteration in range(10):
        opt.zero_grad()
        task_model = gbml.clone()
        loss = compute_loss(task_model)
        task_model.adapt(loss)
        loss.backward()
        opt.step()
    ~~~
    """

    def __init__(self, module, transform, lr=1.0, adapt_transform=False, first_order=False, allow_unused=False, allow_nograd=False, **kwargs):
        super(GBML, self).__init__()
        self.module = module
        self.transform = transform
        self.adapt_transform = adapt_transform
        self.lr = lr
        self.first_order = first_order
        self.allow_unused = allow_unused
        self.allow_nograd = allow_nograd
        if 'compute_update' in kwargs:
            self.compute_update = kwargs.get('compute_update')
        else:
            self.compute_update = l2l.optim.ParameterUpdate(parameters=self.module.parameters(), transform=transform)
        self.diff_sgd = l2l.optim.DifferentiableSGD(lr=self.lr)
        self._params_updated = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None, adapt_transform=None):
        """
        **Description**

        Similar to `MAML.clone()`.

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
        if adapt_transform is None:
            adapt_transform = self.adapt_transform
        module_clone = l2l.clone_module(self.module)
        update_clone = l2l.clone_module(self.compute_update)
        return GBML(module=module_clone, transform=self.transform, lr=self.lr, adapt_transform=adapt_transform, first_order=first_order, allow_unused=allow_unused, allow_nograd=allow_nograd, compute_update=update_clone)

    def adapt(self, loss, first_order=None, allow_nograd=None, allow_unused=None):
        """
        **Description**

        Takes a gradient step on the loss and updates the cloned parameters in place.

        The parameters of the transform are only adapted if `self.adapt_update` is `True`.

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
        if self.adapt_transform and self._params_updated:
            update_grad = torch.autograd.grad(loss, self.compute_update.parameters(), create_graph=second_order, retain_graph=second_order, allow_unused=allow_unused)
            self.diff_sgd(self.compute_update, update_grad)
            self._params_updated = False
        updates = self.compute_update(loss, self.module.parameters(), create_graph=second_order or self.adapt_transform, retain_graph=second_order or self.adapt_transform, allow_unused=allow_unused, allow_nograd=allow_nograd)
        self.diff_sgd(self.module, updates)
        self._params_updated = True


def clone_module(module, memo=None):
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
    if memo is None:
        memo = {}
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
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(module._modules[module_key], memo=memo)
    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone


def update_module(module, updates=None, memo=None):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py)

    **Description**

    Updates the parameters of a module in-place, in a way that preserves differentiability.

    The parameters of the module are swapped with their update values, according to:
    \\[
    p \\gets p + u,
    \\]
    where \\(p\\) is the parameter, and \\(u\\) is its corresponding update.


    **Arguments**

    * **module** (Module) - The module to update.
    * **updates** (list, *optional*, default=None) - A list of gradients for each parameter
        of the model. If None, will use the tensors in .update attributes.

    **Example**
    ~~~python
    error = loss(model(X), y)
    grads = torch.autograd.grad(
        error,
        model.parameters(),
        create_graph=True,
    )
    updates = [-lr * g for g in grads]
    l2l.update_module(model, updates=updates)
    ~~~
    """
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            None
        for p, g in zip(params, updates):
            p.update = g
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            module._parameters[param_key] = memo[p]
        elif p is not None and hasattr(p, 'update') and p.update is not None:
            updated = p + p.update
            p.update = None
            memo[p] = updated
            module._parameters[param_key] = updated
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        elif buff is not None and hasattr(buff, 'update') and buff.update is not None:
            updated = buff + buff.update
            buff.update = None
            memo[buff] = updated
            module._buffers[buffer_key] = updated
    for module_key in module._modules:
        module._modules[module_key] = update_module(module._modules[module_key], updates=None, memo=memo)
    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module


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
            if g is not None:
                p.update = -lr * g
    return update_module(model)


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
            p.grad = None
            p._lr = None
    for buffer_key in model._buffers:
        buff = model._buffers[buffer_key]
        if buff is not None and buff.grad is not None and buff._lr is not None:
            model._buffers[buffer_key] = buff - buff._lr * buff.grad
            buff.grad = None
            buff._lr = None
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


def kronecker_addmm(mat1, mat2, mat3, bias=None, alpha=1.0, beta=1.0):
    """
    Returns alpha * (mat2.t() X mat1) @ vec(mat3) + beta * vec(bias)
    (Assuming bias is not None.)
    """
    res = mat1 @ mat3 @ mat2
    res.mul_(alpha)
    if bias is not None:
        res.add_(beta, bias)
    return res


class KroneckerLinear(nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/kroneckers.py)

    **Description**

    A linear transformation whose parameters are expressed as a Kronecker product.

    This Module maps an input vector \\(x \\in \\mathbb{R}^{nm} \\) to \\(y = Ax + b\\) such that:

    \\[
    A = R^\\top \\otimes L,
    \\]

    where \\(L \\in \\mathbb{R}^{n \\times n}\\) and \\(R \\in \\mathbb{R}^{m \\times m}\\) are the learnable Kronecker factors.
    This implementation can reduce the memory requirement for large linear mapping
    from \\(\\mathcal{O}(n^2 \\cdot m^2)\\) to \\(\\mathcal{O}(n^2 + m^2)\\), but forces \\(y \\in \\mathbb{R}^{nm}\\).

    The matrix \\(A\\) is initialized as the identity, and the bias as a zero vector.

    **Arguments**

    * **n** (int) - Dimensionality of the left Kronecker factor.
    * **m** (int) - Dimensionality of the right Kronecker factor.
    * **bias** (bool, *optional*, default=True) - Whether to include the bias term.
    * **psd** (bool, *optional*, default=False) - Forces the matrix \\(A\\) to be positive semi-definite if True.
    * **device** (device, *optional*, default=None) - The device on which to instantiate the Module.

    **References**

    1. Jose et al. 2018. "Kronecker recurrent units".
    2. Arnold et al. 2019. "When MAML can adapt fast and how to assist when it cannot".

    **Example**
    ~~~python
    m, n = 2, 3
    x = torch.randn(6)
    kronecker = KroneckerLinear(n, m)
    y = kronecker(x)
    y.shape  # (6, )
    ~~~
    """

    def __init__(self, n, m, bias=True, psd=False, device=None):
        super(KroneckerLinear, self).__init__()
        self.left = nn.Parameter(th.eye(n, device=device))
        self.right = nn.Parameter(th.eye(m, device=device))
        self.bias = None
        self.psd = psd
        if bias:
            self.bias = nn.Parameter(th.zeros(n, m, device=device))
        self.device = device
        self

    def forward(self, x):
        old_device = x.device
        if self.device is not None:
            x = x
        left = self.left
        right = self.right
        if self.psd:
            left = left.t() @ left
            right = right.t() @ right
        n = self.left.size(0)
        m = self.right.size(0)
        if len(x.shape) == 1:
            if x.size(0) != n * m:
                raise ValueError('Input vector must have size n*m')
            X = x.reshape(m, n).t()
            Y = kronecker_addmm(left, right, X, self.bias)
            y = Y.t().flatten()
            return y
        if x.shape[-2:] != (n, m):
            raise ValueError('Final two dimensions of input tensor must have shape (n, m)')
        x = kronecker_addmm(left, right, x, self.bias)
        return x


class KroneckerRNN(nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/kroneckers.py)

    **Description**

    Implements a recurrent neural network whose matrices are parameterized via their Kronecker factors.
    (See `KroneckerLinear` for details.)

    **Arguments**

    * **n** (int) - Dimensionality of the left Kronecker factor.
    * **m** (int) - Dimensionality of the right Kronecker factor.
    * **bias** (bool, *optional*, default=True) - Whether to include the bias term.
    * **sigma** (callable, *optional*, default=None) - The activation function.

    **References**

    1. Jose et al. 2018. "Kronecker recurrent units".

    **Example**
    ~~~python
    m, n = 2, 3
    x = torch.randn(6)
    h = torch.randn(6)
    kronecker = KroneckerRNN(n, m)
    y, new_h = kronecker(x, h)
    y.shape  # (6, )
    ~~~
    """

    def __init__(self, n, m, bias=True, sigma=None):
        super(KroneckerRNN, self).__init__()
        self.W_h = KroneckerLinear(n, m, bias=bias)
        self.U_h = KroneckerLinear(n, m, bias=bias)
        self.W_y = KroneckerLinear(n, m, bias=bias)
        if sigma is None:
            sigma = nn.Tanh()
        self.sigma = sigma

    def forward(self, x, hidden):
        new_hidden = self.W_h(x) + self.U_h(hidden)
        new_hidden = self.sigma(new_hidden)
        output = self.W_y(new_hidden)
        return output, new_hidden


class KroneckerLSTM(nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/kroneckers.py)

    **Description**

    Implements an LSTM using a factorization similar to the one of
    `KroneckerLinear`.

    **Arguments**

    * **n** (int) - Dimensionality of the left Kronecker factor.
    * **m** (int) - Dimensionality of the right Kronecker factor.
    * **bias** (bool, *optional*, default=True) - Whether to include the bias term.
    * **sigma** (callable, *optional*, default=None) - The activation function.

    **References**

    1. Jose et al. 2018. "Kronecker recurrent units".

    **Example**
    ~~~python
    m, n = 2, 3
    x = torch.randn(6)
    h = torch.randn(6)
    kronecker = KroneckerLSTM(n, m)
    y, new_h = kronecker(x, h)
    y.shape  # (6, )
    ~~~
    """

    def __init__(self, n, m, bias=True, sigma=None):
        super(KroneckerLSTM, self).__init__()
        self.W_ii = KroneckerLinear(n, m, bias=bias)
        self.W_hi = KroneckerLinear(n, m, bias=bias)
        self.W_if = KroneckerLinear(n, m, bias=bias)
        self.W_hf = KroneckerLinear(n, m, bias=bias)
        self.W_ig = KroneckerLinear(n, m, bias=bias)
        self.W_hg = KroneckerLinear(n, m, bias=bias)
        self.W_io = KroneckerLinear(n, m, bias=bias)
        self.W_ho = KroneckerLinear(n, m, bias=bias)
        if sigma is None:
            sigma = nn.Sigmoid()
        self.sigma = sigma
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        h, c = hidden
        i = self.sigma(self.W_ii(x) + self.W_hi(h))
        f = self.sigma(self.W_if(x) + self.W_hf(h))
        g = self.tanh(self.W_ig(x) + self.W_hg(h))
        o = self.sigma(self.W_io(x) + self.W_ho(h))
        c = f * c + i * g
        h = o * self.tanh(c)
        return h, (h, c)


EPS = 1e-08


def kronecker(A, B):
    return torch.einsum('ab,cd->acbd', A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))


def onehot(x, dim):
    size = x.size(0)
    x = x.long()
    onehot = torch.zeros(size, dim, device=x.device)
    onehot.scatter_(1, x.view(-1, 1), 1.0)
    return onehot


class SVClassifier(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/metaoptnet.py)

    **Description**

    A module for the differentiable SVM classifier of MetaOptNet.

    **Arguments**

    * **support** (Tensor, *optional*, default=None) - Tensor of support features.
    * **labels** (Tensor, *optional*, default=None) - Labels corresponding to the support features.
    * **ways** (str, *optional*, default=None) - Number of classes in the task.
    * **normalize** (bool, *optional*, default=False) - Whether to normalize the inputs.
    * **C_reg** (float, *optional*, default=0.1) - Regularization weight for SVM.
    * **max_iters** (int, *optional*, default=15) - Maximum number of iterations for SVM convergence.

    **References**

    1. Lee et al. 2019. "Prototypical Networks for Few-shot Learning"

    **Example**

    ~~~python
    classifier = SVMClassifier()
    support = features(support_data)
    classifier.fit_(support, labels)
    query = features(query_data)
    preds = classifier(query)
    ~~~
    """

    def __init__(self, support=None, labels=None, ways=None, normalize=False, C_reg=0.1, max_iters=15):
        super(SVClassifier, self).__init__()
        self.C_reg = C_reg
        self.max_iters = max_iters
        self._normalize = normalize
        if support is not None and labels is not None:
            if ways is None:
                ways = len(torch.unique(labels))
            self.fit_(support, labels, ways)

    def fit_(self, support, labels, ways=None, C_reg=None, max_iters=None):
        if C_reg is None:
            C_reg = self.C_reg
        if max_iters is None:
            max_iters = self.max_iters
        if self._normalize:
            support = self.normalize(support)
        if ways is None:
            ways = len(torch.unique(labels))
        num_support = support.size(0)
        device = support.device
        kernel = support @ support.t()
        I_ways = torch.eye(ways)
        block_kernel = kronecker(kernel, I_ways)
        block_kernel.add_(torch.eye(ways * num_support, device=device))
        labels_onehot = onehot(labels, dim=ways).view(1, -1)
        I_sw = torch.eye(num_support * ways, device=device)
        I_s = torch.eye(num_support, device=device)
        h = C_reg * labels_onehot
        A = kronecker(I_s, torch.ones(1, ways, device=device))
        b = torch.zeros(1, num_support, device=device)
        qp = QPFunction(verbose=False, maxIter=max_iters)
        qp_solution = qp(block_kernel, -labels_onehot, I_sw, h, A, b)
        self.qp_solution = qp_solution.reshape(num_support, ways)
        self.support = support
        self.num_support = num_support
        self.ways = ways

    @staticmethod
    def normalize(x, epsilon=EPS):
        x = x / (x.norm(p=2, dim=1, keepdim=True) + epsilon)
        return x

    def forward(self, x):
        if self._normalize:
            x = self.normalize(x)
        num_query = x.size(0)
        qp_solution = self.qp_solution.unsqueeze(1).expand(self.num_support, num_query, self.ways)
        compatibility = self.support @ x.t()
        compatibility = compatibility.unsqueeze(2).expand(self.num_support, num_query, self.ways)
        logits = qp_solution * compatibility
        logits = torch.sum(logits, dim=0)
        return logits


class Lambda(torch.nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class Flatten(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)

    **Description**

    Utility Module to flatten inputs to `(batch_size, -1)` shape.

    **Example**
    ~~~python
    flatten = Flatten()
    x = torch.randn(5, 3, 32, 32)
    x = flatten(x)
    print(x.shape)  # (5, 3072)
    ~~~
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Scale(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/misc.py)

    **Description**

    A per-parameter scaling factor with learnable parameter.

    **Arguments**

    * **shape** (int or tuple, *optional*, default=1) - The shape of the scaling matrix.
    * **alpha** (float, *optional*, default=1.0) - Initial value for the
        scaling factor.

    **Example**
    ~~~python
    x = torch.ones(3)
    scale = Scale(x.shape, alpha=0.5)
    print(scale(x))  # [.5, .5, .5]
    ~~~
    """

    def __init__(self, shape=None, alpha=1.0):
        super(Scale, self).__init__()
        if isinstance(shape, int):
            shape = shape,
        elif shape is None:
            shape = 1,
        alpha = torch.ones(*shape) * alpha
        self.alpha = torch.nn.Parameter(alpha)

    def forward(self, x):
        return x * self.alpha


def compute_prototypes(support, labels):
    classes = torch.unique(labels)
    prototypes = torch.zeros(classes.size(0), *support.shape[1:], device=support.device, dtype=support.dtype)
    for i, cls in enumerate(classes):
        embeddings = support[labels == cls]
        prototypes[i].add_(embeddings.mean(dim=0))
    return prototypes


class PrototypicalClassifier(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/nn/protonet.py)

    **Description**

    A module for the differentiable nearest neighbour classifier of Prototypical Networks.

    **Arguments**

    * **support** (Tensor, *optional*, default=None) - Tensor of support features.
    * **labels** (Tensor, *optional*, default=None) - Labels corresponding to the support features.
    * **distance** (str, *optional*, default='euclidean') - Distance metric between samples. ['euclidean', 'cosine']
    * **normalize** (bool, *optional*, default=False) - Whether to normalize the inputs. Defaults to True when distance='cosine'.

    **References**

    1. Snell et al. 2017. "Prototypical Networks for Few-shot Learning"

    **Example**

    ~~~python
    classifier = PrototypicalClassifier()
    support = features(support_data)
    classifier.fit_(support, labels)
    query = features(query_data)
    preds = classifier(query)
    ~~~
    """

    def __init__(self, support=None, labels=None, distance='euclidean', normalize=False):
        super(PrototypicalClassifier, self).__init__()
        self.distance = 'euclidean'
        self.normalize = normalize
        self._compute_prototypes = compute_prototypes
        if distance == 'euclidean':
            self.distance = PrototypicalClassifier.euclidean_distance
        elif distance == 'cosine':
            self.distance = PrototypicalClassifier.cosine_distance
            self.normalize = True
        else:
            self.distance = distance
        self.prototypes = None
        if support is not None and labels is not None:
            self.fit_(support, labels)

    def fit_(self, support, labels):
        """
        **Description**

        Computes and updates the prototypes given support embeddings and
        corresponding labels.

        """
        prototypes = self._compute_prototypes(support, labels)
        if self.normalize:
            prototypes = PrototypicalClassifier.normalize(prototypes)
        self.prototypes = prototypes
        return prototypes

    @staticmethod
    def cosine_distance(prototypes, queries):
        return -torch.mm(queries, prototypes.t())

    @staticmethod
    def euclidean_distance(prototypes, queries):
        n = prototypes.size(0)
        m = queries.size(0)
        prototypes = prototypes.unsqueeze(0).expand(m, n, -1)
        queries = queries.unsqueeze(1).expand(m, n, -1)
        distance = (prototypes - queries).pow(2).sum(dim=2)
        return distance

    @staticmethod
    def normalize(x, epsilon=EPS):
        x = x / (x.norm(p=2, dim=1, keepdim=True) + epsilon)
        return x

    def forward(self, x):
        assert self.prototypes is not None, 'Prototypes not computed, use compute_prototypes(support, labels)'
        if self.normalize:
            x = PrototypicalClassifier.normalize(x)
        return -self.distance(self.prototypes, x)


class LearnableOptimizer(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/learnable_optimizer.py)

    **Description**

    A PyTorch Optimizer with learnable transform, enabling the implementation
    of meta-descent / hyper-gradient algorithms.

    This optimizer takes a Module and a gradient transform.
    At each step, the gradient of the module is passed through the transforms,
    and the module differentiably update -- i.e. when the next backward is called,
    gradients of both the module and the transform are computed.
    In turn, the transform can be updated via your favorite optmizer.

    **Arguments**

    * **model** (Module) - Module to be updated.
    * **transform** (Module) - Transform used to compute updates of the model.
    * **lr** (float) - Learning rate.

    **References**

    1. Sutton. 1992. “Gain Adaptation Beats Least Squares.”
    2. Schraudolph. 1999. “Local Gain Adaptation in Stochastic Gradient Descent.”
    3. Baydin et al. 2017. “Online Learning Rate Adaptation with Hypergradient Descent.”
    4. Majumder et al. 2019. “Learning the Learning Rate for Gradient Descent by Gradient Descent.”
    5. Jacobsen et al. 2019. “Meta-Descent for Online, Continual Prediction.”

    **Example**

    ~~~python
    linear = nn.Linear(784, 10)
    transform = l2l.optim.ModuleTransform(torch.nn.Linear)
    metaopt = l2l.optim.LearnableOptimizer(linear, transform, lr=0.01)
    opt = torch.optim.SGD(metaopt.parameters(), lr=0.001)

    metaopt.zero_grad()
    opt.zero_grad()
    error = loss(linear(X), y)
    error.backward()
    opt.step()  # update metaopt
    metaopt.step()  # update linear
    ~~~
    """

    def __init__(self, model, transform, lr=1.0):
        super(LearnableOptimizer, self).__init__()
        assert isinstance(model, torch.nn.Module), 'model should inherit from nn.Module.'
        self.info = {'model': model}
        self.transforms = []
        for name, param in model.named_parameters():
            trans = transform(param)
            self.transforms.append(trans)
        self.transforms = torch.nn.ModuleList(self.transforms)
        self.lr = lr

    def step(self, closure=None):
        model = self.info['model']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for param, transform in zip(model.parameters(), self.transforms):
                if hasattr(param, 'grad') and param.grad is not None:
                    grad = param.grad.detach()
                    grad.requires_grad = False
                    update = -self.lr * transform(grad)
                    param.detach_()
                    param.requires_grad = False
                    param.update = update
            l2l.update_module(model, updates=None)
            for param in model.parameters():
                param.retain_grad()

    def zero_grad(self):
        """Only reset target parameters."""
        model = self.info['model']
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                p.grad = torch.zeros_like(p.data)


class ParameterUpdate(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/parameter_update.py)

    **Description**

    Convenience class to implement custom update functions.

    Objects instantiated from this class behave similarly to `torch.autograd.grad`,
    but return parameter updates as opposed to gradients.
    Concretely, the gradients are first computed, then fed to their respective transform
    whose output is finally returned to the user.

    Additionally, this class supports parameters that might not require updates by setting
    the `allow_nograd` flag to True.
    In this case, the returned update is `None`.

    **Arguments**

    * **parameters** (list) - Parameters of the model to update.
    * **transform** (callable) - A callable that returns an instantiated
        transform given a parameter.

    **Example**
    ~~~python
    model = torch.nn.Linear()
    transform = l2l.optim.KroneckerTransform(l2l.nn.KroneckerLinear)
    get_update = ParameterUpdate(model, transform)
    opt = torch.optim.SGD(model.parameters() + get_update.parameters())

    for iteration in range(10):
        opt.zero_grad()
        error = loss(model(X), y)
        updates = get_update(
            error,
            model.parameters(),
            create_graph=True,
        )
        l2l.update_module(model, updates)
        opt.step()
    ~~~
    """

    def __init__(self, parameters, transform):
        super(ParameterUpdate, self).__init__()
        transforms_indices = []
        transform_modules = []
        module_counter = 0
        for param in parameters:
            t = transform(param)
            if t is None:
                idx = None
            elif isinstance(t, torch.nn.Module):
                transform_modules.append(t)
                idx = module_counter
                module_counter += 1
            else:
                msg = 'Transform should be either a Module or None.'
                raise ValueError(msg)
            transforms_indices.append(idx)
        self.transforms_modules = torch.nn.ModuleList(transform_modules)
        self.transforms_indices = transforms_indices

    def forward(self, loss, parameters, create_graph=False, retain_graph=False, allow_unused=False, allow_nograd=False):
        """
        **Description**

        Similar to torch.autograd.grad, but passes the gradients through the
        provided transform.

        **Arguments**

        * **loss** (Tensor) - The loss to differentiate.
        * **parameters** (iterable) - Parameters w.r.t. which we want to compute the update.
        * **create_graph** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **retain_graph** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **allow_unused** (bool, *optional*, default=False) - Same as `torch.autograd.grad`.
        * **allow_nograd** (bool, *optional*, default=False) - Properly handles parameters
            that do not require gradients. (Their update will be `None`.)

        """
        updates = []
        if allow_nograd:
            parameters = list(parameters)
            diff_params = [p for p in parameters if p.requires_grad]
            grad_params = torch.autograd.grad(loss, diff_params, retain_graph=create_graph, create_graph=create_graph, allow_unused=allow_unused)
            gradients = []
            grad_counter = 0
            for param in parameters:
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = torch.autograd.grad(loss, parameters, create_graph=create_graph, retain_graph=retain_graph, allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                msg = 'learn2learn: Maybe try with allow_nograd=True and/or' + 'allow_unused=True ?'
                None
        for g, t in zip(gradients, self.transforms_indices):
            if t is None or g is None:
                update = g
            else:
                transform = self.transforms_modules[t]
                update = transform(g)
            updates.append(update)
        return updates


class MetaCurvatureTransform(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/transforms/module_transform.py)

    **Description**

    Implements the Meta-Curvature transform of Park and Oliva, 2019.

    Unlike `ModuleTranform` and `KroneckerTransform`, this class does not wrap other Modules but is directly
    called on a weight to instantiate the transform.

    **Arguments**

    * **param** (Tensor) - The weight whose gradients will be transformed.
    * **lr** (float, *optional*, default=1.0) - Scaling factor of the udpate. (non-learnable)

    **References**

    1. Park & Oliva. 2019. Meta-curvature.

    **Example**
    ~~~python
    classifier = torch.nn.Linear(784, 10, bias=False)
    metacurvature_update = MetaCurvatureTransform(classifier.weight)
    loss(classifier(X), y).backward()
    update = metacurvature_update(classifier.weight.grad)
    classifier.weight.data.add_(-lr, update)  # Not a differentiable update. See l2l.optim.DifferentiableSGD.
    ~~~
    """

    def __init__(self, param, lr=1.0):
        super(MetaCurvatureTransform, self).__init__()
        self.lr = lr
        shape = param.shape
        if len(shape) == 1:
            self.dim = 1
            self.mc = torch.nn.Parameter(torch.ones_like(param))
        elif len(shape) == 2:
            self.dim = 2
            self.mc_in = torch.nn.Parameter(torch.eye(shape[0]))
            self.mc_out = torch.nn.Parameter(torch.eye(shape[1]))
        elif len(shape) == 4:
            self.dim = 4
            self.n_in = shape[0]
            self.n_out = shape[1]
            self.n_f = int(np.prod(shape) / (self.n_in * self.n_out))
            self.mc_in = torch.nn.Parameter(torch.eye(self.n_in))
            self.mc_out = torch.nn.Parameter(torch.eye(self.n_out))
            self.mc_f = torch.nn.Parameter(torch.eye(self.n_f))
        else:
            raise NotImplementedError('Parameter with shape', shape, 'is not supported by MetaCurvature.')

    def forward(self, grad):
        if self.dim == 1:
            update = self.mc * grad
        elif self.dim == 2:
            update = self.mc_in @ grad @ self.mc_out
        else:
            update = grad.permute(2, 3, 0, 1).contiguous()
            shape = update.shape
            update = update.view(-1, self.n_out) @ self.mc_out
            update = self.mc_f @ update.view(self.n_f, -1)
            update = update.view(self.n_f, self.n_in, self.n_out)
            update = update.permute(1, 0, 2).contiguous().view(self.n_in, -1)
            update = self.mc_in @ update
            update = update.view(self.n_in, self.n_f, self.n_out).permute(1, 0, 2).contiguous().view(shape)
            update = update.permute(2, 3, 0, 1).contiguous()
        return self.lr * update


class ReshapedTransform(torch.nn.Module):
    """
    Helper class to reshape gradients before they are fed to a Module and
    reshape back the update returned by the Module.
    """

    def __init__(self, transform, shape):
        super(ReshapedTransform, self).__init__()
        self.transform = transform
        self.in_shape = shape

    def forward(self, grad):
        """docstring for __forward__"""
        out_shape = grad.shape
        update = grad.view(self.in_shape)
        update = self.transform(update)
        update = update.view(out_shape)
        return update


class DifferentiableSGD(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/optim/update_rules/differentiable_sgd.py)

    **Description**

    A callable object that applies a list of updates to the parameters of a torch.nn.Module in a differentiable manner.

    For each parameter \\(p\\) and corresponding gradient \\(g\\), calling an instance of this class results in updating parameters:

    \\[
    p \\gets p - \\alpha g,
    \\]

    where \\(\\alpha\\) is the learning rate.

    Note: The module is updated in-place.

    **Arguments**

    * **lr** (float) - The learning rate used to update the model.

    **Example**
    ~~~python
    sgd = DifferentiableSGD(0.1)
    gradients = torch.autograd.grad(
        loss,
        model.parameters(),
        create_gaph=True)
    sgd(model, gradients)  # model is updated in-place
    ~~~
    """

    def __init__(self, lr):
        super(DifferentiableSGD, self).__init__()
        self.lr = lr

    def forward(self, module, gradients=None):
        """
        **Arguments**

        * **module** (Module) - The module to update.
        * **gradients** (list, *optional*, default=None) - A list of gradients for each parameter
            of the module. If None, will use the gradients in .grad attributes.

        """
        if gradients is None:
            gradients = [p.grad for p in module.parameters()]
        updates = [(None if g is None else g.mul(-self.lr)) for g in gradients]
        update_module(module, updates)


class LinearBlock(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearBlock, self).__init__()
        self.relu = torch.nn.ReLU()
        self.normalize = torch.nn.BatchNorm1d(output_size, affine=True, momentum=0.999, eps=0.001, track_running_stats=False)
        self.linear = torch.nn.Linear(input_size, output_size)
        fc_init_(self.linear)

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, max_pool=True, max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = int(2 * max_pool_factor), int(2 * max_pool_factor)
        if max_pool:
            self.max_pool = torch.nn.MaxPool2d(kernel_size=stride, stride=stride, ceil_mode=False)
            stride = 1, 1
        else:
            self.max_pool = lambda x: x
        self.normalize = torch.nn.BatchNorm2d(out_channels, affine=True)
        torch.nn.init.uniform_(self.normalize.weight)
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(torch.nn.Sequential):

    def __init__(self, hidden=64, channels=1, max_pool=False, layers=4, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, (3, 3), max_pool=max_pool, max_pool_factor=max_pool_factor)]
        for _ in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, kernel_size=(3, 3), max_pool=max_pool, max_pool_factor=max_pool_factor))
        super(ConvBase, self).__init__(*core)


class OmniglotFC(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

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
        super(OmniglotFC, self).__init__()
        if sizes is None:
            sizes = [256, 128, 64, 64]
        layers = [LinearBlock(input_size, sizes[0])]
        for s_i, s_o in zip(sizes[:-1], sizes[1:]):
            layers.append(LinearBlock(s_i, s_o))
        layers = torch.nn.Sequential(*layers)
        self.features = torch.nn.Sequential(l2l.nn.Flatten(), layers)
        self.classifier = fc_init_(torch.nn.Linear(sizes[-1], output_size))
        self.input_size = input_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class OmniglotCNN(torch.nn.Module):
    """

    [Source](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

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
        self.base = ConvBase(hidden=hidden_size, channels=1, max_pool=False, layers=layers)
        self.features = torch.nn.Sequential(l2l.nn.Lambda(lambda x: x.view(-1, 1, 28, 28)), self.base, l2l.nn.Lambda(lambda x: x.mean(dim=[2, 3])), l2l.nn.Flatten())
        self.classifier = torch.nn.Linear(hidden_size, output_size, bias=True)
        self.classifier.weight.data.normal_()
        self.classifier.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNN4Backbone(ConvBase):

    def __init__(self, hidden_size=64, layers=4, channels=3, max_pool=True, max_pool_factor=None):
        if max_pool_factor is None:
            max_pool_factor = 4 // layers
        super(CNN4Backbone, self).__init__(hidden=hidden_size, layers=layers, channels=channels, max_pool=max_pool, max_pool_factor=max_pool_factor)

    def forward(self, x):
        x = super(CNN4Backbone, self).forward(x)
        x = x.reshape(x.size(0), -1)
        return x


class CNN4(torch.nn.Module):
    """

    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/cnn4.py)

    **Description**

    The convolutional network commonly used for MiniImagenet, as described by Ravi et Larochelle, 2017.

    This network assumes inputs of shapes (3, 84, 84).

    Instantiate `CNN4Backbone` if you only need the feature extractor.

    **References**

    1. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **output_size** (int) - The dimensionality of the network's output.
    * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
    * **layers** (int, *optional*, default=4) - The number of convolutional layers.
    * **channels** (int, *optional*, default=3) - The number of channels in input.
    * **max_pool** (bool, *optional*, default=True) - Whether ConvBlocks use max-pooling.
    * **embedding_size** (int, *optional*, default=None) - Size of feature embedding.
        Defaults to 25 * hidden_size (for mini-Imagenet).

    **Example**
    ~~~python
    model = CNN4(output_size=20, hidden_size=128, layers=3)
    ~~~
    """

    def __init__(self, output_size, hidden_size=64, layers=4, channels=3, max_pool=True, embedding_size=None):
        super(CNN4, self).__init__()
        if embedding_size is None:
            embedding_size = 25 * hidden_size
        self.features = CNN4Backbone(hidden_size=hidden_size, channels=channels, max_pool=max_pool, layers=layers, max_pool_factor=4 // layers)
        self.classifier = torch.nn.Linear(embedding_size, output_size, bias=True)
        maml_init_(self.classifier)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DropBlock(nn.Module):

    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = torch.distributions.Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size(0) * block_mask.size(1) * block_mask.size(2) * block_mask.size(3)
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)
        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero(as_tuple=False)
        nr_blocks = non_zero_idxs.shape[0]
        offsets = torch.stack([torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1), torch.arange(self.block_size).repeat(self.block_size)]).t()
        offsets = torch.cat((torch.zeros(self.block_size ** 2, 2).long(), offsets.long()), dim=1)
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()
            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.0
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
        block_mask = 1 - padded_mask
        return block_mask


def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / 40000 * self.num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        return out


class ResNet12Backbone(nn.Module):

    def __init__(self, avg_pool=True, wider=True, embedding_dropout=0.0, dropblock_dropout=0.1, dropblock_size=5, channels=3):
        super(ResNet12Backbone, self).__init__()
        self.inplanes = channels
        block = BasicBlock
        if wider:
            num_filters = [64, 160, 320, 640]
        else:
            num_filters = [64, 128, 256, 512]
        self.layer1 = self._make_layer(block, num_filters[0], stride=2, dropblock_dropout=dropblock_dropout)
        self.layer2 = self._make_layer(block, num_filters[1], stride=2, dropblock_dropout=dropblock_dropout)
        self.layer3 = self._make_layer(block, num_filters[2], stride=2, dropblock_dropout=dropblock_dropout, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, num_filters[3], stride=2, dropblock_dropout=dropblock_dropout, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        else:
            self.avgpool = l2l.nn.Lambda(lambda x: x)
        self.flatten = l2l.nn.Flatten()
        self.embedding_dropout = embedding_dropout
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=self.embedding_dropout, inplace=False)
        self.dropblock_dropout = dropblock_dropout
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, dropblock_dropout=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dropblock_dropout, drop_block, block_size))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return x


class ResNet12(nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/resnet12.py)

    **Description**

    The 12-layer residual network from Mishra et al, 2017.

    The code is adapted from [Lee et al, 2019](https://github.com/kjunelee/MetaOptNet/)
    who share it under the Apache 2 license.

    Instantiate `ResNet12Backbone` if you only need the feature extractor.

    List of changes:

    * Rename ResNet to ResNet12.
    * Small API modifications.
    * Fix code style to be compatible with PEP8.
    * Support multiple devices in DropBlock

    **References**

    1. Mishra et al. 2017. “A Simple Neural Attentive Meta-Learner.” ICLR 18.
    2. Lee et al. 2019. “Meta-Learning with Differentiable Convex Optimization.” CVPR 19.
    3. Lee et al's code: [https://github.com/kjunelee/MetaOptNet/](https://github.com/kjunelee/MetaOptNet/)
    4. Oreshkin et al. 2018. “TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning.” NeurIPS 18.

    **Arguments**

    * **output_size** (int) - The dimensionality of the output (eg, number of classes).
    * **hidden_size** (list, *optional*, default=640) - Size of the embedding once features are extracted.
        (640 is for mini-ImageNet; used for the classifier layer)
    * **avg_pool** (bool, *optional*, default=True) - Set to False for the 16k-dim embeddings of Lee et al, 2019.
    * **wider** (bool, *optional*, default=True) - True uses (64, 160, 320, 640) filters akin to Lee et al, 2019.
        False uses (64, 128, 256, 512) filters, akin to Oreshkin et al, 2018.
    * **embedding_dropout** (float, *optional*, default=0.0) - Dropout rate on the flattened embedding layer.
    * **dropblock_dropout** (float, *optional*, default=0.1) - Dropout rate for the residual layers.
    * **dropblock_size** (int, *optional*, default=5) - Size of drop blocks.

    **Example**
    ~~~python
    model = ResNet12(output_size=ways, hidden_size=1600, avg_pool=False)
    ~~~
    """

    def __init__(self, output_size, hidden_size=640, avg_pool=True, wider=True, embedding_dropout=0.0, dropblock_dropout=0.1, dropblock_size=5, channels=3):
        super(ResNet12, self).__init__()
        self.features = ResNet12Backbone(avg_pool=avg_pool, wider=wider, embedding_dropout=embedding_dropout, dropblock_dropout=dropblock_dropout, dropblock_size=dropblock_size, channels=channels)
        self.classifier = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class wide_basic(torch.nn.Module):

    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = torch.nn.Sequential(torch.nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(torch.nn.Module):

    def __init__(self, depth, widen_factor, dropout_rate):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor
        nStages = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(nStages[3], momentum=0.9)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 21)
        out = out.view(out.size(0), -1)
        return out


class WRN28Backbone(WideResNet):

    def __init__(self, dropout=0.0):
        super(WRN28Backbone, self).__init__(depth=28, widen_factor=10, dropout_rate=dropout)


class WRN28(torch.nn.Module):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models/wrn28.py)

    **Description**

    The 28-layer 10-depth wide residual network from Dhillon et al, 2020.

    The code is adapted from [Ye et al, 2020](https://github.com/Sha-Lab/FEAT)
    who share it under the MIT license.

    Instantiate `WRN28Backbone` if you only need the feature extractor.

    **References**

    1. Dhillon et al. 2020. “A Baseline for Few-Shot Image Classification.” ICLR 20.
    2. Ye et al. 2020. “Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions.” CVPR 20.
    3. Ye et al's code: [https://github.com/Sha-Lab/FEAT](https://github.com/Sha-Lab/FEAT)

    **Arguments**

    * **output_size** (int) - The dimensionality of the output.
    * **hidden_size** (list, *optional*, default=640) - Size of the embedding once features are extracted.
        (640 is for mini-ImageNet; used for the classifier layer)
    * **dropout** (float, *optional*, default=0.0) - Dropout rate.

    **Example**
    ~~~python
    model = WRN28(output_size=ways, hidden_size=1600, avg_pool=False)
    ~~~
    """

    def __init__(self, output_size, hidden_size=640, dropout=0.0):
        super(WRN28, self).__init__()
        self.features = WRN28Backbone(dropout=dropout)
        self.classifier = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LR(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LR, self).__init__()
        self.lr = torch.ones(input_size)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicBlock,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CNN4Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (CNN4Backbone_BNRS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), 0], {}),
     False),
    (CaviaDiagNormalPolicy,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvBase,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (ConvBase_BNRS,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64]), 0], {}),
     False),
    (ConvBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvBlock_BNRS,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     True),
    (Convnet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (DiagNormalPolicy,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DropBlock,
     lambda: ([], {'block_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Flatten,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (HypergradTransform,
     lambda: ([], {'param': torch.rand([4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (KroneckerLSTM,
     lambda: ([], {'n': 4, 'm': 4}),
     lambda: ([torch.rand([4, 4]), (torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]))], {}),
     False),
    (KroneckerLinear,
     lambda: ([], {'n': 4, 'm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (KroneckerRNN,
     lambda: ([], {'n': 4, 'm': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LR,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Lambda,
     lambda: ([], {'fn': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearBlock,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (LinearBlock_BNRS,
     lambda: ([], {'input_size': 4, 'output_size': 4, 'adaptation_steps': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     True),
    (MAML,
     lambda: ([], {'model': _mock_layer(), 'lr': 4}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (MetaBatchNormLayer,
     lambda: ([], {'num_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), 0], {}),
     True),
    (MetaCurvatureTransform,
     lambda: ([], {'param': torch.rand([4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Model,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReshapedTransform,
     lambda: ([], {'transform': _mock_layer(), 'shape': 4}),
     lambda: ([torch.rand([4])], {}),
     True),
    (Scale,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WRN28,
     lambda: ([], {'output_size': 4}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (WRN28Backbone,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 128, 128])], {}),
     True),
    (wide_basic,
     lambda: ([], {'in_planes': 4, 'planes': 4, 'dropout_rate': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
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

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

    def test_027(self):
        self._check(*TESTCASES[27])

    def test_028(self):
        self._check(*TESTCASES[28])

