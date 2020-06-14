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

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch as th


from torch import nn


from torch import optim


from torch import distributions as dist


import random


from copy import deepcopy


import numpy as np


import torch


from torch import autograd


from torch.distributions.kl import kl_divergence


from torch.nn.utils import parameters_to_vector


from torch.nn.utils import vector_to_parameters


import math


from torch.distributions import Normal


from torch.distributions import Categorical


from torch.nn import functional as F


from torch.utils.data import DataLoader


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import grad


import copy


from scipy.stats import truncnorm


DIM = 5


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.mu = nn.Parameter(th.randn(DIM))
        self.sigma = nn.Parameter(th.randn(DIM))

    def forward(self, x=None):
        return dist.Normal(self.mu, self.sigma)


EPSILON = 1e-08


def linear_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()
    return module


class DiagNormalPolicy(nn.Module):

    def __init__(self, input_size, output_size, hiddens=None, activation='relu'
        ):
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
    """Head for sentence-level classification tasks."""

    def __init__(self, num_classes, input_dim=768, inner_dim=200,
        pooler_dropout=0.3):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = F.log_softmax(self.out_proj(x), dim=1)
        return x


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


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


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = l2l.vision.models.ConvBase(output_size=z_dim, hidden
            =hid_dim, channels=x_dim)
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
        self.normalize = nn.BatchNorm1d(output_size, affine=True, momentum=
            0.999, eps=0.001, track_running_stats=False)
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

    def __init__(self, in_channels, out_channels, kernel_size, max_pool=
        True, max_pool_factor=1.0):
        super(ConvBlock, self).__init__()
        stride = int(2 * max_pool_factor), int(2 * max_pool_factor)
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride,
                ceil_mode=False)
            stride = 1, 1
        else:
            self.max_pool = lambda x: x
        self.normalize = nn.BatchNorm2d(out_channels, affine=True)
        nn.init.uniform_(self.normalize.weight)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=1, bias=True)
        maml_init_(self.conv)

    def forward(self, x):
        x = self.conv(x)
        x = self.normalize(x)
        x = self.relu(x)
        x = self.max_pool(x)
        return x


class ConvBase(nn.Sequential):

    def __init__(self, output_size, hidden=64, channels=1, max_pool=False,
        layers=4, max_pool_factor=1.0):
        core = [ConvBlock(channels, hidden, (3, 3), max_pool=max_pool,
            max_pool_factor=max_pool_factor)]
        for l in range(layers - 1):
            core.append(ConvBlock(hidden, hidden, kernel_size=(3, 3),
                max_pool=max_pool, max_pool_factor=max_pool_factor))
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
        self.base = ConvBase(output_size=hidden_size, hidden=hidden_size,
            channels=1, max_pool=False, layers=layers)
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
        self.base = ConvBase(output_size=hidden_size, hidden=hidden_size,
            channels=3, max_pool=True, layers=layers, max_pool_factor=4 //
            layers)
        self.linear = nn.Linear(25 * hidden_size, output_size, bias=True)
        maml_init_(self.linear)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.base(x)
        x = self.linear(x.view(-1, 25 * self.hidden_size))
        return x


def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn, nn.ReLU(), nn.MaxPool2d(2))


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(conv_block(x_dim, hid_dim), conv_block
            (hid_dim, hid_dim), conv_block(hid_dim, hid_dim), conv_block(
            hid_dim, z_dim))
        self.out_channels = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.
            Tanh(), torch.nn.Linear(64, 2))

    def forward(self, x):
        return self.model(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_learnables_learn2learn(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ConvBase(*[], **{'output_size': 4}), [torch.rand([4, 1, 64, 64])], {})

    def test_001(self):
        self._check(ConvBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Convnet(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(DiagNormalPolicy(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(LinearBlock(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4])], {})

    def test_005(self):
        self._check(Model(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(OmniglotFC(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

