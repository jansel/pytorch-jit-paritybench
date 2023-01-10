import sys
_module = sys.modules[__name__]
del sys
const = _module
data = _module
dgr = _module
gan = _module
main = _module
models = _module
train = _module
utils = _module
visual = _module

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


import copy


import math


from torchvision import datasets


from torchvision import transforms


from torch.utils.data import ConcatDataset


import abc


import torch


from torch import nn


from torch.autograd import Variable


from torch.nn import functional as F


import numpy as np


from functools import reduce


from torch import autograd


from torch import optim


import torchvision


from torch.utils.data import DataLoader


from torch.utils.data.dataloader import default_collate


from torch.cuda import FloatTensor as CUDATensor


class BatchTrainable(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class which defines a generative-replay-based training
    interface for a model.

    """

    @abc.abstractmethod
    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=0.5):
        raise NotImplementedError


class Generator(nn.Module):

    def __init__(self, z_size, image_size, image_channel_size, channel_size):
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size
        self.fc = nn.Linear(z_size, (image_size // 8) ** 2 * channel_size * 8)
        self.bn0 = nn.BatchNorm2d(channel_size * 8)
        self.bn1 = nn.BatchNorm2d(channel_size * 4)
        self.deconv1 = nn.ConvTranspose2d(channel_size * 8, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        self.deconv2 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.deconv3 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(channel_size, image_channel_size, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        g = F.relu(self.bn0(self.fc(z).view(z.size(0), self.channel_size * 8, self.image_size // 8, self.image_size // 8)))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        return F.sigmoid(g)


class Solver(BatchTrainable):
    """Abstract solver module of a scholar module"""

    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.criterion = None

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def solve(self, x):
        scores = self(x)
        _, predictions = torch.max(scores, 1)
        return predictions

    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=0.5):
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()
        batch_size = x.size(0)
        self.optimizer.zero_grad()
        real_scores = self.forward(x)
        real_loss = self.criterion(real_scores, y)
        _, real_predicted = real_scores.max(1)
        real_prec = (y == real_predicted).sum().data[0] / batch_size
        if x_ is not None and y_ is not None:
            replay_scores = self.forward(x_)
            replay_loss = self.criterion(replay_scores, y_)
            _, replay_predicted = replay_scores.max(1)
            replay_prec = (y_ == replay_predicted).sum().data[0] / batch_size
            loss = importance_of_new_task * real_loss + (1 - importance_of_new_task) * replay_loss
            precision = (real_prec + replay_prec) / 2
        else:
            loss = real_loss
            precision = real_prec
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.data[0], 'precision': precision}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion


class GenerativeMixin(object):
    """Mixin which defines a sampling iterface for a generative model."""

    def sample(self, size):
        raise NotImplementedError


class Scholar(GenerativeMixin, nn.Module):
    """Scholar for Deep Generative Replay"""

    def __init__(self, label, generator, solver):
        super().__init__()
        self.label = label
        self.generator = generator
        self.solver = solver

    def train_with_replay(self, dataset, scholar=None, previous_datasets=None, importance_of_new_task=0.5, batch_size=32, generator_iterations=2000, generator_training_callbacks=None, solver_iterations=1000, solver_training_callbacks=None, collate_fn=None):
        mutex_condition_infringed = all([scholar is not None, bool(previous_datasets)])
        assert not mutex_condition_infringed, 'scholar and previous datasets cannot be given at the same time'
        self._train_batch_trainable_with_replay(self.generator, dataset, scholar, previous_datasets=previous_datasets, importance_of_new_task=importance_of_new_task, batch_size=batch_size, iterations=generator_iterations, training_callbacks=generator_training_callbacks, collate_fn=collate_fn)
        self._train_batch_trainable_with_replay(self.solver, dataset, scholar, previous_datasets=previous_datasets, importance_of_new_task=importance_of_new_task, batch_size=batch_size, iterations=solver_iterations, training_callbacks=solver_training_callbacks, collate_fn=collate_fn)

    @property
    def name(self):
        return self.label

    def sample(self, size):
        x = self.generator.sample(size)
        y = self.solver.solve(x)
        return x.data, y.data

    def _train_batch_trainable_with_replay(self, trainable, dataset, scholar=None, previous_datasets=None, importance_of_new_task=0.5, batch_size=32, iterations=1000, training_callbacks=None, collate_fn=None):
        if iterations <= 0:
            return
        data_loader = iter(utils.get_data_loader(dataset, batch_size, cuda=self._is_on_cuda(), collate_fn=collate_fn))
        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets), batch_size, cuda=self._is_on_cuda(), collate_fn=collate_fn)) if previous_datasets else None
        progress = tqdm(range(1, iterations + 1))
        for batch_index in progress:
            from_scholar = scholar is not None
            from_previous_datasets = bool(previous_datasets)
            cuda = self._is_on_cuda()
            x, y = next(data_loader)
            x = Variable(x) if cuda else Variable(x)
            y = Variable(y) if cuda else Variable(y)
            if from_previous_datasets:
                x_, y_ = next(data_loader_previous)
            elif from_scholar:
                x_, y_ = scholar.sample(batch_size)
            else:
                x_ = y_ = None
            if x_ is not None and y_ is not None:
                x_ = Variable(x_) if cuda else Variable(x_)
                y_ = Variable(y_) if cuda else Variable(y_)
            result = trainable.train_a_batch(x, y, x_=x_, y_=y_, importance_of_new_task=importance_of_new_task)
            for callback in (training_callbacks or []):
                callback(trainable, progress, batch_index, result)

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


class Critic(nn.Module):

    def __init__(self, image_size, image_channel_size, channel_size):
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size
        self.conv1 = nn.Conv2d(image_channel_size, channel_size, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(channel_size * 4, channel_size * 8, kernel_size=4, stride=1, padding=1)
        self.fc = nn.Linear((image_size // 8) ** 2 * channel_size * 4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, (self.image_size // 8) ** 2 * self.channel_size * 4)
        return self.fc(x)


EPSILON = 1e-16


class WGAN(dgr.Generator):

    def __init__(self, z_size, image_size, image_channel_size, c_channel_size, g_channel_size):
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.c_channel_size = c_channel_size
        self.g_channel_size = g_channel_size
        self.critic = gan.Critic(image_size=self.image_size, image_channel_size=self.image_channel_size, channel_size=self.c_channel_size)
        self.generator = gan.Generator(z_size=self.z_size, image_size=self.image_size, image_channel_size=self.image_channel_size, channel_size=self.g_channel_size)
        self.generator_optimizer = None
        self.critic_optimizer = None
        self.critic_updates_per_generator_update = None
        self.lamda = None

    def train_a_batch(self, x, y, x_=None, y_=None, importance_of_new_task=0.5):
        assert x_ is None or x.size() == x_.size()
        assert y_ is None or y.size() == y_.size()
        for _ in range(self.critic_updates_per_generator_update):
            self.critic_optimizer.zero_grad()
            z = self._noise(x.size(0))
            c_loss_real, g_real = self._c_loss(x, z, return_g=True)
            c_loss_real_gp = c_loss_real + self._gradient_penalty(x, g_real, self.lamda)
            if x_ is not None and y_ is not None:
                c_loss_replay, g_replay = self._c_loss(x_, z, return_g=True)
                c_loss_replay_gp = c_loss_replay + self._gradient_penalty(x_, g_replay, self.lamda)
                c_loss = importance_of_new_task * c_loss_real + (1 - importance_of_new_task) * c_loss_replay
                c_loss_gp = importance_of_new_task * c_loss_real_gp + (1 - importance_of_new_task) * c_loss_replay_gp
            else:
                c_loss = c_loss_real
                c_loss_gp = c_loss_real_gp
            c_loss_gp.backward()
            self.critic_optimizer.step()
        self.generator_optimizer.zero_grad()
        z = self._noise(x.size(0))
        g_loss = self._g_loss(z)
        g_loss.backward()
        self.generator_optimizer.step()
        return {'c_loss': c_loss.data[0], 'g_loss': g_loss.data[0]}

    def sample(self, size):
        return self.generator(self._noise(size))

    def set_generator_optimizer(self, optimizer):
        self.generator_optimizer = optimizer

    def set_critic_optimizer(self, optimizer):
        self.critic_optimizer = optimizer

    def set_critic_updates_per_generator_update(self, k):
        self.critic_updates_per_generator_update = k

    def set_lambda(self, l):
        self.lamda = l

    def _noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * 0.1
        return z if self._is_on_cuda() else z

    def _c_loss(self, x, z, return_g=False):
        g = self.generator(z)
        c_x = self.critic(x).mean()
        c_g = self.critic(g).mean()
        l = -(c_x - c_g)
        return (l, g) if return_g else l

    def _g_loss(self, z, return_g=False):
        g = self.generator(z)
        l = -self.critic(g).mean()
        return (l, g) if return_g else l

    def _gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a if self._is_on_cuda() else a
        a = a.expand(x.size(0), x.nelement() // x.size(0)).contiguous().view(x.size(0), self.image_channel_size, self.image_size, self.image_size)
        interpolated = Variable(a * x.data + (1 - a) * g.data, requires_grad=True)
        c = self.critic(interpolated)
        gradients = autograd.grad(c, interpolated, grad_outputs=torch.ones(c.size()) if self._is_on_cuda() else torch.ones(c.size()), create_graph=True, retain_graph=True)[0]
        return lamda * ((1 - (gradients + EPSILON).norm(2, dim=1)) ** 2).mean()

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda


class CNN(dgr.Solver):

    def __init__(self, image_size, image_channel_size, classes, depth, channel_size, reducing_layers=3):
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.classes = classes
        self.depth = depth
        self.channel_size = channel_size
        self.reducing_layers = reducing_layers
        self.layers = nn.ModuleList([nn.Conv2d(self.image_channel_size, self.channel_size // 2 ** (depth - 2), 3, 1, 1)])
        for i in range(self.depth - 2):
            previous_conv = [l for l in self.layers if isinstance(l, nn.Conv2d)][-1]
            self.layers.append(nn.Conv2d(previous_conv.out_channels, previous_conv.out_channels * 2, 3, 1 if i >= reducing_layers else 2, 1))
            self.layers.append(nn.BatchNorm2d(previous_conv.out_channels * 2))
            self.layers.append(nn.ReLU())
        self.layers.append(utils.LambdaModule(lambda x: x.view(x.size(0), -1)))
        self.layers.append(nn.Linear((image_size // 2 ** reducing_layers) ** 2 * channel_size, self.classes))

    def forward(self, x):
        return reduce(lambda x, l: l(x), self.layers, x)


class LambdaModule(nn.Module):

    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LambdaModule,
     lambda: ([], {'f': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_kuc2477_pytorch_deep_generative_replay(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

