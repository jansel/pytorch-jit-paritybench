import sys
_module = sys.modules[__name__]
del sys
generator = _module
launch = _module
pydreamer = _module
data = _module
envs = _module
atari = _module
dmc = _module
dmlab = _module
dmm = _module
embodied = _module
minerl = _module
minigrid = _module
miniworld = _module
wrappers = _module
models = _module
a2c = _module
baselines = _module
common = _module
decoders = _module
dreamer = _module
encoders = _module
functions = _module
probes = _module
rnn = _module
rssm = _module
preprocessing = _module
tools = _module
xlauncher = _module
setup = _module
train = _module

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


import logging


import logging.config


import time


from collections import defaultdict


from itertools import chain


from logging import critical


from logging import debug


from logging import error


from logging import info


from logging import warning


from typing import Dict


from typing import List


from typing import Optional


from typing import Tuple


import numpy as np


import torch


import torch.distributions as D


import random


from abc import ABC


from abc import abstractmethod


from torch.utils.data import IterableDataset


from torch.utils.data import get_worker_info


import torch.nn as nn


import torch.nn.functional as F


from torch import Tensor


from typing import Any


from typing import Union


from typing import Callable


from typing import TypeVar


from torch import Size


import torch.jit as jit


from torch.nn import Parameter


import warnings


from logging import exception


from typing import Iterator


import scipy.special


from torch.cuda.amp import GradScaler


from torch.cuda.amp import autocast


from torch.profiler import ProfilerActivity


from torch.utils.data import DataLoader


class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: Tensor) ->Tensor:
        return x


def flatten_batch(x: Tensor, nonbatch_dims=1) ->Tuple[Tensor, Size]:
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))
        return x, batch_dim


def unflatten_batch(x: Tensor, batch_dim: Union[Size, Tuple]) ->Tensor:
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, hidden_layers, layer_norm, activation=nn.ELU):
        super().__init__()
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        dim = in_dim
        for i in range(hidden_layers):
            layers += [nn.Linear(dim, hidden_dim), norm(hidden_dim, eps=0.001), activation()]
            dim = hidden_dim
        layers += [nn.Linear(dim, out_dim)]
        if out_dim == 1:
            layers += [nn.Flatten(0)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) ->Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


TensorHM = Tensor


TensorHMA = Tensor


TensorJM = Tensor


TensorJMF = Tensor


def normal_tanh(x: Tensor, min_std=0.01, max_std=1.0):
    mean_, std_ = x.chunk(2, -1)
    mean = torch.tanh(mean_)
    std = max_std * torch.sigmoid(std_) + min_std
    normal = D.normal.Normal(mean, std)
    normal = D.independent.Independent(normal, 1)
    return normal


def tanh_normal(x: Tensor):
    mean_, std_ = x.chunk(2, -1)
    mean = 5 * torch.tanh(mean_ / 5)
    std = F.softplus(std_) + 0.1
    normal = D.normal.Normal(mean, std)
    normal = D.independent.Independent(normal, 1)
    tanh = D.TransformedDistribution(normal, [D.TanhTransform()])
    tanh.entropy = normal.entropy
    return tanh


class ActorCritic(nn.Module):

    def __init__(self, in_dim, out_actions, hidden_dim=400, hidden_layers=4, layer_norm=True, gamma=0.999, lambda_gae=0.95, entropy_weight=0.001, target_interval=100, actor_grad='reinforce', actor_dist='onehot'):
        super().__init__()
        self.in_dim = in_dim
        self.out_actions = out_actions
        self.gamma = gamma
        self.lambda_ = lambda_gae
        self.entropy_weight = entropy_weight
        self.target_interval = target_interval
        self.actor_grad = actor_grad
        self.actor_dist = actor_dist
        actor_out_dim = out_actions if actor_dist == 'onehot' else 2 * out_actions
        self.actor = MLP(in_dim, actor_out_dim, hidden_dim, hidden_layers, layer_norm)
        self.critic = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)
        self.critic_target.requires_grad_(False)
        self.train_steps = 0

    def forward_actor(self, features: Tensor) ->D.Distribution:
        y = self.actor.forward(features).float()
        if self.actor_dist == 'onehot':
            return D.OneHotCategorical(logits=y)
        if self.actor_dist == 'normal_tanh':
            return normal_tanh(y)
        if self.actor_dist == 'tanh_normal':
            return tanh_normal(y)
        assert False, self.actor_dist

    def forward_value(self, features: Tensor) ->Tensor:
        y = self.critic.forward(features)
        return y

    def training_step(self, features: TensorJMF, actions: TensorHMA, rewards: TensorJM, terminals: TensorJM, log_only=False):
        """
        The ordering is as follows:
            features[0] 
            -> actions[0] -> rewards[1], terminals[1], features[1]
            -> actions[1] -> ...
            ...
            -> actions[H-1] -> rewards[H], terminals[H], features[H]
        """
        if not log_only:
            if self.train_steps % self.target_interval == 0:
                self.update_critic_target()
            self.train_steps += 1
        reward1: TensorHM = rewards[1:]
        terminal0: TensorHM = terminals[:-1]
        terminal1: TensorHM = terminals[1:]
        value_t: TensorJM = self.critic_target.forward(features)
        value0t: TensorHM = value_t[:-1]
        value1t: TensorHM = value_t[1:]
        advantage = -value0t + reward1 + self.gamma * (1.0 - terminal1) * value1t
        advantage_gae = []
        agae = None
        for adv, term in zip(reversed(advantage.unbind()), reversed(terminal1.unbind())):
            if agae is None:
                agae = adv
            else:
                agae = adv + self.lambda_ * self.gamma * (1.0 - term) * agae
            advantage_gae.append(agae)
        advantage_gae.reverse()
        advantage_gae = torch.stack(advantage_gae)
        value_target = advantage_gae + value0t
        reality_weight = (1 - terminal0).log().cumsum(dim=0).exp()
        value: TensorJM = self.critic.forward(features)
        value0: TensorHM = value[:-1]
        loss_critic = 0.5 * torch.square(value_target.detach() - value0)
        loss_critic = (loss_critic * reality_weight).mean()
        policy_distr = self.forward_actor(features[:-1])
        if self.actor_grad == 'reinforce':
            action_logprob = policy_distr.log_prob(actions)
            loss_policy = -action_logprob * advantage_gae.detach()
        elif self.actor_grad == 'dynamics':
            loss_policy = -value_target
        else:
            assert False, self.actor_grad
        policy_entropy = policy_distr.entropy()
        loss_actor = loss_policy - self.entropy_weight * policy_entropy
        loss_actor = (loss_actor * reality_weight).mean()
        assert loss_policy.requires_grad and policy_entropy.requires_grad or not loss_critic.requires_grad
        with torch.no_grad():
            metrics = dict(loss_critic=loss_critic.detach(), loss_actor=loss_actor.detach(), policy_entropy=policy_entropy.mean(), policy_value=value0[0].mean(), policy_value_im=value0.mean(), policy_reward=reward1.mean(), policy_reward_std=reward1.std())
            tensors = dict(value=value.detach(), value_target=value_target.detach(), value_advantage=advantage.detach(), value_advantage_gae=advantage_gae.detach(), value_weight=reality_weight.detach())
        return (loss_actor, loss_critic), metrics, tensors

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())


class ConvEncoder(nn.Module):

    def __init__(self, in_channels=3, cnn_depth=32, activation=nn.ELU):
        super().__init__()
        self.out_dim = cnn_depth * 32
        kernels = 4, 4, 4, 4
        stride = 2
        d = cnn_depth
        self.model = nn.Sequential(nn.Conv2d(in_channels, d, kernels[0], stride), activation(), nn.Conv2d(d, d * 2, kernels[1], stride), activation(), nn.Conv2d(d * 2, d * 4, kernels[2], stride), activation(), nn.Conv2d(d * 4, d * 8, kernels[3], stride), activation(), nn.Flatten())

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


class DenseEncoder(nn.Module):

    def __init__(self, in_dim, out_dim=256, activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = [nn.Flatten()]
        layers += [nn.Linear(in_dim, hidden_dim), norm(hidden_dim, eps=0.001), activation()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), norm(hidden_dim, eps=0.001), activation()]
        layers += [nn.Linear(hidden_dim, out_dim), activation()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x, bd = flatten_batch(x, 3)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y


TensorTBE = Tensor


class MultiEncoder(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.reward_input = conf.reward_input
        if conf.reward_input:
            encoder_channels = conf.image_channels + 2
        else:
            encoder_channels = conf.image_channels
        if conf.image_encoder == 'cnn':
            self.encoder_image = ConvEncoder(in_channels=encoder_channels, cnn_depth=conf.cnn_depth)
        elif conf.image_encoder == 'dense':
            self.encoder_image = DenseEncoder(in_dim=conf.image_size * conf.image_size * encoder_channels, out_dim=256, hidden_layers=conf.image_encoder_layers, layer_norm=conf.layer_norm)
        elif not conf.image_encoder:
            self.encoder_image = None
        else:
            assert False, conf.image_encoder
        if conf.vecobs_size:
            self.encoder_vecobs = MLP(conf.vecobs_size, 256, hidden_dim=400, hidden_layers=2, layer_norm=conf.layer_norm)
        else:
            self.encoder_vecobs = None
        assert self.encoder_image or self.encoder_vecobs, 'Either image_encoder or vecobs_size should be set'
        self.out_dim = (self.encoder_image.out_dim if self.encoder_image else 0) + (self.encoder_vecobs.out_dim if self.encoder_vecobs else 0)

    def forward(self, obs: Dict[str, Tensor]) ->TensorTBE:
        embeds = []
        if self.encoder_image:
            image = obs['image']
            T, B, C, H, W = image.shape
            if self.reward_input:
                reward = obs['reward']
                terminal = obs['terminal']
                reward_plane = reward.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                terminal_plane = terminal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand((T, B, 1, H, W))
                image = torch.cat([image, reward_plane, terminal_plane], dim=-3)
            embed_image = self.encoder_image.forward(image)
            embeds.append(embed_image)
        if self.encoder_vecobs:
            embed_vecobs = self.encoder_vecobs(obs['vecobs'])
            embeds.append(embed_vecobs)
        embed = torch.cat(embeds, dim=-1)
        return embed


class GRUEncoderOnly(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.state_dim = conf.deter_dim
        self.out_dim = self.state_dim
        self.encoder = MultiEncoder(conf)
        self.squeeze = nn.Linear(self.encoder.out_dim, 32)
        self.rnn = nn.GRU(32 + conf.action_dim, self.state_dim)

    def init_state(self, batch_size: int) ->Any:
        device = next(self.rnn.parameters()).device
        return torch.zeros((1, batch_size, self.state_dim), device=device)

    def training_step(self, obs: Dict[str, Tensor], in_state: Any, iwae_samples: int=1, do_open_loop=False, do_image_pred=False):
        assert iwae_samples == 1
        reset_first = obs['reset'].select(0, 0)
        state_mask = ~reset_first.unsqueeze(0).unsqueeze(-1)
        in_state = in_state * state_mask
        embed = self.encoder(obs)
        embed = self.squeeze(embed)
        action_next = obs['action_next']
        embed_act = torch.cat([embed, action_next], -1)
        features, out_state = self.rnn(embed_act, in_state)
        out_state = out_state.detach()
        features = features.unsqueeze(-2)
        loss = 0.0
        return loss, features, None, out_state, {}, {}


TensorTB = Tensor


TensorTBI = Tensor


TensorTBIF = Tensor


def insert_dim(x: Tensor, dim: int, size: int) ->Tensor:
    """Inserts dimension and expands it to size."""
    x = x.unsqueeze(dim)
    x = x.expand(*x.shape[:dim], size, *x.shape[dim + 1:])
    return x


def logavgexp(x: Tensor, dim: int) ->Tensor:
    if x.size(dim) > 1:
        return x.logsumexp(dim=dim) - np.log(x.size(dim))
    else:
        return x.squeeze(dim)


class DenseNormalDecoder(nn.Module):

    def __init__(self, in_dim, out_dim=1, hidden_dim=400, hidden_layers=2, layer_norm=True, std=0.3989422804):
        super().__init__()
        self.model = MLP(in_dim, out_dim, hidden_dim, hidden_layers, layer_norm)
        self.std = std
        self.out_dim = out_dim

    def forward(self, features: Tensor) ->D.Distribution:
        y = self.model.forward(features)
        p = D.Normal(loc=y, scale=torch.ones_like(y) * self.std)
        if self.out_dim > 1:
            p = D.independent.Independent(p, 1)
        return p

    def loss(self, output: D.Distribution, target: Tensor) ->Tensor:
        var = self.std ** 2
        return -output.log_prob(target) * var

    def training_step(self, features: TensorTBIF, target: Tensor) ->Tuple[TensorTBI, TensorTB, Tensor]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)
        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)
        decoded = decoded.mean.mean(dim=2)
        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == (2 if self.out_dim == 1 else 3)
        return loss_tbi, loss_tb, decoded


TensorTBCHW = Tensor


class CatImageDecoder(nn.Module):
    """Dense decoder for categorical image, e.g. map"""

    def __init__(self, in_dim, out_shape=(33, 7, 7), activation=nn.ELU, hidden_dim=400, hidden_layers=2, layer_norm=True, min_prob=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_shape = out_shape
        norm = nn.LayerNorm if layer_norm else NoNorm
        layers = []
        if hidden_layers >= 1:
            layers += [nn.Linear(in_dim, hidden_dim), norm(hidden_dim, eps=0.001), activation()]
            for _ in range(hidden_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), norm(hidden_dim, eps=0.001), activation()]
            layers += [nn.Linear(hidden_dim, np.prod(out_shape)), nn.Unflatten(-1, out_shape)]
        else:
            layers += [nn.Linear(in_dim, np.prod(out_shape)), nn.Unflatten(-1, out_shape)]
        self.model = nn.Sequential(*layers)
        self.min_prob = min_prob

    def forward(self, x: Tensor) ->Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output: Tensor, target: Tensor) ->Tensor:
        if len(output.shape) == len(target.shape):
            target = target.argmax(dim=-3)
        assert target.dtype == torch.int64, 'Target should be categorical'
        output, bd = flatten_batch(output, len(self.out_shape))
        target, _ = flatten_batch(target, len(self.out_shape) - 1)
        if self.min_prob == 0:
            loss = F.nll_loss(F.log_softmax(output, 1), target, reduction='none')
        else:
            prob = F.softmax(output, 1)
            prob = (1.0 - self.min_prob) * prob + self.min_prob * (1.0 / prob.size(1))
            loss = F.nll_loss(prob.log(), target, reduction='none')
        if len(self.out_shape) == 3:
            loss = loss.sum(dim=[-1, -2])
        assert len(loss.shape) == 1
        return unflatten_batch(loss, bd)

    def training_step(self, features: TensorTBIF, target: TensorTBCHW) ->Tuple[TensorTBI, TensorTB, TensorTBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)
        logits = self.forward(features)
        loss_tbi = self.loss(logits, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)
        assert len(logits.shape) == 6
        logits = logits - logits.logsumexp(dim=-3, keepdim=True)
        logits = torch.logsumexp(logits, dim=2)
        logits = logits - logits.logsumexp(dim=-3, keepdim=True)
        decoded = logits
        assert len(loss_tbi.shape) == 3 and len(decoded.shape) == 5
        return loss_tbi, loss_tb, decoded


class ConvDecoder(nn.Module):

    def __init__(self, in_dim, out_channels=3, cnn_depth=32, mlp_layers=0, layer_norm=True, activation=nn.ELU):
        super().__init__()
        self.in_dim = in_dim
        kernels = 5, 5, 6, 6
        stride = 2
        d = cnn_depth
        if mlp_layers == 0:
            layers = [nn.Linear(in_dim, d * 32)]
        else:
            hidden_dim = d * 32
            norm = nn.LayerNorm if layer_norm else NoNorm
            layers = [nn.Linear(in_dim, hidden_dim), norm(hidden_dim, eps=0.001), activation()]
            for _ in range(mlp_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), norm(hidden_dim, eps=0.001), activation()]
        self.model = nn.Sequential(*layers, nn.Unflatten(-1, (d * 32, 1, 1)), nn.ConvTranspose2d(d * 32, d * 4, kernels[0], stride), activation(), nn.ConvTranspose2d(d * 4, d * 2, kernels[1], stride), activation(), nn.ConvTranspose2d(d * 2, d, kernels[2], stride), activation(), nn.ConvTranspose2d(d, out_channels, kernels[3], stride))

    def forward(self, x: Tensor) ->Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y

    def loss(self, output: Tensor, target: Tensor) ->Tensor:
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 3)
        loss = 0.5 * torch.square(output - target).sum(dim=[-1, -2, -3])
        return unflatten_batch(loss, bd)

    def training_step(self, features: TensorTBIF, target: TensorTBCHW) ->Tuple[TensorTBI, TensorTB, TensorTBCHW]:
        assert len(features.shape) == 4 and len(target.shape) == 5
        I = features.shape[2]
        target = insert_dim(target, 2, I)
        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)
        decoded = decoded.mean(dim=2)
        assert len(loss_tbi.shape) == 3 and len(decoded.shape) == 5
        return loss_tbi, loss_tb, decoded


class DenseBernoulliDecoder(nn.Module):

    def __init__(self, in_dim, hidden_dim=400, hidden_layers=2, layer_norm=True):
        super().__init__()
        self.model = MLP(in_dim, 1, hidden_dim, hidden_layers, layer_norm)

    def forward(self, features: Tensor) ->D.Distribution:
        y = self.model.forward(features)
        p = D.Bernoulli(logits=y.float())
        return p

    def loss(self, output: D.Distribution, target: Tensor) ->Tensor:
        return -output.log_prob(target)

    def training_step(self, features: TensorTBIF, target: Tensor) ->Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)
        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)
        decoded = decoded.mean.mean(dim=2)
        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_tbi, loss_tb, decoded


class CategoricalSupport(D.Categorical):

    def __init__(self, logits, sup):
        assert logits.shape[-1:] == sup.shape
        super().__init__(logits=logits)
        self.sup = sup

    @property
    def mean(self):
        return torch.einsum('...i,i->...', self.probs, self.sup)


class DenseCategoricalSupportDecoder(nn.Module):
    """
    Represent continuous variable distribution by discrete set of support values.
    Useful for reward head, which can be e.g. [-10, 0, 1, 10]
    """

    def __init__(self, in_dim, support=[0.0, 1.0], hidden_dim=400, hidden_layers=2, layer_norm=True):
        assert isinstance(support, (list, np.ndarray))
        super().__init__()
        self.model = MLP(in_dim, len(support), hidden_dim, hidden_layers, layer_norm)
        self.support = np.array(support).astype(float)
        self._support = nn.Parameter(torch.tensor(support), requires_grad=False)

    def forward(self, features: Tensor) ->D.Distribution:
        y = self.model.forward(features)
        p = CategoricalSupport(logits=y.float(), sup=self._support.data)
        return p

    def loss(self, output: D.Distribution, target: Tensor) ->Tensor:
        target = self.to_categorical(target)
        return -output.log_prob(target)

    def to_categorical(self, target: Tensor) ->Tensor:
        distances = torch.square(target.unsqueeze(-1) - self._support)
        return distances.argmin(-1)

    def training_step(self, features: TensorTBIF, target: Tensor) ->Tuple[TensorTBI, TensorTB, TensorTB]:
        assert len(features.shape) == 4
        I = features.shape[2]
        target = insert_dim(target, 2, I)
        decoded = self.forward(features)
        loss_tbi = self.loss(decoded, target)
        loss_tb = -logavgexp(-loss_tbi, dim=2)
        decoded = decoded.mean.mean(dim=2)
        assert len(loss_tbi.shape) == 3
        assert len(loss_tb.shape) == 2
        assert len(decoded.shape) == 2
        return loss_tbi, loss_tb, decoded


def clip_rewards_np(x: np.ndarray, type_: Optional[str]=None) ->np.ndarray:
    if not type_:
        return x
    if type_ == 'tanh':
        return np.tanh(x)
    if type_ == 'log1p':
        return np.log1p(x)
    assert False, type_


def nanmean(x: Tensor) ->Tensor:
    return torch.nansum(x) / (~torch.isnan(x)).sum()


class MultiDecoder(nn.Module):

    def __init__(self, features_dim, conf):
        super().__init__()
        self.image_weight = conf.image_weight
        self.vecobs_weight = conf.vecobs_weight
        self.reward_weight = conf.reward_weight
        self.terminal_weight = conf.terminal_weight
        if conf.image_decoder == 'cnn':
            self.image = ConvDecoder(in_dim=features_dim, out_channels=conf.image_channels, cnn_depth=conf.cnn_depth)
        elif conf.image_decoder == 'dense':
            self.image = CatImageDecoder(in_dim=features_dim, out_shape=(conf.image_channels, conf.image_size, conf.image_size), hidden_layers=conf.image_decoder_layers, layer_norm=conf.layer_norm, min_prob=conf.image_decoder_min_prob)
        elif not conf.image_decoder:
            self.image = None
        else:
            assert False, conf.image_decoder
        if conf.reward_decoder_categorical:
            self.reward = DenseCategoricalSupportDecoder(in_dim=features_dim, support=clip_rewards_np(conf.reward_decoder_categorical, conf.clip_rewards), hidden_layers=conf.reward_decoder_layers, layer_norm=conf.layer_norm)
        else:
            self.reward = DenseNormalDecoder(in_dim=features_dim, hidden_layers=conf.reward_decoder_layers, layer_norm=conf.layer_norm)
        self.terminal = DenseBernoulliDecoder(in_dim=features_dim, hidden_layers=conf.terminal_decoder_layers, layer_norm=conf.layer_norm)
        if conf.vecobs_size:
            self.vecobs = DenseNormalDecoder(in_dim=features_dim, out_dim=conf.vecobs_size, hidden_layers=4, layer_norm=conf.layer_norm)
        else:
            self.vecobs = None

    def training_step(self, features: TensorTBIF, obs: Dict[str, Tensor], extra_metrics: bool=False) ->Tuple[TensorTBI, Dict[str, Tensor], Dict[str, Tensor]]:
        tensors = {}
        metrics = {}
        loss_reconstr = 0
        if self.image:
            loss_image_tbi, loss_image, image_rec = self.image.training_step(features, obs['image'])
            loss_reconstr += self.image_weight * loss_image_tbi
            metrics.update(loss_image=loss_image.detach().mean())
            tensors.update(loss_image=loss_image.detach(), image_rec=image_rec.detach())
        if self.vecobs:
            loss_vecobs_tbi, loss_vecobs, vecobs_rec = self.vecobs.training_step(features, obs['vecobs'])
            loss_reconstr += self.vecobs_weight * loss_vecobs_tbi
            metrics.update(loss_vecobs=loss_vecobs.detach().mean())
            tensors.update(loss_vecobs=loss_vecobs.detach(), vecobs_rec=vecobs_rec.detach())
        loss_reward_tbi, loss_reward, reward_rec = self.reward.training_step(features, obs['reward'])
        loss_reconstr += self.reward_weight * loss_reward_tbi
        metrics.update(loss_reward=loss_reward.detach().mean())
        tensors.update(loss_reward=loss_reward.detach(), reward_rec=reward_rec.detach())
        loss_terminal_tbi, loss_terminal, terminal_rec = self.terminal.training_step(features, obs['terminal'])
        loss_reconstr += self.terminal_weight * loss_terminal_tbi
        metrics.update(loss_terminal=loss_terminal.detach().mean())
        tensors.update(loss_terminal=loss_terminal.detach(), terminal_rec=terminal_rec.detach())
        if extra_metrics:
            if isinstance(self.reward, DenseCategoricalSupportDecoder):
                reward_cat = self.reward.to_categorical(obs['reward'])
                for i in range(len(self.reward.support)):
                    mask_rewardp = reward_cat == i
                    loss_rewardp = loss_reward * mask_rewardp / mask_rewardp
                    metrics[f'loss_reward{i}'] = nanmean(loss_rewardp)
                    tensors[f'loss_reward{i}'] = loss_rewardp
            else:
                for sig in [-1, 1]:
                    mask_rewardp = torch.sign(obs['reward']) == sig
                    loss_rewardp = loss_reward * mask_rewardp / mask_rewardp
                    metrics[f'loss_reward{sig}'] = nanmean(loss_rewardp)
                    tensors[f'loss_reward{sig}'] = loss_rewardp
            mask_terminal1 = obs['terminal'] > 0
            loss_terminal1 = loss_terminal * mask_terminal1 / mask_terminal1
            metrics['loss_terminal1'] = nanmean(loss_terminal1)
            tensors['loss_terminal1'] = loss_terminal1
        return loss_reconstr, metrics, tensors


def diag_normal(x: Tensor, min_std=0.1, max_std=2.0):
    mean, std = x.chunk(2, -1)
    std = max_std * torch.sigmoid(std) + min_std
    return D.independent.Independent(D.normal.Normal(mean, std), 1)


class VAEWorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.kl_weight = conf.kl_weight
        self.out_dim = conf.stoch_dim
        self.encoder = MultiEncoder(conf)
        self.post_mlp = nn.Sequential(nn.Linear(self.encoder.out_dim, 256), nn.ELU(), nn.Linear(256, 2 * conf.stoch_dim))
        self.decoder = MultiDecoder(conf.stoch_dim, conf)

    def init_state(self, batch_size: int):
        return None

    def training_step(self, obs: Dict[str, Tensor], in_state: Any=None, iwae_samples: int=1, do_open_loop=False, do_image_pred=False):
        embed = self.encoder(obs)
        post = self.post_mlp(embed)
        post = insert_dim(post, 2, iwae_samples)
        post_distr = diag_normal(post)
        z = post_distr.rsample()
        loss_reconstr, metrics, tensors = self.decoder.training_step(z, obs)
        prior_distr = diag_normal(torch.zeros_like(post))
        loss_kl = D.kl.kl_divergence(post_distr, prior_distr)
        assert loss_kl.shape == loss_reconstr.shape
        loss_model_tbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_tbi, dim=2)
        with torch.no_grad():
            loss_kl = -logavgexp(-loss_kl, dim=2)
            entropy_post = post_distr.entropy().mean(dim=2)
            tensors.update(loss_kl=loss_kl.detach(), entropy_post=entropy_post)
            metrics.update(loss_model=loss_model.mean(), loss_kl=loss_kl.mean(), entropy_post=entropy_post.mean())
        if do_image_pred:
            with torch.no_grad():
                zprior = prior_distr.sample()
                _, mets, tens = self.decoder.training_step(zprior, obs, extra_metrics=True)
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                tensors.update(**tensors_pred)
        return loss_model.mean(), z, None, None, metrics, tensors


class GRUVAEWorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.state_dim = conf.deter_dim
        self.out_dim = self.state_dim
        self.embedding = VAEWorldModel(conf)
        self.rnn = nn.GRU(self.embedding.out_dim + conf.action_dim, self.state_dim)
        self.dynamics = DenseNormalDecoder(self.state_dim, self.embedding.out_dim, hidden_layers=2)

    def init_state(self, batch_size: int) ->Any:
        device = next(self.rnn.parameters()).device
        return torch.zeros((1, batch_size, self.state_dim), device=device)

    def training_step(self, obs: Dict[str, Tensor], in_state: Any, iwae_samples: int=1, do_open_loop=False, do_image_pred=False):
        reset_first = obs['reset'].select(0, 0)
        state_mask = ~reset_first.unsqueeze(0).unsqueeze(-1)
        in_state = in_state * state_mask
        loss, embed, _, _, metrics, tensors = self.embedding.training_step(obs, None, iwae_samples=iwae_samples, do_image_pred=do_image_pred)
        T, B, I = embed.shape[:3]
        embed = embed.reshape((T, B * I, -1))
        embed = embed.detach()
        action_next = obs['action_next']
        embed_act = torch.cat([embed, action_next], -1)
        features, out_state = self.rnn(embed_act, in_state)
        features = features.reshape((T, B, I, -1))
        out_state = out_state.detach()
        embed_next = embed[1:]
        _, loss_dyn, embed_pred = self.dynamics.training_step(features[:-1], embed_next)
        loss += loss_dyn.mean()
        metrics['loss_dyn'] = loss_dyn.detach().mean()
        tensors['loss_dyn'] = loss_dyn.detach()
        if do_image_pred:
            with torch.no_grad():
                z = embed_pred
                z = torch.cat([torch.zeros_like(z[0]).unsqueeze(0), z])
                _, mets, tens = self.embedding.decoder.training_step(z.unsqueeze(2), obs, extra_metrics=True)
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                tensors.update(**tensors_pred)
        return loss, features, None, out_state, metrics, tensors


class GoalsProbe(nn.Module):

    def __init__(self, state_dim, conf):
        super().__init__()
        self.decoders = nn.ModuleDict({'goal_direction': DenseNormalDecoder(in_dim=state_dim, out_dim=2, hidden_layers=4, layer_norm=True), 'goals_direction': DenseNormalDecoder(in_dim=state_dim, out_dim=conf.goals_size * 2, hidden_layers=4, layer_norm=True)})

    def training_step(self, features: TensorTBIF, obs: Dict[str, Tensor]):
        loss_total = 0
        metrics = {}
        tensors = {}
        for key, decoder in self.decoders.items():
            assert isinstance(decoder, DenseNormalDecoder)
            target = obs[key]
            _, loss, pred = decoder.training_step(features, target)
            loss_total += loss.mean()
            metrics[f'loss_{key}'] = loss.detach().mean()
            tensors[f'loss_{key}'] = loss.detach()
            tensors[f'{key}_pred'] = pred.detach()
        with torch.no_grad():
            goals = obs['goals_direction']
            pred = tensors['goals_direction_pred']
            mse_per_coord = (goals - pred) ** 2
            mse_per_goal = mse_per_coord.reshape(mse_per_coord.shape[:-1] + (-1, 2)).sum(-1)
            metrics['mse_goals'] = mse_per_goal.mean(-1).mean()
            var_per_coord = goals.reshape((-1, goals.shape[-1])).var(0)
            var_per_goal = var_per_coord.reshape((-1, 2)).sum(-1)
            metrics['var_goals'] = var_per_goal.mean()
            log_ranges = [-1, 0, 5, 10, 50, 200, 1000]
            visage = obs.get('goals_visage')
            if visage is not None:
                assert mse_per_goal.shape == visage.shape
                for i in range(1, len(log_ranges)):
                    vmin = log_ranges[i - 1] + 1
                    vmax = log_ranges[i]
                    mask = (vmin <= visage) & (visage <= vmax)
                    metrics[f'mse_goal_age{vmax}'] = nanmean(mse_per_goal * mask / mask)
        return loss_total, metrics, tensors


IntTensorTBHW = Tensor


class MapProbeHead(nn.Module):

    def __init__(self, map_state_dim, conf):
        super().__init__()
        if conf.map_decoder == 'dense':
            self.decoder = CatImageDecoder(in_dim=map_state_dim, out_shape=(conf.map_channels, conf.map_size, conf.map_size), hidden_dim=conf.map_hidden_dim, hidden_layers=conf.map_hidden_layers, layer_norm=conf.layer_norm)
        else:
            raise NotImplementedError(conf.map_decoder)

    def training_step(self, features: TensorTBIF, obs: Dict[str, Tensor]):
        I = features.shape[2]
        map_coord = insert_dim(obs['map_coord'], 2, I)
        map_features = torch.cat((features, map_coord), dim=-1)
        _, loss, map_pred = self.decoder.training_step(map_features, obs['map'])
        with torch.no_grad():
            map_pred = map_pred.detach()
            acc_map = self.accuracy(map_pred, obs['map'])
            tensors = dict(map_rec=map_pred, loss_map=loss.detach(), acc_map=acc_map)
            metrics = dict(loss_map=loss.mean(), acc_map=nanmean(acc_map))
            if 'map_seen_mask' in obs:
                acc_map_seen = self.accuracy(map_pred, obs['map'], obs['map_seen_mask'])
                metrics['acc_map_seen'] = nanmean(acc_map_seen)
        return loss.mean(), metrics, tensors

    def accuracy(self, output: TensorTBCHW, target: Union[TensorTBCHW, IntTensorTBHW], map_seen_mask: Optional[Tensor]=None):
        if len(output.shape) == len(target.shape):
            target = target.argmax(dim=-3)
        output, bd = flatten_batch(output, 3)
        target, _ = flatten_batch(target, 2)
        acc = output.argmax(dim=-3) == target
        if map_seen_mask is None:
            acc = acc.mean([-1, -2])
        else:
            map_seen_mask, _ = flatten_batch(map_seen_mask, 2)
            acc = (acc * map_seen_mask).sum([-1, -2]) / map_seen_mask.sum([-1, -2])
        acc = unflatten_batch(acc, bd)
        return acc


class MapGoalsProbe(nn.Module):
    """Combined MapProbeHead and GoalsProbe."""

    def __init__(self, state_dim, conf):
        super().__init__()
        self.map_probe = MapProbeHead(state_dim + 4, conf)
        self.goals_probe = GoalsProbe(state_dim, conf)

    def training_step(self, features: TensorTBIF, obs: Dict[str, Tensor]):
        loss_map, metrics_map, tensors_map = self.map_probe.training_step(features, obs)
        loss_goals, metrics_goals, tensors_goals = self.goals_probe.training_step(features, obs)
        loss_total = loss_map + loss_goals
        metrics = dict(**metrics_map, **metrics_goals)
        tensors = dict(**tensors_map, **tensors_goals)
        return loss_total, metrics, tensors


class NoProbeHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def training_step(self, features: TensorTBIF, obs: Dict[str, Tensor]):
        return torch.square(self.dummy), {}, {}


class TransformerVAEWorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.state_dim = 512
        self.out_dim = self.state_dim
        self.embedding = VAEWorldModel(conf)
        self.transformer_in = nn.Linear(self.embedding.out_dim + conf.action_dim, 512)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(512, nhead=8, dim_feedforward=2048, dropout=0.1), num_layers=6, norm=nn.LayerNorm(512))
        self.dynamics = DenseNormalDecoder(self.state_dim, self.embedding.out_dim, hidden_layers=2)

    def init_state(self, batch_size: int) ->Any:
        return None

    def training_step(self, obs: Dict[str, Tensor], in_state: Any, iwae_samples: int=1, do_open_loop=False, do_image_pred=False):
        loss, embed, _, _, metrics, tensors = self.embedding.training_step(obs, None, iwae_samples=iwae_samples, do_image_pred=do_image_pred)
        T, B, I = embed.shape[:3]
        embed = embed.reshape((T, B * I, -1))
        embed = embed.detach()
        action_next = obs['action_next']
        embed_act = torch.cat([embed, action_next], -1)
        state_in = self.transformer_in(embed_act)
        features = self.transformer(state_in)
        features = features.reshape((T, B, I, -1))
        embed_next = embed[1:]
        _, loss_dyn, embed_pred = self.dynamics.training_step(features[:-1], embed_next)
        loss += loss_dyn.mean()
        metrics['loss_dyn'] = loss_dyn.detach().mean()
        tensors['loss_dyn'] = loss_dyn.detach()
        if do_image_pred:
            with torch.no_grad():
                z = embed_pred
                z = torch.cat([torch.zeros_like(z[0]).unsqueeze(0), z])
                _, mets, tens = self.embedding.decoder.training_step(z.unsqueeze(2), obs, extra_metrics=True)
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                tensors.update(**tensors_pred)
        return loss, features, None, None, metrics, tensors


def init_weights_tf2(m):
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if type(m) == nn.GRUCell or type(m) == rnn.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)
    if type(m) == rnn.NormGRUCell or type(m) == rnn.NormGRUCellLateReset:
        nn.init.xavier_uniform_(m.weight_ih.weight.data)
        nn.init.orthogonal_(m.weight_hh.weight.data)


class WorldModelProbe(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.probe_gradients = conf.probe_gradients
        if conf.model == 'vae':
            self.wm = VAEWorldModel(conf)
        elif conf.model == 'gru_vae':
            self.wm = GRUVAEWorldModel(conf)
        elif conf.model == 'transformer_vae':
            self.wm = TransformerVAEWorldModel(conf)
        elif conf.model == 'gru_probe':
            self.wm = GRUEncoderOnly(conf)
        else:
            raise ValueError(conf.model)
        if conf.probe_model == 'map':
            probe_model = MapProbeHead(self.wm.out_dim + 4, conf)
        elif conf.probe_model == 'goals':
            probe_model = GoalsProbe(self.wm.out_dim, conf)
        elif conf.probe_model == 'map+goals':
            probe_model = MapGoalsProbe(self.wm.out_dim, conf)
        elif conf.probe_model == 'none':
            probe_model = NoProbeHead()
        else:
            raise NotImplementedError(f'Unknown probe_model={conf.probe_model}')
        self.probe_model = probe_model
        for m in self.modules():
            init_weights_tf2(m)

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-05):
        if not self.probe_gradients:
            optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
            optimizer_probe = torch.optim.AdamW(self.probe_model.parameters(), lr=lr, eps=eps)
            return optimizer_wm, optimizer_probe
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, eps=eps)
            return optimizer,

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        if not self.probe_gradients:
            grad_metrics = {'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip), 'grad_norm_probe': nn.utils.clip_grad_norm_(self.probe_model.parameters(), grad_clip)}
        else:
            grad_metrics = {'grad_norm': nn.utils.clip_grad_norm_(self.parameters(), grad_clip)}
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def training_step(self, obs: Dict[str, Tensor], in_state: Any, iwae_samples: int=1, imag_horizon: Optional[int]=None, do_open_loop=False, do_image_pred=False, do_dream_tensors=False):
        loss_model, features, states, out_state, metrics, tensors = self.wm.training_step(obs, in_state, iwae_samples=iwae_samples, do_open_loop=do_open_loop, do_image_pred=do_image_pred)
        if not self.probe_gradients:
            features = features.detach()
        loss_probe, metrics_probe, tensors_probe = self.probe_model.training_step(features, obs)
        metrics.update(**metrics_probe)
        tensors.update(**tensors_probe)
        if not self.probe_gradients:
            losses = loss_model, loss_probe
        else:
            losses = loss_model + loss_probe,
        return losses, out_state, metrics, tensors, {}


StateB = Tuple[Tensor, Tensor]


class RSSMCell(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.stoch_discrete = stoch_discrete
        self.deter_dim = deter_dim
        norm = nn.LayerNorm if layer_norm else NoNorm
        self.z_mlp = nn.Linear(stoch_dim * (stoch_discrete or 1), hidden_dim)
        self.a_mlp = nn.Linear(action_dim, hidden_dim, bias=False)
        self.in_norm = norm(hidden_dim, eps=0.001)
        self.gru = rnn.GRUCellStack(hidden_dim, deter_dim, gru_layers, gru_type)
        self.prior_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.prior_norm = norm(hidden_dim, eps=0.001)
        self.prior_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))
        self.post_mlp_h = nn.Linear(deter_dim, hidden_dim)
        self.post_mlp_e = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.post_norm = norm(hidden_dim, eps=0.001)
        self.post_mlp = nn.Linear(hidden_dim, stoch_dim * (stoch_discrete or 2))

    def init_state(self, batch_size):
        device = next(self.gru.parameters()).device
        return torch.zeros((batch_size, self.deter_dim), device=device), torch.zeros((batch_size, self.stoch_dim * (self.stoch_discrete or 1)), device=device)

    def forward(self, embed: Tensor, action: Tensor, reset_mask: Tensor, in_state: Tuple[Tensor, Tensor]) ->Tuple[Tensor, Tuple[Tensor, Tensor]]:
        in_h, in_z = in_state
        in_h = in_h * reset_mask
        in_z = in_z * reset_mask
        B = action.shape[0]
        x = self.z_mlp(in_z) + self.a_mlp(action)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)
        x = self.post_mlp_h(h) + self.post_mlp_e(embed)
        x = self.post_norm(x)
        post_in = F.elu(x)
        post = self.post_mlp(post_in)
        post_distr = self.zdistr(post)
        sample = post_distr.rsample().reshape(B, -1)
        return post, (h, sample)

    def forward_prior(self, action: Tensor, reset_mask: Optional[Tensor], in_state: Tuple[Tensor, Tensor]) ->Tuple[Tensor, Tuple[Tensor, Tensor]]:
        in_h, in_z = in_state
        if reset_mask is not None:
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask
        B = action.shape[0]
        x = self.z_mlp(in_z) + self.a_mlp(action)
        x = self.in_norm(x)
        za = F.elu(x)
        h = self.gru(za, in_h)
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)
        prior_distr = self.zdistr(prior)
        sample = prior_distr.rsample().reshape(B, -1)
        return prior, (h, sample)

    def batch_prior(self, h: Tensor) ->Tensor:
        x = self.prior_mlp_h(h)
        x = self.prior_norm(x)
        x = F.elu(x)
        prior = self.prior_mlp(x)
        return prior

    def zdistr(self, pp: Tensor) ->D.Distribution:
        if self.stoch_discrete:
            logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_discrete))
            distr = D.OneHotCategoricalStraightThrough(logits=logits.float())
            distr = D.independent.Independent(distr, 1)
            return distr
        else:
            return diag_normal(pp)


class RSSMCore(nn.Module):

    def __init__(self, embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm):
        super().__init__()
        self.cell = RSSMCell(embed_dim, action_dim, deter_dim, stoch_dim, stoch_discrete, hidden_dim, gru_layers, gru_type, layer_norm)

    def forward(self, embed: Tensor, action: Tensor, reset: Tensor, in_state: Tuple[Tensor, Tensor], iwae_samples: int=1, do_open_loop=False):
        T, B = embed.shape[:2]
        I = iwae_samples

        def expand(x):
            return x.unsqueeze(2).expand(T, B, I, -1).reshape(T, B * I, -1)
        embeds = expand(embed).unbind(0)
        actions = expand(action).unbind(0)
        reset_masks = expand(~reset.unsqueeze(2)).unbind(0)
        priors = []
        posts = []
        states_h = []
        samples = []
        h, z = in_state
        for i in range(T):
            if not do_open_loop:
                post, (h, z) = self.cell.forward(embeds[i], actions[i], reset_masks[i], (h, z))
            else:
                post, (h, z) = self.cell.forward_prior(actions[i], reset_masks[i], (h, z))
            posts.append(post)
            states_h.append(h)
            samples.append(z)
        posts = torch.stack(posts)
        states_h = torch.stack(states_h)
        samples = torch.stack(samples)
        priors = self.cell.batch_prior(states_h)
        features = self.to_feature(states_h, samples)
        posts = posts.reshape(T, B, I, -1)
        states_h = states_h.reshape(T, B, I, -1)
        samples = samples.reshape(T, B, I, -1)
        priors = priors.reshape(T, B, I, -1)
        states = states_h, samples
        features = features.reshape(T, B, I, -1)
        return priors, posts, samples, features, states, (h.detach(), z.detach())

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def to_feature(self, h: Tensor, z: Tensor) ->Tensor:
        return torch.cat((h, z), -1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h, _ = features.split([self.cell.deter_dim, z.shape[-1]], -1)
        return self.to_feature(h, z)

    def zdistr(self, pp: Tensor) ->D.Distribution:
        return self.cell.zdistr(pp)


class WorldModel(nn.Module):

    def __init__(self, conf):
        super().__init__()
        self.deter_dim = conf.deter_dim
        self.stoch_dim = conf.stoch_dim
        self.stoch_discrete = conf.stoch_discrete
        self.kl_weight = conf.kl_weight
        self.kl_balance = None if conf.kl_balance == 0.5 else conf.kl_balance
        self.aux_critic_weight = conf.aux_critic_weight
        self.encoder = MultiEncoder(conf)
        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.decoder = MultiDecoder(features_dim, conf)
        self.core = RSSMCore(embed_dim=self.encoder.out_dim, action_dim=conf.action_dim, deter_dim=conf.deter_dim, stoch_dim=conf.stoch_dim, stoch_discrete=conf.stoch_discrete, hidden_dim=conf.hidden_dim, gru_layers=conf.gru_layers, gru_type=conf.gru_type, layer_norm=conf.layer_norm)
        if conf.aux_critic:
            self.ac_aux = ActorCritic(in_dim=features_dim, out_actions=conf.action_dim, layer_norm=conf.layer_norm, gamma=conf.gamma_aux, lambda_gae=conf.lambda_gae_aux, entropy_weight=conf.entropy, target_interval=conf.target_interval_aux, actor_grad=conf.actor_grad, actor_dist=conf.actor_dist)
        else:
            self.ac_aux = None
        for m in self.modules():
            init_weights_tf2(m)

    def init_state(self, batch_size: int) ->Tuple[Any, Any]:
        return self.core.init_state(batch_size)

    def forward(self, obs: Dict[str, Tensor], in_state: Any):
        loss, features, states, out_state, metrics, tensors = self.training_step(obs, in_state, forward_only=True)
        return features, out_state

    def training_step(self, obs: Dict[str, Tensor], in_state: Any, iwae_samples: int=1, do_open_loop=False, do_image_pred=False, forward_only=False):
        embed = self.encoder(obs)
        prior, post, post_samples, features, states, out_state = self.core.forward(embed, obs['action'], obs['reset'], in_state, iwae_samples=iwae_samples, do_open_loop=do_open_loop)
        if forward_only:
            return torch.tensor(0.0), features, states, out_state, {}, {}
        loss_reconstr, metrics, tensors = self.decoder.training_step(features, obs)
        d = self.core.zdistr
        dprior = d(prior)
        dpost = d(post)
        loss_kl_exact = D.kl.kl_divergence(dpost, dprior)
        if iwae_samples == 1:
            if not self.kl_balance:
                loss_kl = loss_kl_exact
            else:
                loss_kl_postgrad = D.kl.kl_divergence(dpost, d(prior.detach()))
                loss_kl_priograd = D.kl.kl_divergence(d(post.detach()), dprior)
                loss_kl = (1 - self.kl_balance) * loss_kl_postgrad + self.kl_balance * loss_kl_priograd
        else:
            z = post_samples.reshape(dpost.batch_shape + dpost.event_shape)
            loss_kl = dpost.log_prob(z) - dprior.log_prob(z)
        if self.ac_aux:
            features_tb = features.select(2, 0)
            (_, loss_critic_aux), metrics_ac, tensors_ac = self.ac_aux.training_step(features_tb, obs['action'][1:], obs['reward'], obs['terminal'])
            metrics.update(loss_critic_aux=metrics_ac['loss_critic'], policy_value_aux=metrics_ac['policy_value_im'])
            tensors.update(policy_value_aux=tensors_ac['value'])
        else:
            loss_critic_aux = 0.0
        assert loss_kl.shape == loss_reconstr.shape
        loss_model_tbi = self.kl_weight * loss_kl + loss_reconstr
        loss_model = -logavgexp(-loss_model_tbi, dim=2)
        loss = loss_model.mean() + self.aux_critic_weight * loss_critic_aux
        with torch.no_grad():
            loss_kl = -logavgexp(-loss_kl_exact, dim=2)
            entropy_prior = dprior.entropy().mean(dim=2)
            entropy_post = dpost.entropy().mean(dim=2)
            tensors.update(loss_kl=loss_kl.detach(), entropy_prior=entropy_prior, entropy_post=entropy_post)
            metrics.update(loss_model=loss_model.mean(), loss_kl=loss_kl.mean(), entropy_prior=entropy_prior.mean(), entropy_post=entropy_post.mean())
        if do_image_pred:
            with torch.no_grad():
                prior_samples = self.core.zdistr(prior).sample().reshape(post_samples.shape)
                features_prior = self.core.feature_replace_z(features, prior_samples)
                _, mets, tens = self.decoder.training_step(features_prior, obs, extra_metrics=True)
                metrics_logprob = {k.replace('loss_', 'logprob_'): v for k, v in mets.items() if k.startswith('loss_')}
                tensors_logprob = {k.replace('loss_', 'logprob_'): v for k, v in tens.items() if k.startswith('loss_')}
                tensors_pred = {k.replace('_rec', '_pred'): v for k, v in tens.items() if k.endswith('_rec')}
                metrics.update(**metrics_logprob)
                tensors.update(**tensors_logprob)
                tensors.update(**tensors_pred)
        return loss, features, states, out_state, metrics, tensors


T = TypeVar('T', Tensor, np.ndarray)


def map_structure(data: Union[Tuple[T, ...], Dict[str, T]], f: Callable[[T], T]) ->Union[Tuple[T, ...], Dict[str, T]]:
    if isinstance(data, tuple):
        return tuple(f(d) for d in data)
    elif isinstance(data, dict):
        return {k: f(v) for k, v in data.items()}
    else:
        raise NotImplementedError(type(data))


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Dreamer(nn.Module):

    def __init__(self, conf):
        super().__init__()
        assert conf.action_dim > 0, 'Need to set action_dim to match environment'
        features_dim = conf.deter_dim + conf.stoch_dim * (conf.stoch_discrete or 1)
        self.iwae_samples = conf.iwae_samples
        self.imag_horizon = conf.imag_horizon
        self.wm = WorldModel(conf)
        self.ac = ActorCritic(in_dim=features_dim, out_actions=conf.action_dim, layer_norm=conf.layer_norm, gamma=conf.gamma, lambda_gae=conf.lambda_gae, entropy_weight=conf.entropy, target_interval=conf.target_interval, actor_grad=conf.actor_grad, actor_dist=conf.actor_dist)
        if conf.probe_model == 'map':
            probe_model = MapProbeHead(features_dim + 4, conf)
        elif conf.probe_model == 'goals':
            probe_model = GoalsProbe(features_dim, conf)
        elif conf.probe_model == 'map+goals':
            probe_model = MapGoalsProbe(features_dim, conf)
        elif conf.probe_model == 'none':
            probe_model = NoProbeHead()
        else:
            raise NotImplementedError(f'Unknown probe_model={conf.probe_model}')
        self.probe_model = probe_model
        self.probe_gradients = conf.probe_gradients

    def init_optimizers(self, lr, lr_actor=None, lr_critic=None, eps=1e-05):
        if not self.probe_gradients:
            optimizer_wm = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
            optimizer_probe = torch.optim.AdamW(self.probe_model.parameters(), lr=lr, eps=eps)
            optimizer_actor = torch.optim.AdamW(self.ac.actor.parameters(), lr=lr_actor or lr, eps=eps)
            optimizer_critic = torch.optim.AdamW(self.ac.critic.parameters(), lr=lr_critic or lr, eps=eps)
            return optimizer_wm, optimizer_probe, optimizer_actor, optimizer_critic
        else:
            optimizer_wmprobe = torch.optim.AdamW(self.wm.parameters(), lr=lr, eps=eps)
            optimizer_actor = torch.optim.AdamW(self.ac.actor.parameters(), lr=lr_actor or lr, eps=eps)
            optimizer_critic = torch.optim.AdamW(self.ac.critic.parameters(), lr=lr_critic or lr, eps=eps)
            return optimizer_wmprobe, optimizer_actor, optimizer_critic

    def grad_clip(self, grad_clip, grad_clip_ac=None):
        if not self.probe_gradients:
            grad_metrics = {'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip), 'grad_norm_probe': nn.utils.clip_grad_norm_(self.probe_model.parameters(), grad_clip), 'grad_norm_actor': nn.utils.clip_grad_norm_(self.ac.actor.parameters(), grad_clip_ac or grad_clip), 'grad_norm_critic': nn.utils.clip_grad_norm_(self.ac.critic.parameters(), grad_clip_ac or grad_clip)}
        else:
            grad_metrics = {'grad_norm': nn.utils.clip_grad_norm_(self.wm.parameters(), grad_clip), 'grad_norm_actor': nn.utils.clip_grad_norm_(self.ac.actor.parameters(), grad_clip_ac or grad_clip), 'grad_norm_critic': nn.utils.clip_grad_norm_(self.ac.critic.parameters(), grad_clip_ac or grad_clip)}
        return grad_metrics

    def init_state(self, batch_size: int):
        return self.wm.init_state(batch_size)

    def inference(self, obs: Dict[str, Tensor], in_state: Any) ->Tuple[D.Distribution, Any, Dict]:
        assert 'action' in obs, 'Observation should contain previous action'
        act_shape = obs['action'].shape
        assert len(act_shape) == 3 and act_shape[0] == 1, f'Expected shape (1,B,A), got {act_shape}'
        features, out_state = self.wm.forward(obs, in_state)
        feature = features[:, :, 0]
        action_distr = self.ac.forward_actor(feature)
        value = self.ac.forward_value(feature)
        metrics = dict(policy_value=value.detach().mean())
        return action_distr, out_state, metrics

    def training_step(self, obs: Dict[str, Tensor], in_state: Any, iwae_samples: Optional[int]=None, imag_horizon: Optional[int]=None, do_open_loop=False, do_image_pred=False, do_dream_tensors=False):
        assert 'action' in obs, '`action` required in observation'
        assert 'reward' in obs, '`reward` required in observation'
        assert 'reset' in obs, '`reset` required in observation'
        assert 'terminal' in obs, '`terminal` required in observation'
        iwae_samples = int(iwae_samples or self.iwae_samples)
        imag_horizon = int(imag_horizon or self.imag_horizon)
        T, B = obs['action'].shape[:2]
        I, H = iwae_samples, imag_horizon
        loss_model, features, states, out_state, metrics, tensors = self.wm.training_step(obs, in_state, iwae_samples=iwae_samples, do_open_loop=do_open_loop, do_image_pred=do_image_pred)
        features_probe = features.detach() if not self.probe_gradients else features
        loss_probe, metrics_probe, tensors_probe = self.probe_model.training_step(features_probe, obs)
        metrics.update(**metrics_probe)
        tensors.update(**tensors_probe)
        in_state_dream: StateB = map_structure(states, lambda x: flatten_batch(x.detach())[0])
        features_dream, actions_dream, rewards_dream, terminals_dream = self.dream(in_state_dream, H, self.ac.actor_grad == 'dynamics')
        (loss_actor, loss_critic), metrics_ac, tensors_ac = self.ac.training_step(features_dream.detach(), actions_dream.detach(), rewards_dream.mean.detach(), terminals_dream.mean.detach())
        metrics.update(**metrics_ac)
        tensors.update(policy_value=unflatten_batch(tensors_ac['value'][0], (T, B, I)).mean(-1))
        dream_tensors = {}
        if do_dream_tensors and self.wm.decoder.image is not None:
            with torch.no_grad():
                in_state_dream: StateB = map_structure(states, lambda x: x.detach()[0, :, 0])
                features_dream, actions_dream, rewards_dream, terminals_dream = self.dream(in_state_dream, T - 1)
                image_dream = self.wm.decoder.image.forward(features_dream)
                _, _, tensors_ac = self.ac.training_step(features_dream, actions_dream, rewards_dream.mean, terminals_dream.mean, log_only=True)
                dream_tensors = dict(action_pred=torch.cat([obs['action'][:1], actions_dream]), reward_pred=rewards_dream.mean, terminal_pred=terminals_dream.mean, image_pred=image_dream, **tensors_ac)
                assert dream_tensors['action_pred'].shape == obs['action'].shape
                assert dream_tensors['image_pred'].shape == obs['image'].shape
        if not self.probe_gradients:
            losses = loss_model, loss_probe, loss_actor, loss_critic
        else:
            losses = loss_model + loss_probe, loss_actor, loss_critic
        return losses, out_state, metrics, tensors, dream_tensors

    def dream(self, in_state: StateB, imag_horizon: int, dynamics_gradients=False):
        features = []
        actions = []
        state = in_state
        self.wm.requires_grad_(False)
        for i in range(imag_horizon):
            feature = self.wm.core.to_feature(*state)
            action_dist = self.ac.forward_actor(feature)
            if dynamics_gradients:
                action = action_dist.rsample()
            else:
                action = action_dist.sample()
            features.append(feature)
            actions.append(action)
            _, state = self.wm.core.cell.forward_prior(action, None, state)
        feature = self.wm.core.to_feature(*state)
        features.append(feature)
        features = torch.stack(features)
        actions = torch.stack(actions)
        rewards = self.wm.decoder.reward.forward(features)
        terminals = self.wm.decoder.terminal.forward(features)
        self.wm.requires_grad_(True)
        return features, actions, rewards, terminals

    def __str__(self):
        s = []
        s.append(f'Model: {param_count(self)} parameters')
        for submodel in (self.wm.encoder, self.wm.decoder, self.wm.core, self.ac, self.probe_model):
            if submodel is not None:
                s.append(f'  {type(submodel).__name__:<15}: {param_count(submodel)} parameters')
        return '\n'.join(s)

    def __repr__(self):
        return super().__repr__()


class GRU2Inputs(nn.Module):

    def __init__(self, input1_dim, input2_dim, mlp_dim=200, state_dim=200, num_layers=1, bidirectional=False, input_activation=F.elu):
        super().__init__()
        self.in_mlp1 = nn.Linear(input1_dim, mlp_dim)
        self.in_mlp2 = nn.Linear(input2_dim, mlp_dim, bias=False)
        self.act = input_activation
        self.gru = nn.GRU(input_size=mlp_dim, hidden_size=state_dim, num_layers=num_layers, bidirectional=bidirectional)
        self.directions = 2 if bidirectional else 1

    def init_state(self, batch_size):
        device = next(self.gru.parameters()).device
        return torch.zeros((self.gru.num_layers * self.directions, batch_size, self.gru.hidden_size), device=device)

    def forward(self, input1_seq: Tensor, input2_seq: Tensor, in_state: Optional[Tensor]=None) ->Tuple[Tensor, Tensor]:
        if in_state is None:
            in_state = self.init_state(input1_seq.size(1))
        inp = self.act(self.in_mlp1(input1_seq) + self.in_mlp2(input2_seq))
        output, out_state = self.gru(inp, in_state)
        return output, out_state.detach()


class NormGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.ln_reset = nn.LayerNorm(hidden_size, eps=0.001)
        self.ln_update = nn.LayerNorm(hidden_size, eps=0.001)
        self.ln_newval = nn.LayerNorm(hidden_size, eps=0.001)

    def forward(self, input: Tensor, state: Tensor) ->Tensor:
        gates_i = self.weight_ih(input)
        gates_h = self.weight_hh(state)
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)
        reset = torch.sigmoid(self.ln_reset(reset_i + reset_h))
        update = torch.sigmoid(self.ln_update(update_i + update_h))
        newval = torch.tanh(self.ln_newval(newval_i + reset * newval_h))
        h = update * newval + (1 - update) * state
        return h


class NormGRUCellLateReset(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size, bias=False)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.lnorm = nn.LayerNorm(3 * hidden_size, eps=0.001)
        self.update_bias = -1

    def forward(self, input: Tensor, state: Tensor) ->Tensor:
        gates = self.weight_ih(input) + self.weight_hh(state)
        gates = self.lnorm(gates)
        reset, update, newval = gates.chunk(3, 1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update + self.update_bias)
        newval = torch.tanh(reset * newval)
        h = update * newval + (1 - update) * state
        return h


class GRUCellStack(nn.Module):
    """Multi-layer stack of GRU cells"""

    def __init__(self, input_size, hidden_size, num_layers, cell_type):
        super().__init__()
        self.num_layers = num_layers
        layer_size = hidden_size // num_layers
        assert layer_size * num_layers == hidden_size, 'Must be divisible'
        if cell_type == 'gru':
            cell = nn.GRUCell
        elif cell_type == 'gru_layernorm':
            cell = NormGRUCell
        elif cell_type == 'gru_layernorm_dv2':
            cell = NormGRUCellLateReset
        else:
            assert False, f'Unknown cell type {cell_type}'
        layers = [cell(input_size, layer_size)]
        layers.extend([cell(layer_size, layer_size) for _ in range(num_layers - 1)])
        self.layers = nn.ModuleList(layers)

    def forward(self, input: Tensor, state: Tensor) ->Tensor:
        input_states = state.chunk(self.num_layers, -1)
        output_states = []
        x = input
        for i in range(self.num_layers):
            x = self.layers[i](x, input_states[i])
            output_states.append(x)
        return torch.cat(output_states, -1)


class GRUCell(jit.ScriptModule):
    """Reproduced regular nn.GRUCell, for reference"""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(input_size, 3 * hidden_size))
        self.weight_hh = Parameter(torch.randn(hidden_size, 3 * hidden_size))
        self.bias_ih = Parameter(torch.randn(3 * hidden_size))
        self.bias_hh = Parameter(torch.randn(3 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tensor) ->Tensor:
        gates_i = torch.mm(input, self.weight_ih) + self.bias_ih
        gates_h = torch.mm(state, self.weight_hh) + self.bias_hh
        reset_i, update_i, newval_i = gates_i.chunk(3, 1)
        reset_h, update_h, newval_h = gates_h.chunk(3, 1)
        reset = torch.sigmoid(reset_i + reset_h)
        update = torch.sigmoid(update_i + update_h)
        newval = torch.tanh(newval_i + reset * newval_h)
        h = update * newval + (1 - update) * state
        return h


class LSTMCell(jit.ScriptModule):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    @jit.script_method
    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) ->Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = torch.mm(input, self.weight_ih.t()) + self.bias_ih + torch.mm(hx, self.weight_hh.t()) + self.bias_hh
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        return hy, (hy, cy)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CatImageDecoder,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvDecoder,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ConvEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (DenseBernoulliDecoder,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseCategoricalSupportDecoder,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DenseNormalDecoder,
     lambda: ([], {'in_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GRU2Inputs,
     lambda: ([], {'input1_dim': 4, 'input2_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (MLP,
     lambda: ([], {'in_dim': 4, 'out_dim': 4, 'hidden_dim': 4, 'hidden_layers': 1, 'layer_norm': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NoNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (NormGRUCellLateReset,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_jurgisp_pydreamer(_paritybench_base):
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

