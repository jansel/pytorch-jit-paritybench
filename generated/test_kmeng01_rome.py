import sys
_module = sys.modules[__name__]
del sys
efk = _module
ft = _module
ft_hparams = _module
ft_main = _module
kn = _module
kn_hparams = _module
kn_main = _module
knowledge_neurons = _module
data = _module
knowledge_neurons = _module
patch = _module
pararel_evaluate = _module
plot_pararel_results = _module
setup = _module
tests = _module
mend = _module
efk = _module
enn = _module
ft = _module
mend = _module
fever = _module
nq = _module
wiki = _module
zsre = _module
editable_model = _module
efk_hparams = _module
efk_main = _module
hooks = _module
losses = _module
mend_hparams = _module
mend_main = _module
models = _module
nn = _module
oracle = _module
run = _module
trainer = _module
utils = _module
dsets = _module
attr_snippets = _module
counterfact = _module
knowns = _module
tfidf_stats = _module
zsre = _module
experiments = _module
causal_trace = _module
evaluate = _module
demo = _module
eval_utils_counterfact = _module
eval_utils_zsre = _module
summarize = _module
sweep = _module
rome = _module
compute_u = _module
compute_v = _module
layer_stats = _module
repr_tools = _module
rome_hparams = _module
rome_main = _module
tok_dataset = _module
ipynb_drop_output = _module
util = _module
generate = _module
globals = _module
hparams = _module
logit_lens = _module
nethook = _module
perplexity = _module
runningstats = _module

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


from copy import deepcopy


from typing import Any


from typing import Dict


from typing import List


from typing import Tuple


import torch


import numpy as np


import collections


import math


from functools import partial


from typing import Callable


from typing import Optional


import torch.nn as nn


import torch.nn.functional as F


import random


from functools import lru_cache


import copy


import logging


import time


from collections import defaultdict


from torch.utils.data import Dataset


import re


import typing


from itertools import chain


import scipy.sparse as sp


from sklearn.feature_extraction.text import TfidfVectorizer


import numpy


from matplotlib import pyplot as plt


from time import time


from typing import Union


import scipy


from matplotlib.style import context


from torch.nn.utils.rnn import pad_sequence


import inspect


from collections import OrderedDict


from torch.utils.data.sampler import Sampler


class Patch(torch.nn.Module):
    """
    Patches a torch module to replace/suppress/enhance the intermediate activations
    """

    def __init__(self, ff_layer: nn.Module, mask_idx: int, replacement_activations: torch.Tensor=None, target_positions: List[List[int]]=None, mode: str='replace', enhance_value: float=2.0):
        super().__init__()
        self.ff = ff_layer
        self.acts = replacement_activations
        self.mask_idx = mask_idx
        self.target_positions = target_positions
        self.enhance_value = enhance_value
        assert mode in ['replace', 'suppress', 'enhance']
        self.mode = mode
        if self.mode == 'replace':
            assert self.acts is not None
        elif self.mode in ['enhance', 'suppress']:
            assert self.target_positions is not None

    def forward(self, x: torch.Tensor):
        x = self.ff(x)
        if self.mode == 'replace':
            x[:, self.mask_idx, :] = self.acts
        elif self.mode == 'suppress':
            for pos in self.target_positions:
                x[:, self.mask_idx, pos] = 0.0
        elif self.mode == 'enhance':
            for pos in self.target_positions:
                x[:, self.mask_idx, pos] *= self.enhance_value
        else:
            raise NotImplementedError
        return x


class ConditionedParameter(torch.nn.Module):

    def __init__(self, parameter, condition_dim=1024, hidden_dim=128, max_scale=1):
        super().__init__()
        self.parameter_shape = parameter.shape
        if len(self.parameter_shape) == 2:
            self.conditioners = torch.nn.Sequential(torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)), torch.nn.Tanh(), torch.nn.utils.weight_norm(torch.nn.Linear(hidden_dim, 2 * (parameter.shape[0] + parameter.shape[1]) + 1)))
        elif len(self.parameter_shape) == 1:
            self.conditioners = torch.nn.Sequential(torch.nn.utils.weight_norm(torch.nn.Linear(condition_dim, hidden_dim)), torch.nn.Tanh(), torch.nn.utils.weight_norm(torch.nn.Linear(hidden_dim, 2 * parameter.shape[0] + 1)))
        else:
            raise RuntimeError()
        self.max_scale = max_scale

    def forward(self, inputs, grad):
        if inputs.shape[0] > 1:
            raise RuntimeError('Can only condition on batches of size 1')
        if len(self.parameter_shape) == 2:
            conditioner_cola, conditioner_rowa, conditioner_colb, conditioner_rowb, conditioner_norm = self.conditioners(inputs).split([self.parameter_shape[1], self.parameter_shape[0], self.parameter_shape[1], self.parameter_shape[0], 1], dim=-1)
            a = conditioner_rowa.softmax(-1).T @ conditioner_cola
            b = conditioner_rowb.softmax(-1).T @ conditioner_colb
        elif len(self.parameter_shape) == 1:
            a, b, conditioner_norm = self.conditioners(inputs).split([self.parameter_shape[0], self.parameter_shape[0], 1], dim=-1)
        else:
            raise RuntimeError()
        if a.squeeze().shape[0] != grad.shape[0]:
            return self.max_scale * conditioner_norm.sigmoid().squeeze() * (grad * a.squeeze().T + b.squeeze().T)
        else:
            return self.max_scale * conditioner_norm.sigmoid().squeeze() * (grad * a.squeeze() + b.squeeze())


class LSTMConditioner(torch.nn.Module):

    def __init__(self, vocab_dim=30522, embedding_dim=768, hidden_dim=256, output_dim=1024, embedding_init=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_dim, embedding_dim=embedding_dim, padding_idx=0, _weight=embedding_init)
        self.lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True, batch_first=True))
        self.linear = FeedForward(input_dim=hidden_dim * 2, num_layers=1, hidden_dims=[output_dim], activations=[torch.nn.Tanh()])

    def forward(self, inputs, masks):
        return self.linear(self.lstm(self.embedding(inputs), masks))


class OneShotLearner(torch.nn.Module):

    def __init__(self, model, vocab_dim, embedding_dim=768, hidden_dim=512, condition_dim=768, include_set={}, max_scale=0.001, embedding_init=None):
        super().__init__()
        self.param2conditioner_map = {n: '{}_conditioner'.format(n).replace('.', '_') for n, p in model.named_parameters() if n in include_set}
        self.conditioners = torch.nn.ModuleDict({self.param2conditioner_map[n]: ConditionedParameter(p, condition_dim, hidden_dim, max_scale=max_scale) for n, p in model.named_parameters() if n in include_set})
        self.condition = LSTMConditioner(vocab_dim, embedding_dim, hidden_dim, condition_dim, embedding_init=embedding_init)

    def forward(self, inputs, masks, grads=None):
        condition = self.condition(inputs, masks)
        return {p: self.conditioners[self.param2conditioner_map[p]](condition, grad=grads[p] if grads else None) for p, c in self.param2conditioner_map.items()}


LOG = logging.getLogger(__name__)


def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)
    return new_m, new_s


class GradientTransform(nn.Module):

    def __init__(self, x_dim: int, delta_dim: int, cfg, n_modes=None):
        super().__init__()
        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.cfg = cfg
        if cfg.combine and (cfg.one_sided or cfg.x_only or cfg.delta_only):
            raise ValueError('cfg.combine cannot be used with one-sided MEND variants')
        self.norm_init = False
        self.register_buffer('u_mean', torch.full((x_dim,), float('nan')))
        self.register_buffer('v_mean', torch.full((delta_dim,), float('nan')))
        self.register_buffer('u_std', torch.full((x_dim,), float('nan')))
        self.register_buffer('v_std', torch.full((delta_dim,), float('nan')))
        self.register_buffer('u_s', torch.full((x_dim,), float('nan')))
        self.register_buffer('v_s', torch.full((delta_dim,), float('nan')))
        self.register_buffer('k', torch.full((1,), float('nan')))
        MlpClass = getattr(local_nn, cfg.mlp_class)
        LOG.info(f'Building Gradient Transform with MLP class {MlpClass}')

        def delta_net():
            return MlpClass(delta_dim, delta_dim, delta_dim * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def x_net():
            return MlpClass(x_dim, x_dim, x_dim * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def combined_net():
            return MlpClass(delta_dim + x_dim, delta_dim + x_dim, (delta_dim + x_dim) * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def ID():
            return lambda x, mode=None: x
        if cfg.combine:
            self.mlp = combined_net()
        elif cfg.one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def forward(self, u, v, param_idx=None):
        u, v = u, v
        u_ = u.view(-1, u.shape[-1])
        v_ = v.view(-1, v.shape[-1])
        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(-1)
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]
        if self.training:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(u_[idx], self.u_mean, self.u_s, self.k)
                    self.v_mean, self.v_s = update_counter(v_[idx], self.v_mean, self.v_s, self.k)
            if self.k < 2:
                raise RuntimeError(f"Can't perform normalization with only {self.k} samples so far")
            self.u_std = (self.u_s / (self.k - 1)) ** 0.5
            self.v_std = (self.v_s / (self.k - 1)) ** 0.5
        if self.cfg.norm:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-07)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-07)
        else:
            u_input = u_
            v_input = v_
        if self.cfg.combine:
            output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(v_input, mode=param_idx)


def _logits(x):
    return x if not hasattr(x, 'logits') else x.logits


def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {'acc': acc, 'log_prob': log_probs.mean(), 'prob': log_probs.exp().mean(), 'nll': -log_probs.mean(), 'n_tokens': log_probs.shape[0]}


def multiclass_log_probs(pred, targ, shift=True):
    NULL_TOKEN = 0
    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:
        pred = pred[:, :-1]
        targ = targ[:, 1:]
    mask = targ != -100
    targ[~mask] = NULL_TOKEN
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    if pred.dim() == 3:
        correct = (pred_ids == targ).all(-1)
    acc = correct.float().mean()
    n_tokens = mask.float().sum()
    log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
    prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
    return {'acc': acc, 'log_prob': log_prob, 'prob': prob, 'n_tokens': n_tokens, 'nll': -log_prob}


def masked_log_probs(pred, targ, shift=True):
    pred = pred
    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f'Expected pred to have 2 or 3 dimensions, got {pred.shape}')
    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(pred, targ, shift=shift)


def shift_targets(config):
    return 't5' not in config.model.name.lower()


class EditableModel(nn.Module):

    def __init__(self, model, config, model_constructor):
        super().__init__()
        self.model = model
        self.config = config
        self.model_constructor = model_constructor

        def _edit_loss_fn(pred, targ):
            return masked_log_probs(pred, targ, shift=shift_targets(self.config))
        self.edit_loss_fn = _edit_loss_fn
        self.loc_loss_fn = _edit_loss_fn

    def edit(self, batch, condition=None, detach_history=False):
        raise NotImplementedError

    def forward(self, *inputs, **kwargs):
        return _logits(self.model(*inputs, **kwargs))

    def outer_parameters(self):
        return self.parameters()

    def base_loss(self, input_ids, attention_masks, label_ids):
        pass


class CastModule(nn.Module):

    def __init__(self, module: nn.Module, in_cast: torch.dtype=torch.float32, out_cast: torch.dtype=None):
        super().__init__()
        self.underlying = module
        self.in_cast = in_cast
        self.out_cast = out_cast

    def cast(self, obj, dtype):
        if dtype is None:
            return obj
        if isinstance(obj, torch.Tensor):
            return obj
        else:
            return obj

    def forward(self, *args, **kwargs):
        args = tuple(self.cast(a, self.in_cast) for a in args)
        kwargs = {k: self.cast(v, self.in_cast) for k, v in kwargs.items()}
        outputs = self.underlying(*args, **kwargs)
        if isinstance(outputs, torch.Tensor):
            outputs = self.cast(outputs, self.out_cast)
        elif isinstance(outputs, tuple):
            outputs = tuple(self.cast(o, self.out_cast) for o in outputs)
        else:
            raise RuntimeError(f'Not sure how to cast type {type(outputs)}')
        return outputs

    def extra_repr(self):
        return f'in_cast: {self.in_cast}\nout_cast: {self.out_cast}'


def scr():
    if os.path.exists('/scr-ssd'):
        scr_dir = '/scr-ssd/' + getpass.getuser()
    elif os.path.exists('/scr'):
        scr_dir = '/scr/' + getpass.getuser()
    else:
        scr_dir = '/tmp/scr-' + getpass.getuser()
    if not os.path.exists(scr_dir):
        os.makedirs(scr_dir)
    return scr_dir


class BertClassifier(torch.nn.Module):

    def __init__(self, model_name, hidden_dim=768):
        super().__init__()
        self.model = transformers.BertModel.from_pretrained(model_name, cache_dir=scr())
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'labels'}
        return self.classifier(self.model(*args, **filtered_kwargs)[1])


class LRLinear(nn.Module):

    def __init__(self, inf, outf, rank: int=None, relu=False, init='id', n_modes=None):
        super().__init__()
        mid_dim = min(rank, inf)
        if init == 'id':
            self.u = nn.Parameter(torch.zeros(outf, mid_dim))
            self.v = nn.Parameter(torch.randn(mid_dim, inf))
        elif init == 'xavier':
            self.u = nn.Parameter(torch.empty(outf, mid_dim))
            self.v = nn.Parameter(torch.empty(mid_dim, inf))
            nn.init.xavier_uniform_(self.u.data, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.v.data, gain=1.0)
        else:
            raise ValueError(f'Unrecognized initialization {init}')
        if n_modes is not None:
            self.mode_shift = nn.Embedding(n_modes, outf)
            self.mode_shift.weight.data.zero_()
            self.mode_scale = nn.Embedding(n_modes, outf)
            self.mode_scale.weight.data.fill_(1)
        self.n_modes = n_modes
        self.bias = nn.Parameter(torch.zeros(outf))
        self.inf = inf
        self.init = init

    def forward(self, x, mode=None):
        if mode is not None:
            assert self.n_modes is not None, "Linear got a mode but wasn't initialized for it"
            assert mode < self.n_modes, f'Input mode {mode} outside of range {self.n_modes}'
        assert x.shape[-1] == self.inf, f'Input wrong dim ({x.shape}, {self.inf})'
        pre_act = (self.u @ (self.v @ x.T)).T
        if self.bias is not None:
            pre_act += self.bias
        if mode is not None:
            if not isinstance(mode, torch.Tensor):
                mode = torch.tensor(mode)
            scale, shift = self.mode_scale(mode), self.mode_shift(mode)
            pre_act = pre_act * scale + shift
        acts = pre_act.clamp(min=0)
        if self.init == 'id':
            return acts + x
        else:
            return acts


class IDMLP(nn.Module):

    def __init__(self, indim: int, outdim: int, hidden_dim: int, n_hidden: int, init: str=None, act: str=None, rank: int=None, n_modes: int=None):
        super().__init__()
        LOG.info(f'Building IDMLP ({init}) {[indim] * (n_hidden + 2)}')
        self.layers = nn.ModuleList([LRLinear(indim, indim, rank=rank, relu=idx < n_hidden, init=init, n_modes=n_modes) for idx in range(n_hidden + 1)])

    def forward(self, x, mode=None):
        for layer in self.layers:
            x = layer(x, mode=mode)
        return x


class MLP(nn.Module):

    def __init__(self, indim: int, outdim: int, hidden_dim: int, n_hidden: int, init: str='xavier_uniform', act: str='relu', rank: int=None):
        super().__init__()
        self.init = init
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'learned':
            self.act = ActMLP(10, 1)
        else:
            raise ValueError(f"Unrecognized activation function '{act}'")
        if hidden_dim is None:
            hidden_dim = outdim * 2
        if init.startswith('id') and outdim != indim:
            LOG.info(f'Overwriting outdim ({outdim}) to be indim ({indim})')
            outdim = indim
        if init == 'id':
            old_hidden_dim = hidden_dim
            if hidden_dim < indim * 2:
                hidden_dim = indim * 2
            if hidden_dim % indim != 0:
                hidden_dim += hidden_dim % indim
            if old_hidden_dim != hidden_dim:
                LOG.info(f'Overwriting hidden dim ({old_hidden_dim}) to be {hidden_dim}')
        if init == 'id_alpha':
            self.alpha = nn.Parameter(torch.zeros(1, outdim))
        dims = [indim] + [hidden_dim] * n_hidden + [outdim]
        LOG.info(f'Building ({init}) MLP: {dims} (rank {rank})')
        layers = []
        for idx, (ind, outd) in enumerate(zip(dims[:-1], dims[1:])):
            if rank is None:
                layers.append(nn.Linear(ind, outd))
            else:
                layers.append(LRLinear(ind, outd, rank=rank))
            if idx < n_hidden:
                layers.append(self.act)
        if rank is None:
            if init == 'id':
                if n_hidden > 0:
                    layers[0].weight.data = torch.eye(indim).repeat(hidden_dim // indim, 1)
                    layers[0].weight.data[hidden_dim // 2:] *= -1
                    layers[-1].weight.data = torch.eye(outdim).repeat(1, hidden_dim // outdim)
                    layers[-1].weight.data[:, hidden_dim // 2:] *= -1
                    layers[-1].weight.data /= hidden_dim // indim / 2.0
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    if init == 'ortho':
                        nn.init.orthogonal_(layer.weight)
                    elif init == 'id':
                        if layer.weight.shape[0] == layer.weight.shape[1]:
                            layer.weight.data = torch.eye(hidden_dim)
                    else:
                        gain = 3 ** 0.5 if layer is layers[-1] else 1.0
                        nn.init.xavier_uniform_(layer.weight, gain=gain)
                    layer.bias.data[:] = 0
        layers[-1].bias = None
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.init == 'id_alpha':
            return x + self.alpha * self.mlp(x)
        else:
            return self.mlp(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CastModule,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (EditableModel,
     lambda: ([], {'model': _mock_layer(), 'config': _mock_config(), 'model_constructor': 4}),
     lambda: ([], {'input': torch.rand([4, 4])}),
     False),
    (MLP,
     lambda: ([], {'indim': 4, 'outdim': 4, 'hidden_dim': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_kmeng01_rome(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

