import sys
_module = sys.modules[__name__]
del sys
download_d4rl_datasets = _module
decision_transformer = _module
d4rl_infos = _module
model = _module
utils = _module
dt_runs = _module
plot = _module
test = _module
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


import math


import torch


import torch.nn as nn


import torch.nn.functional as F


import random


import time


import numpy as np


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


class MaskedCausalAttention(nn.Module):

    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.n_heads = n_heads
        self.max_T = max_T
        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)
        self.proj_net = nn.Linear(h_dim, h_dim)
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)
        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        weights = weights.masked_fill(self.mask[..., :T, :T] == 0, float('-inf'))
        normalized_weights = F.softmax(weights, dim=-1)
        attention = self.att_drop(normalized_weights @ v)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)
        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):

    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(nn.Linear(h_dim, 4 * h_dim), nn.GELU(), nn.Linear(4 * h_dim, h_dim), nn.Dropout(drop_p))
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):

    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len, n_heads, drop_p, max_timestep=4096):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim
        input_seq_len = 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)
        self.embed_action = torch.nn.Linear(act_dim, h_dim)
        use_action_tanh = True
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action = nn.Sequential(*([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else [])))

    def forward(self, timesteps, states, actions, returns_to_go):
        B, T, _ = states.shape
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings
        h = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        h = self.embed_ln(h)
        h = self.transformer(h)
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        return_preds = self.predict_rtg(h[:, 2])
        state_preds = self.predict_state(h[:, 2])
        action_preds = self.predict_action(h[:, 1])
        return state_preds, action_preds, return_preds


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Block,
     lambda: ([], {'h_dim': 4, 'max_T': 4, 'n_heads': 4, 'drop_p': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MaskedCausalAttention,
     lambda: ([], {'h_dim': 4, 'max_T': 4, 'n_heads': 4, 'drop_p': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_nikhilbarhate99_min_decision_transformer(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

