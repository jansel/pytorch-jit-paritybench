import sys
_module = sys.modules[__name__]
del sys
checkpoint = _module
classify = _module
models = _module
optim = _module
pretrain = _module
tokenization = _module
train = _module
utils = _module

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


import numpy as np


import tensorflow as tf


import torch


import itertools


import torch.nn as nn


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import math


from typing import NamedTuple


import torch.nn.functional as F


from torch.optim import Optimizer


from torch.nn.utils import clip_grad_norm_


from random import randint


from random import shuffle


from random import random as rand


import random


import logging


class Classifier(nn.Module):
    """ Classifier with Transformer """

    def __init__(self, cfg, n_labels):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ(self.fc(h[:, (0)]))
        logits = self.classifier(self.drop(pooled_h))
        return logits


class LayerNorm(nn.Module):
    """A layernorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """The embedding module from word, position and token_type embeddings."""

    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim)
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim)
        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))


def merge_last(x, n_dims):
    """merge the last n_dims to a dimension"""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


def split_last(x, shape):
    """split the last dimension to given shape"""
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, (None), (None), :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = merge_last(h, 2)
        self.scores = scores
        return h


def gelu(x):
    """Implementation of the gelu activation function by Hugging Face"""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
        self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """

    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h


class BertModel4Pretrain(nn.Module):
    """Bert Model for Pretrain : Masked LM and next sentence classification"""

    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = models.gelu
        self.norm = models.LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.dim, 2)
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, (0)]))
        masked_pos = masked_pos[:, :, (None)].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)
        return logits_lm, logits_clsf


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Classifier,
     lambda: ([], {'cfg': _mock_config(vocab_size=4, dim=4, max_len=4, n_segments=4, p_drop_hidden=0.5, n_layers=1, p_drop_attn=0.5, n_heads=4, dim_ff=4), 'n_labels': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4, 4], dtype=torch.int64), torch.rand([4, 4])], {}),
     False),
    (Embeddings,
     lambda: ([], {'cfg': _mock_config(vocab_size=4, dim=4, max_len=4, n_segments=4, p_drop_hidden=0.5)}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (LayerNorm,
     lambda: ([], {'cfg': _mock_config(dim=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionWiseFeedForward,
     lambda: ([], {'cfg': _mock_config(dim=4, dim_ff=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Transformer,
     lambda: ([], {'cfg': _mock_config(vocab_size=4, dim=4, max_len=4, n_segments=4, p_drop_hidden=0.5, n_layers=1, p_drop_attn=0.5, n_heads=4, dim_ff=4)}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4, 4], dtype=torch.int64), torch.rand([4, 4])], {}),
     False),
]

class Test_dhlee347_pytorchic_bert(_paritybench_base):
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

