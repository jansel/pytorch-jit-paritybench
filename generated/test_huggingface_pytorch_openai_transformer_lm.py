import sys
_module = sys.modules[__name__]
del sys
analysis = _module
datasets = _module
generate = _module
loss = _module
model_pytorch = _module
opt = _module
text_utils = _module
train = _module
utils = _module

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


import torch.nn as nn


import copy


import math


import re


import collections


import torch.nn.functional as F


from torch.nn.parameter import Parameter


from torch.optim import Optimizer


from torch.nn.utils import clip_grad_norm_


from sklearn.metrics import accuracy_score


from sklearn.utils import shuffle


class LayerNorm(nn.Module):
    """Construct a layernorm module in the OpenAI style (epsilon inside the square root)."""

    def __init__(self, n_state, e=1e-05):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):

    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):

    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        b = self.b[:, :, :w.size(-2), :w.size(-1)]
        w = w * b + -1000000000.0 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {'relu': nn.ReLU, 'swish': swish, 'gelu': gelu}


class MLP(nn.Module):

    def __init__(self, n_state, cfg):
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):

    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x):
        a = self.attn(x)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(TransformerModel, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x):
        x = x.view(-1, x.size(-2), x.size(-1))
        e = self.drop(self.embed(x))
        h = e.sum(dim=2)
        for block in self.h:
            h = block(h)
        return h


class LMHead(nn.Module):
    """ Language Model Head for the transformer """

    def __init__(self, model, cfg, trunc_and_reshape=True):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight
        self.trunc_and_reshape = trunc_and_reshape

    def forward(self, h):
        h_trunc = h[:, :-1].contiguous().view(-1, self.n_embd) if self.trunc_and_reshape else h
        lm_logits = self.decoder(h_trunc)
        return lm_logits


class MultipleChoiceHead(nn.Module):
    """ Classifier Head for the transformer """

    def __init__(self, clf_token, cfg):
        super(MultipleChoiceHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout2d(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, 1)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[(flat == self.clf_token), :]
        clf_h = clf_h.view(-1, x.size(1), self.n_embd, 1)
        clf_h = self.dropout(clf_h.transpose(1, 2)).transpose(1, 2)
        clf_h = clf_h.contiguous().view(-1, self.n_embd)
        clf_logits = self.linear(clf_h)
        return clf_logits.view(-1, x.size(1))


class ClfHead(nn.Module):
    """Classification Head for the transformer

    TODO: test this class."""

    def __init__(self, clf_token, cfg, n_class):
        super(ClfHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, n_class)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        clf_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        clf_h = clf_h[(flat == self.clf_token), :]
        clf_h = self.dropout(clf_h)
        clf_logits = self.linear(clf_h)
        return clf_logits


class SimilarityHead(nn.Module):
    """ Similarity Head for the transformer

        TODO: test this class."""

    def __init__(self, clf_token, cfg):
        super(SimilarityHead, self).__init__()
        self.n_embd = cfg.n_embd
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        self.linear = nn.Linear(cfg.n_embd, 1)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, h, x):
        sim_h = h.view(-1, self.n_embd)
        flat = x[..., 0].contiguous().view(-1)
        sim_h = sim_h[(flat == self.clf_token), :]
        sim_h = self.dropout(sim_h)
        sim_h = sim_h.sum(dim=1)
        sim_logits = self.linear(sim_h)
        return sim_logits


class LMModel(nn.Module):
    """ Transformer with language model head only """

    def __init__(self, cfg, vocab=40990, n_ctx=512, return_probs=False):
        super(LMModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg, trunc_and_reshape=False)
        self.return_probs = return_probs
        if self.return_probs:
            pos_emb_mask = torch.zeros(1, 1, vocab)
            pos_emb_mask[:, :, -n_ctx:] = -1000000000000.0
            self.register_buffer('pos_emb_mask', pos_emb_mask)

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        if self.return_probs:
            lm_logits = F.softmax(lm_logits + self.pos_emb_mask, dim=-1)
        return lm_logits


class DoubleHeadModel(nn.Module):
    """ Transformer with language model and task specific heads """

    def __init__(self, cfg, clf_token, task_head_type, vocab=40990, n_ctx=512):
        super(DoubleHeadModel, self).__init__()
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.lm_head = LMHead(self.transformer, cfg)
        if isinstance(task_head_type, str):
            if task_head_type == 'multiple_choice':
                self.task_head = MultipleChoiceHead(clf_token, cfg)
            elif task_head_type == 'similarity':
                self.task_head = SimilarityHead(clf_token, cfg)
            elif task_head_type == 'inference':
                self.task_head = ClfHead(clf_token, cfg, 3)
            else:
                raise ValueError(f"task_head_type is expected to be 'multiple_choice' 'similarity', 'inference' or ('classification', n_class) got {task_head_type}.")
        elif isinstance(task_head_type, collections.abc.Sequence) and len(task_head_type) == 2 and task_head_type[0] == 'classification':
            n_class = task_head_type[1]
            self.task_head = ClfHead(clf_token, cfg, n_class)
        else:
            raise ValueError(f"task_head_type is expected to be 'multiple_choice' 'similarity', 'inference' or ('classification', n_class) got {task_head_type}.")

    def forward(self, x):
        h = self.transformer(x)
        lm_logits = self.lm_head(h)
        task_logits = self.task_head(h, x)
        return lm_logits, task_logits


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'nx': 4, 'n_ctx': 4, 'cfg': _mock_config(n_head=4, attn_pdrop=0.5, resid_pdrop=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (ClfHead,
     lambda: ([], {'clf_token': 4, 'cfg': _mock_config(n_embd=4, clf_pdrop=0.5), 'n_class': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'n_state': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MultipleChoiceHead,
     lambda: ([], {'clf_token': 4, 'cfg': _mock_config(n_embd=4, clf_pdrop=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_huggingface_pytorch_openai_transformer_lm(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

