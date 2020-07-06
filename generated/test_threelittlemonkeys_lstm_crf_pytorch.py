import sys
_module = sys.modules[__name__]
del sys
dataloader = _module
embedding = _module
evaluate = _module
model = _module
parameters = _module
brown2ptb = _module
filter = _module
predict = _module
prepare = _module
train = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import re


from time import time


from collections import defaultdict


CUDA = torch.cuda.is_available()


DROPOUT = 0.5


EMBED = {'lookup': 300}


EMBED_SIZE = sum(EMBED.values())


PAD_IDX = 0


zeros = lambda *x: torch.zeros(*x) if CUDA else torch.zeros


class embed(nn.Module):

    def __init__(self, cti_size, wti_size, hre=False):
        super().__init__()
        self.hre = hre
        for model, dim in EMBED.items():
            if model == 'char-cnn':
                self.char_embed = self.cnn(cti_size, dim)
            elif model == 'char-rnn':
                self.char_embed = self.rnn(cti_size, dim)
            if model == 'lookup':
                self.word_embed = nn.Embedding(wti_size, dim, padding_idx=PAD_IDX)
            elif model == 'sae':
                self.word_embed = self.sae(wti_size, dim)
        if self.hre:
            self.sent_embed = self.rnn(EMBED_SIZE, EMBED_SIZE, True)
        self = self if CUDA else self

    def forward(self, xc, xw):
        hc, hw = None, None
        if 'char-cnn' in EMBED or 'char-rnn' in EMBED:
            hc = self.char_embed(xc)
        if 'lookup' in EMBED or 'sae' in EMBED:
            hw = self.word_embed(xw)
        h = torch.cat([h for h in [hc, hw] if type(h) == torch.Tensor], 2)
        if self.hre:
            h = self.sent_embed(h) if self.hre else h
        return h


    class cnn(nn.Module):

        def __init__(self, vocab_size, embed_size):
            super().__init__()
            dim = 50
            num_featmaps = 50
            kernel_sizes = [3]
            self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD_IDX)
            self.conv = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_featmaps, kernel_size=(i, dim)) for i in kernel_sizes])
            self.dropout = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(len(kernel_sizes) * num_featmaps, embed_size)

        def forward(self, x):
            b = x.size(0)
            x = x.view(-1, x.size(2))
            x = self.embed(x)
            x = x.unsqueeze(1)
            h = [conv(x) for conv in self.conv]
            h = [F.relu(k).squeeze(3) for k in h]
            h = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h]
            h = torch.cat(h, 1)
            h = self.dropout(h)
            h = self.fc(h)
            h = h.view(b, -1, h.size(1))
            return h


    class rnn(nn.Module):

        def __init__(self, vocab_size, embed_size, embedded=False):
            super().__init__()
            self.dim = embed_size
            self.rnn_type = 'GRU'
            self.num_dirs = 2
            self.num_layers = 2
            self.embedded = embedded
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
            self.rnn = getattr(nn, self.rnn_type)(input_size=self.dim, hidden_size=self.dim // self.num_dirs, num_layers=self.num_layers, bias=True, batch_first=True, dropout=DROPOUT, bidirectional=self.num_dirs == 2)

        def init_state(self, b):
            n = self.num_layers * self.num_dirs
            h = self.dim // self.num_dirs
            hs = zeros(n, b, h)
            if self.rnn_type == 'LSTM':
                cs = zeros(n, b, h)
                return hs, cs
            return hs

        def forward(self, x):
            b = x.size(0)
            s = self.init_state(b * (1 if self.embedded else x.size(1)))
            if not self.embedded:
                x = x.view(-1, x.size(2))
                x = self.embed(x)
            h, s = self.rnn(x, s)
            h = s if self.rnn_type == 'GRU' else s[-1]
            h = torch.cat([x for x in h[-self.num_dirs:]], 1)
            h = h.view(b, -1, h.size(1))
            return h


    class sae(nn.Module):

        def __init__(self, vocab_size, embed_size=512):
            super().__init__()
            dim = embed_size
            num_layers = 1
            self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD_IDX)
            self.pe = self.pos_encoding(dim)
            self.layers = nn.ModuleList([self.layer(dim) for _ in range(num_layers)])

        def forward(self, x):
            mask = x.eq(PAD_IDX).view(x.size(0), 1, 1, -1)
            x = self.embed(x)
            h = x + self.pe[:x.size(1)]
            for layer in self.layers:
                h = layer(h, mask)
            return h

        @staticmethod
        def pos_encoding(dim, maxlen=1000):
            pe = Tensor(maxlen, dim)
            pos = torch.arange(0, maxlen, 1.0).unsqueeze(1)
            k = torch.exp(-np.log(10000) * torch.arange(0, dim, 2.0) / dim)
            pe[:, 0::2] = torch.sin(pos * k)
            pe[:, 1::2] = torch.cos(pos * k)
            return pe


        class layer(nn.Module):

            def __init__(self, dim):
                super().__init__()
                self.attn = embed.sae.attn_mh(dim)
                self.ffn = embed.sae.ffn(dim)

            def forward(self, x, mask):
                z = self.attn(x, x, x, mask)
                z = self.ffn(z)
                return z


        class attn_mh(nn.Module):

            def __init__(self, dim):
                super().__init__()
                self.D = dim
                self.H = 8
                self.Dk = self.D // self.H
                self.Dv = self.D // self.H
                self.Wq = nn.Linear(self.D, self.H * self.Dk)
                self.Wk = nn.Linear(self.D, self.H * self.Dk)
                self.Wv = nn.Linear(self.D, self.H * self.Dv)
                self.Wo = nn.Linear(self.H * self.Dv, self.D)
                self.dropout = nn.Dropout(DROPOUT)
                self.norm = nn.LayerNorm(self.D)

            def attn_sdp(self, q, k, v, mask):
                c = np.sqrt(self.Dk)
                a = torch.matmul(q, k.transpose(2, 3)) / c
                a = a.masked_fill(mask, -10000)
                a = F.softmax(a, -1)
                a = torch.matmul(a, v)
                return a

            def forward(self, q, k, v, mask):
                b = q.size(0)
                x = q
                q = self.Wq(q).view(b, -1, self.H, self.Dk).transpose(1, 2)
                k = self.Wk(k).view(b, -1, self.H, self.Dk).transpose(1, 2)
                v = self.Wv(v).view(b, -1, self.H, self.Dv).transpose(1, 2)
                z = self.attn_sdp(q, k, v, mask)
                z = z.transpose(1, 2).contiguous().view(b, -1, self.H * self.Dv)
                z = self.Wo(z)
                z = self.norm(x + self.dropout(z))
                return z


        class ffn(nn.Module):

            def __init__(self, dim):
                super().__init__()
                dim_ffn = 2048
                self.layers = nn.Sequential(nn.Linear(dim, dim_ffn), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(dim_ffn, dim))
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                z = x + self.layers(x)
                z = self.norm(z)
                return z


UNIT = 'word'


HRE = UNIT == 'sent'


EOS_IDX = 2


LongTensor = torch.LongTensor if CUDA else torch.LongTensor


SOS_IDX = 1


def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))


randn = lambda *x: torch.randn(*x) if CUDA else torch.randn


class crf(nn.Module):

    def __init__(self, num_tags):
        super().__init__()
        self.batch_size = 0
        self.num_tags = num_tags
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[(SOS_IDX), :] = -10000
        self.trans.data[:, (EOS_IDX)] = -10000
        self.trans.data[:, (PAD_IDX)] = -10000
        self.trans.data[(PAD_IDX), :] = -10000
        self.trans.data[PAD_IDX, EOS_IDX] = 0
        self.trans.data[PAD_IDX, PAD_IDX] = 0

    def forward(self, h, mask):
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000)
        score[:, (SOS_IDX)] = 0.0
        trans = self.trans.unsqueeze(0)
        for t in range(h.size(1)):
            mask_t = mask[:, (t)].unsqueeze(1)
            emit_t = h[:, (t)].unsqueeze(2)
            score_t = score.unsqueeze(1) + emit_t + trans
            score_t = log_sum_exp(score_t)
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[EOS_IDX])
        return score

    def score(self, h, y0, mask):
        score = Tensor(self.batch_size).fill_(0.0)
        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)):
            mask_t = mask[:, (t)]
            emit_t = torch.cat([h[t, y0[t + 1]] for h, y0 in zip(h, y0)])
            trans_t = torch.cat([trans[y0[t + 1], y0[t]] for y0 in y0])
            score += (emit_t + trans_t) * mask_t
        last_tag = y0.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[EOS_IDX, last_tag]
        return score

    def decode(self, h, mask):
        bptr = LongTensor()
        score = Tensor(self.batch_size, self.num_tags).fill_(-10000)
        score[:, (SOS_IDX)] = 0.0
        for t in range(h.size(1)):
            mask_t = mask[:, (t)].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans
            score_t, bptr_t = score_t.max(2)
            score_t += h[:, (t)]
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(self.batch_size):
            i = best_tag[b]
            j = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:j]):
                i = bptr_t[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()
        return best_path


HIDDEN_SIZE = 1000


NUM_DIRS = 2


NUM_LAYERS = 2


RNN_TYPE = 'LSTM'


class rnn(nn.Module):

    def __init__(self, cti_size, wti_size, num_tags):
        super().__init__()
        self.batch_size = 0
        self.embed = embed(cti_size, wti_size, HRE)
        self.rnn = getattr(nn, RNN_TYPE)(input_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE // NUM_DIRS, num_layers=NUM_LAYERS, bias=True, batch_first=True, dropout=DROPOUT, bidirectional=NUM_DIRS == 2)
        self.out = nn.Linear(HIDDEN_SIZE, num_tags)

    def init_state(self, b):
        n = NUM_LAYERS * NUM_DIRS
        h = HIDDEN_SIZE // NUM_DIRS
        hs = zeros(n, b, h)
        if RNN_TYPE == 'LSTM':
            cs = zeros(n, b, h)
            return hs, cs
        return hs

    def forward(self, xc, xw, mask):
        hs = self.init_state(self.batch_size)
        x = self.embed(xc, xw)
        if HRE:
            x = x.view(self.batch_size, -1, EMBED_SIZE)
        x = nn.utils.rnn.pack_padded_sequence(x, mask.sum(1).int(), batch_first=True)
        h, _ = self.rnn(x, hs)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        h = self.out(h)
        h *= mask.unsqueeze(2)
        return h


class rnn_crf(nn.Module):

    def __init__(self, cti_size, wti_size, num_tags):
        super().__init__()
        self.rnn = rnn(cti_size, wti_size, num_tags)
        self.crf = crf(num_tags)
        self = self if CUDA else self

    def forward(self, xc, xw, y0):
        self.zero_grad()
        self.rnn.batch_size = y0.size(0)
        self.crf.batch_size = y0.size(0)
        mask = y0[:, 1:].gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        Z = self.crf.forward(h, mask)
        score = self.crf.score(h, y0, mask)
        return torch.mean(Z - score)

    def decode(self, xc, xw, lens):
        self.rnn.batch_size = len(lens)
        self.crf.batch_size = len(lens)
        if HRE:
            mask = Tensor([([1] * x + [PAD_IDX] * (lens[0] - x)) for x in lens])
        else:
            mask = xw.gt(PAD_IDX).float()
        h = self.rnn(xc, xw, mask)
        return self.crf.decode(h, mask)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (embed,
     lambda: ([], {'cti_size': 4, 'wti_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
]

class Test_threelittlemonkeys_lstm_crf_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

