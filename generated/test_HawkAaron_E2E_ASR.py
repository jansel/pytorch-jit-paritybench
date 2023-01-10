import sys
_module = sys.modules[__name__]
del sys
DataLoader = _module
ctc_decoder = _module
eval = _module
eval_att = _module
model = _module
model2012 = _module
seq2seq = _module
attention = _module
decoder = _module
encoder = _module
seq2seq = _module
train_att = _module
train_ctc = _module
train_rnnt = _module

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


import math


import time


import torch


from torch.autograd import Variable


import torch.nn.functional as F


import numpy as np


import copy


from torch import nn


from torch import autograd


import random


class RNNModel(nn.Module):

    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout=0.2, blank=0, bidirectional=False):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.blank = blank
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            hidden_size *= 2
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs, hid=None):
        h, hid = self.lstm(xs, hid)
        return self.linear(h), hid

    def greedy_decode(self, xs):
        xs = self(xs)[0][0]
        xs = F.log_softmax(xs, dim=1)
        logp, pred = torch.max(xs, dim=1)
        return pred.data.cpu().numpy(), -float(logp.sum())

    def beam_search(self, xs, W):
        """ CTC """
        xs = self(xs)[0][0]
        logp = F.log_softmax(xs, dim=1)
        return ctc_beam(logp.data.cpu().numpy(), W)


class Sequence:

    def __init__(self, seq=None, blank=0):
        if seq is None:
            self.g = []
            self.k = [blank]
            self.h = None
            self.logp = 0
        else:
            self.g = seq.g[:]
            self.k = seq.k[:]
            self.h = seq.h
            self.logp = seq.logp

    def __str__(self):
        return 'Prediction: {}\nlog-likelihood {:.2f}\n'.format(' '.join([rephone[i] for i in self.k]), -self.logp)


def log_aplusb(a, b):
    return max(a, b) + math.log1p(math.exp(-math.fabs(a - b)))


class Transducer(nn.Module):

    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout=0.5, blank=0, bidirectional=False):
        super(Transducer, self).__init__()
        self.blank = blank
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.loss = RNNTLoss(size_average=True)
        self.encoder = RNNModel(input_size, vocab_size, hidden_size, num_layers, dropout, bidirectional=bidirectional)
        self.embed = nn.Embedding(vocab_size, vocab_size - 1, padding_idx=blank)
        self.embed.weight.data[1:] = torch.eye(vocab_size - 1)
        self.embed.weight.requires_grad = False
        self.decoder = RNNModel(vocab_size - 1, vocab_size, hidden_size, 1, dropout)

    def forward(self, xs, ys, xlen, ylen):
        xs, _ = self.encoder(xs)
        zero = autograd.Variable(torch.zeros((ys.shape[0], 1)).long())
        if ys.is_cuda:
            zero = zero
        ymat = torch.cat((zero, ys), dim=1)
        ymat = self.embed(ymat)
        ymat, _ = self.decoder(ymat)
        loss = self.loss(xs, ymat, ys, xlen, ylen)
        return loss

    def greedy_decode(self, xs):

        def decode_one(x):
            vy = autograd.Variable(torch.LongTensor([0]), volatile=True).view(1, 1)
            if xs.is_cuda:
                vy = vy
            zeroy, zeroh = self.decoder(self.embed(vy))
            y, h = zeroy, zeroh
            y_seq = []
            for i in x:
                _, pred = torch.max(i + y[0][0], dim=0)
                pred = int(pred)
                if pred != self.blank:
                    y_seq.append(pred)
                    vy.data[0][0] = pred
                    y, h = self.decoder(self.embed(vy), h)
            return y_seq
        xs, _ = self.encoder(xs)
        return [decode_one(x) for x in xs]

    def beam_search(self, xs, W=10, prefix=True):
        """''
        `xs`: acoustic model outputs
        NOTE only support one sequence (batch size = 1)
        TODO skip summation over prefixes unless multiple hypotheses are identical
        """

        def forward_step(label, hidden):
            """ `label`: int """
            label = autograd.Variable(torch.LongTensor([label]), volatile=True).view(1, 1)
            label = self.embed(label)
            pred, hidden = self.decoder(label, hidden)
            return pred[0][0], hidden

        def isprefix(a, b):
            if a == b or len(a) >= len(b):
                return False
            for i in range(len(a)):
                if a[i] != b[i]:
                    return False
            return True
        xs = self.encoder(xs)[0][0]
        B = [Sequence(blank=self.blank)]
        for i, x in enumerate(xs):
            sorted(B, key=lambda a: len(a.k), reverse=True)
            A = B
            B = []
            if prefix:
                for j in range(len(A) - 1):
                    for i in range(j + 1, len(A)):
                        if not isprefix(A[i].k, A[j].k):
                            continue
                        pred, _ = forward_step(A[i].k[-1], A[i].h)
                        idx = len(A[i].k)
                        logp = F.log_softmax(x + pred, dim=0)
                        curlogp = A[i].logp + float(logp[A[j].k[idx]])
                        for k in range(idx, len(A[j].k) - 1):
                            logp = F.log_softmax(x + A[j].g[k], dim=0)
                            curlogp += float(logp[A[j].k[k + 1]])
                        A[j].logp = log_aplusb(A[j].logp, curlogp)
            while True:
                y_hat = max(A, key=lambda a: a.logp)
                A.remove(y_hat)
                pred, hidden = forward_step(y_hat.k[-1], y_hat.h)
                logp = F.log_softmax(x + pred, dim=0)
                for k in range(self.vocab_size):
                    yk = Sequence(y_hat)
                    yk.logp += float(logp[k])
                    if k == self.blank:
                        B.append(yk)
                        continue
                    yk.h = hidden
                    yk.k.append(k)
                    if prefix:
                        yk.g.append(pred)
                    A.append(yk)
                y_hat = max(A, key=lambda a: a.logp)
                yb = max(B, key=lambda a: a.logp)
                if len(B) >= W and yb.logp >= y_hat.logp:
                    break
            sorted(B, key=lambda a: a.logp, reverse=True)
            B = B[:W]
        None
        return B[0].k, -B[0].logp


class NNAttention(nn.Module):

    def __init__(self, n_channels, kernel_size=15, log_t=False):
        super(NNAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, n_channels, kernel_size, padding=padding)
        self.nn = nn.Sequential(nn.ReLU(), nn.Linear(n_channels, 1))
        self.log_t = log_t

    def forward(self, eh, dhx, ax=None):
        """ `eh` (BTH), `dhx` (BH) """
        pax = eh + dhx.unsqueeze(dim=1)
        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).transpose(1, 2)
            pax = pax + ax
        pax = self.nn(pax)
        pax = pax.squeeze(dim=2)
        if self.log_t:
            log_t = math.log(pax.shape[1])
            pax = log_t * pax
        ax = nn.functional.softmax(pax, dim=1)
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1)
        return sx, ax


class Attention(nn.Module):

    def __init(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def forward(self, hidden, enc_out):
        output = enc_out * hidden[-1].unsqueeze(dim=1)
        output = output.sum(dim=2)
        output = F.softmax(output, dim=1)
        output = output.unsqueeze(dim=2) * enc_out
        output = output.sum(dim=1)
        return output


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, sample_rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attention = NNAttention(hidden_size, log_t=True)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size - 1)
        self.vocab_size = vocab_size
        self.sample_rate = sample_rate

    def forward(self, target, enc_out, enc_hid):
        """
        `target`: (batch, length)
        `enc_out`: Encoder output, (batch, length, dim)
        `enc_hid`: last hidden state of encoder
        """
        target = target.transpose(0, 1)
        length = target.shape[0]
        inputs = target[0]
        hidden = enc_hid
        ax = sx = None
        out = []
        align = []
        loss = 0
        for i in range(1, length):
            output, hidden, ax, sx = self._step(inputs, hidden, enc_out, ax, sx)
            out.append(output)
            align.append(ax)
            if random.random() < self.sample_rate:
                inputs = output.max(dim=1)[1]
            else:
                inputs = target[i]
            if i % 40 == 0:
                out = torch.cat(out, dim=0)
                out = out.view(-1, out.shape[-1])
                t = target[i - 39:i + 1].contiguous().view(-1)
                loss = loss + F.cross_entropy(out, t, size_average=False)
                out = []
                loss.backward()
                loss = Variable(loss.data)
                hidden = Variable(hidden.data)
                ax = Variable(ax.data)
                sx = Variable(sx.data)
        left = len(out)
        if left > 0:
            out = torch.cat(out, dim=0)
            out = out.view(-1, out.shape[-1])
            target = target[length - left:].contiguous().view(-1)
            loss = loss + F.cross_entropy(out, target, size_average=False)
            loss.backward()
        return loss

    def _step(self, inputs, hidden, enc_out, ax, sx):
        embeded = self.embedding(inputs)
        if sx is not None:
            embeded = embeded + sx
        out = self.rnn(embeded, hidden)
        sx, ax = self.attention(enc_out, out, ax)
        output = self.fc(out + sx)
        return output, out, ax, sx


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, inputs, hidden=None):
        """
        `inputs`: (batch, length, input_size)
        `hidden`: Initial hidden state (num_layer, batch_size, hidden_size)
        """
        x, h = self.rnn(inputs, hidden)
        dim = h.shape[0]
        h = h.sum(dim=0) / dim
        if self.rnn.bidirectional:
            half = x.shape[-1] // 2
            x = x[:, :, :half] + x[:, :, half:]
        return x, h


class Seq2seq(nn.Module):

    def __init__(self, input_size, vocab_size, hidden_size, num_layers, dropout, bidirectional, sample_rate=0.4, **kwargs):
        super(Seq2seq, self).__init__(**kwargs)
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout, bidirectional)
        self.decoder = Decoder(vocab_size, hidden_size, sample_rate)
        self.vocab_size = vocab_size

    def forward(self, inputs, targets):
        """
        `inputs`: (batch, length, dim)
        `targets`: (batch, length)
        """
        enc_out, enc_hid = self.encoder(inputs)
        loss = self.decoder(targets, enc_out, enc_hid)
        return loss

    def greedy_decode(self, inputs):
        """ only support one sequence """
        enc_out, enc_hid = self.encoder(inputs)
        inputs = torch.autograd.Variable(torch.LongTensor([self.vocab_size - 1]), volatile=True)
        if enc_out.is_cuda:
            inputs = inputs
        hidden = enc_hid
        y_seq = []
        label = 1
        logp = 0
        ax = sx = None
        while label != 0:
            output, hidden, ax, sx = self.decoder._step(inputs, hidden, enc_out, ax, sx)
            output = torch.nn.functional.log_softmax(output, dim=1)
            pred, inputs = output.max(dim=1)
            label = int(inputs.data[0])
            logp += float(pred)
            y_seq.append(label)
        None
        return y_seq, -logp

    def decode_step(self, x, y, state=None, softmax=False):
        """ `x` (TH), `y` (1) """
        if state is None:
            hx, ax, sx = None, None, None
        else:
            hx, ax, sx = state
        out, hx, ax, sx = self.decoder._step(y, hx, x, ax, sx)
        if softmax:
            out = nn.functional.log_softmax(out, dim=1)
        return out, (hx, ax, sx)

    def beam_search(self, xs, beam_size=10, max_len=200):
        start_tok = self.vocab_size - 1
        end_tok = 0
        x, h = self.encode(xs)
        y = torch.autograd.Variable(torch.LongTensor([start_tok]), volatile=True)
        beam = [((start_tok,), 0, (h, None, None))]
        complete = []
        for _ in range(max_len):
            new_beam = []
            for hyp, score, state in beam:
                y[0] = hyp[-1]
                out, state = self.decode_step(x, y, state=state, softmax=True)
                out = out.cpu().data.numpy().squeeze(axis=0).tolist()
                for i, p in enumerate(out):
                    new_score = score + p
                    new_hyp = hyp + (i,)
                    new_beam.append((new_hyp, new_score, state))
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)
            for cand in new_beam[:beam_size]:
                if cand[0][-1] == end_tok:
                    complete.append(cand)
            beam = filter(lambda x: x[0][-1] != end_tok, new_beam)
            beam = beam[:beam_size]
            if len(beam) == 0:
                break
            if sum(c[1] > beam[0][1] for c in complete) >= beam_size:
                break
        complete = sorted(complete, key=lambda x: x[1], reverse=True)
        if len(complete) == 0:
            complete = beam
        hyp, score, _ = complete[0]
        return hyp, score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (NNAttention,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RNNModel,
     lambda: ([], {'input_size': 4, 'vocab_size': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
]

class Test_HawkAaron_E2E_ASR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

