import sys
_module = sys.modules[__name__]
del sys
embed_sequences = _module
eval_contact_casp12 = _module
eval_contact_scop = _module
eval_secstr = _module
eval_similarity = _module
eval_transmembrane = _module
setup = _module
src = _module
alphabets = _module
fasta = _module
models = _module
comparison = _module
embedding = _module
multitask = _module
sequence = _module
parse_utils = _module
pdb = _module
pfam = _module
scop = _module
transmembrane = _module
utils = _module
train_lm_pfam = _module
train_similarity = _module
train_similarity_and_contact = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import PackedSequence


import torch.utils.data


from scipy.stats import pearsonr


from scipy.stats import spearmanr


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.autograd import Variable


class LSTM(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True, batch_first=True
            )
        self.linear = nn.Linear(2 * n_hidden, n_out)

    def forward(self, x):
        if type(x) is not PackedSequence:
            ndim = len(x.size())
            if ndim == 2:
                x = x.unsqueeze(0)
        h, _ = self.rnn(x)
        if type(h) is PackedSequence:
            z = self.linear(h.data)
            return PackedSequence(z, h.batch_sizes)
        else:
            z = self.linear(h.view(h.size(0) * h.size(1), -1))
            z = z.view(h.size(0), h.size(1), -1)
            if ndim == 2:
                z = z.squeeze(0)
            return z


class L1(nn.Module):

    def forward(self, x, y):
        return -torch.sum(torch.abs(x.unsqueeze(1) - y), -1)


class L2(nn.Module):

    def forward(self, x, y):
        return -torch.sum((x.unsqueeze(1) - y) ** 2, -1)


class DotProduct(nn.Module):

    def forward(self, x, y):
        return torch.mm(x, y.t())


def pad_gap_scores(s, gap):
    col = gap.expand(s.size(0), 1)
    s = torch.cat([s, col], 1)
    row = gap.expand(1, s.size(1))
    s = torch.cat([s, row], 0)
    return s


class OrdinalRegression(nn.Module):

    def __init__(self, embedding, n_classes, compare=L1(), align_method=
        'ssa', beta_init=10, allow_insertions=False, gap_init=-10):
        super(OrdinalRegression, self).__init__()
        self.embedding = embedding
        self.n_out = n_classes
        self.compare = compare
        self.align_method = align_method
        self.allow_insertions = allow_insertions
        self.gap = nn.Parameter(torch.FloatTensor([gap_init]))
        self.theta = nn.Parameter(torch.ones(1, n_classes - 1))
        self.beta = nn.Parameter(torch.zeros(n_classes - 1) + beta_init)
        self.clip()

    def forward(self, x):
        return self.embedding(x)

    def clip(self):
        self.theta.data.clamp_(min=0)

    def score(self, z_x, z_y):
        if self.align_method == 'ssa':
            s = self.compare(z_x, z_y)
            if self.allow_insertions:
                s = pad_gap_scores(s, self.gap)
            a = F.softmax(s, 1)
            b = F.softmax(s, 0)
            if self.allow_insertions:
                index = s.size(0) - 1
                index = s.data.new(1).long().fill_(index)
                a = a.index_fill(0, index, 0)
                index = s.size(1) - 1
                index = s.data.new(1).long().fill_(index)
                b = b.index_fill(1, index, 0)
            a = a + b - a * b
            c = torch.sum(a * s) / torch.sum(a)
        elif self.align_method == 'ua':
            s = self.compare(z_x, z_y)
            c = torch.mean(s)
        elif self.align_method == 'me':
            z_x = z_x.mean(0)
            z_y = z_y.mean(0)
            c = self.compare(z_x.unsqueeze(0), z_y.unsqueeze(0)).squeeze(0)
        else:
            raise Exception('Unknown alignment method: ' + self.align_method)
        logits = c * self.theta + self.beta
        return logits.view(-1)


class LMEmbed(nn.Module):

    def __init__(self, nin, nout, lm, padding_idx=-1, transform=nn.ReLU(),
        sparse=False):
        super(LMEmbed, self).__init__()
        if padding_idx == -1:
            padding_idx = nin - 1
        self.lm = lm
        self.embed = nn.Embedding(nin, nout, padding_idx=padding_idx,
            sparse=sparse)
        self.proj = nn.Linear(lm.hidden_size(), nout)
        self.transform = transform
        self.nout = nout

    def forward(self, x):
        packed = type(x) is PackedSequence
        h_lm = self.lm.encode(x)
        if packed:
            h = self.embed(x.data)
            h_lm = h_lm.data
        else:
            h = self.embed(x)
        h_lm = self.proj(h_lm)
        h = self.transform(h + h_lm)
        if packed:
            h = PackedSequence(h, x.batch_sizes)
        return h


class Linear(nn.Module):

    def __init__(self, nin, nhidden, nout, padding_idx=-1, sparse=False, lm
        =None):
        super(Linear, self).__init__()
        if padding_idx == -1:
            padding_idx = nin - 1
        if lm is not None:
            self.embed = LMEmbed(nin, nhidden, lm, padding_idx=padding_idx,
                sparse=sparse)
            self.proj = nn.Linear(self.embed.nout, nout)
            self.lm = True
        else:
            self.proj = nn.Embedding(nin, nout, padding_idx=padding_idx,
                sparse=sparse)
            self.lm = False
        self.nout = nout

    def forward(self, x):
        if self.lm:
            h = self.embed(x)
            if type(h) is PackedSequence:
                h = h.data
                z = self.proj(h)
                z = PackedSequence(z, x.batch_sizes)
            else:
                h = h.view(-1, h.size(2))
                z = self.proj(h)
                z = z.view(x.size(0), x.size(1), -1)
        elif type(x) is PackedSequence:
            z = self.embed(x.data)
            z = PackedSequence(z, x.batch_sizes)
        else:
            z = self.embed(x)
        return z


class StackedRNN(nn.Module):

    def __init__(self, nin, nembed, nunits, nout, nlayers=2, padding_idx=-1,
        dropout=0, rnn_type='lstm', sparse=False, lm=None):
        super(StackedRNN, self).__init__()
        if padding_idx == -1:
            padding_idx = nin - 1
        if lm is not None:
            self.embed = LMEmbed(nin, nembed, lm, padding_idx=padding_idx,
                sparse=sparse)
            nembed = self.embed.nout
            self.lm = True
        else:
            self.embed = nn.Embedding(nin, nembed, padding_idx=padding_idx,
                sparse=sparse)
            self.lm = False
        if rnn_type == 'lstm':
            RNN = nn.LSTM
        elif rnn_type == 'gru':
            RNN = nn.GRU
        self.dropout = nn.Dropout(p=dropout)
        if nlayers == 1:
            dropout = 0
        self.rnn = RNN(nembed, nunits, nlayers, batch_first=True,
            bidirectional=True, dropout=dropout)
        self.proj = nn.Linear(2 * nunits, nout)
        self.nout = nout

    def forward(self, x):
        if self.lm:
            h = self.embed(x)
        elif type(x) is PackedSequence:
            h = self.embed(x.data)
            h = PackedSequence(h, x.batch_sizes)
        else:
            h = self.embed(x)
        h, _ = self.rnn(h)
        if type(h) is PackedSequence:
            h = h.data
            h = self.dropout(h)
            z = self.proj(h)
            z = PackedSequence(z, x.batch_sizes)
        else:
            h = h.view(-1, h.size(2))
            h = self.dropout(h)
            z = self.proj(h)
            z = z.view(x.size(0), x.size(1), -1)
        return z


class SCOPCM(nn.Module):

    def __init__(self, embedding, similarity_kwargs={}, cmap_kwargs={}):
        super(SCOPCM, self).__init__()
        self.embedding = embedding
        embed_dim = embedding.nout
        self.scop_predict = OrdinalRegression(5, **similarity_kwargs)
        self.cmap_predict = ConvContactMap(embed_dim, **cmap_kwargs)

    def clip(self):
        self.scop_predict.clip()
        self.cmap_predict.clip()

    def forward(self, x):
        return self.embedding(x)

    def score(self, z_x, z_y):
        return self.scop_predict(z_x, z_y)

    def predict(self, z):
        return self.cmap_predict(z)


class ConvContactMap(nn.Module):

    def __init__(self, embed_dim, hidden_dim=50, width=7, act=nn.ReLU()):
        super(ConvContactMap, self).__init__()
        self.hidden = nn.Conv2d(2 * embed_dim, hidden_dim, 1)
        self.act = act
        self.conv = nn.Conv2d(hidden_dim, 1, width, padding=width // 2)
        self.clip()

    def clip(self):
        w = self.conv.weight
        self.conv.weight.data[:] = 0.5 * (w + w.transpose(2, 3))

    def forward(self, z):
        return self.predict(z)

    def predict(self, z):
        z = z.transpose(1, 2)
        z_dif = torch.abs(z.unsqueeze(2) - z.unsqueeze(3))
        z_mul = z.unsqueeze(2) * z.unsqueeze(3)
        z = torch.cat([z_dif, z_mul], 1)
        h = self.act(self.hidden(z))
        logits = self.conv(h).squeeze(1)
        return logits


class OrdinalRegression(nn.Module):

    def __init__(self, n_classes, compare=L1(), align_method='ssa',
        beta_init=10, allow_insertions=False, gap_init=-10):
        super(OrdinalRegression, self).__init__()
        self.n_out = n_classes
        self.compare = compare
        self.align_method = align_method
        self.allow_insertions = allow_insertions
        self.gap = nn.Parameter(torch.FloatTensor([gap_init]))
        self.theta = nn.Parameter(torch.ones(1, n_classes - 1))
        self.beta = nn.Parameter(torch.zeros(n_classes - 1) + beta_init)
        self.clip()

    def clip(self):
        self.theta.data.clamp_(min=0)

    def forward(self, z_x, z_y):
        if self.align_method == 'ssa':
            s = self.compare(z_x, z_y)
            if self.allow_insertions:
                s = pad_gap_scores(s, self.gap)
            a = F.softmax(s, 1)
            b = F.softmax(s, 0)
            if self.allow_insertions:
                index = s.size(0) - 1
                index = s.data.new(1).long().fill_(index)
                a = a.index_fill(0, index, 0)
                index = s.size(1) - 1
                index = s.data.new(1).long().fill_(index)
                b = b.index_fill(1, index, 0)
            a = a + b - a * b
            c = torch.sum(a * s) / torch.sum(a)
        elif self.align_method == 'ua':
            s = self.compare(z_x, z_y)
            c = torch.mean(s)
        elif self.align_method == 'me':
            z_x = z_x.mean(0)
            z_y = z_y.mean(0)
            c = self.compare(z_x.unsqueeze(0), z_y.unsqueeze(0)).squeeze(0)
        else:
            raise Exception('Unknown alignment method: ' + self.align_method)
        logits = c * self.theta + self.beta
        return logits.view(-1)


class BiLM(nn.Module):

    def __init__(self, nin, nout, embedding_dim, hidden_dim, num_layers,
        tied=True, mask_idx=None, dropout=0):
        super(BiLM, self).__init__()
        if mask_idx is None:
            mask_idx = nin - 1
        self.mask_idx = mask_idx
        self.embed = nn.Embedding(nin, embedding_dim, padding_idx=mask_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.tied = tied
        if tied:
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.rnn = nn.ModuleList(layers)
        else:
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.lrnn = nn.ModuleList(layers)
            layers = []
            nin = embedding_dim
            for _ in range(num_layers):
                layers.append(nn.LSTM(nin, hidden_dim, 1, batch_first=True))
                nin = hidden_dim
            self.rrnn = nn.ModuleList(layers)
        self.linear = nn.Linear(hidden_dim, nout)

    def hidden_size(self):
        h = 0
        if self.tied:
            for layer in self.rnn:
                h += 2 * layer.hidden_size
        else:
            for layer in self.lrnn:
                h += layer.hidden_size
            for layer in self.rrnn:
                h += layer.hidden_size
        return h

    def reverse(self, h):
        packed = type(h) is PackedSequence
        if packed:
            h, batch_sizes = pad_packed_sequence(h, batch_first=True)
            h_rvs = h.clone().zero_()
            for i in range(h.size(0)):
                n = batch_sizes[i]
                idx = [j for j in range(n - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                h_rvs[(i), :n] = h[i].index_select(0, idx)
            h_rvs = pack_padded_sequence(h_rvs, batch_sizes, batch_first=True)
        else:
            idx = [i for i in range(h.size(1) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            h_rvs = h.index_select(1, idx)
        return h_rvs

    def transform(self, z_fwd, z_rvs, last_only=False):
        if self.tied:
            layers = self.rnn
        else:
            layers = self.lrnn
        h_fwd = []
        h = z_fwd
        for rnn in layers:
            h, _ = rnn(h)
            if type(h) is PackedSequence:
                h = PackedSequence(self.dropout(h.data), h.batch_sizes)
            else:
                h = self.dropout(h)
            if not last_only:
                h_fwd.append(h)
        if last_only:
            h_fwd = h
        if self.tied:
            layers = self.rnn
        else:
            layers = self.rrnn
        h_rvs = []
        h = z_rvs
        for rnn in layers:
            h, _ = rnn(h)
            if type(h) is PackedSequence:
                h = PackedSequence(self.dropout(h.data), h.batch_sizes)
            else:
                h = self.dropout(h)
            if not last_only:
                h_rvs.append(self.reverse(h))
        if last_only:
            h_rvs = self.reverse(h)
        return h_fwd, h_rvs

    def embed_and_split(self, x, pad=False):
        packed = type(x) is PackedSequence
        if packed:
            x, batch_sizes = pad_packed_sequence(x, batch_first=True)
        if pad:
            x = x + 1
            x_ = x.data.new(x.size(0), x.size(1) + 2).zero_()
            if packed:
                for i in range(len(batch_sizes)):
                    n = batch_sizes[i]
                    x_[(i), 1:n + 1] = x[(i), :n]
                batch_sizes = [(s + 2) for s in batch_sizes]
            else:
                x_[:, 1:-1] = x
            x = x_
        z = self.embed(x)
        z_fwd = z[:, :-1]
        z_rvs = z[:, 1:]
        if packed:
            lengths = [(s - 1) for s in batch_sizes]
            z_fwd = pack_padded_sequence(z_fwd, lengths, batch_first=True)
            z_rvs = pack_padded_sequence(z_rvs, lengths, batch_first=True)
        z_rvs = self.reverse(z_rvs)
        return z_fwd, z_rvs

    def encode(self, x):
        z_fwd, z_rvs = self.embed_and_split(x, pad=True)
        h_fwd_layers, h_rvs_layers = self.transform(z_fwd, z_rvs)
        packed = type(z_fwd) is PackedSequence
        concat = []
        for h_fwd, h_rvs in zip(h_fwd_layers, h_rvs_layers):
            if packed:
                h_fwd, batch_sizes = pad_packed_sequence(h_fwd, batch_first
                    =True)
                h_rvs, batch_sizes = pad_packed_sequence(h_rvs, batch_first
                    =True)
            h_fwd = h_fwd[:, :-1]
            h_rvs = h_rvs[:, 1:]
            concat.append(h_fwd)
            concat.append(h_rvs)
        h = torch.cat(concat, 2)
        if packed:
            batch_sizes = [(s - 1) for s in batch_sizes]
            h = pack_padded_sequence(h, batch_sizes, batch_first=True)
        return h

    def forward(self, x):
        z_fwd, z_rvs = self.embed_and_split(x, pad=False)
        h_fwd, h_rvs = self.transform(z_fwd, z_rvs, last_only=True)
        packed = type(z_fwd) is PackedSequence
        if packed:
            h_flat = h_fwd.data
            logp_fwd = self.linear(h_flat)
            logp_fwd = PackedSequence(logp_fwd, h_fwd.batch_sizes)
            h_flat = h_rvs.data
            logp_rvs = self.linear(h_flat)
            logp_rvs = PackedSequence(logp_rvs, h_rvs.batch_sizes)
            logp_fwd, batch_sizes = pad_packed_sequence(logp_fwd,
                batch_first=True)
            logp_rvs, batch_sizes = pad_packed_sequence(logp_rvs,
                batch_first=True)
        else:
            b = h_fwd.size(0)
            n = h_fwd.size(1)
            h_flat = h_fwd.contiguous().view(-1, h_fwd.size(2))
            logp_fwd = self.linear(h_flat)
            logp_fwd = logp_fwd.view(b, n, -1)
            h_flat = h_rvs.contiguous().view(-1, h_rvs.size(2))
            logp_rvs = self.linear(h_flat)
            logp_rvs = logp_rvs.view(b, n, -1)
        b = h_fwd.size(0)
        zero = h_fwd.data.new(b, 1, logp_fwd.size(2)).zero_()
        logp_fwd = torch.cat([zero, logp_fwd], 1)
        logp_rvs = torch.cat([logp_rvs, zero], 1)
        logp = F.log_softmax(logp_fwd + logp_rvs, dim=2)
        if packed:
            batch_sizes = [(s + 1) for s in batch_sizes]
            logp = pack_padded_sequence(logp, batch_sizes, batch_first=True)
        return logp


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_tbepler_protein_sequence_embedding_iclr2019(_paritybench_base):
    pass
    def test_000(self):
        self._check(ConvContactMap(*[], **{'embed_dim': 4}), [torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(DotProduct(*[], **{}), [torch.rand([4, 4]), torch.rand([4, 4])], {})

    def test_002(self):
        self._check(L1(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(L2(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(LSTM(*[], **{'n_in': 4, 'n_hidden': 4, 'n_out': 4}), [torch.rand([4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(OrdinalRegression(*[], **{'n_classes': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

