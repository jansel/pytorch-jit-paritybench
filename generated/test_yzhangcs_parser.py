import sys
_module = sys.modules[__name__]
del sys
parser = _module
cmds = _module
cmd = _module
evaluate = _module
predict = _module
train = _module
config = _module
model = _module
modules = _module
bert = _module
biaffine = _module
bilstm = _module
char_lstm = _module
dropout = _module
mlp = _module
scalar_mix = _module
utils = _module
alg = _module
common = _module
corpus = _module
data = _module
embedding = _module
field = _module
fn = _module
metric = _module
vocab = _module
run = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


from torch.optim import Adam


from torch.optim.lr_scheduler import ExponentialLR


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pad_sequence


from torch.nn.modules.rnn import apply_permutation


from torch.nn.utils.rnn import PackedSequence


from collections.abc import Iterable


from itertools import chain


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch.utils.data import Sampler


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.word_embed = nn.Embedding(num_embeddings=args.n_words, embedding_dim=args.n_embed)
        if args.feat == 'char':
            self.feat_embed = CHAR_LSTM(n_chars=args.n_feats, n_embed=args.n_char_embed, n_out=args.n_embed)
        elif args.feat == 'bert':
            self.feat_embed = BertEmbedding(model=args.bert_model, n_layers=args.n_bert_layers, n_out=args.n_embed)
        else:
            self.feat_embed = nn.Embedding(num_embeddings=args.n_feats, embedding_dim=args.n_embed)
        self.embed_dropout = IndependentDropout(p=args.embed_dropout)
        self.lstm = BiLSTM(input_size=args.n_embed * 2, hidden_size=args.n_lstm_hidden, num_layers=args.n_lstm_layers, dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)
        self.mlp_arc_h = MLP(n_in=args.n_lstm_hidden * 2, n_hidden=args.n_mlp_arc, dropout=args.mlp_dropout)
        self.mlp_arc_d = MLP(n_in=args.n_lstm_hidden * 2, n_hidden=args.n_mlp_arc, dropout=args.mlp_dropout)
        self.mlp_rel_h = MLP(n_in=args.n_lstm_hidden * 2, n_hidden=args.n_mlp_rel, dropout=args.mlp_dropout)
        self.mlp_rel_d = MLP(n_in=args.n_lstm_hidden * 2, n_hidden=args.n_mlp_rel, dropout=args.mlp_dropout)
        self.arc_attn = Biaffine(n_in=args.n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=args.n_mlp_rel, n_out=args.n_rels, bias_x=True, bias_y=True)
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, words, feats):
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        ext_mask = words.ge(self.word_embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        if self.args.feat == 'char':
            feat_embed = self.feat_embed(feats[mask])
            feat_embed = pad_sequence(feat_embed.split(lens.tolist()), True)
        elif self.args.feat == 'bert':
            feat_embed = self.feat_embed(*feats)
        else:
            feat_embed = self.feat_embed(feats)
        word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
        embed = torch.cat((word_embed, feat_embed), dim=-1)
        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)
        s_arc = self.arc_attn(arc_d, arc_h)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        return s_arc, s_rel

    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.is_available() else 'cpu'
        state = torch.load(path, map_location=device)
        model = cls(state['args'])
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model
        return model

    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if hasattr(self, 'pretrained'):
            pretrained = state_dict.pop('pretrained.weight')
        state = {'args': self.args, 'state_dict': state_dict, 'pretrained': pretrained}
        torch.save(state, path)


class BertEmbedding(nn.Module):

    def __init__(self, model, n_layers, n_out, requires_grad=False):
        super(BertEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained(model, output_hidden_states=True)
        self.bert = self.bert.requires_grad_(requires_grad)
        self.n_layers = n_layers
        self.n_out = n_out
        self.requires_grad = requires_grad
        self.hidden_size = self.bert.config.hidden_size
        self.scalar_mix = ScalarMix(n_layers)
        self.projection = nn.Linear(self.hidden_size, n_out, False)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'n_layers={self.n_layers}, n_out={self.n_out}'
        if self.requires_grad:
            s += f', requires_grad={self.requires_grad}'
        s += ')'
        return s

    def forward(self, subwords, bert_lens, bert_mask):
        batch_size, seq_len = bert_lens.shape
        mask = bert_lens.gt(0)
        if not self.requires_grad:
            self.bert.eval()
        _, _, bert = self.bert(subwords, attention_mask=bert_mask)
        bert = bert[-self.n_layers:]
        bert = self.scalar_mix(bert)
        bert = bert[bert_mask].split(bert_lens[mask].tolist())
        bert = torch.stack([i.mean(0) for i in bert])
        bert_embed = bert.new_zeros(batch_size, seq_len, self.hidden_size)
        bert_embed = bert_embed.masked_scatter_(mask.unsqueeze(-1), bert)
        bert_embed = self.projection(bert_embed)
        return bert_embed


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f'n_in={self.n_in}, n_out={self.n_out}'
        if self.bias_x:
            s += f', bias_x={self.bias_x}'
        if self.bias_y:
            s += f', bias_y={self.bias_y}'
        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[(...), :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[(...), :1])), -1)
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.squeeze(1)
        return s


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=hidden_size))
            input_size = hidden_size * 2
        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'{self.input_size}, {self.hidden_size}'
        if self.num_layers > 1:
            s += f', num_layers={self.num_layers}'
        if self.dropout > 0:
            s += f', dropout={self.dropout}'
        s += ')'
        return s

    def reset_parameters(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)
        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)
        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size])) for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)
        return output, hx_n

    def forward(self, sequence, hx=None):
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []
        if hx is None:
            ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)
        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [(i * mask[:len(i)]) for i in x]
            x_f, (h_f, c_f) = self.layer_forward(x=x, hx=(h[i, 0], c[i, 0]), cell=self.f_cells[i], batch_sizes=batch_sizes)
            x_b, (h_b, c_b) = self.layer_forward(x=x, hx=(h[i, 1], c[i, 1]), cell=self.b_cells[i], batch_sizes=batch_sizes, reverse=True)
            x = torch.cat((x_f, x_b), -1)
            h_n.append(torch.stack((h_f, h_b)))
            c_n.append(torch.stack((c_f, c_b)))
        x = PackedSequence(x, sequence.batch_sizes, sequence.sorted_indices, sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)
        return x, hx


class CHAR_LSTM(nn.Module):

    def __init__(self, n_chars, n_embed, n_out):
        super(CHAR_LSTM, self).__init__()
        self.embed = nn.Embedding(num_embeddings=n_chars, embedding_dim=n_embed)
        self.lstm = nn.LSTM(input_size=n_embed, hidden_size=n_out // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        mask = x.gt(0)
        lens = mask.sum(dim=1)
        x = pack_padded_sequence(self.embed(x), lens, True, False)
        x, (hidden, _) = self.lstm(x)
        hidden = torch.cat(torch.unbind(hidden), dim=-1)
        return hidden


class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()
        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f'p={self.p}'
        if self.batch_first:
            s += f', batch_first={self.batch_first}'
        return s

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, (0)], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask
        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)
        return mask


class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def forward(self, *items):
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p) for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [(mask * scale) for mask in masks]
            items = [(item * mask.unsqueeze(dim=-1)) for item, mask in zip(items, masks)]
        return items


class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0):
        super(MLP, self).__init__()
        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ScalarMix(nn.Module):

    def __init__(self, n_layers, dropout=0):
        super(ScalarMix, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def extra_repr(self):
        s = f'n_layers={self.n_layers}'
        if self.dropout.p > 0:
            s += f', dropout={self.dropout.p}'
        return s

    def forward(self, tensors):
        normed_weights = self.dropout(self.weights.softmax(-1))
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))
        return self.gamma * weighted_sum


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Biaffine,
     lambda: ([], {'n_in': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (CHAR_LSTM,
     lambda: ([], {'n_chars': 4, 'n_embed': 4, 'n_out': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     True),
    (IndependentDropout,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (MLP,
     lambda: ([], {'n_in': 4, 'n_hidden': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Model,
     lambda: ([], {'args': _mock_config(n_words=4, n_embed=4, feat=4, n_feats=4, embed_dropout=0.5, n_lstm_hidden=4, n_lstm_layers=1, lstm_dropout=0.5, n_mlp_arc=4, mlp_dropout=0.5, n_mlp_rel=4, n_rels=4, pad_index=4, unk_index=4)}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
    (ScalarMix,
     lambda: ([], {'n_layers': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SharedDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_yzhangcs_parser(_paritybench_base):
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

