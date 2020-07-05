import sys
_module = sys.modules[__name__]
del sys
evaluate = _module
model = _module
data = _module
ema = _module
model = _module
run = _module
utils = _module
nn = _module

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


import torch.nn.functional as F


class BiDAF(nn.Module):

    def __init__(self, args, pretrained):
        super(BiDAF, self).__init__()
        self.args = args
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        self.char_conv = nn.Sequential(nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)), nn.ReLU())
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        assert self.args.hidden_size * 2 == self.args.char_channel_size + self.args.word_dim
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i), nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i), nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2), nn.Sigmoid()))
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size, bidirectional=True, batch_first=True, dropout=args.dropout)
        self.att_weight_c = Linear(args.hidden_size * 2, 1)
        self.att_weight_q = Linear(args.hidden_size * 2, 1)
        self.att_weight_cq = Linear(args.hidden_size * 2, 1)
        self.modeling_LSTM1 = LSTM(input_size=args.hidden_size * 8, hidden_size=args.hidden_size, bidirectional=True, batch_first=True, dropout=args.dropout)
        self.modeling_LSTM2 = LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size, bidirectional=True, batch_first=True, dropout=args.dropout)
        self.p1_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.p1_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.p2_weight_g = Linear(args.hidden_size * 8, 1, dropout=args.dropout)
        self.p2_weight_m = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.output_LSTM = LSTM(input_size=args.hidden_size * 2, hidden_size=args.hidden_size, bidirectional=True, batch_first=True, dropout=args.dropout)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, batch):

        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            x = self.dropout(self.char_emb(x))
            x = x.transpose(2, 3)
            x = x.view(-1, self.args.char_dim, x.size(3)).unsqueeze(1)
            x = self.char_conv(x).squeeze()
            x = F.max_pool1d(x, x.size(2)).squeeze()
            x = x.view(batch_size, -1, self.args.char_channel_size)
            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)
            cq = []
            for i in range(q_len):
                qi = q.select(1, i).unsqueeze(1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            cq = torch.stack(cq, dim=-1)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + cq
            a = F.softmax(s, dim=2)
            c2q_att = torch.bmm(a, q)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            q2c_att = torch.bmm(b, c).squeeze()
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            m2 = self.output_LSTM((m, l))[0]
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()
            return p1, p2
        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]
        g = att_flow_layer(c, q)
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        p1, p2 = output_layer(g, m, c_lens)
        return p1, p2


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)
            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x):
        x, x_len = x
        x = self.dropout(x)
        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)
        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)
        x = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)[0]
        x = x.index_select(dim=0, index=x_ori_idx)
        h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)
        return x, h


class Linear(nn.Module):

    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_galsang_BiDAF_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

