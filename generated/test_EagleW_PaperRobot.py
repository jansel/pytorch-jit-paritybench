import sys
_module = sys.modules[__name__]
del sys
GAT = _module
GATA = _module
TAT = _module
model = _module
graph_attention = _module
test = _module
train = _module
utils = _module
data_loader = _module
eval = _module
eval_final = _module
input = _module
loader = _module
loader = _module
logger = _module
preprocessing = _module
Decoder = _module
Encoder = _module
baseRNN = _module
predictor = _module
seq2seq = _module
utils = _module
pycocoevalcap = _module
bleu = _module
bleu_scorer = _module
cider = _module
cider_scorer = _module
meteor = _module
rouge = _module
test = _module
train = _module
optim = _module

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


import torch.nn as nn


import torch


import torch.nn.functional as F


import numpy as np


from torch.autograd import Variable


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import time


import torch.optim as optim


from torch.utils import data


import math


from collections import OrderedDict


import copy


class GAT(nn.Module):

    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout,
            alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GATA(nn.Module):

    def __init__(self, emb_dim, hid_dim, out_dim, num_voc, num_heads,
        num_ent, num_rel, dropout, alpha, **kwargs):
        super(GATA, self).__init__()
        self.ent_embedding = nn.Embedding(num_ent, emb_dim)
        self.rel_embedding = nn.Embedding(num_rel, emb_dim)
        self.graph = GAT(nfeat=emb_dim, nhid=hid_dim, dropout=dropout,
            nheads=num_heads, alpha=alpha)
        self.text = TAT(emb_dim, num_voc)
        self.gate = nn.Embedding(num_ent, out_dim)

    def forward(self, nodes, adj, pos, shifted_pos, h_sents, h_order,
        h_lengths, t_sents, t_order, t_lengths):
        node_features = self.ent_embedding(nodes)
        graph = self.graph(node_features, adj)
        head_graph = graph[[shifted_pos[:, (0)].squeeze()]]
        tail_graph = graph[[shifted_pos[:, (1)].squeeze()]]
        head_text = self.text(h_sents, h_order, h_lengths, node_features[[
            shifted_pos[:, (0)].squeeze()]])
        tail_text = self.text(t_sents, t_order, t_lengths, node_features[[
            shifted_pos[:, (1)].squeeze()]])
        r_pos = self.rel_embedding(pos[:, (2)].squeeze())
        gate_head = self.gate(pos[:, (0)].squeeze())
        gate_tail = self.gate(pos[:, (1)].squeeze())
        score_pos = self._score(head_graph, head_text, tail_graph,
            tail_text, r_pos, gate_head, gate_tail)
        return score_pos

    def _score(self, hg, ht, tg, tt, r, gh, gt):
        gate_h = torch.sigmoid(gh)
        gate_t = torch.sigmoid(gt)
        head = gate_h * hg + (1 - gate_h) * ht
        tail = gate_t * tg + (1 - gate_t) * tt
        s = (head + r - tail) ** 2
        return s


class TAT(nn.Module):
    """
    A Bi-LSTM layer with attention
    """

    def __init__(self, embedding_dim, voc_size):
        super(TAT, self).__init__()
        self.hidden_dim = embedding_dim
        self.word_embeddings = nn.Embedding(voc_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim // 2,
            bidirectional=True)
        self.attF = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, sentence, orders, lengths, ent_emb):
        embedded = self.word_embeddings(sentence)
        padded_sent = pack_padded_sequence(embedded, lengths, batch_first=True)
        output = padded_sent
        output, hidden = self.lstm(output)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[orders]
        att = torch.unsqueeze(self.attF(ent_emb), 2)
        att_score = F.softmax(torch.bmm(output, att), dim=1)
        o = torch.squeeze(torch.bmm(output.transpose(1, 2), att_score))
        return o


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(
            in_features, out_features).type(torch.cuda.FloatTensor if torch
            .cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)
            ), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(
            out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.
            is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform_(torch.FloatTensor(
            out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.
            is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]
        f_1 = h @ self.a1
        f_2 = h @ self.a2
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))
        zero_vec = -9000000000000000.0 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)
        if self.concat:
            return F.sigmoid(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features
            ) + ' -> ' + str(self.out_features) + ')'


class TermEncoder(nn.Module):

    def __init__(self, embedding, input_dropout_p):
        super(TermEncoder, self).__init__()
        self.embedding = embedding
        self.input_dropout = nn.Dropout(input_dropout_p)

    def forward(self, term):
        mask = term.eq(0).detach()
        embedded = self.embedding(term)
        embedded = self.input_dropout(embedded)
        return embedded, mask


class BaseRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, input_dropout_p, n_layers,
        rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell: {0}'.format(rnn_cell))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class Seq2seq(nn.Module):

    def __init__(self, ref_encoder, term_encoder, decoder):
        super(Seq2seq, self).__init__()
        self.ref_encoder = ref_encoder
        self.term_encoder = term_encoder
        self.decoder = decoder

    def forward(self, batch_s, batch_o_s, source_len, max_source_oov,
        batch_term, batch_o_term, batch_t=None, batch_o_t=None,
        teacher_forcing_ratio=0, beam=False, stopwords=None, sflag=False):
        encoder_outputs, encoder_hidden, enc_mask = self.ref_encoder(batch_s,
            source_len)
        term_output, term_mask = self.term_encoder(batch_term)
        result = self.decoder(max_source_oov, batch_t, batch_o_t, batch_o_s,
            enc_mask, encoder_hidden, encoder_outputs, batch_o_term,
            term_mask, term_output, teacher_forcing_ratio, beam, stopwords,
            sflag)
        return result
