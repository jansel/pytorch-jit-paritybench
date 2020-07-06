import sys
_module = sys.modules[__name__]
del sys
build_all_data = _module
build_pretrained_w2v = _module
core = _module
bamnet = _module
bamnet = _module
ent_modules = _module
entnet = _module
modules = _module
utils = _module
build_data = _module
build_all = _module
freebase = _module
webquestions = _module
config = _module
freebase_utils = _module
generic_utils = _module
metrics = _module
joint_test = _module
run_freebase = _module
run_webquestions = _module
test = _module
test_entnet = _module
train = _module
train_entnet = _module

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


import numpy as np


import torch


from torch import optim


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.nn import MultiLabelMarginLoss


import torch.backends.cudnn as cudnn


import torch.nn as nn


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


import torch.nn.functional as F


from torch.nn import CrossEntropyLoss


from torch.autograd import Variable


class EncoderCNN(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, kernel_size=[2, 3], dropout=None, shared_embed=None, init_word_embed=None, use_cuda=True):
        super(EncoderCNN, self).__init__()
        None
        self.use_cuda = use_cuda
        self.dropout = dropout
        self.embed = shared_embed if shared_embed is not None else nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.cnns = nn.ModuleList([nn.Conv1d(embed_size, hidden_size, kernel_size=k, padding=k - 1) for k in kernel_size])
        if len(kernel_size) > 1:
            self.fc = nn.Linear(len(kernel_size) * hidden_size, hidden_size)
        if shared_embed is None:
            self.init_weights(init_word_embed)

    def init_weights(self, init_word_embed):
        if init_word_embed is not None:
            None
            self.embed.weight.data.copy_(torch.from_numpy(init_word_embed))
        else:
            self.embed.weight.data.uniform_(-0.08, 0.08)

    def forward(self, x, x_len=None):
        """x: [batch_size * max_length]
           x_len: reserved
        """
        x = self.embed(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        z = [conv(x) for conv in self.cnns]
        output = [F.max_pool1d(i, kernel_size=i.size(-1)).squeeze(-1) for i in z]
        if len(output) > 1:
            output = self.fc(torch.cat(output, -1))
        else:
            output = output[0]
        return None, output


def to_cuda(x, use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        x = x
    return x


class EncoderRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, dropout=None, bidirectional=False, shared_embed=None, init_word_embed=None, rnn_type='lstm', use_cuda=True):
        super(EncoderRNN, self).__init__()
        if not rnn_type in ('lstm', 'gru'):
            raise RuntimeError('rnn_type is expected to be lstm or gru, got {}'.format(rnn_type))
        if bidirectional:
            None
        else:
            None
        if bidirectional and hidden_size % 2 != 0:
            raise RuntimeError('hidden_size is expected to be even in the bidirectional mode!')
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.embed = shared_embed if shared_embed is not None else nn.Embedding(vocab_size, embed_size, padding_idx=0)
        model = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.model = model(embed_size, self.hidden_size, 1, batch_first=True, bidirectional=bidirectional)
        if shared_embed is None:
            self.init_weights(init_word_embed)

    def init_weights(self, init_word_embed):
        if init_word_embed is not None:
            None
            self.embed.weight.data.copy_(torch.from_numpy(init_word_embed))
        else:
            self.embed.weight.data.uniform_(-0.08, 0.08)

    def forward(self, x, x_len):
        """x: [batch_size * max_length]
           x_len: [batch_size]
        """
        x = self.embed(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        x = pack_padded_sequence(x[indx], sorted_x_len.data.tolist(), batch_first=True)
        h0 = to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.use_cuda)
        if self.rnn_type == 'lstm':
            c0 = to_cuda(torch.zeros(self.num_directions, x_len.size(0), self.hidden_size), self.use_cuda)
            packed_h, (packed_h_t, _) = self.model(x, (h0, c0))
            if self.num_directions == 2:
                packed_h_t = torch.cat([packed_h_t[i] for i in range(packed_h_t.size(0))], -1)
        else:
            packed_h, packed_h_t = self.model(x, h0)
            if self.num_directions == 2:
                packed_h_t = packed_h_t.transpose(0, 1).contiguous().view(query_lengths.size(0), -1)
        hh, _ = pad_packed_sequence(packed_h, batch_first=True)
        _, inverse_indx = torch.sort(indx, 0)
        restore_hh = hh[inverse_indx]
        restore_packed_h_t = packed_h_t[inverse_indx]
        return restore_hh, restore_packed_h_t


class SeqEncoder(object):
    """Question Encoder"""

    def __init__(self, vocab_size, embed_size, hidden_size, seq_enc_type='lstm', word_emb_dropout=None, cnn_kernel_size=[3], bidirectional=False, shared_embed=None, init_word_embed=None, use_cuda=True):
        if seq_enc_type in ('lstm', 'gru'):
            self.que_enc = EncoderRNN(vocab_size, embed_size, hidden_size, dropout=word_emb_dropout, bidirectional=bidirectional, shared_embed=shared_embed, init_word_embed=init_word_embed, rnn_type=seq_enc_type, use_cuda=use_cuda)
        elif seq_enc_type == 'cnn':
            self.que_enc = EncoderCNN(vocab_size, embed_size, hidden_size, kernel_size=cnn_kernel_size, dropout=word_emb_dropout, shared_embed=shared_embed, init_word_embed=init_word_embed, use_cuda=use_cuda)
        else:
            raise RuntimeError('Unknown SeqEncoder type: {}'.format(seq_enc_type))


class EntEncoder(nn.Module):
    """Entity Encoder"""

    def __init__(self, o_embed_size, hidden_size, num_ent_types, num_relations, vocab_size=None, vocab_embed_size=None, shared_embed=None, seq_enc_type='lstm', word_emb_dropout=None, ent_enc_dropout=None, use_cuda=True):
        super(EntEncoder, self).__init__()
        self.ent_enc_dropout = ent_enc_dropout
        self.hidden_size = hidden_size
        self.relation_embed = nn.Embedding(num_relations, o_embed_size, padding_idx=0)
        self.embed = shared_embed if shared_embed is not None else nn.Embedding(vocab_size, vocab_embed_size, padding_idx=0)
        self.vocab_embed_size = self.embed.weight.data.size(1)
        self.linear_node_name_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_node_type_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_rels_key = nn.Linear(hidden_size + o_embed_size, hidden_size, bias=False)
        self.linear_node_name_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_node_type_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_rels_val = nn.Linear(hidden_size + o_embed_size, hidden_size, bias=False)
        self.kg_enc_ent = SeqEncoder(vocab_size, self.vocab_embed_size, hidden_size, seq_enc_type=seq_enc_type, word_emb_dropout=word_emb_dropout, bidirectional=True, cnn_kernel_size=[3], shared_embed=shared_embed, use_cuda=use_cuda).que_enc
        self.kg_enc_type = SeqEncoder(vocab_size, self.vocab_embed_size, hidden_size, seq_enc_type=seq_enc_type, word_emb_dropout=word_emb_dropout, bidirectional=True, cnn_kernel_size=[3], shared_embed=shared_embed, use_cuda=use_cuda).que_enc
        self.kg_enc_rel = SeqEncoder(vocab_size, self.vocab_embed_size, hidden_size, seq_enc_type=seq_enc_type, word_emb_dropout=word_emb_dropout, bidirectional=True, cnn_kernel_size=[3], shared_embed=shared_embed, use_cuda=use_cuda).que_enc

    def forward(self, x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask):
        node_ent_names, node_type_names, node_types, edge_rel_names, edge_rels = self.enc_kg_features(x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask)
        node_name_key = self.linear_node_name_key(node_ent_names)
        node_type_key = self.linear_node_type_key(node_type_names)
        rel_key = self.linear_rels_key(torch.cat([edge_rel_names, edge_rels], -1))
        node_name_val = self.linear_node_name_val(node_ent_names)
        node_type_val = self.linear_node_type_val(node_type_names)
        rel_val = self.linear_rels_val(torch.cat([edge_rel_names, edge_rels], -1))
        ent_comp_val = [node_name_val, node_type_val, rel_val]
        ent_comp_key = [node_name_key, node_type_key, rel_key]
        return ent_comp_val, ent_comp_key

    def enc_kg_features(self, x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask):
        node_ent_names = self.kg_enc_ent(x_ent_names.view(-1, x_ent_names.size(-1)), x_ent_name_len.view(-1))[1].view(x_ent_names.size(0), x_ent_names.size(1), -1)
        node_type_names = self.kg_enc_type(x_type_names.view(-1, x_type_names.size(-1)), x_type_name_len.view(-1))[1].view(x_type_names.size(0), x_type_names.size(1), -1)
        node_types = None
        edge_rel_names = torch.mean(self.kg_enc_rel(x_rel_names.view(-1, x_rel_names.size(-1)), x_rel_name_len.view(-1))[1].view(x_rel_names.size(0), x_rel_names.size(1), x_rel_names.size(2), -1), 2)
        edge_rels = torch.mean(self.relation_embed(x_rels.view(-1, x_rels.size(-1))), 1).view(x_rels.size(0), x_rels.size(1), -1)
        if self.ent_enc_dropout:
            node_ent_names = F.dropout(node_ent_names, p=self.ent_enc_dropout, training=self.training)
            node_type_names = F.dropout(node_type_names, p=self.ent_enc_dropout, training=self.training)
            edge_rel_names = F.dropout(edge_rel_names, p=self.ent_enc_dropout, training=self.training)
            edge_rels = F.dropout(edge_rels, p=self.ent_enc_dropout, training=self.training)
        return node_ent_names, node_type_names, node_types, edge_rel_names, edge_rels


INF = 1e+20


class Attention(nn.Module):

    def __init__(self, hidden_size, h_state_embed_size=None, in_memory_embed_size=None, atten_type='simple'):
        super(Attention, self).__init__()
        self.atten_type = atten_type
        if not h_state_embed_size:
            h_state_embed_size = hidden_size
        if not in_memory_embed_size:
            in_memory_embed_size = hidden_size
        if atten_type in ('mul', 'add'):
            self.W = torch.Tensor(h_state_embed_size, hidden_size)
            self.W = nn.Parameter(nn.init.xavier_uniform_(self.W))
            if atten_type == 'add':
                self.W2 = torch.Tensor(in_memory_embed_size, hidden_size)
                self.W2 = nn.Parameter(nn.init.xavier_uniform_(self.W2))
                self.W3 = torch.Tensor(hidden_size, 1)
                self.W3 = nn.Parameter(nn.init.xavier_uniform_(self.W3))
        elif atten_type == 'simple':
            pass
        else:
            raise RuntimeError('Unknown atten_type: {}'.format(self.atten_type))

    def forward(self, query_embed, in_memory_embed, atten_mask=None):
        if self.atten_type == 'simple':
            attention = torch.bmm(in_memory_embed, query_embed.unsqueeze(2)).squeeze(2)
        elif self.atten_type == 'mul':
            attention = torch.bmm(in_memory_embed, torch.mm(query_embed, self.W).unsqueeze(2)).squeeze(2)
        elif self.atten_type == 'add':
            attention = torch.tanh(torch.mm(in_memory_embed.view(-1, in_memory_embed.size(-1)), self.W2).view(in_memory_embed.size(0), -1, self.W2.size(-1)) + torch.mm(query_embed, self.W).unsqueeze(1))
            attention = torch.mm(attention.view(-1, attention.size(-1)), self.W3).view(attention.size(0), -1)
        else:
            raise RuntimeError('Unknown atten_type: {}'.format(self.atten_type))
        if atten_mask is not None:
            attention = atten_mask * attention - (1 - atten_mask) * INF
        return attention


class GRUStep(nn.Module):

    def __init__(self, hidden_size, input_size):
        super(GRUStep, self).__init__()
        """GRU module"""
        self.linear_z = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_r = nn.Linear(hidden_size + input_size, hidden_size, bias=False)
        self.linear_t = nn.Linear(hidden_size + input_size, hidden_size, bias=False)

    def forward(self, h_state, input_):
        z = torch.sigmoid(self.linear_z(torch.cat([h_state, input_], -1)))
        r = torch.sigmoid(self.linear_r(torch.cat([h_state, input_], -1)))
        t = torch.tanh(self.linear_t(torch.cat([r * h_state, input_], -1)))
        h_state = (1 - z) * h_state + z * t
        return h_state


class EntRomHop(nn.Module):

    def __init__(self, query_embed_size, in_memory_embed_size, hidden_size, atten_type='add'):
        super(EntRomHop, self).__init__()
        self.atten = Attention(hidden_size, query_embed_size, in_memory_embed_size, atten_type=atten_type)
        self.gru_step = GRUStep(hidden_size, in_memory_embed_size)

    def forward(self, h_state, key_memory_embed, val_memory_embed, atten_mask=None):
        attention = self.atten(h_state, key_memory_embed, atten_mask=atten_mask)
        probs = torch.softmax(attention, dim=-1)
        memory_output = torch.bmm(probs.unsqueeze(1), val_memory_embed).squeeze(1)
        h_state = self.gru_step(h_state, memory_output)
        return h_state


class SelfAttention_CoAtt(nn.Module):

    def __init__(self, hidden_size, use_cuda=True):
        super(SelfAttention_CoAtt, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.model = nn.LSTM(2 * hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)

    def forward(self, x, x_len, atten_mask):
        CoAtt = torch.bmm(x, x.transpose(1, 2))
        CoAtt = atten_mask.unsqueeze(1) * CoAtt - (1 - atten_mask).unsqueeze(1) * INF
        CoAtt = torch.softmax(CoAtt, dim=-1)
        new_x = torch.cat([torch.bmm(CoAtt, x), x], -1)
        sorted_x_len, indx = torch.sort(x_len, 0, descending=True)
        new_x = pack_padded_sequence(new_x[indx], sorted_x_len.data.tolist(), batch_first=True)
        h0 = to_cuda(torch.zeros(2, x_len.size(0), self.hidden_size // 2), self.use_cuda)
        c0 = to_cuda(torch.zeros(2, x_len.size(0), self.hidden_size // 2), self.use_cuda)
        packed_h, (packed_h_t, _) = self.model(new_x, (h0, c0))
        _, inverse_indx = torch.sort(indx, 0)
        packed_h_t = torch.cat([packed_h_t[i] for i in range(packed_h_t.size(0))], -1)
        restore_packed_h_t = packed_h_t[inverse_indx]
        output = restore_packed_h_t
        return output


class Entnet(nn.Module):

    def __init__(self, vocab_size, vocab_embed_size, o_embed_size, hidden_size, num_ent_types, num_relations, seq_enc_type='cnn', word_emb_dropout=None, que_enc_dropout=None, ent_enc_dropout=None, pre_w2v=None, num_hops=1, att='add', use_cuda=True):
        super(Entnet, self).__init__()
        self.use_cuda = use_cuda
        self.seq_enc_type = seq_enc_type
        self.que_enc_dropout = que_enc_dropout
        self.ent_enc_dropout = ent_enc_dropout
        self.num_hops = num_hops
        self.hidden_size = hidden_size
        self.que_enc = SeqEncoder(vocab_size, vocab_embed_size, hidden_size, seq_enc_type=seq_enc_type, word_emb_dropout=word_emb_dropout, bidirectional=True, cnn_kernel_size=[2, 3], init_word_embed=pre_w2v, use_cuda=use_cuda).que_enc
        self.ent_enc = EntEncoder(o_embed_size, hidden_size, num_ent_types, num_relations, vocab_size=vocab_size, vocab_embed_size=vocab_embed_size, shared_embed=self.que_enc.embed, seq_enc_type=seq_enc_type, word_emb_dropout=word_emb_dropout, ent_enc_dropout=ent_enc_dropout, use_cuda=use_cuda)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        if seq_enc_type in ('lstm', 'gru'):
            self.self_atten = SelfAttention_CoAtt(hidden_size)
            None
        self.ent_memory_hop = EntRomHop(hidden_size, hidden_size, hidden_size, atten_type=att)
        None

    def forward(self, memories, queries, query_lengths):
        x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask = memories
        x_rel_mask = self.create_mask_3D(x_rel_mask, x_rels.size(-1), use_cuda=self.use_cuda)
        if self.seq_enc_type in ('lstm', 'gru'):
            Q_r = self.que_enc(queries, query_lengths)[0]
            if self.que_enc_dropout:
                Q_r = F.dropout(Q_r, p=self.que_enc_dropout, training=self.training)
            query_mask = self.create_mask(query_lengths, Q_r.size(1), self.use_cuda)
            q_r = self.self_atten(Q_r, query_lengths, query_mask)
        else:
            q_r = self.que_enc(queries, query_lengths)[1]
            if self.que_enc_dropout:
                q_r = F.dropout(q_r, p=self.que_enc_dropout, training=self.training)
        ent_val, ent_key = self.ent_enc(x_ent_names, x_ent_name_len, x_type_names, x_types, x_type_name_len, x_rel_names, x_rels, x_rel_name_len, x_rel_mask)
        ent_val = torch.cat([each.unsqueeze(2) for each in ent_val], 2)
        ent_key = torch.cat([each.unsqueeze(2) for each in ent_key], 2)
        ent_val = torch.sum(ent_val, 2)
        ent_key = torch.sum(ent_key, 2)
        mem_hop_scores = []
        mid_score = self.clf_score(q_r, ent_key)
        mem_hop_scores.append(mid_score)
        for _ in range(self.num_hops):
            q_r = q_r + self.ent_memory_hop(q_r, ent_key, ent_val)
            q_r = self.batchnorm(q_r)
            mid_score = self.clf_score(q_r, ent_key)
            mem_hop_scores.append(mid_score)
        return mem_hop_scores

    def clf_score(self, q_r, ent_key):
        return torch.matmul(ent_key, q_r.unsqueeze(-1)).squeeze(-1)

    def create_mask(self, x, N, use_cuda=True):
        x = x.data
        mask = np.zeros((x.size(0), N))
        for i in range(x.size(0)):
            mask[(i), :x[i]] = 1
        return to_cuda(torch.Tensor(mask), use_cuda)

    def create_mask_3D(self, x, N, use_cuda=True):
        x = x.data
        mask = np.zeros((x.size(0), x.size(1), N))
        for i in range(x.size(0)):
            for j in range(x.size(1)):
                mask[(i), (j), :x[i, j]] = 1
        return to_cuda(torch.Tensor(mask), use_cuda)


VERY_SMALL_NUMBER = 1e-10


def create_mask(x, N, use_cuda=True):
    x = x.data
    mask = np.zeros((x.size(0), N))
    for i in range(x.size(0)):
        mask[(i), :x[i]] = 1
    return to_cuda(torch.Tensor(mask), use_cuda)


class AnsEncoder(nn.Module):
    """Answer Encoder"""

    def __init__(self, o_embed_size, hidden_size, num_ent_types, num_relations, vocab_size=None, vocab_embed_size=None, shared_embed=None, word_emb_dropout=None, ans_enc_dropout=None, use_cuda=True):
        super(AnsEncoder, self).__init__()
        self.use_cuda = use_cuda
        self.ans_enc_dropout = ans_enc_dropout
        self.hidden_size = hidden_size
        self.ent_type_embed = nn.Embedding(num_ent_types, o_embed_size // 8, padding_idx=0)
        self.relation_embed = nn.Embedding(num_relations, o_embed_size, padding_idx=0)
        self.embed = shared_embed if shared_embed is not None else nn.Embedding(vocab_size, vocab_embed_size, padding_idx=0)
        self.vocab_embed_size = self.embed.weight.data.size(1)
        self.linear_type_bow_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_paths_key = nn.Linear(hidden_size + o_embed_size, hidden_size, bias=False)
        self.linear_ctx_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_type_bow_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear_paths_val = nn.Linear(hidden_size + o_embed_size, hidden_size, bias=False)
        self.linear_ctx_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lstm_enc_type = EncoderRNN(vocab_size, self.vocab_embed_size, hidden_size, dropout=word_emb_dropout, bidirectional=True, shared_embed=shared_embed, rnn_type='lstm', use_cuda=use_cuda)
        self.lstm_enc_path = EncoderRNN(vocab_size, self.vocab_embed_size, hidden_size, dropout=word_emb_dropout, bidirectional=True, shared_embed=shared_embed, rnn_type='lstm', use_cuda=use_cuda)
        self.lstm_enc_ctx = EncoderRNN(vocab_size, self.vocab_embed_size, hidden_size, dropout=word_emb_dropout, bidirectional=True, shared_embed=shared_embed, rnn_type='lstm', use_cuda=use_cuda)

    def forward(self, x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ents, x_ctx_ent_len, x_ctx_ent_num):
        ans_type_bow, ans_types, ans_path_bow, ans_paths, ans_ctx_ent = self.enc_ans_features(x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ents, x_ctx_ent_len, x_ctx_ent_num)
        ans_val, ans_key = self.enc_comp_kv(ans_type_bow, ans_types, ans_path_bow, ans_paths, ans_ctx_ent)
        return ans_val, ans_key

    def enc_comp_kv(self, ans_type_bow, ans_types, ans_path_bow, ans_paths, ans_ctx_ent):
        ans_type_bow_val = self.linear_type_bow_val(ans_type_bow)
        ans_paths_val = self.linear_paths_val(torch.cat([ans_path_bow, ans_paths], -1))
        ans_ctx_val = self.linear_ctx_val(ans_ctx_ent)
        ans_type_bow_key = self.linear_type_bow_key(ans_type_bow)
        ans_paths_key = self.linear_paths_key(torch.cat([ans_path_bow, ans_paths], -1))
        ans_ctx_key = self.linear_ctx_key(ans_ctx_ent)
        ans_comp_val = [ans_type_bow_val, ans_paths_val, ans_ctx_val]
        ans_comp_key = [ans_type_bow_key, ans_paths_key, ans_ctx_key]
        return ans_comp_val, ans_comp_key

    def enc_ans_features(self, x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ents, x_ctx_ent_len, x_ctx_ent_num):
        """
        x_types: answer type
        x_paths: answer path, i.e., bow of relation
        x_ctx_ents: answer context, i.e., bow of entity words, (batch_size, num_cands, num_ctx, L)
        """
        ans_type_bow = self.lstm_enc_type(x_type_bow.view(-1, x_type_bow.size(-1)), x_type_bow_len.view(-1))[1].view(x_type_bow.size(0), x_type_bow.size(1), -1)
        ans_path_bow = self.lstm_enc_path(x_path_bow.view(-1, x_path_bow.size(-1)), x_path_bow_len.view(-1))[1].view(x_path_bow.size(0), x_path_bow.size(1), -1)
        ans_paths = torch.mean(self.relation_embed(x_paths.view(-1, x_paths.size(-1))), 1).view(x_paths.size(0), x_paths.size(1), -1)
        ctx_num_mask = create_mask(x_ctx_ent_num.view(-1), x_ctx_ents.size(2), self.use_cuda).view(x_ctx_ent_num.shape + (-1,))
        ans_ctx_ent = self.lstm_enc_ctx(x_ctx_ents.view(-1, x_ctx_ents.size(-1)), x_ctx_ent_len.view(-1))[1].view(x_ctx_ents.size(0), x_ctx_ents.size(1), x_ctx_ents.size(2), -1)
        ans_ctx_ent = ctx_num_mask.unsqueeze(-1) * ans_ctx_ent
        ans_ctx_ent = torch.sum(ans_ctx_ent, dim=2) / torch.clamp(x_ctx_ent_num.float().unsqueeze(-1), min=VERY_SMALL_NUMBER)
        if self.ans_enc_dropout:
            ans_type_bow = F.dropout(ans_type_bow, p=self.ans_enc_dropout, training=self.training)
            ans_path_bow = F.dropout(ans_path_bow, p=self.ans_enc_dropout, training=self.training)
            ans_paths = F.dropout(ans_paths, p=self.ans_enc_dropout, training=self.training)
            ans_ctx_ent = F.dropout(ans_ctx_ent, p=self.ans_enc_dropout, training=self.training)
        return ans_type_bow, None, ans_path_bow, ans_paths, ans_ctx_ent


class RomHop(nn.Module):

    def __init__(self, query_embed_size, in_memory_embed_size, hidden_size, atten_type='add'):
        super(RomHop, self).__init__()
        self.hidden_size = hidden_size
        self.gru_linear_z = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gru_linear_r = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gru_linear_t = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gru_atten = Attention(hidden_size, query_embed_size, in_memory_embed_size, atten_type=atten_type)

    def forward(self, query_embed, in_memory_embed, out_memory_embed, query_att, atten_mask=None, ctx_mask=None, query_mask=None):
        output = self.update_coatt_cat_maxpool(query_embed, in_memory_embed, out_memory_embed, query_att, atten_mask=atten_mask, ctx_mask=ctx_mask, query_mask=query_mask)
        return output

    def gru_step(self, h_state, in_memory_embed, out_memory_embed, atten_mask=None):
        attention = self.gru_atten(h_state, in_memory_embed, atten_mask=atten_mask)
        probs = torch.softmax(attention, dim=-1)
        memory_output = torch.bmm(probs.unsqueeze(1), out_memory_embed).squeeze(1)
        z = torch.sigmoid(self.gru_linear_z(torch.cat([h_state, memory_output], -1)))
        r = torch.sigmoid(self.gru_linear_r(torch.cat([h_state, memory_output], -1)))
        t = torch.tanh(self.gru_linear_t(torch.cat([r * h_state, memory_output], -1)))
        output = (1 - z) * h_state + z * t
        return output

    def update_coatt_cat_maxpool(self, query_embed, in_memory_embed, out_memory_embed, query_att, atten_mask=None, ctx_mask=None, query_mask=None):
        attention = torch.bmm(query_embed, in_memory_embed.view(in_memory_embed.size(0), -1, in_memory_embed.size(-1)).transpose(1, 2)).view(query_embed.size(0), query_embed.size(1), in_memory_embed.size(1), -1)
        if ctx_mask is not None:
            attention[:, :, :, (-1)] = ctx_mask.unsqueeze(1) * attention[:, :, :, (-1)].clone() - (1 - ctx_mask).unsqueeze(1) * INF
        if atten_mask is not None:
            attention = atten_mask.unsqueeze(1).unsqueeze(-1) * attention - (1 - atten_mask).unsqueeze(1).unsqueeze(-1) * INF
        if query_mask is not None:
            attention = query_mask.unsqueeze(2).unsqueeze(-1) * attention - (1 - query_mask).unsqueeze(2).unsqueeze(-1) * INF
        kb_feature_att = F.max_pool1d(attention.view(attention.size(0), attention.size(1), -1).transpose(1, 2), kernel_size=attention.size(1)).squeeze(-1).view(attention.size(0), -1, attention.size(-1))
        kb_feature_att = torch.softmax(kb_feature_att, dim=-1).view(-1, kb_feature_att.size(-1)).unsqueeze(1)
        in_memory_embed = torch.bmm(kb_feature_att, in_memory_embed.view(-1, in_memory_embed.size(2), in_memory_embed.size(-1))).squeeze(1).view(in_memory_embed.size(0), in_memory_embed.size(1), -1)
        out_memory_embed = out_memory_embed.sum(2)
        attention = F.max_pool1d(attention.view(attention.size(0), -1, attention.size(-1)), kernel_size=attention.size(-1)).squeeze(-1).view(attention.size(0), attention.size(1), attention.size(2))
        probs = torch.softmax(attention, dim=-1)
        new_query_embed = query_embed + query_att.unsqueeze(2) * torch.bmm(probs, out_memory_embed)
        probs2 = torch.softmax(attention, dim=1)
        kb_att = torch.bmm(query_att.unsqueeze(1), probs).squeeze(1)
        in_memory_embed = in_memory_embed + kb_att.unsqueeze(2) * torch.bmm(probs2.transpose(1, 2), new_query_embed)
        return new_query_embed, in_memory_embed, out_memory_embed


class BAMnet(nn.Module):

    def __init__(self, vocab_size, vocab_embed_size, o_embed_size, hidden_size, num_ent_types, num_relations, num_query_words, word_emb_dropout=None, que_enc_dropout=None, ans_enc_dropout=None, pre_w2v=None, num_hops=1, att='add', use_cuda=True):
        super(BAMnet, self).__init__()
        self.use_cuda = use_cuda
        self.word_emb_dropout = word_emb_dropout
        self.que_enc_dropout = que_enc_dropout
        self.ans_enc_dropout = ans_enc_dropout
        self.num_hops = num_hops
        self.hidden_size = hidden_size
        self.que_enc = SeqEncoder(vocab_size, vocab_embed_size, hidden_size, seq_enc_type='lstm', word_emb_dropout=word_emb_dropout, bidirectional=True, init_word_embed=pre_w2v, use_cuda=use_cuda).que_enc
        self.ans_enc = AnsEncoder(o_embed_size, hidden_size, num_ent_types, num_relations, vocab_size=vocab_size, vocab_embed_size=vocab_embed_size, shared_embed=self.que_enc.embed, word_emb_dropout=word_emb_dropout, ans_enc_dropout=ans_enc_dropout, use_cuda=use_cuda)
        self.qw_embed = nn.Embedding(num_query_words, o_embed_size // 8, padding_idx=0)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.init_atten = Attention(hidden_size, hidden_size, hidden_size, atten_type=att)
        self.self_atten = SelfAttention_CoAtt(hidden_size)
        None
        self.memory_hop = RomHop(hidden_size, hidden_size, hidden_size, atten_type=att)
        None

    def kb_aware_query_enc(self, memories, queries, query_lengths, ans_mask, ctx_mask=None):
        Q_r = self.que_enc(queries, query_lengths)[0]
        if self.que_enc_dropout:
            Q_r = F.dropout(Q_r, p=self.que_enc_dropout, training=self.training)
        query_mask = create_mask(query_lengths, Q_r.size(1), self.use_cuda)
        q_r_init = self.self_atten(Q_r, query_lengths, query_mask)
        _, _, _, x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ent, x_ctx_ent_len, x_ctx_ent_num, _, _, _, _ = memories
        ans_comp_val, ans_comp_key = self.ans_enc(x_type_bow, x_types, x_type_bow_len, x_path_bow, x_paths, x_path_bow_len, x_ctx_ent, x_ctx_ent_len, x_ctx_ent_num)
        if self.ans_enc_dropout:
            for _ in range(len(ans_comp_key)):
                ans_comp_key[_] = F.dropout(ans_comp_key[_], p=self.ans_enc_dropout, training=self.training)
        ans_comp_atts = [self.init_atten(q_r_init, each, atten_mask=ans_mask) for each in ans_comp_key]
        if ctx_mask is not None:
            ans_comp_atts[-1] = ctx_mask * ans_comp_atts[-1] - (1 - ctx_mask) * INF
        ans_comp_probs = [torch.softmax(each, dim=-1) for each in ans_comp_atts]
        memory_summary = []
        for i, probs in enumerate(ans_comp_probs):
            memory_summary.append(torch.bmm(probs.unsqueeze(1), ans_comp_val[i]))
        memory_summary = torch.cat(memory_summary, 1)
        CoAtt = torch.bmm(Q_r, memory_summary.transpose(1, 2))
        CoAtt = query_mask.unsqueeze(-1) * CoAtt - (1 - query_mask).unsqueeze(-1) * INF
        if ctx_mask is not None:
            ctx_mask_global = (ctx_mask.sum(-1, keepdim=True) > 0).float()
            CoAtt[:, :, (-1)] = ctx_mask_global * CoAtt[:, :, (-1)].clone() - (1 - ctx_mask_global) * INF
        q_att = F.max_pool1d(CoAtt, kernel_size=CoAtt.size(-1)).squeeze(-1)
        q_att = torch.softmax(q_att, dim=-1)
        return (ans_comp_val, ans_comp_key), (q_att, Q_r), query_mask

    def forward(self, memories, queries, query_lengths, query_words, ctx_mask=None):
        ctx_mask = None
        mem_hop_scores = []
        ans_mask = create_mask(memories[0], memories[2].size(1), self.use_cuda)
        self.qw_vec = torch.mean(self.qw_embed(query_words), 1)
        x_types = memories[4]
        ans_types = torch.mean(self.ans_enc.ent_type_embed(x_types.view(-1, x_types.size(-1))), 1).view(x_types.size(0), x_types.size(1), -1)
        qw_anstype_loss = torch.bmm(ans_types, self.qw_vec.unsqueeze(2)).squeeze(2)
        if ans_mask is not None:
            qw_anstype_loss = ans_mask * qw_anstype_loss - (1 - ans_mask) * INF
        mem_hop_scores.append(qw_anstype_loss)
        (ans_val, ans_key), (q_att, Q_r), query_mask = self.kb_aware_query_enc(memories, queries, query_lengths, ans_mask, ctx_mask=ctx_mask)
        ans_val = torch.cat([each.unsqueeze(2) for each in ans_val], 2)
        ans_key = torch.cat([each.unsqueeze(2) for each in ans_key], 2)
        q_r = torch.bmm(q_att.unsqueeze(1), Q_r).squeeze(1)
        mid_score = self.scoring(ans_key.sum(2), q_r, mask=ans_mask)
        mem_hop_scores.append(mid_score)
        Q_r, ans_key, ans_val = self.memory_hop(Q_r, ans_key, ans_val, q_att, atten_mask=ans_mask, ctx_mask=ctx_mask, query_mask=query_mask)
        q_r = torch.bmm(q_att.unsqueeze(1), Q_r).squeeze(1)
        mid_score = self.scoring(ans_key, q_r, mask=ans_mask)
        mem_hop_scores.append(mid_score)
        for _ in range(self.num_hops):
            q_r_tmp = self.memory_hop.gru_step(q_r, ans_key, ans_val, atten_mask=ans_mask)
            q_r = self.batchnorm(q_r + q_r_tmp)
            mid_score = self.scoring(ans_key, q_r, mask=ans_mask)
            mem_hop_scores.append(mid_score)
        return mem_hop_scores

    def premature_score(self, memories, queries, query_lengths, ctx_mask=None):
        ctx_mask = None
        ans_mask = create_mask(memories[0], memories[2].size(1), self.use_cuda)
        (ans_val, ans_key), (q_att, Q_r), query_mask = self.kb_aware_query_enc(memories, queries, query_lengths, ans_mask, ctx_mask=ctx_mask)
        ans_key = torch.cat([each.unsqueeze(2) for each in ans_key], 2)
        mem_hop_scores = []
        q_r = torch.bmm(q_att.unsqueeze(1), Q_r).squeeze(1)
        score = self.scoring(ans_key.sum(2), q_r, mask=ans_mask)
        return score

    def scoring(self, ans_r, q_r, mask=None):
        score = torch.bmm(ans_r, q_r.unsqueeze(2)).squeeze(2)
        if mask is not None:
            score = mask * score - (1 - mask) * INF
        return score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EncoderCNN,
     lambda: ([], {'vocab_size': 4, 'embed_size': 4, 'hidden_size': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
    (EntRomHop,
     lambda: ([], {'query_embed_size': 4, 'in_memory_embed_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 16, 4])], {}),
     False),
    (GRUStep,
     lambda: ([], {'hidden_size': 4, 'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttention_CoAtt,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64), torch.rand([4, 4])], {}),
     False),
]

class Test_hugochan_BAMnet(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

