import sys
_module = sys.modules[__name__]
del sys
attention = _module
build_emb = _module
data_generator = _module
model = _module
modules = _module
script = _module
train = _module
util = _module

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


import torch


import torch.nn.functional as F


import torch.nn as nn


from torch.autograd import Variable


import copy


import math


import numpy as np


import random


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).
            transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self
            .dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k
            )
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.
            log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SimpleEncoder(nn.Module):
    """
    takes (batch_size, seq_len, embed_dim) as inputs
    calculate MASK, POSITION_ENCODING 
    """

    def __init__(self, embed_dim, head=4, layer=1, dropout=0.1):
        super(SimpleEncoder, self).__init__()
        d_ff = 2 * embed_dim
        self.position = PositionalEncoding(embed_dim, dropout)
        attn = MultiHeadedAttention(head, embed_dim)
        ff = PositionwiseFeedForward(embed_dim, d_ff)
        self.encoder = Encoder(EncoderLayer(embed_dim, attn, ff, dropout),
            layer)

    def forward(self, x, mask):
        mask = mask.unsqueeze(-2)
        x = self.position(x)
        x = self.encoder(x, mask)
        return x


def load_dict(filename):
    word2id = dict()
    with open(filename) as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id


def l_relu(x, n_slope=0.01):
    return F.leaky_relu(x, n_slope)


class KAReader(nn.Module):
    """docstring for ClassName"""

    def __init__(self, args):
        super(KAReader, self).__init__()
        self.entity2id = load_dict(args['data_folder'] + args['entity2id'])
        self.word2id = load_dict(args['data_folder'] + args['word2id'])
        self.relation2id = load_dict(args['data_folder'] + args['relation2id'])
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        self.num_word = len(self.word2id)
        self.num_layer = args['num_layer']
        self.use_doc = args['use_doc']
        self.word_drop = args['word_drop']
        self.hidden_drop = args['hidden_drop']
        self.label_smooth = args['label_smooth']
        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file'):
                setattr(self, k, args['data_folder'] + v)
        self.entity_emb = nn.Embedding(self.num_entity + 1, self.entity_dim,
            padding_idx=self.num_entity)
        self.entity_emb.weight.data.copy_(torch.from_numpy(np.pad(np.load(
            self.entity_emb_file), ((0, 1), (0, 0)), 'constant')))
        self.entity_emb.weight.requires_grad = False
        self.entity_linear = nn.Linear(self.entity_dim, self.entity_dim)
        self.word_emb = nn.Embedding(self.num_word, self.word_dim,
            padding_idx=1)
        self.word_emb.weight.data.copy_(torch.from_numpy(np.load(self.
            word_emb_file)))
        self.word_emb.weight.requires_grad = False
        self.word_emb_match = SeqAttnMatch(self.word_dim)
        self.hidden_dim = self.entity_dim
        self.question_encoder = Packed(nn.LSTM(self.word_dim, self.
            hidden_dim // 2, batch_first=True, bidirectional=True))
        self.self_att_r = AttnEncoder(self.hidden_dim)
        self.self_att_q = AttnEncoder(self.hidden_dim)
        self.combine_q_rel = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ent_info_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.input_proj = nn.Linear(2 * self.word_dim + 1, self.hidden_dim)
        self.doc_encoder = Packed(nn.LSTM(self.hidden_dim, self.hidden_dim //
            2, batch_first=True, bidirectional=True))
        self.doc_to_ent = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ent_info_gate = ConditionGate(self.hidden_dim)
        self.ent_info_gate_out = ConditionGate(self.hidden_dim)
        self.kg_prop = nn.Linear(self.hidden_dim + self.entity_dim, self.
            entity_dim)
        self.kg_gate = nn.Linear(self.hidden_dim + self.entity_dim, self.
            entity_dim)
        self.self_prop = nn.Linear(self.entity_dim, self.entity_dim)
        self.combine_q = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.reader_gate = nn.Linear(2 * self.hidden_dim, self.hidden_dim)
        self.query_update = QueryReform(self.hidden_dim)
        self.attn_match = nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2)
        self.attn_match_q = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.loss = nn.BCEWithLogitsLoss()
        self.word_drop = nn.Dropout(self.word_drop)
        self.hidden_drop = nn.Dropout(self.hidden_drop)

    def forward(self, feed):
        question = feed['questions']
        q_mask = (question != 1).float()
        q_len = q_mask.sum(-1)
        q_word_emb = self.word_drop(self.word_emb(question))
        q_emb, _ = self.question_encoder(q_word_emb, q_len, max_length=
            question.size(1))
        q_emb = self.hidden_drop(q_emb)
        B, max_q_len = question.size(0), question.size(1)
        ent_emb_ = self.entity_emb(feed['candidate_entities'])
        ent_emb = l_relu(self.entity_linear(ent_emb_))
        ent_mask = (feed['candidate_entities'] != self.num_entity).float()
        max_num_neighbors = feed['entity_link_ents'].size(2)
        max_num_candidates = feed['candidate_entities'].size(1)
        neighbor_mask = (feed['entity_link_ents'] != self.num_entity).float()
        rel_word_ids = feed['rel_word_ids']
        rel_word_mask = (rel_word_ids != 1).float()
        rel_word_lens = rel_word_mask.sum(-1)
        rel_word_lens[rel_word_lens == 0] = 1
        rel_encoded, _ = self.question_encoder(self.word_drop(self.word_emb
            (rel_word_ids)), rel_word_lens, max_length=rel_word_ids.size(1))
        rel_encoded = self.hidden_drop(rel_encoded)
        rel_encoded = self.self_att_r(rel_encoded, rel_word_mask)
        neighbor_rel_ids = feed['entity_link_rels'].long().view(-1)
        neighbor_rel_emb = torch.index_select(rel_encoded, dim=0, index=
            neighbor_rel_ids).view(B * max_num_candidates,
            max_num_neighbors, self.hidden_dim)
        neighbor_ent_local_index = feed['entity_link_ents'].long()
        neighbor_ent_local_index = neighbor_ent_local_index.view(B, -1)
        neighbor_ent_local_mask = (neighbor_ent_local_index != -1).long()
        fix_index = torch.arange(B).long() * max_num_candidates
        fix_index = fix_index.to(torch.device('cuda'))
        neighbor_ent_local_index = neighbor_ent_local_index + fix_index.view(
            -1, 1)
        neighbor_ent_local_index = (neighbor_ent_local_index + 1
            ) * neighbor_ent_local_mask
        neighbor_ent_local_index = neighbor_ent_local_index.view(-1)
        ent_seed_info = feed['query_entities'].float()
        ent_is_seed = torch.cat([torch.zeros(1).to(torch.device('cuda')),
            ent_seed_info.view(-1)], dim=0)
        ent_seed_indicator = torch.index_select(ent_is_seed, dim=0, index=
            neighbor_ent_local_index).view(B * max_num_candidates,
            max_num_neighbors)
        q_emb_expand = q_emb.unsqueeze(1).expand(B, max_num_candidates,
            max_q_len, -1).contiguous()
        q_emb_expand = q_emb_expand.view(B * max_num_candidates, max_q_len, -1)
        q_mask_expand = q_mask.unsqueeze(1).expand(B, max_num_candidates, -1
            ).contiguous()
        q_mask_expand = q_mask_expand.view(B * max_num_candidates, -1)
        q_n_affinity = torch.bmm(q_emb_expand, neighbor_rel_emb.transpose(1, 2)
            )
        q_n_affinity_mask_q = q_n_affinity - (1 - q_mask_expand.unsqueeze(2)
            ) * 1e+20
        q_n_affinity_mask_n = q_n_affinity - (1 - neighbor_mask.view(B *
            max_num_candidates, 1, max_num_neighbors))
        normalize_over_q = F.softmax(q_n_affinity_mask_q, dim=1)
        normalize_over_n = F.softmax(q_n_affinity_mask_n, dim=2)
        retrieve_q = torch.bmm(normalize_over_q.transpose(1, 2), q_emb_expand)
        q_rel_simi = torch.sum(neighbor_rel_emb * retrieve_q, dim=2)
        init_q_emb = self.self_att_r(q_emb, q_mask)
        retrieve_r = torch.bmm(normalize_over_n, neighbor_rel_emb)
        q_and_rel = torch.cat([q_emb_expand, retrieve_r], dim=2)
        rel_aware_q = self.combine_q_rel(q_and_rel).tanh().view(B,
            max_num_candidates, -1, self.hidden_dim)
        q_node_emb = rel_aware_q.max(2)[0]
        ent_emb = l_relu(self.combine_q(torch.cat([ent_emb, q_node_emb],
            dim=2)))
        ent_emb_for_lookup = ent_emb.view(-1, self.entity_dim)
        ent_emb_for_lookup = torch.cat([torch.zeros(1, self.entity_dim).to(
            torch.device('cuda')), ent_emb_for_lookup], dim=0)
        neighbor_ent_emb = torch.index_select(ent_emb_for_lookup, dim=0,
            index=neighbor_ent_local_index)
        neighbor_ent_emb = neighbor_ent_emb.view(B * max_num_candidates,
            max_num_neighbors, -1)
        neighbor_vec = torch.cat([neighbor_rel_emb, neighbor_ent_emb], dim=-1
            ).view(B * max_num_candidates, max_num_neighbors, -1)
        neighbor_scores = q_rel_simi * ent_seed_indicator
        neighbor_scores = neighbor_scores - (1 - neighbor_mask.view(B *
            max_num_candidates, max_num_neighbors)) * 100000000.0
        attn_score = F.softmax(neighbor_scores, dim=1)
        aggregate = self.kg_prop(neighbor_vec) * attn_score.unsqueeze(2)
        aggregate = l_relu(aggregate.sum(1)).view(B, max_num_candidates, -1)
        self_prop_ = l_relu(self.self_prop(ent_emb))
        gate_value = self.kg_gate(torch.cat([aggregate, ent_emb], dim=-1)
            ).sigmoid()
        ent_emb = gate_value * self_prop_ + (1 - gate_value) * aggregate
        if self.use_doc:
            q_for_text = self.query_update(init_q_emb, ent_emb,
                ent_seed_info, ent_mask)
            q_node_emb = torch.cat([q_node_emb, q_for_text.unsqueeze(1).
                expand_as(q_node_emb).contiguous()], dim=-1)
            ent_linked_doc_spans = feed['ent_link_doc_spans']
            doc = feed['documents']
            max_num_doc = doc.size(1)
            max_d_len = doc.size(2)
            doc_mask = (doc != 1).float()
            doc_len = doc_mask.sum(-1)
            doc_len += (doc_len == 0).float()
            doc_len = doc_len.view(-1)
            d_word_emb = self.word_drop(self.word_emb(doc.view(-1, doc.size
                (-1))))
            q_word_emb = q_word_emb.unsqueeze(1).expand(B, max_num_doc,
                max_q_len, self.word_dim).contiguous()
            q_word_emb = q_word_emb.view(B * max_num_doc, max_q_len, -1)
            q_mask_ = (question == 1).unsqueeze(1).expand(B, max_num_doc,
                max_q_len).contiguous()
            q_mask_ = q_mask_.view(B * max_num_doc, -1)
            q_weighted_emb = self.word_emb_match(d_word_emb, q_word_emb,
                q_mask_)
            doc_em = feed['documents_em'].float().view(B * max_num_doc,
                max_d_len, 1)
            doc_input = torch.cat([d_word_emb, q_weighted_emb, doc_em], dim=-1)
            doc_input = self.input_proj(doc_input).tanh()
            word_entity_id = ent_linked_doc_spans.view(B,
                max_num_candidates, -1).transpose(1, 2)
            word_ent_info_mask = (word_entity_id.sum(-1, keepdim=True) != 0
                ).float()
            word_ent_info = torch.bmm(word_entity_id.float(), ent_emb)
            word_ent_info = self.ent_info_proj(word_ent_info).tanh()
            doc_input = self.ent_info_gate(q_for_text.unsqueeze(1),
                word_ent_info, doc_input.view(B, max_num_doc * max_d_len, -
                1), word_ent_info_mask)
            d_emb, _ = self.doc_encoder(doc_input.view(B * max_num_doc,
                max_d_len, -1), doc_len, max_length=doc.size(2))
            d_emb = self.hidden_drop(d_emb)
            d_emb = self.ent_info_gate_out(q_for_text.unsqueeze(1),
                word_ent_info, d_emb.view(B, max_num_doc * max_d_len, -1),
                word_ent_info_mask).view(B * max_num_doc, max_d_len, -1)
            q_for_text = q_for_text.unsqueeze(1).expand(B, max_num_doc,
                self.hidden_dim).contiguous()
            q_for_text = q_for_text.view(B * max_num_doc, -1)
            d_emb = d_emb.view(B * max_num_doc, max_d_len, -1)
            q_over_d = torch.bmm(q_for_text.unsqueeze(1), d_emb.transpose(1, 2)
                ).squeeze(1)
            q_over_d = F.softmax(q_over_d - (1 - doc_mask.view(B *
                max_num_doc, max_d_len)) * 100000000.0, dim=-1)
            q_retrieve_d = torch.bmm(q_over_d.unsqueeze(1), d_emb).view(B,
                max_num_doc, -1)
            ent_linked_doc = (ent_linked_doc_spans.sum(-1) != 0).float()
            ent_emb_from_doc = torch.bmm(ent_linked_doc, q_retrieve_d)
            ent_emb_from_span = torch.bmm(feed['ent_link_doc_norm_spans'].
                float().view(B, max_num_candidates, -1), d_emb.view(B, 
                max_num_doc * max_d_len, -1))
            ent_emb_from_span = F.dropout(ent_emb_from_span, 0.2, self.training
                )
        if self.use_doc:
            ent_emb = l_relu(self.attn_match(torch.cat([ent_emb,
                ent_emb_from_doc, ent_emb_from_span], dim=-1)))
        ent_scores = (q_node_emb * ent_emb).sum(2)
        answers = feed['answers'].float()
        if self.label_smooth:
            answers = (1.0 - self.label_smooth
                ) * answers + self.label_smooth / answers.size(1)
        loss = self.loss(ent_scores, feed['answers'].float())
        pred_dist = (ent_scores - (1 - ent_mask) * 100000000.0).sigmoid(
            ) * ent_mask
        pred = torch.max(ent_scores, dim=1)[1]
        return loss, pred, pred_dist


class Packed(nn.Module):

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    @property
    def batch_first(self):
        return self.rnn.batch_first

    def forward(self, inputs, lengths, hidden=None, max_length=None):
        lens, indices = torch.sort(lengths, 0, True)
        inputs = inputs[indices] if self.batch_first else inputs[:, (indices)]
        outputs, (h, c) = self.rnn(nn.utils.rnn.pack_padded_sequence(inputs,
            lens.tolist(), batch_first=self.batch_first), hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=
            self.batch_first, total_length=max_length)
        _, _indices = torch.sort(indices, 0)
        outputs = outputs[_indices] if self.batch_first else outputs[:, (
            _indices)]
        h, c = h[:, (_indices), :], c[:, (_indices), :]
        return outputs, (h, c)


class ConditionGate(nn.Module):
    """docstring for ConditionGate"""

    def __init__(self, h_dim):
        super(ConditionGate, self).__init__()
        self.gate = nn.Linear(2 * h_dim, h_dim, bias=False)

    def forward(self, q, x, y, gate_mask):
        q_x_sim = x * q
        q_y_sim = y * q
        gate_val = self.gate(torch.cat([q_x_sim, q_y_sim], dim=-1)).sigmoid()
        gate_val = gate_val * gate_mask
        return gate_val * x + (1 - gate_val) * y


class Fusion(nn.Module):
    """docstring for Fusion"""

    def __init__(self, d_hid):
        super(Fusion, self).__init__()
        self.r = nn.Linear(d_hid * 4, d_hid, bias=False)
        self.g = nn.Linear(d_hid * 4, d_hid, bias=False)

    def forward(self, x, y):
        r_ = self.r(torch.cat([x, y, x - y, x * y], dim=-1)).tanh()
        g_ = torch.sigmoid(self.g(torch.cat([x, y, x - y, x * y], dim=-1)))
        return g_ * r_ + (1 - g_) * x


class AttnEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self, d_hid):
        super(AttnEncoder, self).__init__()
        self.attn_linear = nn.Linear(d_hid, 1, bias=False)

    def forward(self, x, x_mask):
        """
        x: (B, len, d_hid)
        x_mask: (B, len)
        return: (B, d_hid)
        """
        x_attn = self.attn_linear(x)
        x_attn = x_attn - (1 - x_mask.unsqueeze(2)) * 100000000.0
        x_attn = F.softmax(x_attn, dim=1)
        return (x * x_attn).sum(1)


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize:
            if self.training:
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
            y_mask: batch * len2 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * hdim
        """
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), dim=-1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))
        matched_seq = alpha.bmm(y)
        return matched_seq


class QueryReform(nn.Module):
    """docstring for QueryReform"""

    def __init__(self, h_dim):
        super(QueryReform, self).__init__()
        self.fusion = Fusion(h_dim)
        self.q_ent_attn = nn.Linear(h_dim, h_dim)

    def forward(self, q_node, ent_emb, seed_info, ent_mask):
        """
        q: (B,q_len,h_dim)
        q_mask: (B,q_len)
        q_ent_span: (B,q_len)
        ent_emb: (B,C,h_dim)
        seed_info: (B, C)
        ent_mask: (B, C)
        """
        q_ent_attn = (self.q_ent_attn(q_node).unsqueeze(1) * ent_emb).sum(2,
            keepdim=True)
        q_ent_attn = F.softmax(q_ent_attn - (1 - ent_mask.unsqueeze(2)) * 
            100000000.0, dim=1)
        seed_retrieve = torch.bmm(seed_info.unsqueeze(1), ent_emb).squeeze(1)
        return self.fusion(q_node, seed_retrieve)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_xwhan_Knowledge_Aware_Reader(_paritybench_base):
    pass
    def test_000(self):
        self._check(AttnEncoder(*[], **{'d_hid': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ConditionGate(*[], **{'h_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Fusion(*[], **{'d_hid': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(LayerNorm(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(MultiHeadedAttention(*[], **{'h': 4, 'd_model': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(PositionalEncoding(*[], **{'d_model': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(PositionwiseFeedForward(*[], **{'d_model': 4, 'd_ff': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(QueryReform(*[], **{'h_dim': 4}), [torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(SublayerConnection(*[], **{'size': 4, 'dropout': 0.5}), [torch.rand([4, 4, 4, 4]), ReLU()], {})

