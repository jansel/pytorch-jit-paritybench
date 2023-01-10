import sys
_module = sys.modules[__name__]
del sys
code_bert = _module
lsr_bert = _module
Config = _module
ConfigBert = _module
config = _module
evaluation = _module
gen_data = _module
gen_data_bert = _module
models = _module
attention = _module
bert = _module
encoder = _module
gcn = _module
lockedropout = _module
lsr = _module
reasoner = _module
test = _module
train = _module
utils = _module
constant = _module
helper = _module
tokenizer = _module
torch_utils = _module

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


import torch


from torch import nn


from torch.nn.utils.rnn import pad_sequence


import torch.nn as nn


import torch.optim as optim


import numpy as np


import time


import sklearn.metrics


import random


from collections import defaultdict


import torch.nn.functional as F


import copy


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.autograd import Variable


import math


from torch import optim


from torch.optim import Optimizer


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers, dropout, self_loop=False):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = dropout
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * i, self.head_dim))
        self.weight_list = self.weight_list
        self.linear_output = self.linear_output
        self.self_loop = self_loop

    def forward(self, adj, gcn_inputs):
        denom = adj.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []
        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            if self.self_loop:
                AxW = AxW + self.weight_list[l](outputs)
            else:
                AxW = AxW
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))
        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs
        out = self.linear_output(gcn_outputs)
        return out


class StructInduction(nn.Module):

    def __init__(self, sem_dim_size, sent_hiddent_size, bidirectional):
        super(StructInduction, self).__init__()
        self.bidirectional = bidirectional
        self.sem_dim_size = sem_dim_size
        self.str_dim_size = sent_hiddent_size - self.sem_dim_size
        self.tp_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tp_linear.weight)
        nn.init.constant_(self.tp_linear.bias, 0)
        self.tc_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tc_linear.weight)
        nn.init.constant_(self.tc_linear.bias, 0)
        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fi_linear.weight)
        self.bilinear = nn.Bilinear(self.str_dim_size, self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.bilinear.weight)
        self.exparam = nn.Parameter(torch.Tensor(1, 1, self.sem_dim_size))
        torch.nn.init.xavier_uniform_(self.exparam)
        self.fzlinear = nn.Linear(3 * self.sem_dim_size, 2 * self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.fzlinear.weight)
        nn.init.constant_(self.fzlinear.bias, 0)

    def forward(self, input):
        batch_size, token_size, dim_size = input.size()
        """STEP1: Calculating Attention Matrix"""
        if self.bidirectional:
            input = input.view(batch_size, token_size, 2, dim_size // 2)
            sem_v = torch.cat((input[:, :, 0, :self.sem_dim_size // 2], input[:, :, 1, :self.sem_dim_size // 2]), 2)
            str_v = torch.cat((input[:, :, 0, self.sem_dim_size // 2:], input[:, :, 1, self.sem_dim_size // 2:]), 2)
        else:
            sem_v = input[:, :, :self.sem_dim_size]
            str_v = input[:, :, self.sem_dim_size:]
        tp = torch.tanh(self.tp_linear(str_v))
        tc = torch.tanh(self.tc_linear(str_v))
        tp = tp.unsqueeze(2).expand(tp.size(0), tp.size(1), tp.size(1), tp.size(2)).contiguous()
        tc = tc.unsqueeze(2).expand(tc.size(0), tc.size(1), tc.size(1), tc.size(2)).contiguous()
        f_ij = self.bilinear(tp, tc).squeeze()
        f_i = torch.exp(self.fi_linear(str_v)).squeeze()
        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1))
        A_ij = torch.exp(f_ij) * mask
        """STEP: Incude Latent Structure"""
        tmp = torch.sum(A_ij, dim=1)
        res = torch.zeros(batch_size, token_size, token_size)
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        L_ij = -A_ij + res
        L_ij_bar = L_ij
        L_ij_bar[:, 0, :] = f_i
        LLinv = torch.inverse(L_ij_bar)
        d0 = f_i * LLinv[:, :, 0]
        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)
        tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
        tmp2 = A_ij * LLinv.transpose(1, 2)
        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)
        temp12 = torch.ones(batch_size, token_size, token_size - 1)
        temp22 = torch.ones(batch_size, token_size - 1, token_size)
        mask1 = torch.cat([temp11, temp12], 2)
        mask2 = torch.cat([temp21, temp22], 1)
        dx = mask1 * tmp1 - mask2 * tmp2
        d = torch.cat([d0.unsqueeze(1), dx], dim=1)
        df = d.transpose(1, 2)
        ssr = torch.cat([self.exparam.repeat(batch_size, 1, 1), sem_v], 1)
        pinp = torch.bmm(df, ssr)
        cinp = torch.bmm(dx, sem_v)
        finp = torch.cat([sem_v, pinp, cinp], dim=2)
        output = F.relu(self.fzlinear(finp))
        return output, df


class DynamicReasoner(nn.Module):

    def __init__(self, hidden_size, gcn_layer, dropout_gcn):
        super(DynamicReasoner, self).__init__()
        self.hidden_size = hidden_size
        self.gcn_layer = gcn_layer
        self.dropout_gcn = dropout_gcn
        self.struc_att = StructInduction(hidden_size // 2, hidden_size, True)
        self.gcn = GraphConvLayer(hidden_size, self.gcn_layer, self.dropout_gcn, self_loop=True)

    def forward(self, input):
        """
        :param input:
        :return:
        """
        """Structure Induction"""
        _, att = self.struc_att(input)
        """Perform reasoning"""
        output = self.gcn(att[:, :, 1:], input)
        return output


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, dropout_embedding, dropout_encoder):
        super(Encoder, self).__init__()
        self.wordEmbeddingDim = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size, self.hidden_size, bidirectional=True)
        self.dropout_embedding = nn.Dropout(p=dropout_embedding)
        self.dropout_encoder = nn.Dropout(p=dropout_encoder)

    def forward(self, seq, lens):
        batch_size = seq.shape[0]
        lens_sorted, lens_argsort = torch.sort(lens, 0, True)
        _, lens_argsort_argsort = torch.sort(lens_argsort, 0)
        seq_ = torch.index_select(seq, 0, lens_argsort)
        seq_embd = self.dropout_embedding(seq_)
        packed = pack_padded_sequence(seq_embd, lens_sorted, batch_first=True)
        self.encoder.flatten_parameters()
        output, h = self.encoder(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output.contiguous()
        output = torch.index_select(output, 0, lens_argsort_argsort)
        h = h.permute(1, 0, 2).contiguous().view(batch_size, 1, -1)
        h = torch.index_select(h, 0, lens_argsort_argsort)
        output = self.dropout_encoder(output)
        h = self.dropout_encoder(h)
        return output, h


class SelfAttention(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / input_size ** 0.5))

    def forward(self, input, memory, mask):
        input_dot = self.input_linear(input)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + cross_dot
        att = att - 1e+30 * (1 - mask[:, None])
        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        return torch.cat([input, output_one], dim=-1)


class LSR(nn.Module):

    def __init__(self, config):
        super(LSR, self).__init__()
        self.config = config
        self.finetune_emb = config.finetune_emb
        self.word_emb = nn.Embedding(config.data_word_vec.shape[0], config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))
        if not self.finetune_emb:
            self.word_emb.weight.requires_grad = False
        self.ner_emb = nn.Embedding(13, config.entity_type_size, padding_idx=0)
        self.coref_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)
        hidden_size = config.rnn_hidden
        input_size = config.data_word_vec.shape[1] + config.coref_size + config.entity_type_size
        self.linear_re = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_sent = nn.Linear(hidden_size * 2, hidden_size)
        self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)
        self.self_att = SelfAttention(hidden_size)
        self.bili = torch.nn.Bilinear(hidden_size + config.dis_size, hidden_size + config.dis_size, hidden_size)
        self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
        self.linear_output = nn.Linear(2 * hidden_size, config.relation_num)
        self.relu = nn.ReLU()
        self.dropout_rate = nn.Dropout(config.dropout_rate)
        self.rnn_sent = Encoder(input_size, hidden_size, config.dropout_emb, config.dropout_rate)
        self.hidden_size = hidden_size
        self.use_struct_att = config.use_struct_att
        if self.use_struct_att == True:
            self.structInduction = StructInduction(hidden_size // 2, hidden_size, True)
        self.dropout_gcn = nn.Dropout(config.dropout_gcn)
        self.reasoner_layer_first = config.reasoner_layer_first
        self.reasoner_layer_second = config.reasoner_layer_second
        self.use_reasoning_block = config.use_reasoning_block
        if self.use_reasoning_block:
            self.reasoner = nn.ModuleList()
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_first, self.dropout_gcn))
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_second, self.dropout_gcn))

    def doc_encoder(self, input_sent, context_seg):
        """
        :param sent: sent emb
        :param context_seg: segmentation mask for sentences in a document
        :return:
        """
        batch_size = context_seg.shape[0]
        docs_emb = []
        docs_len = []
        sents_emb = []
        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = (context_seg[batch_no] == 1).nonzero().squeeze(-1).tolist()
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index:index + 1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index + 1:index + 1])
                        sent_lens.append(index - pre_index)
                pre_index = index
            sents = pad_sequence(sent_list).permute(1, 0, 2)
            sent_lens_t = torch.LongTensor(sent_lens)
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(sents, sent_lens_t)
            doc_emb = None
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim=0)
            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))
        docs_emb = pad_sequence(docs_emb).permute(1, 0, 2)
        sents_emb = pad_sequence(sents_emb).permute(1, 0, 2)
        return docs_emb, sents_emb

    def forward(self, context_idxs, pos, context_ner, h_mapping, t_mapping, relation_mask, dis_h_2_t, dis_t_2_h, context_seg, mention_node_position, entity_position, mention_node_sent_num, all_node_num, entity_num_list, sdp_pos, sdp_num_list):
        """
        :param context_idxs: Token IDs
        :param pos: coref pos IDs
        :param context_ner: NER tag IDs
        :param h_mapping: Head
        :param t_mapping: Tail
        :param relation_mask: There are multiple relations for each instance so we need a mask in a batch
        :param dis_h_2_t: distance for head
        :param dis_t_2_h: distance for tail
        :param context_seg: mask for different sentences in a document
        :param mention_node_position: Mention node position
        :param entity_position: Entity node position
        :param mention_node_sent_num: number of mention nodes in each sentences of a document
        :param all_node_num: the number of nodes  (mention, entity, MDP) in a document
        :param entity_num_list: the number of entity nodes in each document
        :param sdp_pos: MDP node position
        :param sdp_num_list: the number of MDP node in each document
        :return:
        """
        """===========STEP1: Encode the document============="""
        sent_emb = torch.cat([self.word_emb(context_idxs), self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1)
        docs_rep, sents_rep = self.doc_encoder(sent_emb, context_seg)
        max_doc_len = docs_rep.shape[1]
        context_output = self.dropout_rate(torch.relu(self.linear_re(docs_rep)))
        """===========STEP2: Extract all node reps of a document graph============="""
        """extract Mention node representations"""
        mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()
        max_mention_num = max(mention_num_list)
        mentions_rep = torch.bmm(mention_node_position[:, :max_mention_num, :max_doc_len], context_output)
        """extract MDP(meta dependency paths) node representations"""
        sdp_num_list = sdp_num_list.long().tolist()
        max_sdp_num = max(sdp_num_list)
        sdp_rep = torch.bmm(sdp_pos[:, :max_sdp_num, :max_doc_len], context_output)
        """extract Entity node representations"""
        entity_rep = torch.bmm(entity_position[:, :, :max_doc_len], context_output)
        """concatenate all nodes of an instance"""
        gcn_inputs = []
        all_node_num_batch = []
        for batch_no, (m_n, e_n, s_n) in enumerate(zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)):
            m_rep = mentions_rep[batch_no][:m_n]
            e_rep = entity_rep[batch_no][:e_n]
            s_rep = sdp_rep[batch_no][:s_n]
            gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep), dim=0))
            node_num = m_n + e_n + s_n
            all_node_num_batch.append(node_num)
        gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
        output = gcn_inputs
        """===========STEP3: Induce the Latent Structure============="""
        if self.use_reasoning_block:
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)
        elif self.use_struct_att:
            gcn_inputs, _ = self.structInduction(gcn_inputs)
            max_all_node_num = torch.max(all_node_num).item()
            assert gcn_inputs.shape[1] == max_all_node_num
        mention_node_position = mention_node_position.permute(0, 2, 1)
        output = torch.bmm(mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
        context_output = torch.add(context_output, output)
        start_re_output = torch.matmul(h_mapping[:, :, :max_doc_len], context_output)
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output)
        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)
        re_rep = self.dropout_rate(self.relu(self.bili(s_rep, t_rep)))
        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        return self.linear_output(re_rep)


class LockedDropout(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x


class BiAttention(nn.Module):

    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)
        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / input_size ** 0.5))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)
        input = self.dropout(input)
        memory = self.dropout(memory)
        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e+30 * (1 - mask[:, None])
        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)
        return torch.cat([input, output_one, input * output_one, output_two * output_one], dim=-1)


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key))]
        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn


class MultiGraphConvLayer(nn.Module):
    """ A GCN module operated on multihead attention """

    def __init__(self, mem_dim, layers, heads, dropout):
        super(MultiGraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.heads = heads
        self.gcn_drop = dropout
        self.Linear = nn.Linear(self.mem_dim * self.heads, self.mem_dim)
        self.weight_list = nn.ModuleList()
        for i in range(self.heads):
            for j in range(self.layers):
                self.weight_list.append(nn.Linear(self.mem_dim + self.head_dim * j, self.head_dim))
        self.weight_list = self.weight_list
        self.Linear = self.Linear

    def forward(self, adj_list, gcn_inputs):
        multi_head_list = []
        for i in range(self.heads):
            adj = adj_list[:, i]
            denom = adj.sum(2).unsqueeze(2) + 1
            outputs = gcn_inputs
            cache_list = [outputs]
            output_list = []
            for l in range(self.layers):
                index = i * self.layers + l
                Ax = adj.bmm(outputs)
                AxW = self.weight_list[index](Ax)
                AxW = AxW + self.weight_list[index](outputs)
                AxW = AxW / denom
                gAxW = F.relu(AxW)
                cache_list.append(gAxW)
                outputs = torch.cat(cache_list, dim=2)
                output_list.append(self.gcn_drop(gAxW))
            gcn_ouputs = torch.cat(output_list, dim=2)
            gcn_ouputs = gcn_ouputs + gcn_inputs
            multi_head_list.append(gcn_ouputs)
        final_output = torch.cat(multi_head_list, dim=2)
        out = self.Linear(final_output)
        return out


class StructInductionNoSplit(nn.Module):

    def __init__(self, sent_hiddent_size, bidirectional):
        super(StructInductionNoSplit, self).__init__()
        self.bidirectional = bidirectional
        self.str_dim_size = sent_hiddent_size
        self.model_dim = sent_hiddent_size
        self.linear_keys = nn.Linear(self.model_dim, self.model_dim)
        self.linear_query = nn.Linear(self.model_dim, self.model_dim)
        self.linear_root = nn.Linear(self.model_dim, 1)

    def forward(self, input):
        batch_size, token_size, dim_size = input.size()
        key = self.linear_keys(input)
        query = self.linear_query(input)
        f_i = self.linear_root(input).squeeze(-1)
        query = query / math.sqrt(self.model_dim)
        f_ij = torch.matmul(query, key.transpose(1, 2))
        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1))
        A_ij = torch.exp(f_ij) * mask
        tmp = torch.sum(A_ij, dim=1)
        res = torch.zeros(batch_size, token_size, token_size)
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        L_ij = -A_ij + res
        L_ij_bar = L_ij
        L_ij_bar[:, 0, :] = f_i
        LLinv = torch.inverse(L_ij_bar)
        d0 = f_i * LLinv[:, :, 0]
        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)
        tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
        tmp2 = A_ij * LLinv.transpose(1, 2)
        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)
        temp12 = torch.ones(batch_size, token_size, token_size - 1)
        temp22 = torch.ones(batch_size, token_size - 1, token_size)
        mask1 = torch.cat([temp11, temp12], 2)
        mask2 = torch.cat([temp21, temp22], 1)
        dx = mask1 * tmp1 - mask2 * tmp2
        d = torch.cat([d0.unsqueeze(1), dx], dim=1)
        df = d.transpose(1, 2)
        output = None
        return output, df


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LockedDropout,
     lambda: ([], {'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (StructInductionNoSplit,
     lambda: ([], {'sent_hiddent_size': 4, 'bidirectional': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
]

class Test_nanguoshun_LSR(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

