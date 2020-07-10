import sys
_module = sys.modules[__name__]
del sys
CoQA_eval = _module
detail_model = _module
layers = _module
model_CoQA = _module
model_QuAC = _module
utils = _module
general_utils = _module
predict_CoQA = _module
predict_QuAC = _module
preprocess_CoQA = _module
preprocess_QuAC = _module
train_CoQA = _module
train_QuAC = _module

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
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


import torch.nn.functional as F


import math


import random


from torch.autograd import Variable


from torch.nn.parameter import Parameter


from torch.nn.utils.rnn import pad_packed_sequence as unpack


from torch.nn.utils.rnn import pack_padded_sequence as pack


import torch.optim as optim


import numpy as np


import logging


from torch.nn import Parameter


import re


import string


from collections import Counter


class FlowQA(nn.Module):
    """Network for the FlowQA Module."""

    def __init__(self, opt, embedding=None, padding_idx=0):
        super(FlowQA, self).__init__()
        doc_input_size = 0
        que_input_size = 0
        layers.set_my_dropout_prob(opt['my_dropout_p'])
        layers.set_seq_dropout(opt['do_seq_dropout'])
        if opt['use_wemb']:
            self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
            if embedding is not None:
                self.embedding.weight.data = embedding
                if opt['fix_embeddings'] or opt['tune_partial'] == 0:
                    opt['fix_embeddings'] = True
                    opt['tune_partial'] = 0
                    for p in self.embedding.parameters():
                        p.requires_grad = False
                else:
                    assert opt['tune_partial'] < embedding.size(0)
                    fixed_embedding = embedding[opt['tune_partial']:]
                    self.register_buffer('fixed_embedding', fixed_embedding)
                    self.fixed_embedding = fixed_embedding
            embedding_dim = opt['embedding_dim']
            doc_input_size += embedding_dim
            que_input_size += embedding_dim
        else:
            opt['fix_embeddings'] = True
            opt['tune_partial'] = 0
        if opt['CoVe_opt'] > 0:
            self.CoVe = layers.MTLSTM(opt, embedding)
            CoVe_size = self.CoVe.output_size
            doc_input_size += CoVe_size
            que_input_size += CoVe_size
        if opt['use_elmo']:
            options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
            weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)
            doc_input_size += 1024
            que_input_size += 1024
        if opt['use_pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            doc_input_size += opt['pos_dim']
        if opt['use_ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            doc_input_size += opt['ner_dim']
        if opt['do_prealign']:
            self.pre_align = layers.GetAttentionHiddens(embedding_dim, opt['prealign_hidden'], similarity_attention=True)
            doc_input_size += embedding_dim
        if opt['no_em']:
            doc_input_size += opt['num_features'] - 3
        else:
            doc_input_size += opt['num_features']
        doc_hidden_size, que_hidden_size = doc_input_size, que_input_size
        None
        flow_size = opt['hidden_size']
        self.doc_rnn1 = layers.StackedBRNN(doc_hidden_size, opt['hidden_size'], num_layers=1)
        self.dialog_flow1 = layers.StackedBRNN(opt['hidden_size'] * 2, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
        self.doc_rnn2 = layers.StackedBRNN(opt['hidden_size'] * 2 + flow_size + CoVe_size, opt['hidden_size'], num_layers=1)
        self.dialog_flow2 = layers.StackedBRNN(opt['hidden_size'] * 2, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
        doc_hidden_size = opt['hidden_size'] * 2
        self.question_rnn, que_hidden_size = layers.RNN_from_opt(que_hidden_size, opt['hidden_size'], opt, num_layers=2, concat_rnn=opt['concat_rnn'], add_feat=CoVe_size)
        None
        self.deep_attn = layers.DeepAttention(opt, abstr_list_cnt=2, deep_att_hidden_size_per_abstr=opt['deep_att_hidden_size_per_abstr'], do_similarity=opt['deep_inter_att_do_similar'], word_hidden_size=embedding_dim + CoVe_size, no_rnn=True)
        self.deep_attn_rnn, doc_hidden_size = layers.RNN_from_opt(self.deep_attn.att_final_size + flow_size, opt['hidden_size'], opt, num_layers=1)
        self.dialog_flow3 = layers.StackedBRNN(doc_hidden_size, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
        self.high_lvl_qrnn, que_hidden_size = layers.RNN_from_opt(que_hidden_size * 2, opt['hidden_size'], opt, num_layers=1, concat_rnn=True)
        att_size = doc_hidden_size + 2 * opt['hidden_size'] * 2
        if opt['self_attention_opt'] > 0:
            self.highlvl_self_att = layers.GetAttentionHiddens(att_size, opt['deep_att_hidden_size_per_abstr'])
            self.high_lvl_crnn, doc_hidden_size = layers.RNN_from_opt(doc_hidden_size * 2 + flow_size, opt['hidden_size'], opt, num_layers=1, concat_rnn=False)
            None
        elif opt['self_attention_opt'] == 0:
            self.high_lvl_crnn, doc_hidden_size = layers.RNN_from_opt(doc_hidden_size + flow_size, opt['hidden_size'], opt, num_layers=1, concat_rnn=False)
        None
        self.self_attn = layers.LinearSelfAttn(que_hidden_size)
        if opt['do_hierarchical_query']:
            self.hier_query_rnn = layers.StackedBRNN(que_hidden_size, opt['hidden_size'], num_layers=1, rnn_type=nn.GRU, bidir=False)
            que_hidden_size = opt['hidden_size']
        self.get_answer = layers.GetSpanStartEnd(doc_hidden_size, que_hidden_size, opt, opt['ptr_net_indep_attn'], opt['ptr_net_attn_type'], opt['do_ptr_update'])
        self.ans_type_prediction = layers.BilinearLayer(doc_hidden_size * 2, que_hidden_size, opt['answer_type_num'])
        self.opt = opt

    def forward(self, x1, x1_c, x1_f, x1_pos, x1_ner, x1_mask, x2_full, x2_c, x2_full_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_c = document char indices           [batch * len_d * len_w] or [1]
        x1_f = document word features indices  [batch * q_num * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2_full = question word indices        [batch * q_num * len_q]
        x2_c = question char indices           [(batch * q_num) * len_q * len_w]
        x2_full_mask = question padding mask   [batch * q_num * len_q]
        """
        if self.opt['use_elmo'] and self.opt['elmo_batch_size'] > self.opt['batch_size']:
            if x1_c.dim() != 1:
                precomputed_bilm_output = self.elmo._elmo_lstm(x1_c)
                self.precomputed_layer_activations = [t.detach().cpu() for t in precomputed_bilm_output['activations']]
                self.precomputed_mask_with_bos_eos = precomputed_bilm_output['mask'].detach().cpu()
                self.precomputed_cnt = 0
            layer_activations = [t[x1.size(0) * self.precomputed_cnt:x1.size(0) * (self.precomputed_cnt + 1), :, :] for t in self.precomputed_layer_activations]
            mask_with_bos_eos = self.precomputed_mask_with_bos_eos[x1.size(0) * self.precomputed_cnt:x1.size(0) * (self.precomputed_cnt + 1), :]
            if x1.is_cuda:
                layer_activations = [t for t in layer_activations]
                mask_with_bos_eos = mask_with_bos_eos
            representations = []
            for i in range(len(self.elmo._scalar_mixes)):
                scalar_mix = getattr(self.elmo, 'scalar_mix_{}'.format(i))
                representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(representation_with_bos_eos, mask_with_bos_eos)
                representations.append(self.elmo._dropout(representation_without_bos_eos))
            x1_elmo = representations[0][:, :x1.size(1), :]
            self.precomputed_cnt += 1
            precomputed_elmo = True
        else:
            precomputed_elmo = False
        """
        x1_full = document word indices        [batch * q_num * len_d]
        x1_full_mask = document padding mask   [batch * q_num * len_d]
        """
        x1_full = x1.unsqueeze(1).expand(x2_full.size(0), x2_full.size(1), x1.size(1)).contiguous()
        x1_full_mask = x1_mask.unsqueeze(1).expand(x2_full.size(0), x2_full.size(1), x1.size(1)).contiguous()
        drnn_input_list, qrnn_input_list = [], []
        x2 = x2_full.view(-1, x2_full.size(-1))
        x2_mask = x2_full_mask.view(-1, x2_full.size(-1))
        if self.opt['use_wemb']:
            emb = self.embedding if self.training else self.eval_embed
            x1_emb = emb(x1)
            x2_emb = emb(x2)
            if self.opt['dropout_emb'] > 0:
                x1_emb = layers.dropout(x1_emb, p=self.opt['dropout_emb'], training=self.training)
                x2_emb = layers.dropout(x2_emb, p=self.opt['dropout_emb'], training=self.training)
            drnn_input_list.append(x1_emb)
            qrnn_input_list.append(x2_emb)
        if self.opt['CoVe_opt'] > 0:
            x1_cove_mid, x1_cove_high = self.CoVe(x1, x1_mask)
            x2_cove_mid, x2_cove_high = self.CoVe(x2, x2_mask)
            if self.opt['dropout_emb'] > 0:
                x1_cove_mid = layers.dropout(x1_cove_mid, p=self.opt['dropout_emb'], training=self.training)
                x1_cove_high = layers.dropout(x1_cove_high, p=self.opt['dropout_emb'], training=self.training)
                x2_cove_mid = layers.dropout(x2_cove_mid, p=self.opt['dropout_emb'], training=self.training)
                x2_cove_high = layers.dropout(x2_cove_high, p=self.opt['dropout_emb'], training=self.training)
            drnn_input_list.append(x1_cove_mid)
            qrnn_input_list.append(x2_cove_mid)
        if self.opt['use_elmo']:
            if not precomputed_elmo:
                x1_elmo = self.elmo(x1_c)['elmo_representations'][0]
            x2_elmo = self.elmo(x2_c)['elmo_representations'][0]
            if self.opt['dropout_emb'] > 0:
                x1_elmo = layers.dropout(x1_elmo, p=self.opt['dropout_emb'], training=self.training)
                x2_elmo = layers.dropout(x2_elmo, p=self.opt['dropout_emb'], training=self.training)
            drnn_input_list.append(x1_elmo)
            qrnn_input_list.append(x2_elmo)
        if self.opt['use_pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['use_ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            drnn_input_list.append(x1_ner_emb)
        x1_input = torch.cat(drnn_input_list, dim=2)
        x2_input = torch.cat(qrnn_input_list, dim=2)

        def expansion_for_doc(z):
            return z.unsqueeze(1).expand(z.size(0), x2_full.size(1), z.size(1), z.size(2)).contiguous().view(-1, z.size(1), z.size(2))
        x1_emb_expand = expansion_for_doc(x1_emb)
        x1_cove_high_expand = expansion_for_doc(x1_cove_high)
        if self.opt['no_em']:
            x1_f = x1_f[:, :, :, 3:]
        x1_input = torch.cat([expansion_for_doc(x1_input), x1_f.view(-1, x1_f.size(-2), x1_f.size(-1))], dim=2)
        x1_mask = x1_full_mask.view(-1, x1_full_mask.size(-1))
        if self.opt['do_prealign']:
            x1_atten = self.pre_align(x1_emb_expand, x2_emb, x2_mask)
            x1_input = torch.cat([x1_input, x1_atten], dim=2)

        def flow_operation(cur_h, flow):
            flow_in = cur_h.transpose(0, 1).view(x1_full.size(2), x1_full.size(0), x1_full.size(1), -1)
            flow_in = flow_in.transpose(0, 2).contiguous().view(x1_full.size(1), x1_full.size(0) * x1_full.size(2), -1).transpose(0, 1)
            flow_out = flow(flow_in)
            if self.opt['no_dialog_flow']:
                flow_out = flow_out * 0
            flow_out = flow_out.transpose(0, 1).view(x1_full.size(1), x1_full.size(0), x1_full.size(2), -1).transpose(0, 2).contiguous()
            flow_out = flow_out.view(x1_full.size(2), x1_full.size(0) * x1_full.size(1), -1).transpose(0, 1)
            return flow_out
        doc_abstr_ls = []
        doc_hiddens = self.doc_rnn1(x1_input, x1_mask)
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow1)
        doc_abstr_ls.append(doc_hiddens)
        doc_hiddens = self.doc_rnn2(torch.cat((doc_hiddens, doc_hiddens_flow, x1_cove_high_expand), dim=2), x1_mask)
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow2)
        doc_abstr_ls.append(doc_hiddens)
        _, que_abstr_ls = self.question_rnn(x2_input, x2_mask, return_list=True, additional_x=x2_cove_high)
        question_hiddens = self.high_lvl_qrnn(torch.cat(que_abstr_ls, 2), x2_mask)
        que_abstr_ls += [question_hiddens]
        doc_info = self.deep_attn([torch.cat([x1_emb_expand, x1_cove_high_expand], 2)], doc_abstr_ls, [torch.cat([x2_emb, x2_cove_high], 2)], que_abstr_ls, x1_mask, x2_mask)
        doc_hiddens = self.deep_attn_rnn(torch.cat((doc_info, doc_hiddens_flow), dim=2), x1_mask)
        doc_hiddens_flow = flow_operation(doc_hiddens, self.dialog_flow3)
        doc_abstr_ls += [doc_hiddens]
        x1_att = torch.cat(doc_abstr_ls, 2)
        if self.opt['self_attention_opt'] > 0:
            highlvl_self_attn_hiddens = self.highlvl_self_att(x1_att, x1_att, x1_mask, x3=doc_hiddens, drop_diagonal=True)
            doc_hiddens = self.high_lvl_crnn(torch.cat([doc_hiddens, highlvl_self_attn_hiddens, doc_hiddens_flow], dim=2), x1_mask)
        elif self.opt['self_attention_opt'] == 0:
            doc_hiddens = self.high_lvl_crnn(torch.cat([doc_hiddens, doc_hiddens_flow], dim=2), x1_mask)
        doc_abstr_ls += [doc_hiddens]
        q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_avg_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        if self.opt['do_hierarchical_query']:
            question_avg_hidden = self.hier_query_rnn(question_avg_hidden.view(x1_full.size(0), x1_full.size(1), -1))
            question_avg_hidden = question_avg_hidden.contiguous().view(-1, question_avg_hidden.size(-1))
        start_scores, end_scores = self.get_answer(doc_hiddens, question_avg_hidden, x1_mask)
        all_start_scores = start_scores.view_as(x1_full)
        all_end_scores = end_scores.view_as(x1_full)
        doc_avg_hidden = torch.cat((torch.max(doc_hiddens, dim=1)[0], torch.mean(doc_hiddens, dim=1)), dim=1)
        class_scores = self.ans_type_prediction(doc_avg_hidden, question_avg_hidden)
        all_class_scores = class_scores.view(x1_full.size(0), x1_full.size(1), -1)
        all_class_scores = all_class_scores.squeeze(-1)
        return all_start_scores, all_end_scores, all_class_scores


def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training == False or p == 0:
        return x
    dropout_mask = 1.0 / (1 - p) * torch.bernoulli((1 - p) * (x.new_zeros(x.size(0), x.size(2)) + 1))
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3:
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


class StackedBRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, rnn_type=nn.LSTM, concat_layers=False, do_residual=False, add_feat=0, dialog_flow=False, bidir=True):
        super(StackedBRNN, self).__init__()
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.do_residual = do_residual
        self.dialog_flow = dialog_flow
        self.hidden_size = hidden_size
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size + add_feat if i == 1 else 2 * hidden_size
            if self.dialog_flow == True:
                input_size += 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=bidir))

    def forward(self, x, x_mask=None, return_list=False, additional_x=None, previous_hiddens=None):
        x = x.transpose(0, 1)
        if additional_x is not None:
            additional_x = additional_x.transpose(0, 1)
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = hiddens[-1]
            if i == 1 and additional_x is not None:
                rnn_input = torch.cat((rnn_input, additional_x), 2)
            if my_dropout_p > 0:
                rnn_input = dropout(rnn_input, p=my_dropout_p, training=self.training)
            if self.dialog_flow == True:
                if previous_hiddens is not None:
                    dialog_memory = previous_hiddens[i - 1].transpose(0, 1)
                else:
                    dialog_memory = rnn_input.new_zeros((rnn_input.size(0), rnn_input.size(1), self.hidden_size * 2))
                rnn_input = torch.cat((rnn_input, dialog_memory), 2)
            rnn_output = self.rnns[i](rnn_input)[0]
            if self.do_residual and i > 0:
                rnn_output = rnn_output + hiddens[-1]
            hiddens.append(rnn_output)
        hiddens = [h.transpose(0, 1) for h in hiddens]
        if self.concat_layers:
            output = torch.cat(hiddens[1:], 2)
        else:
            output = hiddens[-1]
        if return_list:
            return output, hiddens[1:]
        else:
            return output


class MemoryLasagna_Time(nn.Module):

    def __init__(self, input_size, hidden_size, rnn_type='lstm'):
        super(MemoryLasagna_Time, self).__init__()
        RNN_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell}
        self.rnn = RNN_TYPES[rnn_type](input_size, hidden_size)
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, memory):
        if self.training:
            x = x * self.dropout_mask
        memory = self.rnn(x.contiguous().view(-1, x.size(-1)), memory)
        if self.rnn_type == 'lstm':
            h = memory[0].view(x.size(0), x.size(1), -1)
        else:
            h = memory.view(x.size(0), x.size(1), -1)
        return h, memory

    def get_init(self, sample_tensor):
        global my_dropout_p
        self.dropout_mask = 1.0 / (1 - my_dropout_p) * torch.bernoulli((1 - my_dropout_p) * (sample_tensor.new_zeros(sample_tensor.size(0), sample_tensor.size(1), self.input_size) + 1))
        h = sample_tensor.new_zeros(sample_tensor.size(0), sample_tensor.size(1), self.hidden_size).float()
        memory = sample_tensor.new_zeros(sample_tensor.size(0) * sample_tensor.size(1), self.hidden_size).float()
        if self.rnn_type == 'lstm':
            memory = memory, memory
        return h, memory


class MTLSTM(nn.Module):

    def __init__(self, opt, embedding=None, padding_idx=0):
        """Initialize an MTLSTM

        Arguments:
            embedding (Float Tensor): If not None, initialize embedding matrix with specified embedding vectors
        """
        super(MTLSTM, self).__init__()
        self.embedding = nn.Embedding(opt['vocab_size'], opt['embedding_dim'], padding_idx=padding_idx)
        if embedding is not None:
            self.embedding.weight.data = embedding
        state_dict = torch.load(opt['MTLSTM_path'])
        self.rnn1 = nn.LSTM(300, 300, num_layers=1, bidirectional=True)
        self.rnn2 = nn.LSTM(600, 300, num_layers=1, bidirectional=True)
        state_dict1 = dict([((name, param.data) if isinstance(param, Parameter) else (name, param)) for name, param in state_dict.items() if '0' in name])
        state_dict2 = dict([((name.replace('1', '0'), param.data) if isinstance(param, Parameter) else (name.replace('1', '0'), param)) for name, param in state_dict.items() if '1' in name])
        self.rnn1.load_state_dict(state_dict1)
        self.rnn2.load_state_dict(state_dict2)
        for p in self.embedding.parameters():
            p.requires_grad = False
        for p in self.rnn1.parameters():
            p.requires_grad = False
        for p in self.rnn2.parameters():
            p.requires_grad = False
        self.output_size = 600

    def setup_eval_embed(self, eval_embed, padding_idx=0):
        """Allow evaluation vocabulary size to be greater than training vocabulary size

        Arguments:
            eval_embed (Float Tensor): Initialize eval_embed to be the specified embedding vectors
        """
        self.eval_embed = nn.Embedding(eval_embed.size(0), eval_embed.size(1), padding_idx=padding_idx)
        self.eval_embed.weight.data = eval_embed
        for p in self.eval_embed.parameters():
            p.requires_grad = False

    def forward(self, x_idx, x_mask):
        """A pretrained MT-LSTM (McCann et. al. 2017).
        This LSTM was trained with 300d 840B GloVe on the WMT 2017 machine translation dataset.

        Arguments:
            x_idx (Long Tensor): a Long Tensor of size (batch * len).
            x_mask (Byte Tensor): a Byte Tensor of mask for the input tensor (batch * len).
        """
        emb = self.embedding if self.training else self.eval_embed
        x_hiddens = emb(x_idx)
        lengths = x_mask.data.eq(0).long().sum(dim=1)
        lens, indices = torch.sort(lengths, 0, True)
        output1, _ = self.rnn1(pack(x_hiddens[indices], lens.tolist(), batch_first=True))
        output2, _ = self.rnn2(output1)
        output1 = unpack(output1, batch_first=True)[0]
        output2 = unpack(output2, batch_first=True)[0]
        _, _indices = torch.sort(indices, 0)
        output1 = output1[_indices]
        output2 = output2[_indices]
        return output1, output2


class AttentionScore(nn.Module):
    """
    sij = Relu(Wx1)DRelu(Wx2)
    """

    def __init__(self, input_size, attention_hidden_size, similarity_score=False):
        super(AttentionScore, self).__init__()
        self.linear = nn.Linear(input_size, attention_hidden_size, bias=False)
        if similarity_score:
            self.linear_final = Parameter(torch.ones(1, 1, 1) / attention_hidden_size ** 0.5, requires_grad=False)
        else:
            self.linear_final = Parameter(torch.ones(1, 1, attention_hidden_size), requires_grad=True)

    def forward(self, x1, x2):
        """
        x1: batch * len1 * input_size
        x2: batch * len2 * input_size
        scores: batch * len1 * len2 <the scores are not masked>
        """
        x1 = dropout(x1, p=my_dropout_p, training=self.training)
        x2 = dropout(x2, p=my_dropout_p, training=self.training)
        x1_rep = self.linear(x1.contiguous().view(-1, x1.size(-1))).view(x1.size(0), x1.size(1), -1)
        x2_rep = self.linear(x2.contiguous().view(-1, x2.size(-1))).view(x2.size(0), x2.size(1), -1)
        x1_rep = F.relu(x1_rep)
        x2_rep = F.relu(x2_rep)
        final_v = self.linear_final.expand_as(x2_rep)
        x2_rep_v = final_v * x2_rep
        scores = x1_rep.bmm(x2_rep_v.transpose(1, 2))
        return scores


class GetAttentionHiddens(nn.Module):

    def __init__(self, input_size, attention_hidden_size, similarity_attention=False):
        super(GetAttentionHiddens, self).__init__()
        self.scoring = AttentionScore(input_size, attention_hidden_size, similarity_score=similarity_attention)

    def forward(self, x1, x2, x2_mask, x3=None, scores=None, return_scores=False, drop_diagonal=False):
        """
        Using x1, x2 to calculate attention score, but x1 will take back info from x3.
        If x3 is not specified, x1 will attend on x2.

        x1: batch * len1 * x1_input_size
        x2: batch * len2 * x2_input_size
        x2_mask: batch * len2

        x3: batch * len2 * x3_input_size (or None)
        """
        if x3 is None:
            x3 = x2
        if scores is None:
            scores = self.scoring(x1, x2)
        x2_mask = x2_mask.unsqueeze(1).expand_as(scores)
        scores.data.masked_fill_(x2_mask.data, -float('inf'))
        if drop_diagonal:
            assert scores.size(1) == scores.size(2)
            diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
            scores.data.masked_fill_(diag_mask, -float('inf'))
        alpha = F.softmax(scores, dim=2)
        matched_seq = alpha.bmm(x3)
        if return_scores:
            return matched_seq, scores
        else:
            return matched_seq


def RNN_from_opt(input_size_, hidden_size_, opt, num_layers=-1, concat_rnn=None, add_feat=0, dialog_flow=False):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    new_rnn = StackedBRNN(input_size=input_size_, hidden_size=hidden_size_, num_layers=num_layers if num_layers > 0 else opt['rnn_layers'], rnn_type=RNN_TYPES[opt['rnn_type']], concat_layers=concat_rnn if concat_rnn is not None else opt['concat_rnn'], do_residual=opt['do_residual_rnn'] or opt['do_residual_everything'], add_feat=add_feat, dialog_flow=dialog_flow)
    output_size = 2 * hidden_size_
    if concat_rnn if concat_rnn is not None else opt['concat_rnn']:
        output_size *= num_layers if num_layers > 0 else opt['rnn_layers']
    return new_rnn, output_size


class DeepAttention(nn.Module):

    def __init__(self, opt, abstr_list_cnt, deep_att_hidden_size_per_abstr, do_similarity=False, word_hidden_size=None, do_self_attn=False, dialog_flow=False, no_rnn=False):
        super(DeepAttention, self).__init__()
        self.no_rnn = no_rnn
        word_hidden_size = opt['embedding_dim'] if word_hidden_size is None else word_hidden_size
        abstr_hidden_size = opt['hidden_size'] * 2
        att_size = abstr_hidden_size * abstr_list_cnt + word_hidden_size
        self.int_attn_list = nn.ModuleList()
        for i in range(abstr_list_cnt + 1):
            self.int_attn_list.append(GetAttentionHiddens(att_size, deep_att_hidden_size_per_abstr, similarity_attention=do_similarity))
        rnn_input_size = abstr_hidden_size * abstr_list_cnt * 2 + opt['hidden_size'] * 2
        self.att_final_size = rnn_input_size
        if not self.no_rnn:
            self.rnn, self.output_size = RNN_from_opt(rnn_input_size, opt['hidden_size'], opt, num_layers=1, dialog_flow=dialog_flow)
        self.opt = opt
        self.do_self_attn = do_self_attn

    def forward(self, x1_word, x1_abstr, x2_word, x2_abstr, x1_mask, x2_mask, return_bef_rnn=False, previous_hiddens=None):
        """
        x1_word, x2_word, x1_abstr, x2_abstr are list of 3D tensors.
        3D tensor: batch_size * length * hidden_size
        """
        x1_att = torch.cat(x1_word + x1_abstr, 2)
        x2_att = torch.cat(x2_word + x2_abstr[:-1], 2)
        x1 = torch.cat(x1_abstr, 2)
        x2_list = x2_abstr
        for i in range(len(x2_list)):
            attn_hiddens = self.int_attn_list[i](x1_att, x2_att, x2_mask, x3=x2_list[i], drop_diagonal=self.do_self_attn)
            x1 = torch.cat((x1, attn_hiddens), 2)
        if not self.no_rnn:
            x1_hiddens = self.rnn(x1, x1_mask, previous_hiddens=previous_hiddens)
            if return_bef_rnn:
                return x1_hiddens, x1
            else:
                return x1_hiddens
        else:
            return x1


class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x = dropout(x, p=my_dropout_p, training=self.training)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)
        return alpha


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """

    def __init__(self, x_size, y_size, opt, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = dropout(x, p=my_dropout_p, training=self.training)
        y = dropout(y, p=my_dropout_p, training=self.training)
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        return xWy


class GetSpanStartEnd(nn.Module):

    def __init__(self, x_size, h_size, opt, do_indep_attn=True, attn_type='Bilinear', do_ptr_update=True):
        super(GetSpanStartEnd, self).__init__()
        self.attn = BilinearSeqAttn(x_size, h_size, opt)
        self.attn2 = BilinearSeqAttn(x_size, h_size, opt) if do_indep_attn else None
        self.rnn = nn.GRUCell(x_size, h_size) if do_ptr_update else None

    def forward(self, x, h0, x_mask):
        """
        x = batch * len * x_size
        h0 = batch * h_size
        x_mask = batch * len
        """
        st_scores = self.attn(x, h0, x_mask)
        if self.rnn is not None:
            ptr_net_in = torch.bmm(F.softmax(st_scores, dim=1).unsqueeze(1), x).squeeze(1)
            ptr_net_in = dropout(ptr_net_in, p=my_dropout_p, training=self.training)
            h0 = dropout(h0, p=my_dropout_p, training=self.training)
            h1 = self.rnn(ptr_net_in, h0)
        else:
            h1 = h0
        end_scores = self.attn(x, h1, x_mask) if self.attn2 is None else self.attn2(x, h1, x_mask)
        return st_scores, end_scores


class BilinearLayer(nn.Module):

    def __init__(self, x_size, y_size, class_num):
        super(BilinearLayer, self).__init__()
        self.linear = nn.Linear(y_size, x_size * class_num)
        self.class_num = class_num

    def forward(self, x, y):
        """
        x = batch * h1
        y = batch * h2
        """
        x = dropout(x, p=my_dropout_p, training=self.training)
        y = dropout(y, p=my_dropout_p, training=self.training)
        Wy = self.linear(y)
        Wy = Wy.view(Wy.size(0), self.class_num, x.size(1))
        xWy = torch.sum(x.unsqueeze(1).expand_as(Wy) * Wy, dim=2)
        return xWy.squeeze(-1)

