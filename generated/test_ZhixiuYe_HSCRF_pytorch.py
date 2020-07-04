import sys
_module = sys.modules[__name__]
del sys
eval = _module
model = _module
crf_layer = _module
data_packer = _module
evaluator = _module
highway_layer = _module
hscrf_layer = _module
model = _module
utils = _module
word_rep_layer = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
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


from torch.autograd import Variable


import numpy as np


import itertools


from functools import reduce


import torch.nn.init


import time


import torch.optim as optim


import functools


class CRF(nn.Module):
    """
    Conditional Random Field (CRF) layer.

    """

    def __init__(self, start_tag, end_tag, hidden_dim, tagset_size):
        """

        args:
            start_tag  (scalar) : special start tag for CRF
            end_tag    (scalar) : special end tag for CRF
            hidden_dim (scalar) : input dim size
            tagset_size(scalar) : target_set_size

        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size * self.
            tagset_size)
        self.rand_init()

    def rand_init(self):
        """
        random initialization

        """
        utils.init_linear(self.hidden2tag)

    def cal_score(self, feats):
        """
        calculate CRF score

        :param feats (sentlen, batch_size, feature_num) : input features
        """
        sentlen = feats.size(0)
        batch_size = feats.size(1)
        crf_scores = self.hidden2tag(feats).view(-1, self.tagset_size, self
            .tagset_size)
        self.crf_scores = crf_scores.view(sentlen, batch_size, self.
            tagset_size, self.tagset_size)
        return self.crf_scores

    def forward(self, feats, target, mask):
        """
        calculate viterbi loss

        args:
            feats  (batch_size, seq_len, hidden_dim) : input features from word_rep layers
            target (batch_size, seq_len, 1) : crf label
            mask   (batch_size, seq_len) : mask for crf label

        """
        crf_scores = self.cal_score(feats)
        loss = self.get_loss(crf_scores, target, mask)
        return loss

    def get_loss(self, scores, target, mask):
        """
        calculate viterbi loss

        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : class score for CRF
            target (seq_len, bat_size, 1) : crf label
            mask   (seq_len, bat_size) : mask for crf label

        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target
            ).view(seq_len, bat_size)
        tg_energy = tg_energy.masked_select(mask).sum()
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.next()
        partition = inivalues[:, (self.start_tag), :].clone()
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(bat_size,
                self.tagset_size, 1).expand(bat_size, self.tagset_size,
                self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values, self.tagset_size)
            mask_idx = mask[(idx), :].view(bat_size, 1).expand(bat_size,
                self.tagset_size)
            partition.masked_scatter_(mask_idx, cur_partition.masked_select
                (mask_idx))
        partition = partition[:, (self.end_tag)].sum()
        loss = (partition - tg_energy) / bat_size
        return loss

    def decode(self, feats, mask):
        """
        decode with dynamic programming

        args:
            feats (sentlen, batch_size, feature_num) : input features
            mask (seq_len, bat_size) : mask for padding

        """
        scores = self.cal_score(feats)
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        mask = Variable(1 - mask.data, volatile=True)
        decode_idx = Variable(torch.LongTensor(seq_len - 1, bat_size),
            volatile=True)
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.next()
        forscores = inivalues[:, (self.start_tag), :]
        back_points = list()
        for idx, cur_values in seq_iter:
            cur_values = cur_values + forscores.contiguous().view(bat_size,
                self.tagset_size, 1).expand(bat_size, self.tagset_size,
                self.tagset_size)
            forscores, cur_bp = torch.max(cur_values, 1)
            cur_bp.masked_fill_(mask[idx].view(bat_size, 1).expand(bat_size,
                self.tagset_size), self.end_tag)
            back_points.append(cur_bp)
        pointer = back_points[-1][:, (self.end_tag)]
        decode_idx[-1] = pointer
        for idx in range(len(back_points) - 2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous(
                ).view(bat_size, 1))
            decode_idx[idx] = pointer
        return decode_idx


class hw(nn.Module):
    """
    Highway layers

    """

    def __init__(self, size, num_layers=1, dropout_ratio=0.5):
        super(hw, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_ratio)
        for i in range(num_layers):
            tmptrans = nn.Linear(size, size)
            tmpgate = nn.Linear(size, size)
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)

    def rand_init(self):
        """
        random initialization

        """
        for i in range(self.num_layers):
            utils.init_linear(self.trans[i])
            utils.init_linear(self.gate[i])

    def forward(self, x):
        """
        update statics for f1 score

        args: 
            x (ins_num, hidden_dim): input tensor

        """
        t = self.gate[0](x)
        g = nn.functional.sigmoid(t)
        h = nn.functional.relu(self.trans[0](x))
        x = g * h + (1 - g) * x
        for i in range(1, self.num_layers):
            x = self.dropout(x)
            g = nn.functional.sigmoid(self.gate[i](x))
            h = nn.functional.relu(self.trans[i](x))
            x = g * h + (1 - g) * x
        return x


class HSCRF(nn.Module):

    def __init__(self, tag_to_ix, word_rep_dim=300, SCRF_feature_dim=100,
        index_embeds_dim=10, ALLOWED_SPANLEN=6, start_id=4, stop_id=5,
        noBIES=False, no_index=False, no_sub=False, grconv=False):
        super(HSCRF, self).__init__()
        self.tag_to_ix = tag_to_ix
        self.ix_to_tag = {v: k for k, v in self.tag_to_ix.items()}
        self.tagset_size = len(tag_to_ix)
        self.index_embeds_dim = index_embeds_dim
        self.SCRF_feature_dim = SCRF_feature_dim
        self.ALLOWED_SPANLEN = ALLOWED_SPANLEN
        self.start_id = start_id
        self.stop_id = stop_id
        self.grconv = grconv
        self.index_embeds = nn.Embedding(self.ALLOWED_SPANLEN, self.
            index_embeds_dim)
        self.init_embedding(self.index_embeds.weight)
        self.dense = nn.Linear(word_rep_dim, self.SCRF_feature_dim)
        self.init_linear(self.dense)
        self.CRF_tagset_size = 4 * (self.tagset_size - 3) + 2
        self.transition = nn.Parameter(torch.zeros(self.tagset_size, self.
            tagset_size))
        span_word_embedding_dim = (2 * self.SCRF_feature_dim + self.
            index_embeds_dim)
        self.new_hidden2CRFtag = nn.Linear(span_word_embedding_dim, self.
            CRF_tagset_size)
        self.init_linear(self.new_hidden2CRFtag)
        if self.grconv:
            self.Wl = nn.Linear(self.SCRF_feature_dim, self.SCRF_feature_dim)
            self.Wr = nn.Linear(self.SCRF_feature_dim, self.SCRF_feature_dim)
            self.Gl = nn.Linear(self.SCRF_feature_dim, 3 * self.
                SCRF_feature_dim)
            self.Gr = nn.Linear(self.SCRF_feature_dim, 3 * self.
                SCRF_feature_dim)
            self.toSCRF = nn.Linear(self.SCRF_feature_dim, self.tagset_size)
            self.init_linear(self.Wl)
            self.init_linear(self.Wr)
            self.init_linear(self.Gl)
            self.init_linear(self.Gr)
            self.init_linear(self.toSCRF)

    def init_embedding(self, input_embedding):
        """
        Initialize embedding

        """
        bias = np.sqrt(3.0 / input_embedding.size(1))
        nn.init.uniform(input_embedding, -bias, bias)

    def init_linear(self, input_linear):
        """
        Initialize linear transformation

        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.
            weight.size(1)))
        nn.init.uniform(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    def get_logloss_denominator(self, scores, mask):
        """
        calculate all path scores of SCRF with dynamic programming

        args:
            scores (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : features for SCRF
            mask   (batch_size) : mask for words

        """
        logalpha = Variable(torch.FloatTensor(self.batch_size, self.
            sent_len + 1, self.tagset_size).fill_(-10000.0))
        logalpha[:, (0), (self.start_id)] = 0.0
        istarts = [0] * self.ALLOWED_SPANLEN + range(self.sent_len - self.
            ALLOWED_SPANLEN + 1)
        for i in range(1, self.sent_len + 1):
            tmp = scores[:, istarts[i]:i, (i - 1)] + logalpha[:, istarts[i]:i
                ].unsqueeze(3).expand(self.batch_size, i - istarts[i], self
                .tagset_size, self.tagset_size)
            tmp = tmp.transpose(1, 3).contiguous().view(self.batch_size,
                self.tagset_size, (i - istarts[i]) * self.tagset_size)
            max_tmp, _ = torch.max(tmp, dim=2)
            tmp = tmp - max_tmp.view(self.batch_size, self.tagset_size, 1)
            logalpha[:, (i)] = max_tmp + torch.log(torch.sum(torch.exp(tmp),
                dim=2))
        mask = mask.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1,
            self.tagset_size)
        alpha = torch.gather(logalpha, 1, mask).squeeze(1)
        return alpha[:, (self.stop_id)].sum()

    def decode(self, factexprscalars, mask):
        """
        decode SCRF labels with dynamic programming

        args:
            factexprscalars (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : features for SCRF
            mask            (batch_size) : mask for words

        """
        batch_size = factexprscalars.size(0)
        sentlen = factexprscalars.size(1)
        factexprscalars = factexprscalars.data
        logalpha = torch.FloatTensor(batch_size, sentlen + 1, self.tagset_size
            ).fill_(-10000.0)
        logalpha[:, (0), (self.start_id)] = 0.0
        starts = torch.zeros((batch_size, sentlen, self.tagset_size))
        ys = torch.zeros((batch_size, sentlen, self.tagset_size))
        for j in range(1, sentlen + 1):
            istart = 0
            if j > self.ALLOWED_SPANLEN:
                istart = max(0, j - self.ALLOWED_SPANLEN)
            f = factexprscalars[:, istart:j, (j - 1)].permute(0, 3, 1, 2
                ).contiguous().view(batch_size, self.tagset_size, -1
                ) + logalpha[:, istart:j].contiguous().view(batch_size, 1, -1
                ).expand(batch_size, self.tagset_size, (j - istart) * self.
                tagset_size)
            logalpha[:, (j), :], argm = torch.max(f, dim=2)
            starts[:, (j - 1), :] = argm / self.tagset_size + istart
            ys[:, (j - 1), :] = argm % self.tagset_size
        batch_scores = []
        batch_spans = []
        for i in range(batch_size):
            spans = []
            batch_scores.append(max(logalpha[i, mask[i] - 1]))
            end = mask[i] - 1
            y = self.stop_id
            while end >= 0:
                start = int(starts[i, end, y])
                y_1 = int(ys[i, end, y])
                spans.append((start, end, y_1, y))
                y = y_1
                end = start - 1
            batch_spans.append(spans)
        return batch_spans, batch_scores

    def get_logloss_numerator(self, goldfactors, scores, mask):
        """
        get scores of best path

        args:
            goldfactors (batch_size, tag_len, 4) : path labels
            scores      (batch_size, sent_len, sent_len, self.tagset_size, self.tagset_size) : all tag scores
            mask        (batch_size, tag_len) : mask for goldfactors

        """
        batch_size = scores.size(0)
        sent_len = scores.size(1)
        tagset_size = scores.size(3)
        goldfactors = goldfactors[:, :, (0)
            ] * sent_len * tagset_size * tagset_size + goldfactors[:, :, (1)
            ] * tagset_size * tagset_size + goldfactors[:, :, (2)
            ] * tagset_size + goldfactors[:, :, (3)]
        factorexprs = scores.view(batch_size, -1)
        val = torch.gather(factorexprs, 1, goldfactors)
        numerator = val.masked_select(mask)
        return numerator

    def grConv_scores(self, feats):
        """
        calculate SCRF scores with grConv

        args:
            feats (batch_size, sentence_len, featsdim) : word representations

        """
        scores = Variable(torch.zeros(self.batch_size, self.sent_len, self.
            sent_len, self.SCRF_feature_dim))
        diag0 = torch.LongTensor(range(self.sent_len))
        ht = feats
        scores[:, (diag0), (diag0)] = ht
        if self.sent_len == 1:
            return self.toSCRF(scores).unsqueeze(3
                ) + self.transition.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        for span_len in range(1, min(self.ALLOWED_SPANLEN, self.sent_len)):
            ht_1_l = ht[:, :-1]
            ht_1_r = ht[:, 1:]
            h_t_hat = 4 * nn.functional.sigmoid(self.Wl(ht_1_l) + self.Wr(
                ht_1_r)) - 2
            w = torch.exp(self.Gl(ht_1_l) + self.Gr(ht_1_r)).view(self.
                batch_size, self.sent_len - span_len, 3, self.SCRF_feature_dim
                ).permute(2, 0, 1, 3)
            w = w / w.sum(0).unsqueeze(0).expand(3, self.batch_size, self.
                sent_len - span_len, self.SCRF_feature_dim)
            ht = w[0] * h_t_hat + w[1] * ht_1_l + w[2] * ht_1_r
            scores[:, (diag0[:-span_len]), (diag0[span_len:])] = ht
        return self.toSCRF(scores).unsqueeze(3) + self.transition.unsqueeze(0
            ).unsqueeze(0).unsqueeze(0)

    def HSCRF_scores(self, feats):
        """
        calculate SCRF scores with HSCRF

        args:
            feats (batch_size, sentence_len, featsdim) : word representations

        """
        validtag_size = self.tagset_size - 3
        scores = Variable(torch.zeros(self.batch_size, self.sent_len, self.
            sent_len, self.tagset_size, self.tagset_size))
        diag0 = torch.LongTensor(range(self.sent_len))
        m10000 = Variable(torch.FloatTensor([-10000.0]).expand(self.
            batch_size, self.sent_len, self.tagset_size, 1))
        m30000 = Variable(torch.FloatTensor([-10000.0]).expand(self.
            batch_size, self.sent_len, self.tagset_size, 3))
        for span_len in range(min(self.ALLOWED_SPANLEN, self.sent_len)):
            emb_x = self.concat_features(feats, span_len)
            emb_x = self.new_hidden2CRFtag(emb_x)
            if span_len == 0:
                tmp = torch.cat((self.transition[:, :validtag_size].
                    unsqueeze(0).unsqueeze(0) + emb_x[:, (0), :, :
                    validtag_size].unsqueeze(2), m10000, self.transition[:,
                    -2:].unsqueeze(0).unsqueeze(0) + emb_x[:, (0), :, -2:].
                    unsqueeze(2)), 3)
                scores[:, (diag0), (diag0)] = tmp
            elif span_len == 1:
                tmp = torch.cat((self.transition[:, :validtag_size].
                    unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.
                    sent_len - 1, self.tagset_size, validtag_size) + (emb_x
                    [:, (0), :, validtag_size:2 * validtag_size] + emb_x[:,
                    (1), :, 3 * validtag_size:4 * validtag_size]).unsqueeze
                    (2), m30000[:, 1:]), 3)
                scores[:, (diag0[:-1]), (diag0[1:])] = tmp
            elif span_len == 2:
                tmp = torch.cat((self.transition[:, :validtag_size].
                    unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.
                    sent_len - 2, self.tagset_size, validtag_size) + (emb_x
                    [:, (0), :, validtag_size:2 * validtag_size] + emb_x[:,
                    (1), :, 2 * validtag_size:3 * validtag_size] + emb_x[:,
                    (2), :, 3 * validtag_size:4 * validtag_size]).unsqueeze
                    (2), m30000[:, 2:]), 3)
                scores[:, (diag0[:-2]), (diag0[2:])] = tmp
            elif span_len >= 3:
                tmp0 = self.transition[:, :validtag_size].unsqueeze(0
                    ).unsqueeze(0).expand(self.batch_size, self.sent_len -
                    span_len, self.tagset_size, validtag_size) + (emb_x[:,
                    (0), :, validtag_size:2 * validtag_size] + emb_x[:, 1:
                    span_len, :, 2 * validtag_size:3 * validtag_size].sum(1
                    ) + emb_x[:, (span_len), :, 3 * validtag_size:4 *
                    validtag_size]).unsqueeze(2)
                tmp = torch.cat((tmp0, m30000[:, span_len:]), 3)
                scores[:, (diag0[:-span_len]), (diag0[span_len:])] = tmp
        return scores

    def concat_features(self, emb_z, span_len):
        """
        concatenate two features

        args:

            emb_z (batch_size, sentence_len, featsdim) : word representations
            span_len: a number (from 0)

        """
        batch_size = emb_z.size(0)
        sent_len = emb_z.size(1)
        hidden_dim = emb_z.size(2)
        emb_z = emb_z.unsqueeze(1).expand(batch_size, sent_len, sent_len,
            hidden_dim)
        new_emb_z1 = [emb_z[:, i:i + 1, i:i + span_len + 1] for i in range(
            sent_len - span_len)]
        new_emb_z1 = torch.cat(new_emb_z1, 1)
        new_emb_z2 = (new_emb_z1[:, :, (0)] - new_emb_z1[:, :, (span_len)]
            ).unsqueeze(2).expand(batch_size, sent_len - span_len, span_len +
            1, hidden_dim)
        index = Variable(torch.LongTensor(range(span_len + 1)))
        index = self.index_embeds(index).unsqueeze(0).unsqueeze(0).expand(
            batch_size, sent_len - span_len, span_len + 1, self.
            index_embeds_dim)
        new_emb = torch.cat((new_emb_z1, new_emb_z2, index), 3).transpose(1, 2
            ).contiguous()
        return new_emb

    def forward(self, feats, mask_word, tags, mask_tag):
        """
        calculate loss

        args:
            feats (batch_size, sent_len, featsdim) : word representations
            mask_word (batch_size) : sentence lengths
            tags (batch_size, tag_len, 4) : target
            mask_tag (batch_size, tag_len) : tag_len <= sentence_len

        """
        self.batch_size = feats.size(0)
        self.sent_len = feats.size(1)
        feats = self.dense(feats)
        if self.grconv:
            self.SCRF_scores = self.grConv_scores(feats)
        else:
            self.SCRF_scores = self.HSCRF_scores(feats)
        forward_score = self.get_logloss_denominator(self.SCRF_scores,
            mask_word)
        numerator = self.get_logloss_numerator(tags, self.SCRF_scores, mask_tag
            )
        return (forward_score - numerator.sum()) / self.batch_size

    def get_scrf_decode(self, feats, mask):
        """
        decode with SCRF

        args:
            feats (batch_size, sent_len, featsdim) : word representations
            mask  (batch_size) : mask for words

        """
        self.batch_size = feats.size(0)
        self.sent_len = feats.size(1)
        feats = self.dense(feats)
        if self.grconv:
            self.SCRF_scores = self.grConv_scores(feats)
        else:
            self.SCRF_scores = self.HSCRF_scores(feats)
        batch_spans, batch_scores = self.decode(self.SCRF_scores, mask)
        batch_answer = self.tuple_to_seq(batch_spans)
        return batch_answer, np.array(batch_scores)

    def tuple_to_seq(self, batch_spans):
        batch_answer = []
        for spans in batch_spans:
            answer = utils.tuple_to_seq_BIOES(spans, self.ix_to_tag)
            batch_answer.append(answer[:-1])
        return batch_answer


class ner_model(nn.Module):

    def __init__(self, word_embedding_dim, word_hidden_dim,
        word_lstm_layers, vocab_size, char_size, char_embedding_dim,
        char_lstm_hidden_dim, cnn_filter_num, char_lstm_layers, char_lstm,
        dropout_ratio, if_highway, highway_layers, crf_start_tag,
        crf_end_tag, crf_target_size, scrf_tag_map, scrf_dense_dim,
        in_doc_words, index_embeds_dim, ALLOWED_SPANLEN, scrf_start_tag,
        scrf_end_tag, grconv):
        super(ner_model, self).__init__()
        self.char_lstm = char_lstm
        self.word_rep = WORD_REP(char_size, char_embedding_dim,
            char_lstm_hidden_dim, cnn_filter_num, char_lstm_layers,
            word_embedding_dim, word_hidden_dim, word_lstm_layers,
            vocab_size, dropout_ratio, if_highway=if_highway, in_doc_words=
            in_doc_words, highway_layers=highway_layers, char_lstm=char_lstm)
        self.crf = CRF(crf_start_tag, crf_end_tag, word_hidden_dim,
            crf_target_size)
        self.hscrf = HSCRF(scrf_tag_map, word_rep_dim=word_hidden_dim,
            SCRF_feature_dim=scrf_dense_dim, index_embeds_dim=
            index_embeds_dim, ALLOWED_SPANLEN=ALLOWED_SPANLEN, start_id=
            scrf_start_tag, stop_id=scrf_end_tag, grconv=grconv)

    def forward(self, forw_sentence, forw_position, back_sentence,
        back_position, word_seq, cnn_features, crf_target, crf_mask,
        scrf_mask_words, scrf_target, scrf_mask_target, onlycrf=True):
        """
        calculate loss

        :param forw_sentence   (char_seq_len, batch_size) : char-level representation of sentence
        :param forw_position   (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
        :param back_sentence   (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
        :param back_position   (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
        :param word_seq        (word_seq_len, batch_size) : word-level representation of sentence
        :param cnn_features    (word_seq_len, batch_size, word_len) : char-level representation of words
        :param crf_target      (word_seq_len, batch_size, 1): labels for CRF
        :param crf_mask        (word_seq_len, batch_size) : mask for crf_target and word_seq
        :param scrf_mask_words (batch_size) : lengths of sentences
        :param scrf_target     (batch_size, tag_len, 4) : labels for SCRF
        :param scrf_mask_target(batch_size, tag_len) : mask for scrf_target
        :param onlycrf         (True or False) : whether training data is suitable for SCRF
        :return:
        """
        word_representations = self.word_rep(forw_sentence, forw_position,
            back_sentence, back_position, word_seq, cnn_features)
        loss_crf = self.crf(word_representations, crf_target, crf_mask)
        loss = loss_crf
        if not onlycrf:
            loss_scrf = self.hscrf(word_representations.transpose(0, 1),
                scrf_mask_words, scrf_target, scrf_mask_target)
            loss = loss + loss_scrf
        if self.char_lstm:
            loss_lm = self.word_rep.lm_loss(forw_sentence, forw_position,
                back_sentence, back_position, word_seq)
            loss = loss + loss_lm
        return loss


class WORD_REP(nn.Module):

    def __init__(self, char_size, char_embedding_dim, char_hidden_dim,
        cnn_filter_num, char_lstm_layers, word_embedding_dim,
        word_hidden_dim, word_lstm_layers, vocab_size, dropout_ratio,
        if_highway=False, in_doc_words=2, highway_layers=1, char_lstm=True):
        super(WORD_REP, self).__init__()
        self.char_embedding_dim = char_embedding_dim
        self.char_hidden_dim = char_hidden_dim
        self.cnn_filter_num = cnn_filter_num
        self.char_size = char_size
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_size = vocab_size
        self.char_lstm = char_lstm
        self.if_highway = if_highway
        self.char_embeds = nn.Embedding(char_size, char_embedding_dim)
        self.word_embeds = nn.Embedding(vocab_size, word_embedding_dim)
        if char_lstm:
            self.crit_lm = nn.CrossEntropyLoss()
            self.forw_char_lstm = nn.LSTM(char_embedding_dim,
                char_hidden_dim, num_layers=char_lstm_layers, bidirectional
                =False, dropout=dropout_ratio)
            self.back_char_lstm = nn.LSTM(char_embedding_dim,
                char_hidden_dim, num_layers=char_lstm_layers, bidirectional
                =False, dropout=dropout_ratio)
            self.word_lstm_lm = nn.LSTM(word_embedding_dim + 
                char_hidden_dim * 2, word_hidden_dim // 2, num_layers=
                word_lstm_layers, bidirectional=True, dropout=dropout_ratio)
            self.char_pre_train_out = nn.Linear(char_hidden_dim, char_size)
            self.word_pre_train_out = nn.Linear(char_hidden_dim, in_doc_words)
            if self.if_highway:
                self.forw2char = highway_layer.hw(char_hidden_dim,
                    num_layers=highway_layers, dropout_ratio=dropout_ratio)
                self.back2char = highway_layer.hw(char_hidden_dim,
                    num_layers=highway_layers, dropout_ratio=dropout_ratio)
                self.forw2word = highway_layer.hw(char_hidden_dim,
                    num_layers=highway_layers, dropout_ratio=dropout_ratio)
                self.back2word = highway_layer.hw(char_hidden_dim,
                    num_layers=highway_layers, dropout_ratio=dropout_ratio)
                self.fb2char = highway_layer.hw(2 * char_hidden_dim,
                    num_layers=highway_layers, dropout_ratio=dropout_ratio)
        else:
            self.cnn = nn.Conv2d(1, cnn_filter_num, (3, char_embedding_dim),
                padding=(2, 0))
            self.word_lstm_cnn = nn.LSTM(word_embedding_dim +
                cnn_filter_num, word_hidden_dim // 2, num_layers=
                word_lstm_layers, bidirectional=True, dropout=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.batch_size = 1
        self.word_seq_length = 1

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding

        """
        assert pre_word_embeddings.size()[1] == self.word_embedding_dim
        self.word_embeds.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self):
        """
        random initialization

        args:
            init_char_embedding: random initialize char embedding or not

        """
        utils.init_embedding(self.char_embeds.weight)
        if self.char_lstm:
            utils.init_lstm(self.forw_char_lstm)
            utils.init_lstm(self.back_char_lstm)
            utils.init_lstm(self.word_lstm_lm)
            utils.init_linear(self.char_pre_train_out)
            utils.init_linear(self.word_pre_train_out)
            if self.if_highway:
                self.forw2char.rand_init()
                self.back2char.rand_init()
                self.forw2word.rand_init()
                self.back2word.rand_init()
                self.fb2char.rand_init()
        else:
            utils.init_lstm(self.word_lstm_cnn)

    def word_pre_train_forward(self, sentence, position):
        """
        output of forward language model

        args:
            sentence (char_seq_len, batch_size): char-level representation of sentence
            position (word_seq_len, batch_size): position of blank space in char-level representation of sentence

        """
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.forw_char_lstm(d_embeds)
        tmpsize = position.size()
        position = position.unsqueeze(2).expand(tmpsize[0], tmpsize[1],
            self.char_hidden_dim)
        select_lstm_out = torch.gather(lstm_out, 0, position)
        d_lstm_out = self.dropout(select_lstm_out).view(-1, self.
            char_hidden_dim)
        if self.if_highway:
            char_out = self.forw2word(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out
        pre_score = self.word_pre_train_out(d_char_out)
        return pre_score, hidden

    def word_pre_train_backward(self, sentence, position):
        """
        output of backward language model

        args:
            sentence (char_seq_len, batch_size): char-level representation of sentence (inverse order)
            position (word_seq_len, batch_size): position of blank space in inversed char-level representation of sentence

        """
        embeds = self.char_embeds(sentence)
        d_embeds = self.dropout(embeds)
        lstm_out, hidden = self.back_char_lstm(d_embeds)
        tmpsize = position.size()
        position = position.unsqueeze(2).expand(tmpsize[0], tmpsize[1],
            self.char_hidden_dim)
        select_lstm_out = torch.gather(lstm_out, 0, position)
        d_lstm_out = self.dropout(select_lstm_out).view(-1, self.
            char_hidden_dim)
        if self.if_highway:
            char_out = self.back2word(d_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = d_lstm_out
        pre_score = self.word_pre_train_out(d_char_out)
        return pre_score, hidden

    def lm_loss(self, f_f, f_p, b_f, b_p, w_f):
        """
        language model loss

        args:
            f_f (char_seq_len, batch_size) : char-level representation of sentence
            f_p (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
            b_f (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
            b_p (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
            w_f (word_seq_len, batch_size) : word-level representation of sentence

        """
        cf_p = f_p[0:-1, :].contiguous()
        cb_p = b_p[1:, :].contiguous()
        cf_y = w_f[1:, :].contiguous()
        cb_y = w_f[0:-1, :].contiguous()
        cfs, _ = self.word_pre_train_forward(f_f, cf_p)
        loss = self.crit_lm(cfs, cf_y.view(-1))
        cbs, _ = self.word_pre_train_backward(b_f, cb_p)
        loss = loss + self.crit_lm(cbs, cb_y.view(-1))
        return loss

    def cnn_lstm(self, word_seq, cnn_features):
        """
        return word representations with character-cnn

        args:
            word_seq:     word_seq_len, batch_size
            cnn_features: word_seq_len, batch_size, word_len

        """
        self.set_batch_seq_size(word_seq)
        cnn_features = cnn_features.view(cnn_features.size(0) *
            cnn_features.size(1), -1)
        cnn_features = self.char_embeds(cnn_features).view(cnn_features.
            size(0), 1, cnn_features.size(1), -1)
        cnn_features = self.cnn(cnn_features)
        d_char_out = nn.functional.max_pool2d(cnn_features, kernel_size=(
            cnn_features.size(2), 1)).view(self.word_seq_length, self.
            batch_size, self.cnn_filter_num)
        word_emb = self.word_embeds(word_seq)
        word_input = torch.cat((word_emb, d_char_out), dim=2)
        word_input = self.dropout(word_input)
        lstm_out, _ = self.word_lstm_cnn(word_input)
        lstm_out = self.dropout(lstm_out)
        return lstm_out

    def lm_lstm(self, forw_sentence, forw_position, back_sentence,
        back_position, word_seq):
        """
        return word representations with character-language-model

        args:
            forw_sentence (char_seq_len, batch_size) : char-level representation of sentence
            forw_position (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
            back_sentence (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
            back_position (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
            word_seq (word_seq_len, batch_size) : word-level representation of sentence

        """
        self.set_batch_seq_size(forw_position)
        forw_emb = self.char_embeds(forw_sentence)
        back_emb = self.char_embeds(back_sentence)
        d_f_emb = self.dropout(forw_emb)
        d_b_emb = self.dropout(back_emb)
        forw_lstm_out, _ = self.forw_char_lstm(d_f_emb)
        back_lstm_out, _ = self.back_char_lstm(d_b_emb)
        forw_position = forw_position.unsqueeze(2).expand(self.
            word_seq_length, self.batch_size, self.char_hidden_dim)
        select_forw_lstm_out = torch.gather(forw_lstm_out, 0, forw_position)
        back_position = back_position.unsqueeze(2).expand(self.
            word_seq_length, self.batch_size, self.char_hidden_dim)
        select_back_lstm_out = torch.gather(back_lstm_out, 0, back_position)
        fb_lstm_out = self.dropout(torch.cat((select_forw_lstm_out,
            select_back_lstm_out), dim=2))
        if self.if_highway:
            char_out = self.fb2char(fb_lstm_out)
            d_char_out = self.dropout(char_out)
        else:
            d_char_out = fb_lstm_out
        word_emb = self.word_embeds(word_seq)
        d_word_emb = self.dropout(word_emb)
        word_input = torch.cat((d_word_emb, d_char_out), dim=2)
        lstm_out, _ = self.word_lstm_lm(word_input)
        d_lstm_out = self.dropout(lstm_out)
        return d_lstm_out

    def forward(self, forw_sentence, forw_position, back_sentence,
        back_position, word_seq, cnn_features):
        """
        word representations

        args:
            forw_sentence (char_seq_len, batch_size) : char-level representation of sentence
            forw_position (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
            back_sentence (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
            back_position (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
            word_seq (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        """
        if self.char_lstm:
            return self.lm_lstm(forw_sentence, forw_position, back_sentence,
                back_position, word_seq)
        else:
            return self.cnn_lstm(word_seq, cnn_features)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ZhixiuYe_HSCRF_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(hw(*[], **{'size': 4}), [torch.rand([4, 4, 4, 4])], {})

