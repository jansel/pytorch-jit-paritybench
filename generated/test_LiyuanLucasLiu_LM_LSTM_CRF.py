import sys
_module = sys.modules[__name__]
del sys
conf = _module
eval_w = _module
eval_wc = _module
model = _module
crf = _module
evaluator = _module
highway = _module
lm_lstm_crf = _module
lstm_crf = _module
ner_dataset = _module
predictor = _module
utils = _module
seq_w = _module
seq_wc = _module
train_w = _module
train_wc = _module

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


import time


import torch


import torch.autograd as autograd


import torch.nn as nn


import torch.optim as optim


import itertools


import functools


import torch.sparse as sparse


import numpy as np


from functools import reduce


import torch.nn.init


class CRF_L(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Ma et al. 2016, has more parameters than CRF_S

    args:
        hidden_dim : input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans
    """

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_L, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size * self.
            tagset_size, bias=if_bias)

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)

    def forward(self, feats):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer (batch_size, seq_len, tag_size, tag_size)
        """
        return self.hidden2tag(feats).view(-1, self.tagset_size, self.
            tagset_size)


class CRF_S(nn.Module):
    """Conditional Random Field (CRF) layer. This version is used in Lample et al. 2016, has less parameters than CRF_L.

    args:
        hidden_dim: input dim size
        tagset_size: target_set_size
        if_biase: whether allow bias in linear trans

    """

    def __init__(self, hidden_dim, tagset_size, if_bias=True):
        super(CRF_S, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self
            .tagset_size))

    def rand_init(self):
        """random initialization
        """
        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        args:
            feats (batch_size, seq_len, hidden_dim) : input score from previous layers
        return:
            output from crf layer ( (batch_size * seq_len), tag_size, tag_size)
        """
        scores = self.hidden2tag(feats).view(-1, self.tagset_size, 1)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size
            ) + self.transitions.view(1, self.tagset_size, self.tagset_size
            ).expand(ins_num, self.tagset_size, self.tagset_size)
        return crf_scores


class CRFLoss_gd(nn.Module):
    """loss for greedy decode loss, i.e., although its for CRF Layer, we calculate the loss as

    .. math::
        \\sum_{j=1}^n \\log (p(\\hat{y}_{j+1}|z_{j+1}, \\hat{y}_{j}))

    instead of

    .. math::
        \\sum_{j=1}^n \\log (\\phi(\\hat{y}_{j-1}, \\hat{y}_j, \\mathbf{z}_j)) - \\log (\\sum_{\\mathbf{y}' \\in \\mathbf{Y}(\\mathbf{Z})} \\prod_{j=1}^n \\phi(y'_{j-1}, y'_j, \\mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_gd, self).__init__()
        self.tagset_size = tagset_size
        self.average_batch = average_batch
        self.crit = nn.CrossEntropyLoss(size_average=self.average_batch)

    def forward(self, scores, target, current):
        """
        args:
            scores (Word_Seq_len, Batch_size, target_size_from, target_size_to): crf scores
            target (Word_Seq_len, Batch_size): golden list
            current (Word_Seq_len, Batch_size): current state
        return:
            crf greedy loss
        """
        ins_num = current.size(0)
        current = current.expand(ins_num, 1, self.tagset_size)
        scores = scores.view(ins_num, self.tagset_size, self.tagset_size)
        current_score = torch.gather(scores, 1, current).squeeze()
        return self.crit(current_score, target)


class CRFLoss_vb(nn.Module):
    """loss for viterbi decode

    .. math::
        \\sum_{j=1}^n \\log (\\phi(\\hat{y}_{j-1}, \\hat{y}_j, \\mathbf{z}_j)) - \\log (\\sum_{\\mathbf{y}' \\in \\mathbf{Y}(\\mathbf{Z})} \\prod_{j=1}^n \\phi(y'_{j-1}, y'_j, \\mathbf{z}_j) )

    args:
        tagset_size: target_set_size
        start_tag: ind for <start>
        end_tag: ind for <pad>
        average_batch: whether average the loss among batch

    """

    def __init__(self, tagset_size, start_tag, end_tag, average_batch=True):
        super(CRFLoss_vb, self).__init__()
        self.tagset_size = tagset_size
        self.start_tag = start_tag
        self.end_tag = end_tag
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        args:
            scores (seq_len, bat_size, target_size_from, target_size_to) : crf scores
            target (seq_len, bat_size, 1) : golden state
            mask (size seq_len, bat_size) : mask for padding
        return:
            loss
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target
            ).view(seq_len, bat_size)
        tg_energy = tg_energy.masked_select(mask).sum()
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
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
        if self.average_batch:
            loss = (partition - tg_energy) / bat_size
        else:
            loss = partition - tg_energy
        return loss


class hw(nn.Module):
    """Highway layers

    args: 
        size: input and output dimension
        dropout_ratio: dropout ratio
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
        return:
            output tensor (ins_num, hidden_dim)
        """
        g = nn.functional.sigmoid(self.gate[0](x))
        h = nn.functional.relu(self.trans[0](x))
        x = g * h + (1 - g) * x
        for i in range(1, self.num_layers):
            x = self.dropout(x)
            g = nn.functional.sigmoid(self.gate[i](x))
            h = nn.functional.relu(self.trans[i](x))
            x = g * h + (1 - g) * x
        return x


class LM_LSTM_CRF(nn.Module):
    """LM_LSTM_CRF model

    args:
        tagset_size: size of label set
        char_size: size of char dictionary
        char_dim: size of char embedding
        char_hidden_dim: size of char-level lstm hidden dim
        char_rnn_layers: number of char-level lstm layers
        embedding_dim: size of word embedding
        word_hidden_dim: size of word-level blstm hidden dim
        word_rnn_layers: number of word-level lstm layers
        vocab_size: size of word dictionary
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
        if_highway: use highway layers or not
        in_doc_words: number of words that occurred in the corpus (used for language model prediction)
        highway_layers: number of highway layers
    """

    def __init__(self, tagset_size, char_size, char_dim, char_hidden_dim,
        char_rnn_layers, embedding_dim, word_hidden_dim, word_rnn_layers,
        vocab_size, dropout_ratio, large_CRF=True, if_highway=False,
        in_doc_words=2, highway_layers=1):
        super(LM_LSTM_CRF, self).__init__()
        self.char_dim = char_dim
        self.char_hidden_dim = char_hidden_dim
        self.char_size = char_size
        self.word_dim = embedding_dim
        self.word_hidden_dim = word_hidden_dim
        self.word_size = vocab_size
        self.if_highway = if_highway
        self.char_embeds = nn.Embedding(char_size, char_dim)
        self.forw_char_lstm = nn.LSTM(char_dim, char_hidden_dim, num_layers
            =char_rnn_layers, bidirectional=False, dropout=dropout_ratio)
        self.back_char_lstm = nn.LSTM(char_dim, char_hidden_dim, num_layers
            =char_rnn_layers, bidirectional=False, dropout=dropout_ratio)
        self.char_rnn_layers = char_rnn_layers
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.word_lstm = nn.LSTM(embedding_dim + char_hidden_dim * 2, 
            word_hidden_dim // 2, num_layers=word_rnn_layers, bidirectional
            =True, dropout=dropout_ratio)
        self.word_rnn_layers = word_rnn_layers
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.tagset_size = tagset_size
        if large_CRF:
            self.crf = crf.CRF_L(word_hidden_dim, tagset_size)
        else:
            self.crf = crf.CRF_S(word_hidden_dim, tagset_size)
        if if_highway:
            self.forw2char = highway.hw(char_hidden_dim, num_layers=
                highway_layers, dropout_ratio=dropout_ratio)
            self.back2char = highway.hw(char_hidden_dim, num_layers=
                highway_layers, dropout_ratio=dropout_ratio)
            self.forw2word = highway.hw(char_hidden_dim, num_layers=
                highway_layers, dropout_ratio=dropout_ratio)
            self.back2word = highway.hw(char_hidden_dim, num_layers=
                highway_layers, dropout_ratio=dropout_ratio)
            self.fb2char = highway.hw(2 * char_hidden_dim, num_layers=
                highway_layers, dropout_ratio=dropout_ratio)
        self.char_pre_train_out = nn.Linear(char_hidden_dim, char_size)
        self.word_pre_train_out = nn.Linear(char_hidden_dim, in_doc_words)
        self.batch_size = 1
        self.word_seq_length = 1

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def rand_init_embedding(self):
        """
        random initialize char-level embedding
        """
        utils.init_embedding(self.char_embeds.weight)

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert pre_word_embeddings.size()[1] == self.word_dim
        self.word_embeds.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self, init_char_embedding=True, init_word_embedding=False):
        """
        random initialization

        args:
            init_char_embedding: random initialize char embedding or not
            init_word_embedding: random initialize word embedding or not
        """
        if init_char_embedding:
            utils.init_embedding(self.char_embeds.weight)
        if init_word_embedding:
            utils.init_embedding(self.word_embeds.weight)
        if self.if_highway:
            self.forw2char.rand_init()
            self.back2char.rand_init()
            self.forw2word.rand_init()
            self.back2word.rand_init()
            self.fb2char.rand_init()
        utils.init_lstm(self.forw_char_lstm)
        utils.init_lstm(self.back_char_lstm)
        utils.init_lstm(self.word_lstm)
        utils.init_linear(self.char_pre_train_out)
        utils.init_linear(self.word_pre_train_out)
        self.crf.rand_init()

    def word_pre_train_forward(self, sentence, position, hidden=None):
        """
        output of forward language model

        args:
            sentence (char_seq_len, batch_size): char-level representation of sentence
            position (word_seq_len, batch_size): position of blank space in char-level representation of sentence
            hidden: initial hidden state

        return:
            language model output (word_seq_len, in_doc_word), hidden
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

    def word_pre_train_backward(self, sentence, position, hidden=None):
        """
        output of backward language model

        args:
            sentence (char_seq_len, batch_size): char-level representation of sentence (inverse order)
            position (word_seq_len, batch_size): position of blank space in inversed char-level representation of sentence
            hidden: initial hidden state

        return:
            language model output (word_seq_len, in_doc_word), hidden
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

    def forward(self, forw_sentence, forw_position, back_sentence,
        back_position, word_seq, hidden=None):
        """
        args:
            forw_sentence (char_seq_len, batch_size) : char-level representation of sentence
            forw_position (word_seq_len, batch_size) : position of blank space in char-level representation of sentence
            back_sentence (char_seq_len, batch_size) : char-level representation of sentence (inverse order)
            back_position (word_seq_len, batch_size) : position of blank space in inversed char-level representation of sentence
            word_seq (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
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
        lstm_out, _ = self.word_lstm(word_input)
        d_lstm_out = self.dropout(lstm_out)
        crf_out = self.crf(d_lstm_out)
        crf_out = crf_out.view(self.word_seq_length, self.batch_size, self.
            tagset_size, self.tagset_size)
        return crf_out


class LSTM_CRF(nn.Module):
    """LSTM_CRF model

    args: 
        vocab_size: size of word dictionary
        tagset_size: size of label set
        embedding_dim: size of word embedding
        hidden_dim: size of word-level blstm hidden dim
        rnn_layers: number of word-level lstm layers
        dropout_ratio: dropout ratio
        large_CRF: use CRF_L or not, refer model.crf.CRF_L and model.crf.CRF_S for more details
    """

    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim,
        rnn_layers, dropout_ratio, large_CRF=True):
        super(LSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=
            rnn_layers, bidirectional=True, dropout=dropout_ratio)
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout_ratio)
        self.dropout2 = nn.Dropout(p=dropout_ratio)
        self.tagset_size = tagset_size
        if large_CRF:
            self.crf = crf.CRF_L(hidden_dim, tagset_size)
        else:
            self.crf = crf.CRF_S(hidden_dim, tagset_size)
        self.batch_size = 1
        self.seq_length = 1

    def rand_init_hidden(self):
        """
        random initialize hidden variable
        """
        return autograd.Variable(torch.randn(2 * self.rnn_layers, self.
            batch_size, self.hidden_dim // 2)), autograd.Variable(torch.
            randn(2 * self.rnn_layers, self.batch_size, self.hidden_dim // 2))

    def set_batch_size(self, bsize):
        """
        set batch size
        """
        self.batch_size = bsize

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_embedding(self, pre_embeddings):
        """
        load pre-trained word embedding

        args:
            pre_word_embeddings (self.word_size, self.word_dim) : pre-trained embedding
        """
        assert pre_embeddings.size()[1] == self.embedding_dim
        self.word_embeds.weight = nn.Parameter(pre_embeddings)

    def rand_init_embedding(self):
        utils.init_embedding(self.word_embeds.weight)

    def rand_init(self, init_embedding=False):
        """
        random initialization

        args:
            init_embedding: random initialize embedding or not
        """
        if init_embedding:
            utils.init_embedding(self.word_embeds.weight)
        utils.init_lstm(self.lstm)
        self.crf.rand_init()

    def forward(self, sentence, hidden=None):
        """
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        """
        self.set_batch_seq_size(sentence)
        embeds = self.word_embeds(sentence)
        d_embeds = self.dropout1(embeds)
        lstm_out, hidden = self.lstm(d_embeds, hidden)
        lstm_out = lstm_out.view(-1, self.hidden_dim)
        d_lstm_out = self.dropout2(lstm_out)
        crf_out = self.crf(d_lstm_out)
        crf_out = crf_out.view(self.seq_length, self.batch_size, self.
            tagset_size, self.tagset_size)
        return crf_out, hidden


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_LiyuanLucasLiu_LM_LSTM_CRF(_paritybench_base):
    pass
    def test_000(self):
        self._check(CRF_L(*[], **{'hidden_dim': 4, 'tagset_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(CRF_S(*[], **{'hidden_dim': 4, 'tagset_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(hw(*[], **{'size': 4}), [torch.rand([4, 4, 4, 4])], {})

