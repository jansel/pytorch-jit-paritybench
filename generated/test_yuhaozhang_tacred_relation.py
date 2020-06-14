import sys
_module = sys.modules[__name__]
del sys
loader = _module
ensemble = _module
eval = _module
layers = _module
rnn = _module
prepare_vocab = _module
train = _module
constant = _module
helper = _module
scorer = _module
torch_utils = _module
vocab = _module

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


import random


import torch


import torch.nn as nn


import torch.optim as optim


from torch import nn


from torch.nn import init


import torch.nn.functional as F


import math


import numpy as np


import time


class LSTMLayer(nn.Module):
    """ A wrapper for LSTM with sequence packing. """

    def __init__(self, emb_dim, hidden_dim, num_layers, dropout, use_cuda):
        super(LSTMLayer, self).__init__()
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=
            True, dropout=dropout)
        self.use_cuda = use_cuda

    def forward(self, x, x_mask, init_state):
        """
        x: batch_size * feature_size * seq_len
        x_mask : batch_size * seq_len
        """
        x_lens = x_mask.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lens = list(x_lens[idx_sort])
        x = x.index_select(0, idx_sort)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True
            )
        rnn_output, (ht, ct) = self.rnn(rnn_input, init_state)
        rnn_output = nn.utils.rnn.pad_packed_sequence(rnn_output,
            batch_first=True)[0]
        rnn_output = rnn_output.index_select(0, idx_unsort)
        ht = ht.index_select(0, idx_unsort)
        ct = ct.index_select(0, idx_unsort)
        return rnn_output, (ht, ct)


class PositionAwareAttention(nn.Module):
    """
    A position-augmented attention layer where the attention weight is
    a = T' . tanh(Ux + Vq + Wf)
    where x is the input, q is the query, and f is additional position features.
    """

    def __init__(self, input_size, query_size, feature_size, attn_size):
        super(PositionAwareAttention, self).__init__()
        self.input_size = input_size
        self.query_size = query_size
        self.feature_size = feature_size
        self.attn_size = attn_size
        self.ulinear = nn.Linear(input_size, attn_size)
        self.vlinear = nn.Linear(query_size, attn_size, bias=False)
        if feature_size > 0:
            self.wlinear = nn.Linear(feature_size, attn_size, bias=False)
        else:
            self.wlinear = None
        self.tlinear = nn.Linear(attn_size, 1)
        self.init_weights()

    def init_weights(self):
        self.ulinear.weight.data.normal_(std=0.001)
        self.vlinear.weight.data.normal_(std=0.001)
        if self.wlinear is not None:
            self.wlinear.weight.data.normal_(std=0.001)
        self.tlinear.weight.data.zero_()

    def forward(self, x, x_mask, q, f):
        """
        x : batch_size * seq_len * input_size
        q : batch_size * query_size
        f : batch_size * seq_len * feature_size
        """
        batch_size, seq_len, _ = x.size()
        x_proj = self.ulinear(x.contiguous().view(-1, self.input_size)).view(
            batch_size, seq_len, self.attn_size)
        q_proj = self.vlinear(q.view(-1, self.query_size)).contiguous().view(
            batch_size, self.attn_size).unsqueeze(1).expand(batch_size,
            seq_len, self.attn_size)
        if self.wlinear is not None:
            f_proj = self.wlinear(f.view(-1, self.feature_size)).contiguous(
                ).view(batch_size, seq_len, self.attn_size)
            projs = [x_proj, q_proj, f_proj]
        else:
            projs = [x_proj, q_proj]
        scores = self.tlinear(torch.tanh(sum(projs)).view(-1, self.attn_size)
            ).view(batch_size, seq_len)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        weights = F.softmax(scores, dim=1)
        outputs = weights.unsqueeze(1).bmm(x).squeeze(1)
        return outputs


class PositionAwareRNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'],
            padding_idx=constant.PAD_ID)
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt[
                'pos_dim'], padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt[
                'ner_dim'], padding_idx=constant.PAD_ID)
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'],
            batch_first=True, dropout=opt['dropout'])
        self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])
        if opt['attn']:
            self.attn_layer = layers.PositionAwareAttention(opt[
                'hidden_dim'], opt['hidden_dim'], 2 * opt['pe_dim'], opt[
                'attn_dim'])
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])
        self.opt = opt
        self.topn = self.opt.get('topn', 10000000000.0)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.init_weights()

    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1)
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)
        if self.topn <= 0:
            None
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            None
            self.emb.weight.register_hook(lambda x: torch_utils.
                keep_partial_grad(x, self.topn))
        else:
            None

    def zero_state(self, batch_size):
        state_shape = self.opt['num_layers'], batch_size, self.opt['hidden_dim'
            ]
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0, c0
        else:
            return h0, c0

    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]
        inputs = self.drop(torch.cat(inputs, dim=2))
        input_size = inputs.size(2)
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens,
            batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs,
            batch_first=True)
        hidden = self.drop(ht[(-1), :, :])
        outputs = self.drop(outputs)
        if self.opt['attn']:
            subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
            obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
            pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
            final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
        else:
            final_hidden = hidden
        logits = self.linear(final_hidden)
        return logits, final_hidden


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_yuhaozhang_tacred_relation(_paritybench_base):
    pass
