import sys
_module = sys.modules[__name__]
del sys
config = _module
data = _module
layers = _module
m_reader = _module
model = _module
predictor = _module
r_net = _module
rnn_reader = _module
interactive = _module
predict = _module
preprocess = _module
train = _module
spacy_tokenizer = _module
utils = _module
vector = _module

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


from torch.autograd import Variable


import math


import random


import torch.optim as optim


import numpy as np


import logging


import copy


class StackedBRNN(nn.Module):
    """Stacked Bi-directional RNNs.

    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM, concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))

    def forward(self, x, x_mask):
        """Encode either padded or non-padded sequences.

        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        if x_mask.data.sum() == 0 or x_mask.data.eq(1).long().sum(1).min() == 0:
            output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training:
            output = self._forward_padded(x, x_mask)
        else:
            output = self._forward_unpadded(x, x_mask)
        return output.contiguous()

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        x = x.transpose(0, 1)
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input, p=self.dropout_rate, training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return output

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise, encoding that handles
        padding.
        """
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)
        x = x.index_select(0, idx_sort)
        x = x.transpose(0, 1)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input, rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)
        if output.size(1) != x_mask.size(1):
            padding = torch.zeros(output.size(0), x_mask.size(1) - output.size(1), output.size(2)).type(output.data.type())
            output = torch.cat([output, Variable(padding)], 1)
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return output


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj


class PointerNetwork(nn.Module):

    def __init__(self, x_size, y_size, hidden_size, dropout_rate=0, cell_type=nn.GRUCell, normalize=True):
        super(PointerNetwork, self).__init__()
        self.normalize = normalize
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.linear = nn.Linear(x_size + y_size, hidden_size, bias=False)
        self.weights = nn.Linear(hidden_size, 1, bias=False)
        self.self_attn = NonLinearSeqAttn(y_size, hidden_size)
        self.cell = cell_type(x_size, y_size)

    def init_hiddens(self, y, y_mask):
        attn = self.self_attn(y, y_mask)
        res = attn.unsqueeze(1).bmm(y).squeeze(1)
        return res

    def pointer(self, x, state, x_mask):
        x_ = torch.cat([x, state.unsqueeze(1).repeat(1, x.size(1), 1)], 2)
        s0 = F.tanh(self.linear(x_))
        s = self.weights(s0).view(x.size(0), x.size(1))
        s.data.masked_fill_(x_mask.data, -float('inf'))
        a = F.softmax(s)
        res = a.unsqueeze(1).bmm(x).squeeze(1)
        if self.normalize:
            if self.training:
                scores = F.log_softmax(s)
            else:
                scores = F.softmax(s)
        else:
            scores = a.exp()
        return res, scores

    def forward(self, x, y, x_mask, y_mask):
        hiddens = self.init_hiddens(y, y_mask)
        c, start_scores = self.pointer(x, hiddens, x_mask)
        c_ = F.dropout(c, p=self.dropout_rate, training=self.training)
        hiddens = self.cell(c_, hiddens)
        c, end_scores = self.pointer(x, hiddens, x_mask)
        return start_scores, end_scores


class MemoryAnsPointer(nn.Module):

    def __init__(self, x_size, y_size, hidden_size, hop=1, dropout_rate=0, normalize=True):
        super(MemoryAnsPointer, self).__init__()
        self.normalize = normalize
        self.hidden_size = hidden_size
        self.hop = hop
        self.dropout_rate = dropout_rate
        self.FFNs_start = nn.ModuleList()
        self.SFUs_start = nn.ModuleList()
        self.FFNs_end = nn.ModuleList()
        self.SFUs_end = nn.ModuleList()
        for i in range(self.hop):
            self.FFNs_start.append(FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 1, dropout_rate))
            self.SFUs_start.append(SFU(y_size, 2 * hidden_size))
            self.FFNs_end.append(FeedForwardNetwork(x_size + y_size + 2 * hidden_size, hidden_size, 1, dropout_rate))
            self.SFUs_end.append(SFU(y_size, 2 * hidden_size))

    def forward(self, x, y, x_mask, y_mask):
        z_s = y[:, (-1), :].unsqueeze(1)
        z_e = None
        s = None
        e = None
        p_s = None
        p_e = None
        for i in range(self.hop):
            z_s_ = z_s.repeat(1, x.size(1), 1)
            s = self.FFNs_start[i](torch.cat([x, z_s_, x * z_s_], 2)).squeeze(2)
            s.data.masked_fill_(x_mask.data, -float('inf'))
            p_s = F.softmax(s, dim=1)
            u_s = p_s.unsqueeze(1).bmm(x)
            z_e = self.SFUs_start[i](z_s, u_s)
            z_e_ = z_e.repeat(1, x.size(1), 1)
            e = self.FFNs_end[i](torch.cat([x, z_e_, x * z_e_], 2)).squeeze(2)
            e.data.masked_fill_(x_mask.data, -float('inf'))
            p_e = F.softmax(e, dim=1)
            u_e = p_e.unsqueeze(1).bmm(x)
            z_s = self.SFUs_end[i](z_e, u_e)
        if self.normalize:
            if self.training:
                p_s = F.log_softmax(s, dim=1)
                p_e = F.log_softmax(e, dim=1)
            else:
                p_s = F.softmax(s, dim=1)
                p_e = F.softmax(e, dim=1)
        else:
            p_s = s.exp()
            p_e = e.exp()
        return p_s, p_e


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
        alpha = F.softmax(scores, dim=2)
        matched_seq = alpha.bmm(y)
        return matched_seq


class SelfAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.

    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """

    def __init__(self, input_size, identity=False, diag=True):
        super(SelfAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None
        self.diag = diag

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len1 * dim1
            x_mask: batch * len1 (1 for padding, 0 for true)
        Output:
            matched_seq: batch * len1 * dim1
        """
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
        else:
            x_proj = x
        scores = x_proj.bmm(x_proj.transpose(2, 1))
        if not self.diag:
            x_len = x.size(1)
            for i in range(x_len):
                scores[:, (i), (i)] = 0
        x_mask = x_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=2)
        matched_seq = alpha.bmm(x)
        return matched_seq


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
                alpha = F.log_softmax(xWy)
            else:
                alpha = F.softmax(xWy)
        else:
            alpha = xWy.exp()
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


class NonLinearSeqAttn(nn.Module):
    """Self attention over a sequence:

    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

    def __init__(self, input_size, hidden_size):
        super(NonLinearSeqAttn, self).__init__()
        self.FFN = FeedForwardNetwork(input_size, hidden_size, 1)

    def forward(self, x, x_mask):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        scores = self.FFN(x).squeeze(2)
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


class Gate(nn.Module):
    """Gate Unit
    g = sigmoid(Wx)
    x = g * x
    """

    def __init__(self, input_size):
        super(Gate, self).__init__()
        self.linear = nn.Linear(input_size, input_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: batch * len * dim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            res: batch * len * dim
        """
        x_proj = self.linear(x)
        gate = F.sigmoid(x)
        return x_proj * gate


class SFU(nn.Module):
    """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """

    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        self.linear_r = nn.Linear(input_size + fusion_size, input_size)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size)

    def forward(self, x, fusions):
        r_f = torch.cat([x, fusions], 2)
        r = F.tanh(self.linear_r(r_f))
        g = F.sigmoid(self.linear_g(r_f))
        o = g * r + (1 - g) * x
        return o


class MnemonicReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}

    def __init__(self, args, normalize=True):
        super(MnemonicReader, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(args.char_size, args.char_embedding_dim, padding_idx=0)
        self.char_rnn = layers.StackedBRNN(input_size=args.char_embedding_dim, hidden_size=args.char_hidden_size, num_layers=1, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=False, rnn_type=self.RNN_TYPES[args.rnn_type], padding=False)
        doc_input_size = args.embedding_dim + args.char_hidden_size * 2 + args.num_features
        self.encoding_rnn = layers.StackedBRNN(input_size=doc_input_size, hidden_size=args.hidden_size, num_layers=1, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=False, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        doc_hidden_size = 2 * args.hidden_size
        self.interactive_aligners = nn.ModuleList()
        self.interactive_SFUs = nn.ModuleList()
        self.self_aligners = nn.ModuleList()
        self.self_SFUs = nn.ModuleList()
        self.aggregate_rnns = nn.ModuleList()
        for i in range(args.hop):
            self.interactive_aligners.append(layers.SeqAttnMatch(doc_hidden_size, identity=True))
            self.interactive_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            self.self_aligners.append(layers.SelfAttnMatch(doc_hidden_size, identity=True, diag=False))
            self.self_SFUs.append(layers.SFU(doc_hidden_size, 3 * doc_hidden_size))
            self.aggregate_rnns.append(layers.StackedBRNN(input_size=doc_hidden_size, hidden_size=args.hidden_size, num_layers=1, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=False, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding))
        self.mem_ans_ptr = layers.MemoryAnsPointer(x_size=2 * args.hidden_size, y_size=2 * args.hidden_size, hidden_size=args.hidden_size, hop=args.hop, dropout_rate=args.dropout_rnn, normalize=normalize)

    def forward(self, x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_c = document char indices           [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_c = document char indices           [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        x1_c_emb = self.char_embedding(x1_c)
        x2_c_emb = self.char_embedding(x2_c)
        if self.args.dropout_emb > 0:
            x1_emb = F.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
            x2_emb = F.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
            x1_c_emb = F.dropout(x1_c_emb, p=self.args.dropout_emb, training=self.training)
            x2_c_emb = F.dropout(x2_c_emb, p=self.args.dropout_emb, training=self.training)
        x1_c_features = self.char_rnn(x1_c_emb.reshape((x1_c_emb.size(0) * x1_c_emb.size(1), x1_c_emb.size(2), x1_c_emb.size(3))), x1_mask.unsqueeze(2).repeat(1, 1, x1_c_emb.size(2)).reshape((x1_c_emb.size(0) * x1_c_emb.size(1), x1_c_emb.size(2)))).reshape((x1_c_emb.size(0), x1_c_emb.size(1), x1_c_emb.size(2), -1))[:, :, (-1), :]
        x2_c_features = self.char_rnn(x2_c_emb.reshape((x2_c_emb.size(0) * x2_c_emb.size(1), x2_c_emb.size(2), x2_c_emb.size(3))), x2_mask.unsqueeze(2).repeat(1, 1, x2_c_emb.size(2)).reshape((x2_c_emb.size(0) * x2_c_emb.size(1), x2_c_emb.size(2)))).reshape((x2_c_emb.size(0), x2_c_emb.size(1), x2_c_emb.size(2), -1))[:, :, (-1), :]
        crnn_input = [x1_emb, x1_c_features]
        qrnn_input = [x2_emb, x2_c_features]
        if self.args.num_features > 0:
            crnn_input.append(x1_f)
            qrnn_input.append(x2_f)
        c = self.encoding_rnn(torch.cat(crnn_input, 2), x1_mask)
        q = self.encoding_rnn(torch.cat(qrnn_input, 2), x2_mask)
        c_check = c
        for i in range(self.args.hop):
            q_tilde = self.interactive_aligners[i].forward(c_check, q, x2_mask)
            c_bar = self.interactive_SFUs[i].forward(c_check, torch.cat([q_tilde, c_check * q_tilde, c_check - q_tilde], 2))
            c_tilde = self.self_aligners[i].forward(c_bar, x1_mask)
            c_hat = self.self_SFUs[i].forward(c_bar, torch.cat([c_tilde, c_bar * c_tilde, c_bar - c_tilde], 2))
            c_check = self.aggregate_rnns[i].forward(c_hat, x1_mask)
        start_scores, end_scores = self.mem_ans_ptr.forward(c_check, q, x1_mask, x2_mask)
        return start_scores, end_scores


class R_Net(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}

    def __init__(self, args, normalize=True):
        super(R_Net, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        self.char_embedding = nn.Embedding(args.char_size, args.char_embedding_dim, padding_idx=0)
        self.char_rnn = layers.StackedBRNN(input_size=args.char_embedding_dim, hidden_size=args.char_hidden_size, num_layers=1, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=False, rnn_type=self.RNN_TYPES[args.rnn_type], padding=False)
        doc_input_size = args.embedding_dim + args.char_hidden_size * 2
        self.encode_rnn = layers.StackedBRNN(input_size=doc_input_size, hidden_size=args.hidden_size, num_layers=args.doc_layers, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=args.concat_rnn_layers, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers
        self.question_attn = layers.SeqAttnMatch(question_hidden_size, identity=False)
        self.question_attn_gate = layers.Gate(doc_hidden_size + question_hidden_size)
        self.question_attn_rnn = layers.StackedBRNN(input_size=doc_hidden_size + question_hidden_size, hidden_size=args.hidden_size, num_layers=1, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=False, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        question_attn_hidden_size = 2 * args.hidden_size
        self.doc_self_attn = layers.SelfAttnMatch(question_attn_hidden_size, identity=False)
        self.doc_self_attn_gate = layers.Gate(question_attn_hidden_size + question_attn_hidden_size)
        self.doc_self_attn_rnn = layers.StackedBRNN(input_size=question_attn_hidden_size + question_attn_hidden_size, hidden_size=args.hidden_size, num_layers=1, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=False, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        doc_self_attn_hidden_size = 2 * args.hidden_size
        self.doc_self_attn_rnn2 = layers.StackedBRNN(input_size=doc_self_attn_hidden_size, hidden_size=args.hidden_size, num_layers=1, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=False, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        self.ptr_net = layers.PointerNetwork(x_size=doc_self_attn_hidden_size, y_size=question_hidden_size, hidden_size=args.hidden_size, dropout_rate=args.dropout_rnn, cell_type=nn.GRUCell, normalize=normalize)

    def forward(self, x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_c = document char indices           [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_c = document char indices           [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        x1_c_emb = self.char_embedding(x1_c)
        x2_c_emb = self.char_embedding(x2_c)
        if self.args.dropout_emb > 0:
            x1_emb = F.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
            x2_emb = F.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
            x1_c_emb = F.dropout(x1_c_emb, p=self.args.dropout_emb, training=self.training)
            x2_c_emb = F.dropout(x2_c_emb, p=self.args.dropout_emb, training=self.training)
        x1_c_features = self.char_rnn(x1_c_emb.reshape((x1_c_emb.size(0) * x1_c_emb.size(1), x1_c_emb.size(2), x1_c_emb.size(3))), x1_mask.unsqueeze(2).repeat(1, 1, x1_c_emb.size(2)).reshape((x1_c_emb.size(0) * x1_c_emb.size(1), x1_c_emb.size(2)))).reshape((x1_c_emb.size(0), x1_c_emb.size(1), x1_c_emb.size(2), -1))[:, :, (-1), :]
        x2_c_features = self.char_rnn(x2_c_emb.reshape((x2_c_emb.size(0) * x2_c_emb.size(1), x2_c_emb.size(2), x2_c_emb.size(3))), x2_mask.unsqueeze(2).repeat(1, 1, x2_c_emb.size(2)).reshape((x2_c_emb.size(0) * x2_c_emb.size(1), x2_c_emb.size(2)))).reshape((x2_c_emb.size(0), x2_c_emb.size(1), x2_c_emb.size(2), -1))[:, :, (-1), :]
        crnn_input = [x1_emb, x1_c_features]
        qrnn_input = [x2_emb, x2_c_features]
        c = self.encode_rnn(torch.cat(crnn_input, 2), x1_mask)
        q = self.encode_rnn(torch.cat(qrnn_input, 2), x2_mask)
        question_attn_hiddens = self.question_attn(c, q, x2_mask)
        rnn_input = self.question_attn_gate(torch.cat([c, question_attn_hiddens], 2))
        c = self.question_attn_rnn(rnn_input, x1_mask)
        doc_self_attn_hiddens = self.doc_self_attn(c, x1_mask)
        rnn_input = self.doc_self_attn_gate(torch.cat([c, doc_self_attn_hiddens], 2))
        c = self.doc_self_attn_rnn(rnn_input, x1_mask)
        c = self.doc_self_attn_rnn2(c, x1_mask)
        start_scores, end_scores = self.ptr_net(c, q, x1_mask, x2_mask)
        return start_scores, end_scores


class RnnDocReader(nn.Module):
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
    CELL_TYPES = {'lstm': nn.LSTMCell, 'gru': nn.GRUCell, 'rnn': nn.RNNCell}

    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        self.args = args
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_dim, padding_idx=0)
        if args.use_qemb:
            self.qemb_match = layers.SeqAttnMatch(args.embedding_dim)
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb:
            doc_input_size += args.embedding_dim
        self.doc_rnn = layers.StackedBRNN(input_size=doc_input_size, hidden_size=args.hidden_size, num_layers=args.doc_layers, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=args.concat_rnn_layers, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        self.question_rnn = layers.StackedBRNN(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.question_layers, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=args.concat_rnn_layers, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding)
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers
        if args.question_merge not in ['avg', 'self_attn']:
            raise NotImplementedError('merge_mode = %s' % args.merge_mode)
        if args.question_merge == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)
        self.start_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size, normalize=normalize)
        self.end_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size, normalize=normalize)

    def forward(self, x1, x1_c, x1_f, x1_mask, x2, x2_c, x2_f, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        if self.args.dropout_emb > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.args.dropout_emb, training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
        drnn_input = [x1_emb]
        if self.args.use_qemb:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input.append(x2_weighted_emb)
        if self.args.num_features > 0:
            drnn_input.append(x1_f)
        doc_hiddens = self.doc_rnn(torch.cat(drnn_input, 2), x1_mask)
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.args.question_merge == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.args.question_merge == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (FeedForwardNetwork,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Gate,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SFU,
     lambda: ([], {'input_size': 4, 'fusion_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (StackedBRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_HKUST_KnowComp_MnemonicReader(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

