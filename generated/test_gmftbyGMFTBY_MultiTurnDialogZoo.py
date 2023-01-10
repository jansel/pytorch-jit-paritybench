import sys
_module = sys.modules[__name__]
del sys
master = _module
empchat_process = _module
plato_process = _module
data_loader = _module
eval = _module
metric = _module
bleu = _module
bleu_scorer = _module
DSHRED = _module
DSHRED_RA = _module
GatedGCN = _module
HRAN = _module
HRAN_ablation = _module
HRED = _module
KgCVAE = _module
MReCoSa = _module
MReCoSa_RA = _module
MTGAT = _module
MTGCN = _module
VHRED = _module
WSeq = _module
WSeq_RA = _module
model = _module
layers = _module
seq2seq_attention = _module
seq2seq_gpt2 = _module
seq2seq_multi_head_attention = _module
seq2seq_transformer = _module
train = _module
translate = _module
utils = _module

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


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import torch.nn as nn


import numpy as np


import random


import torch.nn.functional as F


import torch.nn.init as init


import math


from collections import Counter


import types


from torch.nn.utils import clip_grad_norm_


from torch.optim import lr_scheduler


import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter


import re


from scipy.linalg import norm


class DSUtterance_encoder(nn.Module):
    """
    Bidirectional GRU
    """

    def __init__(self, input_size, embedding_size, hidden_size, dropout=0.5, n_layer=1, pretrained=None):
        super(DSUtterance_encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=n_layer, dropout=0 if n_layer == 1 else dropout, bidirectional=True)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, lengths, hidden=None):
        embedded = self.embed(inpt)
        if not hidden:
            hidden = torch.randn(self.n_layer * 2, len(lengths), self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        output, hidden = self.gru(embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = hidden.sum(axis=0)
        hidden = torch.tanh(hidden)
        output = torch.tanh(output)
        return output, hidden


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1.0 / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class DSContext_encoder(nn.Module):
    """
    input_size is 2 * utterance_hidden_size
    """

    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(DSContext_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, bidirectional=True)
        self.attn = Attention(hidden_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, hidden=None):
        if not hidden:
            hidden = torch.randn(2, inpt.shape[1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
        output, hidden = self.gru(inpt, hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        static_attn = self.attn(output[0].unsqueeze(0), output)
        static_attn = static_attn.bmm(output.transpose(0, 1))
        static_attn = static_attn.transpose(0, 1)
        hidden = torch.tanh(hidden)
        return static_attn, output, hidden


class DSDecoder(nn.Module):
    """
    Max likelyhood for decoding the utterance
    input_size is the size of the input vocabulary

    Attention module should satisfy that the decoder_hidden size is the same as 
    the Context encoder hidden size
    """

    def __init__(self, utter_hidden, context_hidden, output_size, embed_size, hidden_size, n_layer=2, dropout=0.5, pretrained=None):
        super(DSDecoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size + self.hidden_size * 2, self.hidden_size, num_layers=n_layer, dropout=0 if n_layer == 1 else dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)
        self.word_level_attn = Attention(hidden_size)
        self.context_encoder = DSContext_encoder(utter_hidden, context_hidden, dropout=dropout)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, last_hidden, encoder_outputs):
        embedded = self.embed(inpt).unsqueeze(0)
        key = last_hidden.mean(axis=0)
        context_output = []
        for turn in encoder_outputs:
            word_attn_weights = self.word_level_attn(key, turn)
            context = word_attn_weights.bmm(turn.transpose(0, 1))
            context = context.transpose(0, 1).squeeze(0)
            context_output.append(context)
        context_output = torch.stack(context_output)
        static_attn, context_output, hidden = self.context_encoder(context_output)
        attn_weights = self.attn(key, context_output)
        context = attn_weights.bmm(context_output.transpose(0, 1))
        context = context.transpose(0, 1)
        rnn_input = torch.cat([embedded, context, static_attn], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden


class DSHRED(nn.Module):

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(DSHRED, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = DSUtterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.context_encoder = DSContext_encoder(utter_hidden, context_hidden, dropout=dropout)
        self.decoder = DSDecoder(output_size, embed_size, decoder_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        for i in range(turn_size):
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        static_attn, context_output, hidden = self.context_encoder(turns)
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output, static_attn)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output, static_attn)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            for i in range(turn_size):
                hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)
            static_attn, context_output, hidden = self.context_encoder(turns)
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            try:
                for i in range(1, maxlen):
                    output, hidden = self.decoder(output, hidden, context_output, static_attn)
                    floss[i] = output
                    output = output.max(1)[1]
                    outputs[i] = output
            except:
                ipdb.set_trace()
            if loss:
                return outputs, floss
            else:
                return outputs


class DSHRED_RA(nn.Module):

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(DSHRED_RA, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_n_layer = utter_n_layer
        self.hidden_size = decoder_hidden
        self.utter_encoder = DSUtterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.decoder = DSDecoder(utter_hidden, context_hidden, output_size, embed_size, decoder_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        turns_output = []
        for i in range(turn_size):
            output, hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
            turns_output.append(output)
        turns = torch.stack(turns)
        hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = hidden
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            turns_output = []
            for i in range(turn_size):
                output, hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
                turns_output.append(output)
            turns = torch.stack(turns)
            hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            try:
                for i in range(1, maxlen):
                    output, hidden = self.decoder(output, hidden, turns_output)
                    floss[i] = output
                    output = output.max(1)[1]
                    outputs[i] = output
            except:
                ipdb.set_trace()
            if loss:
                return outputs, floss
            else:
                return outputs


class Utterance_encoder_ggcn(nn.Module):
    """
    Bidirectional GRU
    """

    def __init__(self, input_size, embedding_size, hidden_size, dropout=0.5, n_layer=1, pretrained=False):
        super(Utterance_encoder_ggcn, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=n_layer, dropout=dropout, bidirectional=True)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, lengths, hidden=None):
        embedded = self.embed(inpt)
        if not hidden:
            hidden = torch.randn(self.n_layer * 2, len(lengths), self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        _, hidden = self.gru(embedded, hidden)
        hidden = hidden.sum(axis=0)
        hidden = torch.tanh(hidden)
        return hidden


class GatedGCNContext(nn.Module):
    """
    GCN Context encoder

    It should be noticed that PyG merges all the subgraph in the batch into a big graph
    which is a sparse block diagonal adjacency matrices.
    Refer: Mini-batches in 
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

    Our implementation is the three layers GCN with the position embedding
    
    ========== Make sure the inpt_size == output_size ==========
    """

    def __init__(self, inpt_size, output_size, user_embed_size, posemb_size, dropout=0.5, threshold=2):
        super(GatedGCNContext, self).__init__()
        size = inpt_size + user_embed_size + posemb_size
        self.threshold = threshold
        self.kernel_rnn1 = nn.GRUCell(size, size)
        self.kernel_rnn2 = nn.GRUCell(size, size)
        self.conv1 = My_DoubleGatedGCN(size, inpt_size, self.kernel_rnn1, self.kernel_rnn2)
        self.conv2 = My_DoubleGatedGCN(size, inpt_size, self.kernel_rnn1, self.kernel_rnn2)
        self.conv3 = My_DoubleGatedGCN(size, inpt_size, self.kernel_rnn1, self.kernel_rnn2)
        self.layer_norm = nn.LayerNorm(inpt_size)
        self.rnn = nn.GRU(inpt_size + user_embed_size, inpt_size, bidirectional=True)
        self.linear1 = nn.Linear(inpt_size * 2, inpt_size)
        self.linear2 = nn.Linear(inpt_size * 2, output_size)
        self.drop = nn.Dropout(p=dropout)
        self.posemb = nn.Embedding(100, posemb_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.kernel_rnn1.weight_hh)
        init.xavier_normal_(self.kernel_rnn1.weight_ih)
        self.kernel_rnn1.bias_ih.data.fill_(0.0)
        self.kernel_rnn1.bias_hh.data.fill_(0.0)
        init.xavier_normal_(self.kernel_rnn2.weight_hh)
        init.xavier_normal_(self.kernel_rnn2.weight_ih)
        self.kernel_rnn2.bias_ih.data.fill_(0.0)
        self.kernel_rnn2.bias_hh.data.fill_(0.0)
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def create_batch(self, gbatch, utter_hidden):
        """create one graph batch
        :param: gbatch [batch_size, ([2, edge_num], [edge_num])]
        :param: utter_hidden [turn_len(node), batch, hidden_size]"""
        utter_hidden = utter_hidden.permute(1, 0, 2)
        batch_size = len(utter_hidden)
        data_list, weights = [], []
        for idx, example in enumerate(gbatch):
            edge_index, edge_w = example
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_w = torch.tensor(edge_w, dtype=torch.float)
            data_list.append(Data(x=utter_hidden[idx], edge_index=edge_index))
            weights.append(edge_w)
        loader = DataLoader(data_list, batch_size=batch_size)
        batch = list(loader)
        assert len(batch) == 1
        batch = batch[0]
        weights = torch.cat(weights)
        return batch, weights

    def forward(self, gbatch, utter_hidden, ub):
        rnn_x, rnnh = self.rnn(torch.cat([utter_hidden, ub], dim=-1))
        rnn_x = torch.tanh(self.linear1(rnn_x))
        turn_size = utter_hidden.size(0)
        rnnh = torch.tanh(rnnh.sum(axis=0))
        if turn_size <= self.threshold:
            return rnn_x, rnnh
        batch, weights = self.create_batch(gbatch, rnn_x)
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        batch_size = torch.max(batch).item() + 1
        pos = []
        for i in range(batch_size):
            pos.append(torch.arange(turn_size, dtype=torch.long))
        pos = torch.cat(pos)
        ub = ub.reshape(-1, ub.size(-1))
        if torch.cuda.is_available():
            x = x
            edge_index = edge_index
            batch = batch
            weights = weights
            pos = pos
        pos = self.posemb(pos)
        x = torch.cat([x, pos, ub], dim=1)
        x1 = torch.tanh(self.conv1(x, edge_index, edge_weight=weights))
        x1_ = torch.cat([x1, pos, ub], dim=1)
        x2 = torch.tanh(self.conv2(x1_, edge_index, edge_weight=weights))
        x2_ = torch.cat([x2, pos, ub], dim=1)
        x3 = torch.tanh(self.conv3(x2_, edge_index, edge_weight=weights))
        x = x1 + x2 + x3
        x = self.layer_norm(self.drop(torch.tanh(x)))
        x = torch.stack(x.chunk(batch_size, dim=0)).permute(1, 0, 2)
        x = torch.cat([rnn_x, x], dim=2)
        x = torch.tanh(self.linear2(x))
        return x, rnnh


class Decoder_ggcn(nn.Module):

    def __init__(self, output_size, embed_size, hidden_size, user_embed_size=10, n_layer=2, dropout=0.5, pretrained=None):
        super(Decoder_ggcn, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size, num_layers=n_layer, dropout=0 if n_layer == 1 else dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, last_hidden, gcncontext):
        embedded = self.embed(inpt).unsqueeze(0)
        key = last_hidden.sum(axis=0)
        attn_weights = self.attn(key, gcncontext)
        context = attn_weights.bmm(gcncontext.transpose(0, 1))
        context = context.transpose(0, 1)
        rnn_inpt = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_inpt, last_hidden)
        output = output.squeeze(0)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden


class GatedGCN(nn.Module):
    """
    When2Talk model
    1. utterance encoder
    2. GCN context encoder
    3. (optional) RNN Context encoder
    4. Attention RNN decoder
    """

    def __init__(self, input_size, output_size, embed_size, utter_hidden_size, context_hidden_size, decoder_hidden_size, position_embed_size, teach_force=0.5, pad=0, sos=0, dropout=0.5, user_embed_size=10, utter_n_layer=1, bn=False, context_threshold=2):
        super(GatedGCN, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder_ggcn(input_size, embed_size, utter_hidden_size, dropout=dropout, n_layer=utter_n_layer)
        self.gcncontext = GatedGCNContext(utter_hidden_size, context_hidden_size, user_embed_size, position_embed_size, dropout=dropout, threshold=context_threshold)
        self.decoder = Decoder_ggcn(output_size, embed_size, decoder_hidden_size, n_layer=utter_n_layer, dropout=dropout)
        self.hidden_proj = nn.Linear(context_hidden_size + user_embed_size, decoder_hidden_size)
        self.hidden_drop = nn.Dropout(p=dropout)
        self.user_embed = nn.Embedding(2, user_embed_size)

    def forward(self, src, tgt, gbatch, subatch, tubatch, lengths):
        """
        :param: src, [turns, lengths, bastch]
        :param: tgt, [lengths, batch]
        :param: gbatch, [batch, ([2, num_edges], [num_edges])]
        :param: subatch, [turn, batch]
        :param: tubatch, [batch]
        :param: lengths, [turns, batch]
        """
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        subatch = self.user_embed(subatch)
        tubatch = self.user_embed(tubatch)
        tubatch = tubatch.unsqueeze(0).repeat(2, 1, 1)
        turns = []
        for i in range(turn_size):
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        context_output, rnnh = self.gcncontext(gbatch, turns, subatch)
        ghidden = context_output[-1]
        hidden = torch.stack([rnnh, ghidden])
        hidden = torch.cat([hidden, tubatch], 2)
        hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t].clone().detach()
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = torch.max(output, 1)[1]
        return outputs

    def predict(self, src, gbatch, subatch, tubatch, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            subatch = self.user_embed(subatch)
            tubatch = self.user_embed(tubatch)
            tubatch = tubatch.unsqueeze(0).repeat(2, 1, 1)
            turns = []
            for i in range(turn_size):
                hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)
            context_output, rnnh = self.gcncontext(gbatch, turns, subatch)
            ghidden = context_output[-1]
            hidden = torch.stack([rnnh, ghidden])
            hidden = torch.cat([hidden, tubatch], 2)
            hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for i in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class Utterance_encoder(nn.Module):
    """
    Bidirectional GRU
    """

    def __init__(self, input_size, embedding_size, hidden_size, dropout=0.5, n_layer=1, pretrained=None):
        super(Utterance_encoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=n_layer, dropout=dropout, bidirectional=True)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, lengths, hidden=None):
        embedded = self.embed(inpt)
        if not hidden:
            hidden = torch.randn(self.n_layer * 2, len(lengths), self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        output, hidden = self.gru(embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = hidden.sum(axis=0)
        hidden = torch.tanh(hidden)
        output = torch.tanh(output)
        return output, hidden


class Context_encoder(nn.Module):
    """
    input_size is 2 * utterance_hidden_size
    """

    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(Context_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, bidirectional=True)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, hidden=None):
        if not hidden:
            hidden = torch.randn(2, inpt.shape[1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
        output, hidden = self.gru(inpt, hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = torch.tanh(hidden)
        return output, hidden


class Decoder(nn.Module):
    """
    Add the multi-head attention for GRU
    """

    def __init__(self, embed_size, hidden_size, output_size, n_layers=2, dropout=0.5, nhead=8):
        super(Decoder, self).__init__()
        self.embed_size, self.hidden_size = embed_size, hidden_size
        self.output_size = output_size
        self.embed = nn.Embedding(output_size, embed_size)
        self.multi_head_attention = nn.ModuleList([Attention(hidden_size) for _ in range(nhead)])
        self.attention = Attention(hidden_size)
        self.rnn = nn.GRU(hidden_size + embed_size, hidden_size, num_layers=n_layers, dropout=0 if n_layers == 1 else dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.ffn = nn.Linear(nhead * hidden_size, hidden_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, last_hidden, encoder_outputs):
        embedded = self.embed(inpt).unsqueeze(0)
        key = last_hidden.sum(axis=0)
        context_collector = []
        for attention_head in self.multi_head_attention:
            attn_weights = attention_head(key, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            context = context.squeeze(1).transpose(0, 1)
            context_collector.append(context)
        context = torch.stack(context_collector).view(-1, context.shape[-1]).transpose(0, 1)
        context = torch.tanh(self.ffn(context)).unsqueeze(0)
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(0)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden


class HRAN(nn.Module):
    """
    utter_n_layer should be the same with the one in the utterance encoder
    """

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(HRAN, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_n_layer = utter_n_layer
        self.hidden_size = decoder_hidden
        self.utter_encoder = Utterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.decoder = Decoder(utter_hidden, context_hidden, output_size, embed_size, decoder_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        turns_output = []
        for i in range(turn_size):
            output, hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
            turns_output.append(output)
        turns = torch.stack(turns)
        hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = hidden
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            turns_output = []
            for i in range(turn_size):
                output, hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
                turns_output.append(output)
            turns = torch.stack(turns)
            hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            try:
                for i in range(1, maxlen):
                    output, hidden = self.decoder(output, hidden, turns_output)
                    floss[i] = output
                    output = output.max(1)[1]
                    outputs[i] = output
            except:
                ipdb.set_trace()
            if loss:
                return outputs, floss
            else:
                return outputs


class HRAN_ablation(nn.Module):
    """
    utter_n_layer should be the same with the one in the utterance encoder
    """

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(HRAN_ablation, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_n_layer = utter_n_layer
        self.hidden_size = decoder_hidden
        self.utter_encoder = Utterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.decoder = Decoder(utter_hidden, context_hidden, output_size, embed_size, decoder_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.word_level_attention = Attention(utter_hidden)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        turns_output = []
        for i in range(turn_size):
            output, hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
            turns_output.append(output)
        turns = torch.stack(turns)
        hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = hidden
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            turns_output = []
            for i in range(turn_size):
                output, hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
                turns_output.append(output)
            turns = torch.stack(turns)
            hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            try:
                for i in range(1, maxlen):
                    output, hidden = self.decoder(output, hidden, turns_output)
                    floss[i] = output
                    output = output.max(1)[1]
                    outputs[i] = output
            except:
                ipdb.set_trace()
            if loss:
                return outputs, floss
            else:
                return outputs


class HRED(nn.Module):

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(HRED, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.context_encoder = Context_encoder(utter_hidden, context_hidden, dropout=dropout)
        self.decoder = Decoder(output_size, embed_size, decoder_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        for i in range(turn_size):
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        context_output, hidden = self.context_encoder(turns)
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            for i in range(turn_size):
                hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)
            context_output, hidden = self.context_encoder(turns)
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            try:
                for i in range(1, maxlen):
                    output, hidden = self.decoder(output, hidden, context_output)
                    floss[i] = output
                    output = output.max(1)[1]
                    outputs[i] = output
            except:
                ipdb.set_trace()
            if loss:
                return outputs, floss
            else:
                return outputs


class VariableLayer(nn.Module):
    """
    VHRED
    """

    def __init__(self, context_hidden, encoder_hidden, z_hidden):
        super(VariableLayer, self).__init__()
        self.context_hidden = context_hidden
        self.encoder_hidden = encoder_hidden
        self.z_hidden = z_hidden
        self.prior_h = nn.ModuleList([nn.Linear(context_hidden, context_hidden), nn.Linear(context_hidden, context_hidden)])
        self.prior_mu = nn.Linear(context_hidden, z_hidden)
        self.prior_var = nn.Linear(context_hidden, z_hidden)
        self.posterior_h = nn.ModuleList([nn.Linear(context_hidden + encoder_hidden, context_hidden), nn.Linear(context_hidden, context_hidden)])
        self.posterior_mu = nn.Linear(context_hidden, z_hidden)
        self.posterior_var = nn.Linear(context_hidden, z_hidden)
        self.softplus = nn.Softplus()

    def prior(self, context_outputs):
        h_prior = context_outputs
        for linear in self.prior_h:
            h_prior = torch.tanh(linear(h_prior))
        mu_prior = self.prior_mu(h_prior)
        var_prior = self.softplus(self.prior_var(h_prior))
        return mu_prior, var_prior

    def posterior(self, context_outputs, encoder_hidden):
        h_posterior = torch.cat([context_outputs, encoder_hidden], 1)
        for linear in self.posterior_h:
            h_posterior = torch.tanh(linear(h_posterior))
        mu_posterior = self.posterior_mu(h_posterior)
        var_posterior = self.softplus(self.posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def kl_div(self, mu1, var1, mu2, var2):
        one = torch.FloatTensor([1.0])
        if torch.cuda.is_available():
            one = one
        kl_div = torch.sum(0.5 * (torch.log(var2) - torch.log(var1) + (var1 + (mu1 - mu2).pow(2)) / var2 - one), 1)
        return kl_div

    def forward(self, context_outputs, encoder_hidden=None, train=True):
        mu_prior, var_prior = self.prior(context_outputs)
        eps = torch.randn((context_outputs.shape[0], self.z_hidden))
        if torch.cuda.is_available():
            eps = eps
        if train:
            mu_posterior, var_posterior = self.posterior(context_outputs, encoder_hidden)
            z_sent = mu_posterior + torch.sqrt(var_posterior) * eps
            kl_div = self.kl_div(mu_posterior, var_posterior, mu_prior, var_prior)
            kl_div = torch.sum(kl_div)
        else:
            z_sent = mu_prior + torch.sqrt(var_prior) * eps
            kl_div = None
        return z_sent, kl_div


def bag_of_words_loss(bow_logits, target_bow, weight=None):
    """ Calculate bag of words representation loss
    Args
        - bow_logits: [batch_size, vocab_size]
        - target_bow: [batch_size, vocab_size]
    """
    log_probs = F.log_softmax(bow_logits, dim=1)
    target_distribution = target_bow / (target_bow.sum(1).view(-1, 1) + 1e-23) + 1e-23
    entropy = -(torch.log(target_distribution) * target_bow).sum()
    loss = -(log_probs * target_bow).sum() - entropy
    loss = loss / target_bow.sum()
    return loss


def to_bow(sentence, vocab_size, pad, sos, eos, unk):
    """  Convert a sentence into a bag of words representation
    Args
        - sentence: a list of token ids
        - vocab_size: V
    Returns
        - bow: a integer vector of size V, numpy ndarray
    """
    sentence = sentence.cpu().numpy()
    bow = Counter(sentence)
    bow[pad], bow[eos], bow[sos], bow[unk] = 0, 0, 0, 0
    x = np.zeros(vocab_size, dtype=np.int64)
    x[list(bow.keys())] = list(bow.values())
    x = torch.tensor(x, dtype=torch.long)
    return x


class KgCVAE(nn.Module):
    """
    Source and Target vocabulary is the same
    """

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, eos=24743, pad=24745, sos=24742, unk=24745, dropout=0.5, utter_n_layer=1, z_hidden=100, pretrained=None):
        super(KgCVAE, self).__init__()
        self.teach_force = teach_force
        assert input_size == output_size, 'The src and tgt vocab size must be the same'
        self.vocab_size = input_size
        self.eos, self.pad, self.sos, self.unk = eos, pad, sos, unk
        self.utter_encoder = Utterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.context_encoder = Context_encoder(utter_hidden, context_hidden, dropout=dropout)
        self.decoder = Decoder(output_size, embed_size, decoder_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.variablelayer = VariableLayer(context_hidden, utter_hidden, z_hidden)
        self.context2decoder = nn.Linear(context_hidden + z_hidden, context_hidden)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.mlp_h = nn.Linear(z_hidden, decoder_hidden)
        self.mlp_p = nn.Linear(decoder_hidden, self.vocab_size)

    def compute_bow_loss(self, z, tgt):
        target_bow = [to_bow(i, self.vocab_size, self.pad, self.sos, self.eos, self.unk) for i in tgt.transpose(0, 1)]
        target_bow = torch.stack(target_bow, dim=0)
        if torch.cuda.is_available():
            target_bow = target_bow
        bow_logits = self.mlp_p(self.mlp_h(z))
        bow_loss = bag_of_words_loss(bow_logits, target_bow)
        return bow_loss

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.vocab_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        for i in range(turn_size):
            inpt_ = self.embedding(src[i])
            hidden = self.utter_encoder(inpt_, lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        tgt_lengths = []
        for i in range(batch_size):
            seq = tgt[:, i]
            counter = 0
            for j in seq:
                if j.item() == self.pad:
                    break
                counter += 1
            tgt_lengths.append(counter)
        tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            tgt_lengths = tgt_lengths
        tgt_ = self.embedding(tgt)
        with torch.no_grad():
            tgt_encoder_hidden = self.utter_encoder(tgt_, tgt_lengths)
        context_output, hidden = self.context_encoder(turns)
        z_sent, kl_div = self.variablelayer(hidden.sum(axis=0), encoder_hidden=tgt_encoder_hidden, train=True)
        bow_loss = self.compute_bow_loss(z_sent, tgt)
        z_sent = z_sent.repeat(2, 1, 1)
        hidden = torch.cat([hidden, z_sent], dim=2)
        hidden = torch.tanh(self.context2decoder(hidden))
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output = self.embedding(output)
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output = self.embedding(output)
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs, kl_div, bow_loss

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.vocab_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            for i in range(turn_size):
                inpt_ = self.embedding(src[i])
                hidden = self.utter_encoder(inpt_, lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)
            context_output, hidden = self.context_encoder(turns)
            z_sent, kl_div = self.variablelayer(hidden.sum(axis=0), encoder_hidden=None, train=False)
            z_sent = z_sent.repeat(2, 1, 1)
            hidden = torch.cat([hidden, z_sent], dim=2)
            hidden = torch.tanh(self.context2decoder(hidden))
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for i in range(1, maxlen):
                output = self.embedding(output)
                output, hidden = self.decoder(output, hidden, context_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class Encoder(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5, pretrained=None):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layer = n_layers
        self.embed = nn.Embedding(self.input_size, self.embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=n_layers, dropout=0 if n_layers == 1 else dropout, bidirectional=True)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def forward(self, src, inpt_lengths, hidden=None):
        embedded = self.embed(src)
        if not hidden:
            hidden = torch.randn(2 * self.n_layer, src.shape[-1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, inpt_lengths, enforce_sorted=False)
        output, hidden = self.rnn(embedded, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = torch.tanh(hidden)
        return output, hidden


class PositionEmbedding(nn.Module):
    """
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MReCoSa(nn.Module):

    def __init__(self, input_size, embed_size, output_size, utter_hidden, decoder_hidden, teach_force=0.5, pad=1, sos=1, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(MReCoSa, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden, n_layers=utter_n_layer, dropout=dropout, pretrained=pretrained)
        self.decoder = Decoder(embed_size, decoder_hidden, output_size, n_layer=utter_n_layer, dropout=dropout, pretrained=pretrained)
        self.teach_force = teach_force
        self.pad, self.sos = pad, sos
        self.output_size = output_size
        self.pos_emb = PositionEmbedding(embed_size, dropout=dropout)
        self.self_attention_context1 = nn.MultiheadAttention(embed_size, 8)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.self_attention_context2 = nn.MultiheadAttention(embed_size, 8)
        self.layer_norm2 = nn.LayerNorm(embed_size)
        self.self_attention_context3 = nn.MultiheadAttention(embed_size, 8)
        self.layer_norm3 = nn.LayerNorm(embed_size)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, max_len = len(src), tgt.size(1), tgt.size(0)
        turns = []
        for i in range(turn_size):
            hidden = self.encoder(src[i], lengths[i])
            turns.append(hidden.sum(axis=0))
        turns = torch.stack(turns)
        turns = self.pos_emb(turns)
        context, _ = self.self_attention_context1(turns, turns, turns)
        turns = self.layer_norm1(context + turns)
        context, _ = self.self_attention_context2(turns, turns, turns)
        turns = self.layer_norm2(context + turns)
        context, _ = self.self_attention_context3(turns, turns, turns)
        turns = self.layer_norm3(context + turns)
        outputs = torch.zeros(max_len, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, turns)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, turns)
                outputs[t] = output
                output = torch.max(output, 1)[1]
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            turns = []
            for i in range(turn_size):
                hidden = self.encoder(src[i], lengths[i])
                turns.append(hidden.sum(axis=0))
            turns = torch.stack(turns)
            turns = self.pos_emb(turns)
            outputs = torch.zeros(maxlen, batch_size)
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                output = output
                floss = floss
            context, _ = self.self_attention_context1(turns, turns, turns)
            turns = self.layer_norm1(context + turns)
            context, _ = self.self_attention_context2(turns, turns, turns)
            turns = self.layer_norm2(context + turns)
            context, _ = self.self_attention_context3(turns, turns, turns)
            turns = self.layer_norm3(context + turns)
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns)
                floss[t] = output
                output = torch.max(output, 1)[1]
                outputs[t] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class MReCoSa_RA(nn.Module):

    def __init__(self, input_size, embed_size, output_size, utter_hidden, decoder_hidden, teach_force=0.5, pad=1, sos=1, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(MReCoSa_RA, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden, n_layers=utter_n_layer, dropout=dropout, pretrained=pretrained)
        self.decoder = Decoder(embed_size, decoder_hidden, output_size, n_layer=utter_n_layer, dropout=dropout, pretrained=pretrained)
        self.teach_force = teach_force
        self.pad, self.sos = pad, sos
        self.output_size = output_size
        self.utter_n_layer = utter_n_layer
        self.hidden_size = decoder_hidden

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, max_len = len(src), tgt.size(1), tgt.size(0)
        turns_output = []
        for i in range(turn_size):
            output, hidden = self.encoder(src[i], lengths[i])
            turns_output.append(output)
        hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = hidden
        outputs = torch.zeros(max_len, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = torch.max(output, 1)[1]
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            turns_output = []
            for i in range(turn_size):
                output, hidden = self.encoder(src[i], lengths[i])
                turns_output.append(output)
            outputs = torch.zeros(maxlen, batch_size)
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                output = output
                floss = floss
            hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                floss[t] = output
                output = torch.max(output, 1)[1]
                outputs[t] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class Utterance_encoder_mt(nn.Module):
    """
    Bidirectional GRU
    """

    def __init__(self, input_size, embedding_size, hidden_size, dropout=0.5, n_layer=1, pretrained=False):
        super(Utterance_encoder_mt, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=n_layer, dropout=dropout, bidirectional=True)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, lengths, hidden=None):
        embedded = self.embed(inpt)
        if not hidden:
            hidden = torch.randn(self.n_layer * 2, len(lengths), self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        _, hidden = self.gru(embedded, hidden)
        hidden = hidden.sum(axis=0)
        hidden = torch.tanh(hidden)
        return hidden


class GATContext(nn.Module):
    """
    GCN Context encoder

    It should be noticed that PyG merges all the subgraph in the batch into a big graph
    which is a sparse block diagonal adjacency matrices.
    Refer: Mini-batches in 
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

    Our implementation is the three layers GCN with the position embedding
    
    ========== Make sure the inpt_size == output_size ==========
    """

    def __init__(self, inpt_size, output_size, user_embed_size, posemb_size, dropout=0.5, threshold=2, head=5):
        super(GATContext, self).__init__()
        size = inpt_size + user_embed_size + posemb_size
        self.threshold = threshold
        self.conv1 = GATConv(size, inpt_size, heads=head, dropout=dropout)
        self.conv2 = GATConv(size, inpt_size, heads=head, dropout=dropout)
        self.conv3 = GATConv(size, inpt_size, heads=head, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(inpt_size)
        self.layer_norm2 = nn.LayerNorm(inpt_size)
        self.layer_norm3 = nn.LayerNorm(inpt_size)
        self.layer_norm4 = nn.LayerNorm(inpt_size)
        self.compress = nn.Linear(head * inpt_size, inpt_size)
        self.rnn = nn.GRU(inpt_size + user_embed_size, inpt_size, bidirectional=True)
        self.linear1 = nn.Linear(inpt_size * 2, inpt_size)
        self.linear2 = nn.Linear(inpt_size, output_size)
        self.drop = nn.Dropout(p=dropout)
        self.posemb = nn.Embedding(100, posemb_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def create_batch(self, gbatch, utter_hidden):
        """create one graph batch
        :param: gbatch [batch_size, ([2, edge_num], [edge_num])]
        :param: utter_hidden [turn_len(node), batch, hidden_size]"""
        utter_hidden = utter_hidden.permute(1, 0, 2)
        batch_size = len(utter_hidden)
        data_list, weights = [], []
        for idx, example in enumerate(gbatch):
            edge_index, edge_w = example
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_w = torch.tensor(edge_w, dtype=torch.float)
            data_list.append(Data(x=utter_hidden[idx], edge_index=edge_index))
            weights.append(edge_w)
        loader = DataLoader(data_list, batch_size=batch_size)
        batch = list(loader)
        assert len(batch) == 1
        batch = batch[0]
        weights = torch.cat(weights)
        return batch, weights

    def forward(self, gbatch, utter_hidden, ub):
        rnn_x, rnnh = self.rnn(torch.cat([utter_hidden, ub], dim=-1))
        rnn_x = torch.tanh(self.linear1(rnn_x))
        turn_size = utter_hidden.size(0)
        rnnh = torch.tanh(rnnh.sum(axis=0))
        if turn_size <= self.threshold:
            return rnn_x, rnnh
        batch, weights = self.create_batch(gbatch, rnn_x)
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        batch_size = torch.max(batch).item() + 1
        pos = []
        for i in range(batch_size):
            pos.append(torch.arange(turn_size, dtype=torch.long))
        pos = torch.cat(pos)
        ub = ub.reshape(-1, ub.size(-1))
        if torch.cuda.is_available():
            x = x
            edge_index = edge_index
            batch = batch
            weights = weights
            pos = pos
        pos = self.posemb(pos)
        x = torch.cat([x, pos, ub], dim=1)
        x1 = torch.tanh(self.conv1(x, edge_index))
        x1 = torch.tanh(self.compress(x1))
        x1 = self.layer_norm1(x1)
        x1_ = torch.cat([x1, pos, ub], dim=1)
        x2 = torch.tanh(self.conv2(x1_, edge_index))
        x2 = torch.tanh(self.compress(x2))
        x2 = self.layer_norm2(x2)
        x2_ = torch.cat([x2, pos, ub], dim=1)
        x3 = torch.tanh(self.conv3(x2_, edge_index))
        x3 = torch.tanh(self.compress(x3))
        x3 = self.layer_norm3(x3)
        x = x1 + x2 + x3
        x = self.drop(torch.tanh(x))
        x = torch.stack(x.chunk(batch_size, dim=0)).permute(1, 0, 2)
        x = self.layer_norm4(rnn_x + x)
        x = torch.tanh(self.linear2(x))
        return x, rnnh


class Decoder_mt(nn.Module):

    def __init__(self, output_size, embed_size, hidden_size, user_embed_size=10, n_layer=2, dropout=0.5, pretrained=None):
        super(Decoder_mt, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size + self.hidden_size, self.hidden_size, num_layers=n_layer, dropout=0 if n_layer == 1 else dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.gru.weight_hh_l0)
        init.xavier_normal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, last_hidden, gcncontext):
        embedded = self.embed(inpt).unsqueeze(0)
        key = last_hidden.sum(axis=0)
        attn_weights = self.attn(key, gcncontext)
        context = attn_weights.bmm(gcncontext.transpose(0, 1))
        context = context.transpose(0, 1)
        rnn_inpt = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_inpt, last_hidden)
        output = output.squeeze(0)
        output = self.out(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden


class MTGAT(nn.Module):
    """
    When2Talk model
    1. utterance encoder
    2. GCN context encoder
    3. (optional) RNN Context encoder
    4. Attention RNN decoder
    """

    def __init__(self, input_size, output_size, embed_size, utter_hidden_size, context_hidden_size, decoder_hidden_size, position_embed_size, teach_force=0.5, pad=0, sos=0, dropout=0.5, user_embed_size=10, utter_n_layer=1, bn=False, context_threshold=2, heads=5):
        super(MTGAT, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder_mt(input_size, embed_size, utter_hidden_size, dropout=dropout, n_layer=utter_n_layer)
        self.gcncontext = GATContext(utter_hidden_size, context_hidden_size, user_embed_size, position_embed_size, dropout=dropout, threshold=context_threshold, head=heads)
        self.decoder = Decoder_mt(output_size, embed_size, decoder_hidden_size, n_layer=utter_n_layer, dropout=dropout)
        self.hidden_proj = nn.Linear(context_hidden_size + user_embed_size, decoder_hidden_size)
        self.hidden_drop = nn.Dropout(p=dropout)
        self.user_embed = nn.Embedding(2, user_embed_size)

    def forward(self, src, tgt, gbatch, subatch, tubatch, lengths):
        """
        :param: src, [turns, lengths, bastch]
        :param: tgt, [lengths, batch]
        :param: gbatch, [batch, ([2, num_edges], [num_edges])]
        :param: subatch, [turn, batch]
        :param: tubatch, [batch]
        :param: lengths, [turns, batch]
        """
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        subatch = self.user_embed(subatch)
        tubatch = self.user_embed(tubatch)
        tubatch = tubatch.unsqueeze(0).repeat(2, 1, 1)
        turns = []
        for i in range(turn_size):
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        context_output, rnnh = self.gcncontext(gbatch, turns, subatch)
        ghidden = context_output[-1]
        hidden = torch.stack([rnnh, ghidden])
        hidden = torch.cat([hidden, tubatch], 2)
        hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t].clone().detach()
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = torch.max(output, 1)[1]
        return outputs

    def predict(self, src, gbatch, subatch, tubatch, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            subatch = self.user_embed(subatch)
            tubatch = self.user_embed(tubatch)
            tubatch = tubatch.unsqueeze(0).repeat(2, 1, 1)
            turns = []
            for i in range(turn_size):
                hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)
            context_output, rnnh = self.gcncontext(gbatch, turns, subatch)
            ghidden = context_output[-1]
            hidden = torch.stack([rnnh, ghidden])
            hidden = torch.cat([hidden, tubatch], 2)
            hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for i in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class OGCNContext(nn.Module):
    """
    GCN Context encoder

    It should be noticed that PyG merges all the subgraph in the batch into a big graph
    which is a sparse block diagonal adjacency matrices.
    Refer: Mini-batches in 
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

    Our implementation is the three layers GCN with the position embedding
    
    ========== Make sure the inpt_size == output_size ==========
    """

    def __init__(self, inpt_size, output_size, user_embed_size, posemb_size, dropout=0.5, threshold=2):
        super(OGCNContext, self).__init__()
        size = inpt_size + user_embed_size + posemb_size
        self.threshold = threshold
        self.conv1 = GCNConv(size, inpt_size)
        self.conv2 = GCNConv(size, inpt_size)
        self.conv3 = GCNConv(size, inpt_size)
        self.rnn = nn.GRU(inpt_size + user_embed_size, inpt_size, bidirectional=True)
        self.linear1 = nn.Linear(inpt_size * 2, inpt_size)
        self.linear2 = nn.Linear(inpt_size * 2, output_size)
        self.drop = nn.Dropout(p=dropout)
        self.posemb = nn.Embedding(100, posemb_size)
        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.rnn.weight_hh_l0)
        init.xavier_normal_(self.rnn.weight_ih_l0)
        self.rnn.bias_ih_l0.data.fill_(0.0)
        self.rnn.bias_hh_l0.data.fill_(0.0)

    def create_batch(self, gbatch, utter_hidden):
        """create one graph batch
        :param: gbatch [batch_size, ([2, edge_num], [edge_num])]
        :param: utter_hidden [turn_len(node), batch, hidden_size]"""
        utter_hidden = utter_hidden.permute(1, 0, 2)
        batch_size = len(utter_hidden)
        data_list, weights = [], []
        for idx, example in enumerate(gbatch):
            edge_index, edge_w = example
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_w = torch.tensor(edge_w, dtype=torch.float)
            data_list.append(Data(x=utter_hidden[idx], edge_index=edge_index))
            weights.append(edge_w)
        loader = DataLoader(data_list, batch_size=batch_size)
        batch = list(loader)
        assert len(batch) == 1
        batch = batch[0]
        weights = torch.cat(weights)
        return batch, weights

    def forward(self, gbatch, utter_hidden, ub):
        rnn_x, rnnh = self.rnn(torch.cat([utter_hidden, ub], dim=-1))
        rnn_x = torch.tanh(self.linear1(rnn_x))
        turn_size = utter_hidden.size(0)
        rnnh = torch.tanh(rnnh.sum(axis=0))
        if turn_size <= self.threshold:
            return rnn_x, rnnh
        batch, weights = self.create_batch(gbatch, rnn_x)
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        batch_size = torch.max(batch).item() + 1
        pos = []
        for i in range(batch_size):
            pos.append(torch.arange(turn_size, dtype=torch.long))
        pos = torch.cat(pos)
        ub = ub.reshape(-1, ub.size(-1))
        if torch.cuda.is_available():
            x = x
            edge_index = edge_index
            batch = batch
            weights = weights
            pos = pos
        pos = self.posemb(pos)
        x = torch.cat([x, pos, ub], dim=1)
        x1 = torch.tanh(self.conv1(x, edge_index, edge_weight=weights))
        x1_ = torch.cat([x1, pos, ub], dim=1)
        x2 = torch.tanh(self.conv2(x1_, edge_index, edge_weight=weights))
        x2_ = torch.cat([x2, pos, ub], dim=1)
        x3 = torch.tanh(self.conv3(x2_, edge_index, edge_weight=weights))
        x = x1 + x2 + x3
        x = self.drop(torch.tanh(x))
        x = torch.stack(x.chunk(batch_size, dim=0)).permute(1, 0, 2)
        x = torch.cat([rnn_x, x], dim=2)
        x = torch.tanh(self.linear2(x))
        return x, rnnh


class MTGCN(nn.Module):
    """
    When2Talk model
    1. utterance encoder
    2. GCN context encoder
    3. (optional) RNN Context encoder
    4. Attention RNN decoder
    """

    def __init__(self, input_size, output_size, embed_size, utter_hidden_size, context_hidden_size, decoder_hidden_size, position_embed_size, teach_force=0.5, pad=0, sos=0, dropout=0.5, user_embed_size=10, utter_n_layer=1, bn=False, context_threshold=2):
        super(MTGCN, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder_mt(input_size, embed_size, utter_hidden_size, dropout=dropout, n_layer=utter_n_layer)
        self.gcncontext = OGCNContext(utter_hidden_size, context_hidden_size, user_embed_size, position_embed_size, dropout=dropout, threshold=context_threshold)
        self.decoder = Decoder_mt(output_size, embed_size, decoder_hidden_size, n_layer=utter_n_layer, dropout=dropout)
        self.hidden_proj = nn.Linear(context_hidden_size + user_embed_size, decoder_hidden_size)
        self.hidden_drop = nn.Dropout(p=dropout)
        self.user_embed = nn.Embedding(2, user_embed_size)

    def forward(self, src, tgt, gbatch, subatch, tubatch, lengths):
        """
        :param: src, [turns, lengths, bastch]
        :param: tgt, [lengths, batch]
        :param: gbatch, [batch, ([2, num_edges], [num_edges])]
        :param: subatch, [turn, batch]
        :param: tubatch, [batch]
        :param: lengths, [turns, batch]
        """
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        subatch = self.user_embed(subatch)
        tubatch = self.user_embed(tubatch)
        tubatch = tubatch.unsqueeze(0).repeat(2, 1, 1)
        turns = []
        for i in range(turn_size):
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        context_output, rnnh = self.gcncontext(gbatch, turns, subatch)
        ghidden = context_output[-1]
        hidden = torch.stack([rnnh, ghidden])
        hidden = torch.cat([hidden, tubatch], 2)
        hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t].clone().detach()
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = torch.max(output, 1)[1]
        return outputs

    def predict(self, src, gbatch, subatch, tubatch, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            subatch = self.user_embed(subatch)
            tubatch = self.user_embed(tubatch)
            tubatch = tubatch.unsqueeze(0).repeat(2, 1, 1)
            turns = []
            for i in range(turn_size):
                hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)
            context_output, rnnh = self.gcncontext(gbatch, turns, subatch)
            ghidden = context_output[-1]
            hidden = torch.stack([rnnh, ghidden])
            hidden = torch.cat([hidden, tubatch], 2)
            hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for i in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class VHRED(nn.Module):
    """
    Source and Target vocabulary is the same
    """

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, z_hidden=100, pretrained=None):
        super(VHRED, self).__init__()
        self.teach_force = teach_force
        assert input_size == output_size, 'The src and tgt vocab size must be the same'
        self.vocab_size = input_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.context_encoder = Context_encoder(utter_hidden, context_hidden, dropout=dropout)
        self.decoder = Decoder(output_size, embed_size, decoder_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.variablelayer = VariableLayer(context_hidden, utter_hidden, z_hidden)
        self.context2decoder = nn.Linear(context_hidden + z_hidden, context_hidden)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.vocab_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        for i in range(turn_size):
            inpt_ = self.embedding(src[i])
            hidden = self.utter_encoder(inpt_, lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        tgt_lengths = []
        for i in range(batch_size):
            seq = tgt[:, i]
            counter = 0
            for j in seq:
                if j.item() == self.pad:
                    break
                counter += 1
            tgt_lengths.append(counter)
        tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            tgt_lengths = tgt_lengths
        tgt_ = self.embedding(tgt)
        with torch.no_grad():
            tgt_encoder_hidden = self.utter_encoder(tgt_, tgt_lengths)
        context_output, hidden = self.context_encoder(turns)
        z_sent, kl_div = self.variablelayer(hidden.sum(axis=0), encoder_hidden=tgt_encoder_hidden, train=True)
        z_sent = z_sent.repeat(2, 1, 1)
        hidden = torch.cat([hidden, z_sent], dim=2)
        hidden = torch.tanh(self.context2decoder(hidden))
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output = self.embedding(output)
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output = self.embedding(output)
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs, kl_div

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.vocab_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            for i in range(turn_size):
                inpt_ = self.embedding(src[i])
                hidden = self.utter_encoder(inpt_, lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)
            context_output, hidden = self.context_encoder(turns)
            z_sent, kl_div = self.variablelayer(hidden.sum(axis=0), encoder_hidden=None, train=False)
            z_sent = z_sent.repeat(2, 1, 1)
            hidden = torch.cat([hidden, z_sent], dim=2)
            hidden = torch.tanh(self.context2decoder(hidden))
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for i in range(1, maxlen):
                output = self.embedding(output)
                output, hidden = self.decoder(output, hidden, context_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class WSeq(nn.Module):

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(WSeq, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.context_encoder = Context_encoder(utter_hidden, context_hidden, dropout=dropout)
        self.decoder = Decoder(output_size, embed_size, decoder_hidden, n_layer=utter_n_layer, dropout=dropout, pretrained=pretrained)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        for i in range(turn_size):
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        context_output, hidden = self.context_encoder(turns)
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            for i in range(turn_size):
                hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
            turns = torch.stack(turns)
            context_output, hidden = self.context_encoder(turns)
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for i in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, context_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class WSeq_RA(nn.Module):

    def __init__(self, embed_size, input_size, output_size, utter_hidden, context_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, pretrained=None):
        super(WSeq_RA, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_n_layer = utter_n_layer
        self.hidden_size = decoder_hidden
        self.utter_encoder = Utterance_encoder(input_size, embed_size, utter_hidden, dropout=dropout, n_layer=utter_n_layer, pretrained=pretrained)
        self.decoder = Decoder(utter_hidden, context_hidden, output_size, embed_size, decoder_hidden, n_layer=utter_n_layer, dropout=dropout, pretrained=pretrained)

    def forward(self, src, tgt, lengths):
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        turns = []
        turns_output = []
        for i in range(turn_size):
            output, hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
            turns_output.append(output)
        turns = torch.stack(turns)
        hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = hidden
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=False):
        with torch.no_grad():
            turn_size, batch_size = len(src), src[0].size(1)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            turns = []
            turns_output = []
            for i in range(turn_size):
                output, hidden = self.utter_encoder(src[i], lengths[i])
                turns.append(hidden)
                turns_output.append(output)
            turns = torch.stack(turns)
            hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for i in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, turns_output)
                floss[i] = output
                output = output.max(1)[1]
                outputs[i] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class Multi_head_attention(nn.Module):
    """
    Multi head attention for RNN, Layernorm and residual connection are used.
    By the way, Transformer sucks.
    """

    def __init__(self, hidden_size, nhead=4):
        super(Multi_head_attention, self).__init__()
        self.nhead = nhead
        self.hidden_size = hidden_size
        self.multi_head_attention = nn.ModuleList([Attention(hidden_size) for _ in range(nhead)])
        self.ffn = nn.Linear(self.nhead * self.hidden_size, self.hidden_size)

    def forward(self, hidden, encoder_outputs):
        context_collector = []
        for attention_head in self.multi_head_attention:
            attn_weights = attention_head(hidden, encoder_outputs)
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            context = context.squeeze(1).transpose(0, 1)
            context_collector.append(context)
        context = torch.stack(context_collector).view(-1, context.shape[-1]).transpose(0, 1)
        context = torch.tanh(self.ffn(context)).unsqueeze(0)
        return context


class Multi_head_attention_trs(nn.Module):
    """
    make sure the hidden_size can be divisible by nhead
    Recommand: 512, 8
    
    1. Multi head attention for encoder hidden state
    2. Use the hidden state to query the context encoder
    """

    def __init__(self, hidden_size, nhead=8, dropout=0.3):
        super(Multi_head_attention_trs, self).__init__()
        self.nhead = nhead
        self.hidden_size = hidden_size
        if hidden_size % nhead != 0:
            raise Exception(f'hidden_size must be divisble by nhead, but got {hidden_size}/{nhead}.')
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, nhead)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.final_attn = Attention(hidden_size)

    def forward(self, hidden, encoder_outputs):
        context, _ = self.multi_head_attention(encoder_outputs, encoder_outputs, encoder_outputs)
        context = context + encoder_outputs
        context = torch.tanh(self.layer_norm(context))
        attn_weights = self.final_attn(hidden.unsqueeze(0), context)
        context = attn_weights.bmm(context.transpose(0, 1))
        context = context.transpose(0, 1)
        return context


class WSeq_attention(nn.Module):
    """
    Cosine similarity defined in ACL 2017 paper: 
    How to Make Context More Useful?
    An Empirical Study on context-Aware Neural Conversational Models

    mode: sum, concat is very hard to be implemented
    """

    def __init__(self, mode='sum'):
        super(WSeq_attention, self).__init__()

    def forward(self, query, utterances):
        utterances = utterances.permute(1, 2, 0)
        query = query.reshape(query.shape[0], 1, query.shape[1])
        p = torch.bmm(query, utterances).squeeze(1)
        query_norm = query.squeeze(1).norm(dim=1)
        utterances_norm = utterances.norm(dim=1)
        p = p / query_norm.reshape(-1, 1)
        p = p / utterances_norm
        sq = torch.ones(p.shape[0], 1)
        if torch.cuda.is_available():
            sq = sq
        p = torch.cat([p, sq], 1)
        p = F.softmax(p, dim=1)
        utterances = utterances.permute(0, 2, 1)
        vector = torch.cat([utterances, query], 1)
        p = p.unsqueeze(1)
        vector = torch.bmm(p, vector).squeeze(1)
        return vector


class PretrainedEmbedding(nn.Module):
    """
    Pretrained English BERT contextual word embeddings
    make sure the embedding size is the same as the embed_size setted in the model
    or the error will be thrown.
    """

    def __init__(self, vocab_size, embed_size, path):
        super(PretrainedEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)
        with open(path, 'rb') as f:
            emb = pickle.load(f)
        self.emb.weight.data.copy_(torch.from_numpy(emb))

    def forward(self, x):
        return self.emb(x)


class Seq2Seq(nn.Module):
    """
    Compose the Encoder and Decoder into the Seq2Seq model
    """

    def __init__(self, input_size, embed_size, output_size, utter_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, src_vocab=None, tgt_vocab=None, pretrained=None):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden, n_layers=utter_n_layer, dropout=dropout, pretrained=pretrained)
        self.decoder = Decoder(embed_size, decoder_hidden, output_size, n_layers=utter_n_layer, dropout=dropout, pretrained=pretrained)
        self.teach_force = teach_force
        self.utter_n_layer = utter_n_layer
        self.pad, self.sos = pad, sos
        self.output_size = output_size

    def forward(self, src, tgt, lengths):
        batch_size, max_len = src.shape[1], tgt.shape[0]
        outputs = torch.zeros(max_len, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        encoder_output, hidden = self.encoder(src, lengths)
        hidden = hidden[-self.utter_n_layer:]
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=True):
        with torch.no_grad():
            batch_size = src.shape[1]
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            encoder_output, hidden = self.encoder(src, lengths)
            hidden = hidden[-self.utter_n_layer:]
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, encoder_output)
                floss[t] = output
                output = output.topk(1)[1].squeeze()
                outputs[t] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class transformer_gpt2(nn.Module):
    """
    GPT2 for seq2seq modeling
    """

    def __init__(self, config_path):
        super(transformer_gpt2, self).__init__()
        self.tokenzier = BertTokenizer(vocab_file='config/vocab_en.txt')
        self.vocab_size = len(self.tokenzier)
        self.model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(config_path)
        self.model = GPT2LMHeadModel(config=self.model_config)
        self.model.resize_token_embeddings(self.vocab_size)
        self.n_ctx = self.model.config.to_dict().get('n_ctx')

    def forward(self, inpt):
        """
        inpt: [seq, batch]
        """
        inpt = inpt.transpose(0, 1)
        opt = self.model.forward(input_ids=inpt)[0]
        opt = F.log_softmax(opt, dim=-1)
        return opt

    def predict(self, inpt, maxlen, loss=True):
        """
        Different from the forward function, auto-regression
        inpt: [seq, batch]
        """
        with torch.no_grad():
            inpt = inpt.transpose(0, 1)
            batch_size = inpt.shape[0]
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.vocab_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            for t in range(maxlen):
                opt = self.model.forward(input_ids=inpt)[0]
                opt = opt[:, -1, :]
                next_token = F.log_softmax(opt, dim=-1)
                floss[t] = next_token
                next_token = next_token.topk(1)[1]
                outputs[t] = next_token.squeeze()
                inpt = torch.cat((inpt, next_token), dim=-1)
            if loss:
                return outputs, floss
            else:
                return outpus


class Seq2Seq_Multi_Head(nn.Module):
    """
    Compose the Encoder and Decoder into the Seq2Seq model
    """

    def __init__(self, input_size, embed_size, output_size, utter_hidden, decoder_hidden, teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1, src_vocab=None, tgt_vocab=None, nhead=8, pretrained=None):
        super(Seq2Seq_Multi_Head, self).__init__()
        self.encoder = Encoder(input_size, embed_size, utter_hidden, n_layers=utter_n_layer, dropout=dropout, pretrained=pretrained)
        self.decoder = Decoder(embed_size, decoder_hidden, output_size, n_layers=utter_n_layer, dropout=dropout, nhead=nhead, pretrained=pretrained)
        self.teach_force = teach_force
        self.utter_n_layer = utter_n_layer
        self.pad, self.sos = pad, sos
        self.output_size = output_size

    def forward(self, src, tgt, lengths):
        batch_size, max_len = src.shape[1], tgt.shape[0]
        outputs = torch.zeros(max_len, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs
        encoder_output, hidden = self.encoder(src, lengths)
        hidden = hidden[-self.utter_n_layer:]
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, encoder_output)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=True):
        with torch.no_grad():
            batch_size = src.shape[1]
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            encoder_output, hidden = self.encoder(src, lengths)
            hidden = hidden[-self.utter_n_layer:]
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, encoder_output)
                floss[t] = output
                output = output.topk(1)[1].squeeze()
                outputs[t] = output
            if loss:
                return outputs, floss
            else:
                return outputs


class Transformer(nn.Module):
    """
    Transformer encoder and GRU decoder
    
    Multi-head attention for GRU
    """

    def __init__(self, input_vocab_size, opt_vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, position_embed_size=300, utter_n_layer=2, dropout=0.3, sos=0, pad=0, teach_force=1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.hidden_size = d_model
        self.embed_src = nn.Embedding(input_vocab_size, d_model)
        self.pos_enc = PositionEmbedding(d_model, dropout=dropout, max_len=position_embed_size)
        self.input_vocab_size = input_vocab_size
        self.utter_n_layer = utter_n_layer
        self.opt_vocab_size = opt_vocab_size
        self.pad, self.sos = pad, sos
        self.teach_force = teach_force
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = Decoder(d_model, d_model, opt_vocab_size, n_layers=utter_n_layer, dropout=dropout, nhead=nhead)

    def generate_key_mask(self, x, lengths):
        seq_length = x.shape[0]
        masks = []
        for sentence_l in lengths:
            masks.append([(False) for _ in range(sentence_l)] + [(True) for _ in range(seq_length - sentence_l)])
        masks = torch.tensor(masks)
        if torch.cuda.is_available():
            masks = masks
        return masks

    def forward(self, src, tgt, lengths):
        batch_size, max_len = src.shape[1], tgt.shape[0]
        src_key_padding_mask = self.generate_key_mask(src, lengths)
        outputs = torch.zeros(max_len, batch_size, self.opt_vocab_size)
        if torch.cuda.is_available():
            outputs = outputs
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            hidden = hidden
        output = tgt[0, :]
        use_teacher = random.random() < self.teach_force
        if use_teacher:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, memory)
                outputs[t] = output
                output = tgt[t]
        else:
            for t in range(1, max_len):
                output, hidden = self.decoder(output, hidden, memory)
                outputs[t] = output
                output = output.topk(1)[1].squeeze().detach()
        return outputs

    def predict(self, src, maxlen, lengths, loss=True):
        with torch.no_grad():
            batch_size = src.shape[1]
            src_key_padding_mask = self.generate_key_mask(src, lengths)
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.opt_vocab_size)
            if torch.cuda.is_available():
                outputs = outputs
                floss = floss
            src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
            memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
            hidden = torch.randn(self.utter_n_layer, batch_size, self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden
            output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
            if torch.cuda.is_available():
                output = output
            for t in range(1, maxlen):
                output, hidden = self.decoder(output, hidden, memory)
                floss[t] = output
                output = output.topk(1)[1].squeeze()
                outputs[t] = output
            if loss:
                return outputs, floss
            else:
                return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (Context_encoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (DSContext_encoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (Multi_head_attention,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (PositionEmbedding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_gmftbyGMFTBY_MultiTurnDialogZoo(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

