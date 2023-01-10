import sys
_module = sys.modules[__name__]
del sys
config = _module
process_data = _module
data_utils = _module
inference = _module
main = _module
model = _module
qgevalcap = _module
bleu = _module
bleu_scorer = _module
cider = _module
cider_scorer = _module
eval = _module
meteor = _module
rouge = _module
trainer = _module

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


import time


from collections import defaultdict


from copy import deepcopy


import numpy as np


import torch


import torch.utils.data as data


import torch.nn.functional as F


import torch.nn as nn


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


import torch.optim as optim


class Encoder(nn.Module):

    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.tag_embedding = nn.Embedding(3, 3)
        lstm_input_size = embedding_size + 3
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size).from_pretrained(embeddings, freeze=config.freeze_embedding)
        self.num_layers = num_layers
        if self.num_layers == 1:
            dropout = 0.0
        self.lstm = nn.LSTM(lstm_input_size, hidden_size, dropout=dropout, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear_trans = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.update_layer = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)
        self.gate = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=False)

    def gated_self_attn(self, queries, memories, mask):
        energies = torch.matmul(queries, memories.transpose(1, 2))
        mask = mask.unsqueeze(1)
        energies = energies.masked_fill(mask == 0, value=-1000000000000.0)
        scores = F.softmax(energies, dim=2)
        context = torch.matmul(scores, queries)
        inputs = torch.cat([queries, context], dim=2)
        f_t = torch.tanh(self.update_layer(inputs))
        g_t = torch.sigmoid(self.gate(inputs))
        updated_output = g_t * f_t + (1 - g_t) * queries
        return updated_output

    def forward(self, src_seq, src_len, tag_seq):
        total_length = src_seq.size(1)
        embedded = self.embedding(src_seq)
        tag_embedded = self.tag_embedding(tag_seq)
        embedded = torch.cat((embedded, tag_embedded), dim=2)
        packed = pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)
        outputs, states = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=total_length)
        h, c = states
        mask = torch.sign(src_seq)
        memories = self.linear_trans(outputs)
        outputs = self.gated_self_attn(outputs, memories, mask)
        _, b, d = h.size()
        h = h.view(2, 2, b, d)
        h = torch.cat((h[:, 0, :, :], h[:, 1, :, :]), dim=-1)
        c = c.view(2, 2, b, d)
        c = torch.cat((c[:, 0, :, :], c[:, 1, :, :]), dim=-1)
        concat_states = h, c
        return outputs, concat_states


INF = 1000000000000.0


UNK_ID = 1


class Decoder(nn.Module):

    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embeddings is not None:
            self.embedding = nn.Embedding(vocab_size, embedding_size).from_pretrained(embeddings, freeze=config.freeze_embedding)
        if num_layers == 1:
            dropout = 0.0
        self.encoder_trans = nn.Linear(hidden_size, hidden_size)
        self.reduce_layer = nn.Linear(embedding_size + hidden_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=False, dropout=dropout)
        self.concat_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.logit_layer = nn.Linear(hidden_size, vocab_size)

    @staticmethod
    def attention(query, memories, mask):
        energy = torch.matmul(query, memories.transpose(1, 2))
        energy = energy.squeeze(1).masked_fill(mask == 0, value=-1000000000000.0)
        attn_dist = F.softmax(energy, dim=1).unsqueeze(dim=1)
        context_vector = torch.matmul(attn_dist, memories)
        return context_vector, energy

    def get_encoder_features(self, encoder_outputs):
        return self.encoder_trans(encoder_outputs)

    def forward(self, trg_seq, ext_src_seq, init_states, encoder_outputs, encoder_mask):
        device = trg_seq.device
        batch_size, max_len = trg_seq.size()
        hidden_size = encoder_outputs.size(-1)
        memories = self.get_encoder_features(encoder_outputs)
        logits = []
        prev_states = init_states
        prev_context = torch.zeros((batch_size, 1, hidden_size))
        prev_context = prev_context
        for i in range(max_len):
            y_i = trg_seq[:, i].unsqueeze(1)
            embedded = self.embedding(y_i)
            lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], 2))
            output, states = self.lstm(lstm_inputs, prev_states)
            context, energy = self.attention(output, memories, encoder_mask)
            concat_input = torch.cat((output, context), dim=2).squeeze(dim=1)
            logit_input = torch.tanh(self.concat_layer(concat_input))
            logit = self.logit_layer(logit_input)
            if config.use_pointer:
                num_oov = max(torch.max(ext_src_seq - self.vocab_size + 1), 0)
                zeros = torch.zeros((batch_size, num_oov), device=config.device)
                extended_logit = torch.cat([logit, zeros], dim=1)
                out = torch.zeros_like(extended_logit) - INF
                out, _ = scatter_max(energy, ext_src_seq, out=out)
                out = out.masked_fill(out == -INF, 0)
                logit = extended_logit + out
                logit = logit.masked_fill(logit == 0, -INF)
            logits.append(logit)
            prev_states = states
            prev_context = context
        logits = torch.stack(logits, dim=1)
        return logits

    def decode(self, y, ext_x, prev_states, prev_context, encoder_features, encoder_mask):
        embedded = self.embedding(y.unsqueeze(1))
        lstm_inputs = self.reduce_layer(torch.cat([embedded, prev_context], 2))
        output, states = self.lstm(lstm_inputs, prev_states)
        context, energy = self.attention(output, encoder_features, encoder_mask)
        concat_input = torch.cat((output, context), 2).squeeze(1)
        logit_input = torch.tanh(self.concat_layer(concat_input))
        logit = self.logit_layer(logit_input)
        if config.use_pointer:
            batch_size = y.size(0)
            num_oov = max(torch.max(ext_x - self.vocab_size + 1), 0)
            zeros = torch.zeros((batch_size, num_oov), device=config.device)
            extended_logit = torch.cat([logit, zeros], dim=1)
            out = torch.zeros_like(extended_logit) - INF
            out, _ = scatter_max(energy, ext_x, out=out)
            out = out.masked_fill(out == -INF, 0)
            logit = extended_logit + out
            logit = logit.masked_fill(logit == -INF, 0)
            logit[:, UNK_ID] = -INF
        return logit, states, context


class Seq2seq(nn.Module):

    def __init__(self, embedding=None):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(embedding, config.vocab_size, config.embedding_size, config.hidden_size, config.num_layers, config.dropout)
        self.decoder = Decoder(embedding, config.vocab_size, config.embedding_size, 2 * config.hidden_size, config.num_layers, config.dropout)

    def forward(self, src_seq, tag_seq, ext_src_seq, trg_seq):
        enc_mask = torch.sign(src_seq)
        src_len = torch.sum(enc_mask, 1)
        enc_outputs, enc_states = self.encoder(src_seq, src_len, tag_seq)
        sos_trg = trg_seq[:, :-1].contiguous()
        logits = self.decoder(sos_trg, ext_src_seq, enc_states, enc_outputs, enc_mask)
        return logits

