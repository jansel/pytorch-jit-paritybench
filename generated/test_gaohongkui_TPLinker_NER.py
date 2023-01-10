import sys
_module = sys.modules[__name__]
del sys
main = _module
common = _module
components = _module
utils = _module
convert_dataset = _module
tplinker_plus_ner = _module
config = _module
evaluate = _module
evaluate_only_bert = _module
tplinker_plus_ner = _module
train = _module
train_only_bert = _module

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


import torch.nn as nn


from torch.nn.parameter import Parameter


import numpy as np


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import re


import copy


import math


import time


class LayerNorm(nn.Module):

    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False, hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))
        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma
        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta
        return outputs


class HandshakingKernel(nn.Module):

    def __init__(self, hidden_size, shaking_type, inner_enc_type):
        super().__init__()
        self.shaking_type = shaking_type
        if shaking_type == 'cat':
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == 'cat_plus':
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == 'cln':
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        elif shaking_type == 'cln_plus':
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional=True)
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == 'mix_pooling':
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == 'lstm':
            self.inner_context_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type='lstm'):

        def pool(seqence, pooling_type):
            if pooling_type == 'mean_pooling':
                pooling = torch.mean(seqence, dim=-2)
            elif pooling_type == 'max_pooling':
                pooling, _ = torch.max(seqence, dim=-2)
            elif pooling_type == 'mix_pooling':
                pooling = self.lamtha * torch.mean(seqence, dim=-2) + (1 - self.lamtha) * torch.max(seqence, dim=-2)[0]
            return pooling
        if 'pooling' in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i + 1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim=1)
        elif inner_enc_type == 'lstm':
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
        return inner_context

    def forward(self, seq_hiddens):
        """
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        """
        seq_len = seq_hiddens.size()[-2]
        shaking_hiddens_list = []
        for ind in range(seq_len):
            hidden_each_step = seq_hiddens[:, ind, :]
            visible_hiddens = seq_hiddens[:, ind:, :]
            repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)
            if self.shaking_type == 'cat':
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == 'cat_plus':
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim=-1)
                shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
            elif self.shaking_type == 'cln':
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
            elif self.shaking_type == 'cln_plus':
                inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
                shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
                shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)
            shaking_hiddens_list.append(shaking_hiddens)
        long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim=1)
        return long_shaking_hiddens


class TPLinkerPlusBert(nn.Module):

    def __init__(self, encoder, tag_size, shaking_type, inner_enc_type, tok_pair_sample_rate=1):
        super().__init__()
        self.encoder = encoder
        self.tok_pair_sample_rate = tok_pair_sample_rate
        shaking_hidden_size = encoder.config.hidden_size
        self.fc = nn.Linear(shaking_hidden_size, tag_size)
        self.handshaking_kernel = HandshakingKernel(shaking_hidden_size, shaking_type, inner_enc_type)

    def forward(self, input_ids, attention_mask, token_type_ids):
        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0]
        seq_len = last_hidden_state.size()[1]
        shaking_hiddens = self.handshaking_kernel(last_hidden_state)
        sampled_tok_pair_indices = None
        if self.training:
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0], 1)
            sampled_tok_pair_indices = sampled_tok_pair_indices
            shaking_hiddens = shaking_hiddens.gather(1, sampled_tok_pair_indices[:, :, None].repeat(1, 1, shaking_hiddens.size()[-1]))
        outputs = self.fc(shaking_hiddens)
        return outputs, sampled_tok_pair_indices


class TPLinkerPlusBiLSTM(nn.Module):

    def __init__(self, init_word_embedding_matrix, emb_dropout_rate, enc_hidden_size, dec_hidden_size, rnn_dropout_rate, tag_size, shaking_type, inner_enc_type, tok_pair_sample_rate=1):
        super().__init__()
        self.word_embeds = nn.Embedding.from_pretrained(init_word_embedding_matrix, freeze=False)
        self.emb_dropout = nn.Dropout(emb_dropout_rate)
        self.enc_lstm = nn.LSTM(init_word_embedding_matrix.size()[-1], enc_hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dec_lstm = nn.LSTM(enc_hidden_size, dec_hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.rnn_dropout = nn.Dropout(rnn_dropout_rate)
        self.tok_pair_sample_rate = tok_pair_sample_rate
        shaking_hidden_size = dec_hidden_size
        self.fc = nn.Linear(shaking_hidden_size, tag_size)
        self.handshaking_kernel = HandshakingKernel(shaking_hidden_size, shaking_type, inner_enc_type)

    def forward(self, input_ids):
        embedding = self.word_embeds(input_ids)
        embedding = self.emb_dropout(embedding)
        lstm_outputs, _ = self.enc_lstm(embedding)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        lstm_outputs, _ = self.dec_lstm(lstm_outputs)
        lstm_outputs = self.rnn_dropout(lstm_outputs)
        seq_len = lstm_outputs.size()[1]
        shaking_hiddens = self.handshaking_kernel(lstm_outputs)
        sampled_tok_pair_indices = None
        if self.training:
            shaking_seq_len = shaking_hiddens.size()[1]
            segment_len = int(shaking_seq_len * self.tok_pair_sample_rate)
            seg_num = math.ceil(shaking_seq_len // segment_len)
            start_ind = torch.randint(seg_num, []) * segment_len
            end_ind = min(start_ind + segment_len, shaking_seq_len)
            sampled_tok_pair_indices = torch.arange(start_ind, end_ind)[None, :].repeat(shaking_hiddens.size()[0], 1)
            sampled_tok_pair_indices = sampled_tok_pair_indices
            shaking_hiddens = shaking_hiddens.gather(1, sampled_tok_pair_indices[:, :, None].repeat(1, 1, shaking_hiddens.size()[-1]))
        outputs = self.fc(shaking_hiddens)
        return outputs, sampled_tok_pair_indices


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LayerNorm,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_gaohongkui_TPLinker_NER(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

