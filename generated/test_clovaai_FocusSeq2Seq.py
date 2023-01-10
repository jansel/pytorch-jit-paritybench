import sys
_module = sys.modules[__name__]
del sys
CNNDM_data_loader = _module
QG_data_loader = _module
build_utils = _module
configs = _module
evaluate = _module
layers = _module
beam_search = _module
bridge = _module
copy_attention = _module
decoder = _module
encoder = _module
selector = _module
models = _module
train = _module
utils = _module
bleu = _module
data_utils = _module
initializer = _module
rouge = _module
perl_rouge = _module
tensor_utils = _module

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


import pandas as pd


from collections import Counter


from functools import partial


import torch


from torch.nn.utils.rnn import pad_sequence


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import re


import numpy as np


import time


import torch.nn as nn


import math


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import random


import torch.optim as optim


class LinearBridge(nn.Module):

    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, rnn='GRU', activation='tanh'):
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        if rnn == 'GRU':
            self.rnn_type = 'GRU'
            self.linear = nn.Linear(enc_hidden_size, dec_hidden_size)
        elif rnn == 'LSTM':
            self.rnn_type = 'LSTM'
            self.linear_h = nn.Linear(enc_hidden_size, dec_hidden_size)
            self.linear_c = nn.Linear(enc_hidden_size, dec_hidden_size)
        if activation.lower() == 'tanh':
            self.act = nn.Tanh()
        elif activation.lower() == 'relu':
            self.act = nn.ReLU()

    def forward(self, hidden):
        """
           [2, B, enc_hidden_size // 2]
        => [B, 2, enc_hidden_size // 2] (transpose)
        => [B, enc_hidden_size]         (view)
        => [B, dec_hidden_size]         (linear)
        """
        if self.rnn_type == 'GRU':
            h = hidden
            B = h.size(1)
            h = h.transpose(0, 1).contiguous().view(B, self.enc_hidden_size)
            return self.act(self.linear(h))
        elif self.rnn_type == 'LSTM':
            h, c = hidden
            B = h.size(1)
            h = h.transpose(0, 1).contiguous().view(B, self.enc_hidden_size)
            c = h.transpose(0, 1).contiguous().view(B, self.enc_hidden_size)
            h = self.act(self.linear_h(h))
            c = self.act(self.linear_c(c))
            h = h, c
            return h


class BahdanauAttention(nn.Module):

    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, attention_size=700, coverage=False, weight_norm=False, bias=True, pointer_end_bias=False):
        """Bahdanau Attention (+ Coverage)"""
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.coverage = coverage
        self.bias = bias
        self.weight_norm = weight_norm
        self.end_bias = pointer_end_bias
        self.Wh = nn.Linear(enc_hidden_size, attention_size, bias=False)
        self.Ws = nn.Linear(dec_hidden_size, attention_size, bias=False)
        if coverage:
            self.Wc = nn.Linear(1, attention_size, bias=False)
        if bias:
            self.b = nn.Parameter(torch.randn(1, 1, attention_size))
        if weight_norm:
            v = nn.Linear(attention_size, 1, bias=False)
            self.v = nn.utils.weight_norm(v)
        else:
            self.v = nn.Linear(attention_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        if self.end_bias:
            self.end_energy = nn.Parameter(torch.randn(1, 1))

    def forward(self, encoder_outputs, decoder_state, mask, coverage=None):
        """
        Args:
            encoder_outputs [B, source_len, hidden_size]
            decoder_state [B, hidden_size]
            mask [B, source_len]
            coverage [B, source_len] (optional)
        Return:
            attention [B, source_len]

        e = v T tanh (Wh @ h + Ws @ s + b)
        a = softmax(e)

        e = v T tanh (Wh @ h + Ws @ s + Wc @ c + b) <= coverage
        a = softmax(e; bias) <= bias
        """
        B, source_len, _ = encoder_outputs.size()
        enc_out_energy = self.Wh(encoder_outputs)
        dec_state_energy = self.Ws(decoder_state).unsqueeze(1)
        energy = enc_out_energy + dec_state_energy
        if self.coverage:
            try:
                cov_energy = self.Wc(coverage.unsqueeze(2))
                energy = energy + cov_energy
            except RuntimeError:
                None
                None
        if self.bias:
            energy = energy + self.b
        energy = self.tanh(energy)
        energy = self.v(energy).squeeze(2)
        energy.masked_fill_(mask, -math.inf)
        if self.end_bias:
            end_energy = self.end_energy.expand(B, 1)
            energy = torch.cat([energy, end_energy], dim=1)
        attention = self.softmax(energy)
        return attention


class CopySwitch(nn.Module):

    def __init__(self, enc_hidden_size=512, dec_hidden_size=256):
        """Pointing the Unknown Words (ACL 2016)"""
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.W = nn.Linear(enc_hidden_size + dec_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, decoder_state, context):
        """
        Args:
            decoder_state [B, hidden_size]
            context [B, hidden_size]
        Return:
            p [B, 1]

        p = sigmoid(W @ s + U @ c + b)
        """
        p = self.W(torch.cat([decoder_state, context], dim=1))
        p = self.sigmoid(p)
        return p


class PointerGenerator(nn.Module):

    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, embed_size=128, rnn_type='LSTM'):
        """Estimation of Word Generation (vs Copying) Probability
        Get To The Point: Summarization with Pointer-Generator Networks (ACL 2017)"""
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.rnn_type = rnn_type
        self.W = nn.Linear(enc_hidden_size + dec_hidden_size + embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, context, decoder_state, decoder_input):
        """
        Args:
            context [B, hidden_size]
            decoder_state [B, hidden_size]
            decoder_input [B, embed_size]
        Return:
            p_gen [B, 1]

        p = sigmoid(wh @ h + ws @ s + wx @ x + b)
        """
        p_gen = self.W(torch.cat([context, decoder_state, decoder_input], dim=1))
        p_gen = self.sigmoid(p_gen)
        return p_gen


class Maxout(nn.Module):

    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[-1] % self._pool_size == 0, 'Wrong input last dim size ({}) for Maxout({})'.format(x.shape[-1], self._pool_size)
        m, i = x.view(*x.shape[:-1], x.shape[-1] // self._pool_size, self._pool_size).max(-1)
        return m


class NQGReadout(nn.Module):

    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, embed_size=300, vocab_size=30000, dropout_p=0.5, pool_size=2, tie=False):
        super().__init__()
        self.tie = tie
        if tie:
            self.W = nn.Linear(embed_size + enc_hidden_size + dec_hidden_size, embed_size)
            self.Wo = nn.Linear(embed_size, vocab_size, bias=False)
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.W = nn.Linear(embed_size + enc_hidden_size + dec_hidden_size, dec_hidden_size)
            self.maxout = Maxout(pool_size=pool_size)
            self.dropout = nn.Dropout(dropout_p)
            self.Wo = nn.Linear(dec_hidden_size // pool_size, vocab_size, bias=False)

    def forward(self, word_emb, context, decoder_state):
        if self.tie:
            r = self.W(torch.cat([word_emb, context, decoder_state], dim=1))
            r = torch.tanh(r)
            r = self.dropout(r)
            energy = self.Wo(r)
        else:
            r = self.W(torch.cat([word_emb, context, decoder_state], dim=1))
            r = self.maxout(r)
            r = self.dropout(r)
            energy = self.Wo(r)
        return energy


class PGReadout(nn.Module):

    def __init__(self, enc_hidden_size=512, dec_hidden_size=256, embed_size=128, vocab_size=50000, dropout_p=0.5, tie=False):
        super().__init__()
        self.tie = tie
        if tie:
            self.W1 = nn.Linear(enc_hidden_size + dec_hidden_size, embed_size)
            self.dropout = nn.Dropout(dropout_p)
            self.tanh = nn.Tanh()
            self.Wo = nn.Linear(embed_size, vocab_size, bias=False)
        elif dropout_p > 0:
            self.mlp = nn.Sequential(nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size), nn.Dropout(dropout_p), nn.Linear(dec_hidden_size, vocab_size))
        else:
            self.mlp = nn.Sequential(nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size), nn.Linear(dec_hidden_size, vocab_size))

    def forward(self, context, decoder_state):
        if self.tie:
            r = self.W1(torch.cat([context, decoder_state], dim=1))
            r = self.dropout(r)
            r = self.tanh(r)
            energy = self.Wo(r)
        else:
            energy = self.mlp(torch.cat([context, decoder_state], dim=1))
        return energy


class Beam(object):

    def __init__(self, batch_size, beam_size, EOS_ID=3):
        """Beam class for beam search"""
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.EOS_ID = EOS_ID
        self.back_pointers = []
        self.token_ids = []
        self.scores = []

    def backtrack(self):
        """Backtracks over batch to generate optimal k-sequences

        back_pointer [B, K]
        token_id [B, K]
        attention [B, K, source_L]

        Returns:
            prediction ([B, K, max_unroll])
                A list of Tensors containing predicted sequence
        """
        B = self.batch_size
        K = self.beam_size
        device = self.token_ids[0].device
        max_unroll = len(self.back_pointers)
        score = self.scores[-1].clone()
        n_eos_found = [0] * B
        back_pointer = torch.arange(0, K).unsqueeze(0).repeat(B, 1)
        prediction = []
        for t in reversed(range(max_unroll)):
            token_id = self.token_ids[t].gather(1, back_pointer)
            back_pointer = self.back_pointers[t].gather(1, back_pointer)
            where_EOS = self.token_ids[t] == self.EOS_ID
            if where_EOS.any():
                for eos_idx in reversed(where_EOS.nonzero().tolist()):
                    batch_idx, beam_idx = eos_idx
                    back_pointer[batch_idx, K - 1 - n_eos_found[batch_idx] % K] = self.back_pointers[t][batch_idx, beam_idx]
                    token_id[batch_idx, K - 1 - n_eos_found[batch_idx] % K] = self.token_ids[t][batch_idx, beam_idx]
                    score[batch_idx, K - 1 - n_eos_found[batch_idx] % K] = self.scores[t][batch_idx, beam_idx]
                    n_eos_found[batch_idx] += 1
            prediction.append(token_id)
        prediction = list(reversed(prediction))
        prediction = torch.stack(prediction, 2)
        score, score_idx = score.topk(K, dim=1)
        batch_starting_indices = torch.arange(0, B * K, step=K, device=device)
        batch_indices = (batch_starting_indices.unsqueeze(1) + score_idx).flatten()
        prediction = prediction.view(B * K, max_unroll).index_select(0, batch_indices).view(B, K, max_unroll)
        return prediction, score


def repeat(tensor, K):
    """
    [B, ...] => [B*K, ...]

    #-- Important --#
    Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
    """
    if isinstance(tensor, torch.Tensor):
        B, *size = tensor.size()
        repeat_size = [1] + [K] + [1] * (tensor.dim() - 1)
        tensor = tensor.unsqueeze(1).repeat(*repeat_size).view(B * K, *size)
        return tensor
    elif isinstance(tensor, list):
        out = []
        for x in tensor:
            for _ in range(K):
                out.append(x.copy())
        return out


class NQGDecoder(nn.Module):

    def __init__(self, embed_size=300, enc_hidden_size=512, dec_hidden_size=512, attention_size=512, vocab_size=20000, dropout_p=0.5, rnn='GRU', tie=False, n_mixture=None):
        """Neural Question Generation from Text: A Preliminary Study (2017)
        incorporated with copying mechanism of Pointing the Unknown Words (ACL 2016)"""
        super().__init__()
        input_size = embed_size + dec_hidden_size
        self.hidden_size = dec_hidden_size
        self.vocab_size = vocab_size
        if rnn == 'GRU':
            self.rnn_type = 'GRU'
            self.rnncell = nn.GRUCell(input_size=input_size, hidden_size=dec_hidden_size)
        elif rnn == 'LSTM':
            self.rnn_type = 'LSTM'
            self.rnncell = nn.LSTMCell(input_size=input_size, hidden_size=dec_hidden_size)
        self.attention = BahdanauAttention(enc_hidden_size, dec_hidden_size, attention_size=attention_size)
        self.copy_switch = CopySwitch(enc_hidden_size, dec_hidden_size)
        self.readout = NQGReadout(enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size, embed_size=embed_size, vocab_size=vocab_size, dropout_p=dropout_p, tie=tie)
        self.tie = tie
        self.n_mixture = n_mixture
        if n_mixture:
            self.mixture_embedding = nn.Embedding(n_mixture, embed_size)

    def forward(self, enc_outputs, s, source_WORD_encoding, answer_WORD_encoding=None, mixture_id=None, target_WORD_encoding=None, source_WORD_encoding_extended=None, train=True, decoding_type='beam', K=10, max_dec_len=30, temperature=1.0, diversity_lambda=0.5):
        device = enc_outputs.device
        B, max_source_len = source_WORD_encoding.size()
        V = self.vocab_size + max_source_len
        PAD_ID = 0
        UNK_ID = 1
        SOS_ID = 2
        EOS_ID = 3
        dec_input_word = torch.tensor([SOS_ID] * B, dtype=torch.long, device=device)
        pad_mask = source_WORD_encoding == PAD_ID
        if train:
            max_dec_len = target_WORD_encoding.size(1)
        else:
            max_dec_len = max_dec_len
            if decoding_type in ['beam', 'diverse_beam', 'topk_sampling']:
                dec_input_word = repeat(dec_input_word, K)
                if self.rnn_type == 'GRU':
                    s = repeat(s, K)
                elif self.rnn_type == 'LSTM':
                    s = repeat(s[0], K), repeat(s[1], K)
                enc_outputs = repeat(enc_outputs, K)
                pad_mask = repeat(pad_mask, K)
                if decoding_type in ['beam', 'diverse_beam']:
                    score = torch.zeros(B, K, device=device)
                    score[:, 1:] = -math.inf
                    beam = Beam(B, K, EOS_ID)
                    n_finished = torch.zeros(B, dtype=torch.long, device=device)
                elif decoding_type == 'topk_sampling':
                    score = torch.zeros(B * K, device=device)
                    finished = torch.zeros(B * K, dtype=torch.uint8, device=device)
            else:
                finished = torch.zeros(B, dtype=torch.uint8, device=device)
                score = torch.zeros(B, device=device)
        out_log_p = []
        output_sentence = []
        self.attention_list = []
        for i in range(max_dec_len):
            if i == 0 and self.n_mixture:
                dec_input_word_embed = self.mixture_embedding(mixture_id)
            else:
                dec_input_word_embed = self.word_embed(dec_input_word)
            if i == 0:
                context = torch.zeros_like(enc_outputs[:, 0])
            dec_input = torch.cat([dec_input_word_embed, context], dim=1)
            s = self.rnncell(dec_input, s)
            if self.rnn_type == 'GRU':
                attention = self.attention(enc_outputs, s, pad_mask)
            if self.rnn_type == 'LSTM':
                attention = self.attention(enc_outputs, s[0], pad_mask)
            self.attention_list.append(attention)
            context = torch.bmm(attention.unsqueeze(1), enc_outputs)
            context = context.squeeze(1)
            if self.rnn_type == 'GRU':
                p_copy = self.copy_switch(s, context)
            if self.rnn_type == 'LSTM':
                p_copy = self.copy_switch(s[0], context)
            if self.rnn_type == 'GRU':
                p_vocab = self.readout(dec_input_word_embed, context, s)
            if self.rnn_type == 'LSTM':
                p_vocab = self.readout(dec_input_word_embed, context, s[0])
            if not train:
                p_vocab[:, PAD_ID] = -math.inf
                p_vocab[:, UNK_ID] = -math.inf
            p_vocab = F.softmax(p_vocab, dim=1)
            p_out = torch.cat([(1 - p_copy) * p_vocab, p_copy * attention], dim=1)
            p_out = p_out + 1e-12
            log_p = p_out.log()
            out_log_p.append(log_p)
            if train:
                dec_input_word = target_WORD_encoding[:, i]
                unk = torch.full_like(dec_input_word, UNK_ID)
                dec_input_word = torch.where(dec_input_word >= self.vocab_size, unk, dec_input_word)
            elif decoding_type in ['beam', 'diverse_beam']:
                current_score = log_p.view(B, K, V)
                if decoding_type == 'diverse_beam':
                    diversity_penalty = torch.zeros(B, V, device=device)
                    for k in range(K):
                        current_beam_score = current_score[:, k]
                        if k > 0:
                            diversity_penalty.scatter_add_(1, beam_word_id, torch.ones(B, V, device=device))
                            current_beam_score -= diversity_lambda * diversity_penalty
                        beam_word_id = current_beam_score.argmax(dim=1, keepdim=True)
                score = score.view(B, K, 1) + current_score
                score = score.view(B, -1)
                topk_score, topk_idx = score.topk(K, dim=1)
                topk_beam_idx = topk_idx // V
                topk_word_id = topk_idx % V
                beam.back_pointers.append(topk_beam_idx.clone())
                beam.token_ids.append(topk_word_id.clone())
                beam.scores.append(topk_score.clone())
                batch_starting_indices = torch.arange(0, B * K, step=K, device=device)
                batch_indices = batch_starting_indices.unsqueeze(1)
                topk_beam_idx_flat = (batch_indices + topk_beam_idx).flatten()
                if self.rnn_type == 'GRU':
                    s = s.index_select(0, topk_beam_idx_flat)
                elif self.rnn_type == 'LSTM':
                    s = s[0].index_select(0, topk_beam_idx_flat), s[1].index_select(0, topk_beam_idx_flat)
                score = topk_score
                where_EOS = topk_word_id == EOS_ID
                score.masked_fill_(where_EOS, -math.inf)
                predicted_word_id = topk_word_id.flatten()
                where_oov = predicted_word_id >= self.vocab_size
                dec_input_word = predicted_word_id.masked_fill(where_oov, UNK_ID)
                generated_eos = topk_word_id == EOS_ID
                if generated_eos.any():
                    n_finished += generated_eos.long().sum(dim=1)
                    if n_finished.min().item() >= K:
                        break
            elif decoding_type in ['greedy', 'topk_sampling']:
                if decoding_type == 'greedy':
                    log_p_sampled, predicted_word_id = log_p.max(dim=1)
                elif decoding_type == 'topk_sampling':
                    topk = 10
                    log_p_topk, predicted_word_id_topk = log_p.topk(topk, dim=1)
                    temperature_scaled_score = (log_p_topk / temperature).exp()
                    sampled_idx = temperature_scaled_score.multinomial(1)
                    log_p_sampled = temperature_scaled_score.gather(1, sampled_idx).squeeze(1)
                    predicted_word_id = predicted_word_id_topk.gather(1, sampled_idx).squeeze(1)
                log_p_sampled.masked_fill_(finished, 0)
                score += log_p_sampled
                where_oov = predicted_word_id >= self.vocab_size
                dec_input_word = predicted_word_id.masked_fill(where_oov, UNK_ID)
                output_sentence.append(predicted_word_id)
                generated_eos = predicted_word_id == EOS_ID
                if generated_eos.any():
                    finished += generated_eos
                    finished.clamp_(0, 1)
                    if finished.min().item() > 0:
                        break
        if train:
            log_p = torch.stack(out_log_p, dim=1)
            return log_p
        elif decoding_type in ['beam', 'diverse_beam']:
            output_sentence, score = beam.backtrack()
            return output_sentence, score
        else:
            output_sentence = torch.stack(output_sentence, dim=-1).view(B, 1, -1)
            return output_sentence, score


class PGDecoder(nn.Module):

    def __init__(self, embed_size=128, enc_hidden_size=512, dec_hidden_size=256, attention_size=700, vocab_size=50000, dropout_p=0.0, rnn='LSTM', tie=False, n_mixture=None):
        """Get To The Point: Summarization with Pointer-Generator Networks (ACL 2017)"""
        super().__init__()
        input_size = embed_size
        self.input_linear = nn.Linear(embed_size + enc_hidden_size, input_size)
        self.hidden_size = dec_hidden_size
        self.vocab_size = vocab_size
        if rnn == 'GRU':
            self.rnn_type = 'GRU'
            self.rnncell = nn.GRUCell(input_size=input_size, hidden_size=dec_hidden_size)
        elif rnn == 'LSTM':
            self.rnn_type = 'LSTM'
            self.rnncell = nn.LSTMCell(input_size=input_size, hidden_size=dec_hidden_size)
        self.attention = BahdanauAttention(enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size if rnn == 'GRU' else 2 * dec_hidden_size, attention_size=attention_size, coverage=True)
        self.pointer_switch = PointerGenerator(enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size if rnn == 'GRU' else 2 * dec_hidden_size, embed_size=embed_size)
        self.readout = PGReadout(enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size, embed_size=embed_size, vocab_size=vocab_size, dropout_p=dropout_p, tie=tie)
        self.tie = tie
        self.n_mixture = n_mixture
        if n_mixture:
            self.mixture_embedding = nn.Embedding(n_mixture, embed_size)

    def forward(self, enc_outputs, s, source_WORD_encoding, answer_WORD_encoding=None, mixture_id=None, target_WORD_encoding=None, source_WORD_encoding_extended=None, train=True, decoding_type='beam', K=10, max_dec_len=100, temperature=1.0, diversity_lambda=0.5):
        device = enc_outputs.device
        B, max_source_len = source_WORD_encoding.size()
        PAD_ID = 0
        UNK_ID = 1
        SOS_ID = 2
        EOS_ID = 3
        dec_input_word = torch.tensor([SOS_ID] * B, dtype=torch.long, device=device)
        pad_mask = source_WORD_encoding == PAD_ID
        coverage = torch.zeros(B, max_source_len, device=device)
        max_n_oov = source_WORD_encoding_extended.max().item() - self.vocab_size + 1
        max_n_oov = max(max_n_oov, 1)
        V = self.vocab_size + max_n_oov
        if train:
            max_dec_len = target_WORD_encoding.size(1)
        else:
            max_dec_len = max_dec_len
            if decoding_type in ['beam', 'diverse_beam', 'topk_sampling']:
                dec_input_word = repeat(dec_input_word, K)
                if self.rnn_type == 'GRU':
                    s = repeat(s, K)
                elif self.rnn_type == 'LSTM':
                    s = repeat(s[0], K), repeat(s[1], K)
                enc_outputs = repeat(enc_outputs, K)
                pad_mask = repeat(pad_mask, K)
                source_WORD_encoding_extended = repeat(source_WORD_encoding_extended, K)
                coverage = repeat(coverage, K)
                if decoding_type in ['beam', 'diverse_beam']:
                    score = torch.zeros(B, K, device=device)
                    score[:, 1:] = -math.inf
                    beam = Beam(B, K, EOS_ID)
                    n_finished = torch.zeros(B, dtype=torch.long, device=device)
                elif decoding_type == 'topk_sampling':
                    score = torch.zeros(B * K, device=device)
                    finished = torch.zeros(B * K, dtype=torch.uint8, device=device)
            else:
                finished = torch.zeros(B, dtype=torch.uint8, device=device)
                score = torch.zeros(B, device=device)
        out_log_p = []
        output_sentence = []
        coverage_loss_list = []
        self.attention_list = []
        for i in range(max_dec_len):
            if i == 0 and self.n_mixture:
                dec_input_word_embed = self.mixture_embedding(mixture_id)
            else:
                dec_input_word_embed = self.word_embed(dec_input_word)
            if i == 0:
                context = torch.zeros_like(enc_outputs[:, 0])
            dec_input = self.input_linear(torch.cat([dec_input_word_embed, context], dim=1))
            s = self.rnncell(dec_input, s)
            if self.rnn_type == 'LSTM':
                s_cat = torch.cat([s[0], s[1]], dim=1)
            if self.rnn_type == 'GRU':
                attention = self.attention(enc_outputs, s, pad_mask, coverage)
            if self.rnn_type == 'LSTM':
                attention = self.attention(enc_outputs, s_cat, pad_mask, coverage)
            self.attention_list.append(attention)
            context = torch.bmm(attention.unsqueeze(1), enc_outputs)
            context = context.squeeze(1)
            if train:
                step_coverage_loss = torch.sum(torch.min(attention, coverage), dim=1)
                coverage_loss_list.append(step_coverage_loss)
            coverage = coverage + attention
            if self.rnn_type == 'GRU':
                p_gen = self.pointer_switch(context, s, dec_input)
            if self.rnn_type == 'LSTM':
                p_gen = self.pointer_switch(context, s_cat, dec_input)
            if self.rnn_type == 'GRU':
                p_vocab = self.readout(context, s)
            if self.rnn_type == 'LSTM':
                p_vocab = self.readout(context, s[0])
            if not train:
                p_vocab[:, PAD_ID] = -math.inf
                p_vocab[:, UNK_ID] = -math.inf
            p_vocab = F.softmax(p_vocab, dim=1)
            ext_zeros = torch.zeros(p_vocab.size(0), max_n_oov, device=device)
            p_out = torch.cat([p_vocab, ext_zeros], dim=1)
            p_out = p_out * p_gen
            p_out.scatter_add_(1, source_WORD_encoding_extended, (1 - p_gen) * attention)
            if not train:
                p_out[:, UNK_ID] = 0
            p_out = p_out + 1e-12
            log_p = p_out.log()
            out_log_p.append(log_p)
            if train:
                dec_input_word = target_WORD_encoding[:, i]
                unk = torch.full_like(dec_input_word, UNK_ID)
                dec_input_word = torch.where(dec_input_word >= self.vocab_size, unk, dec_input_word)
            elif decoding_type in ['beam', 'diverse_beam']:
                current_score = log_p.view(B, K, V)
                if max_dec_len > 30:
                    min_dec_len = 35
                    if i + 1 < min_dec_len:
                        current_score[:, :, EOS_ID] = -math.inf
                if decoding_type == 'diverse_beam':
                    diversity_penalty = torch.zeros(B, V, device=device)
                    for k in range(K):
                        current_beam_score = current_score[:, k]
                        if k > 0:
                            diversity_penalty.scatter_add_(1, beam_word_id, torch.ones(B, V, device=device))
                            current_beam_score -= diversity_lambda * diversity_penalty
                        beam_word_id = current_beam_score.argmax(dim=1, keepdim=True)
                score = score.view(B, K, 1) + current_score
                score = score.view(B, -1)
                topk_score, topk_idx = score.topk(K, dim=1)
                topk_beam_idx = topk_idx // V
                topk_word_id = topk_idx % V
                beam.back_pointers.append(topk_beam_idx)
                beam.token_ids.append(topk_word_id)
                beam.scores.append(topk_score)
                batch_starting_indices = torch.arange(0, B * K, step=K, device=device)
                batch_indices = batch_starting_indices.unsqueeze(1)
                topk_beam_idx_flat = (batch_indices + topk_beam_idx).flatten()
                if self.rnn_type == 'GRU':
                    s = s.index_select(0, topk_beam_idx_flat)
                elif self.rnn_type == 'LSTM':
                    s = s[0].index_select(0, topk_beam_idx_flat), s[1].index_select(0, topk_beam_idx_flat)
                attention = attention.index_select(0, topk_beam_idx_flat)
                coverage = coverage.index_select(0, topk_beam_idx_flat)
                score = topk_score
                where_EOS = topk_word_id == EOS_ID
                score.masked_fill_(where_EOS, -math.inf)
                predicted_word_id = topk_word_id.flatten()
                where_oov = predicted_word_id >= self.vocab_size
                dec_input_word = predicted_word_id.masked_fill(where_oov, UNK_ID)
                generated_eos = topk_word_id == EOS_ID
                if generated_eos.any():
                    n_finished += generated_eos.long().sum(dim=1)
                    if n_finished.min().item() >= K:
                        break
            else:
                if decoding_type == 'greedy':
                    log_p_sampled, predicted_word_id = log_p.max(dim=1)
                elif decoding_type == 'topk_sampling':
                    topk = 10
                    log_p_topk, predicted_word_id_topk = log_p.topk(topk, dim=1)
                    temperature_scaled_score = (log_p_topk / temperature).exp()
                    sampled_idx = temperature_scaled_score.multinomial(1)
                    log_p_sampled = temperature_scaled_score.gather(1, sampled_idx).squeeze(1)
                    predicted_word_id = predicted_word_id_topk.gather(1, sampled_idx).squeeze(1)
                log_p_sampled.masked_fill_(finished, 0)
                score += log_p_sampled
                where_oov = predicted_word_id >= self.vocab_size
                dec_input_word = predicted_word_id.masked_fill(where_oov, UNK_ID)
                output_sentence.append(predicted_word_id)
                generated_eos = predicted_word_id == EOS_ID
                if generated_eos.any():
                    finished += generated_eos
                    finished.clamp_(0, 1)
                    if finished.min().item() > 0:
                        break
        if train:
            log_p = torch.stack(out_log_p, dim=1)
            coverage_loss = torch.stack(coverage_loss_list, dim=1)
            return log_p, coverage_loss
        elif decoding_type in ['beam', 'diverse_beam']:
            output_sentence, score = beam.backtrack()
            return output_sentence, score
        else:
            output_sentence = torch.stack(output_sentence, dim=-1).view(B, 1, -1)
            return output_sentence, score


class FocusedEncoder(nn.Module):

    def __init__(self, word_embed_size=300, answer_position_embed_size=16, ner_embed_size=16, pos_embed_size=16, case_embed_size=16, focus_embed_size=16, hidden_size=512, num_layers=1, dropout_p=0.5, use_focus=True, model='NQG', rnn_type='GRU', feature_rich=False):
        super().__init__()
        self.model = model
        self.feature_rich = feature_rich
        rnn_input_size = word_embed_size
        if feature_rich:
            rnn_input_size += answer_position_embed_size + ner_embed_size + pos_embed_size + case_embed_size
        if use_focus:
            rnn_input_size += focus_embed_size
        if rnn_type == 'GRU':
            self.rnn_type = 'GRU'
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=hidden_size // 2, bidirectional=True, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn_type = 'LSTM'
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_size // 2, bidirectional=True, batch_first=True)
        if dropout_p > 0:
            self.use_dropout = True
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.use_dropout = False
        self.use_focus = use_focus

    def forward(self, source_WORD_encoding, answer_position_BIO_encoding=None, ner_encoding=None, pos_encoding=None, case_encoding=None, focus_mask=None, PAD_ID=0):
        pad_mask = source_WORD_encoding == PAD_ID
        source_len = (~pad_mask).long().sum(dim=1)
        source_len_sorted, idx_sorted = torch.sort(source_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sorted, dim=0)
        source_WORD_encoding = source_WORD_encoding[idx_sorted]
        word_embedding = self.word_embed(source_WORD_encoding)
        enc_input = word_embedding
        if self.feature_rich:
            answer_position_BIO_encoding = answer_position_BIO_encoding[idx_sorted]
            ner_encoding = ner_encoding[idx_sorted]
            pos_encoding = pos_encoding[idx_sorted]
            case_encoding = case_encoding[idx_sorted]
            answer_position_embedding = self.answer_position_embed(answer_position_BIO_encoding)
            ner_embedding = self.ner_embed(ner_encoding)
            pos_embedding = self.pos_embed(pos_encoding)
            case_embedding = self.case_embed(case_encoding)
            enc_input = torch.cat([enc_input, answer_position_embedding, ner_embedding, pos_embedding, case_embedding], dim=2)
        if self.use_focus:
            focus_mask = focus_mask[idx_sorted]
            focus_embedding = self.focus_embed(focus_mask)
            enc_input = torch.cat([enc_input, focus_embedding], dim=2)
        if self.use_dropout:
            enc_input = self.dropout(enc_input)
        enc_input = pack_padded_sequence(enc_input, source_len_sorted, batch_first=True)
        enc_outputs, h = self.rnn(enc_input)
        enc_outputs, _ = pad_packed_sequence(enc_outputs, batch_first=True)
        enc_outputs = enc_outputs[idx_unsort]
        if self.rnn_type == 'LSTM':
            h, c = h
            h = h.index_select(1, idx_unsort)
            c = c.index_select(1, idx_unsort)
            h = h, c
        elif self.rnn_type == 'GRU':
            h = h.index_select(1, idx_unsort)
        return enc_outputs, h


class GRUEncoder(nn.Module):

    def __init__(self, word_embed_size=300, answer_position_embed_size=16, ner_embed_size=16, pos_embed_size=16, case_embed_size=16, hidden_size=300, dropout_p=0.2, task='QG'):
        super().__init__()
        self.task = task
        if task == 'QG':
            input_size = word_embed_size + answer_position_embed_size + ner_embed_size + pos_embed_size + case_embed_size
        else:
            input_size = word_embed_size
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size // 2, batch_first=True, num_layers=1, bidirectional=True)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, source_WORD_encoding, answer_position_BIO_encoding=None, ner_encoding=None, pos_encoding=None, case_encoding=None, PAD_ID=0):
        pad_mask = source_WORD_encoding == PAD_ID
        source_len = (~pad_mask).long().sum(dim=1)
        source_len_sorted, idx_sorted = torch.sort(source_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sorted, dim=0)
        if self.task == 'QG':
            source_WORD_encoding = source_WORD_encoding[idx_sorted]
            answer_position_BIO_encoding = answer_position_BIO_encoding[idx_sorted]
            ner_encoding = ner_encoding[idx_sorted]
            pos_encoding = pos_encoding[idx_sorted]
            case_encoding = case_encoding[idx_sorted]
            word_embedding = self.word_embed(source_WORD_encoding)
            answer_position_embedding = self.answer_position_embed(answer_position_BIO_encoding)
            ner_embedding = self.ner_embed(ner_encoding)
            pos_embedding = self.pos_embed(pos_encoding)
            case_embedding = self.case_embed(case_encoding)
            enc_input = torch.cat([word_embedding, answer_position_embedding, ner_embedding, pos_embedding, case_embedding], dim=2)
        else:
            source_WORD_encoding = source_WORD_encoding[idx_sorted]
            word_embedding = self.word_embed(source_WORD_encoding)
            enc_input = word_embedding
        enc_input = self.dropout(enc_input)
        enc_input = pack_padded_sequence(enc_input, source_len_sorted, batch_first=True)
        enc_outputs, h = self.rnn(enc_input)
        enc_outputs, _ = pad_packed_sequence(enc_outputs, batch_first=True)
        enc_outputs = enc_outputs[idx_unsort]
        h = h.index_select(1, idx_unsort)
        return enc_outputs, h


class ParallelDecoder(nn.Module):

    def __init__(self, embed_size=300, enc_hidden_size=512, dec_hidden_size=512, n_mixture=5, threshold=0.15, task='QG'):
        """Parallel Decoder for Focus Selector"""
        super().__init__()
        self.mixture_embedding = nn.Embedding(n_mixture, embed_size)
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.out_mlp = nn.Sequential(nn.Linear(enc_hidden_size + dec_hidden_size + embed_size, dec_hidden_size), nn.Tanh(), nn.Linear(dec_hidden_size, 1))
        self.threshold = threshold

    def forward(self, enc_outputs, s, source_WORD_encoding, mixture_id, focus_input=None, train=True, max_decoding_len=None):
        B, max_source_len = source_WORD_encoding.size()
        mixture_embedding = self.mixture_embedding(mixture_id)
        concat_h = torch.cat([enc_outputs, s.unsqueeze(1).expand(-1, max_source_len, -1), mixture_embedding.unsqueeze(1).expand(-1, max_source_len, -1)], dim=2)
        focus_logit = self.out_mlp(concat_h).squeeze(2)
        if train:
            return focus_logit
        else:
            focus_p = torch.sigmoid(focus_logit)
            return focus_p


class ParallelSelector(nn.Module):

    def __init__(self, word_embed_size=300, answer_position_embed_size=16, ner_embed_size=16, pos_embed_size=16, case_embed_size=16, enc_hidden_size=300, dec_hidden_size=300, num_layers=1, dropout_p=0.2, n_mixture=5, task='QG', threshold=0.15):
        super().__init__()
        self.task = task
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.n_mixture = n_mixture
        self.encoder = GRUEncoder(word_embed_size, answer_position_embed_size, ner_embed_size, pos_embed_size, case_embed_size, enc_hidden_size, task=task)
        self.dec_init = nn.Sequential(nn.Linear(enc_hidden_size, dec_hidden_size), nn.LeakyReLU())
        self.decoder = ParallelDecoder(word_embed_size, enc_hidden_size, dec_hidden_size, n_mixture=n_mixture, task=task, threshold=threshold)

    def forward(self, source_WORD_encoding, answer_position_BIO_encoding=None, ner_encoding=None, pos_encoding=None, case_encoding=None, mixture_id=None, focus_input=None, train=True, max_decoding_len=None):
        enc_outputs, h = self.encoder(source_WORD_encoding, answer_position_BIO_encoding, ner_encoding, pos_encoding, case_encoding)
        B = h.size(1)
        h = h.transpose(0, 1).contiguous().view(B, self.enc_hidden_size)
        s = self.dec_init(h)
        if mixture_id is None:
            enc_outputs = repeat(enc_outputs, self.n_mixture)
            s = repeat(s, self.n_mixture)
            source_WORD_encoding = repeat(source_WORD_encoding, self.n_mixture)
            focus_input = repeat(focus_input, self.n_mixture)
            mixture_id = torch.arange(self.n_mixture, dtype=torch.long, device=s.device).unsqueeze(0).repeat(B, 1).flatten()
        else:
            assert mixture_id.size(0) == B
        dec_output = self.decoder(enc_outputs, s, source_WORD_encoding, mixture_id, focus_input, train, max_decoding_len=None)
        if train:
            focus_logit = dec_output
            return focus_logit
        else:
            focus_p = dec_output
            return focus_p


class Model(nn.Module):

    def __init__(self, seq2seq, selector=None):
        super().__init__()
        self.selector = selector
        self.seq2seq = seq2seq


class FocusSelector(nn.Module):
    """Sample focus (sequential binary masks) from source sequence"""

    def __init__(self, word_embed_size: int=300, answer_position_embed_size: int=16, ner_embed_size: int=16, pos_embed_size: int=16, case_embed_size: int=16, focus_embed_size: int=16, enc_hidden_size: int=300, dec_hidden_size: int=300, num_layers: int=1, dropout_p: float=0.2, rnn: str='GRU', n_mixture: int=1, seq2seq_model: str='NQG', task: str='QG', threshold: float=0.15, feature_rich: bool=False):
        super().__init__()
        self.task = task
        self.seq2seq_model = seq2seq_model
        self.feature_rich = feature_rich
        self.selector = ParallelSelector(word_embed_size, answer_position_embed_size, ner_embed_size, pos_embed_size, case_embed_size, enc_hidden_size, dec_hidden_size, num_layers=num_layers, dropout_p=dropout_p, n_mixture=n_mixture, task=task, threshold=threshold)

    def add_embedding(self, word_embed, answer_position_embed=None, ner_embed=None, pos_embed=None, case_embed=None):
        if self.feature_rich:
            self.selector.encoder.word_embed = word_embed
            self.selector.encoder.answer_position_embed = answer_position_embed
            self.selector.encoder.ner_embed = ner_embed
            self.selector.encoder.pos_embed = pos_embed
            self.selector.encoder.case_embed = case_embed
        else:
            self.selector.encoder.word_embed = word_embed

    def forward(self, source_WORD_encoding, answer_position_BIO_encoding=None, ner_encoding=None, pos_encoding=None, case_encoding=None, focus_POS_prob=None, mixture_id=None, focus_input=None, train=True, max_decoding_len=30):
        out = self.selector(source_WORD_encoding, answer_position_BIO_encoding, ner_encoding, pos_encoding, case_encoding, mixture_id, focus_input, train, max_decoding_len)
        if train:
            focus_logit = out
            return focus_logit
        else:
            generated_focus_mask = out
            return generated_focus_mask


class Seq2Seq(nn.Module):

    def __init__(self, vocab_size: int=20000, word_embed_size: int=300, answer_position_embed_size: int=16, ner_embed_size: int=16, pos_embed_size: int=16, case_embed_size: int=16, position_embed_size: int=16, focus_embed_size: int=16, enc_hidden_size: int=512, dec_hidden_size: int=256, num_layers: int=1, dropout_p: float=0.5, tie: bool=False, rnn: str='GRU', use_focus: bool=False, task: str='QG', model: str='NQG', feature_rich: bool=False, n_mixture=None):
        """Neural Question Generation from Text: A Preliminary Study (Zhou et al. NLPCC 2017)
        Get To The Point: Summarization with Pointer-Generator Networks (See et al. ACL 2017)
        """
        super().__init__()
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.vocab_size = vocab_size
        self.tie = tie
        self.rnn_type = rnn
        self.use_focus = use_focus
        self.n_mixture = n_mixture
        self.model = model
        self.task = task
        self.feature_rich = feature_rich
        if model == 'NQG':
            assert feature_rich == True
        self.word_embed = nn.Embedding(vocab_size, word_embed_size, padding_idx=0)
        self.encoder = FocusedEncoder(word_embed_size, answer_position_embed_size, ner_embed_size, pos_embed_size, case_embed_size, focus_embed_size, enc_hidden_size, num_layers, dropout_p, rnn_type=self.rnn_type, use_focus=use_focus, model=model, feature_rich=feature_rich)
        self.encoder.word_embed = self.word_embed
        if feature_rich:
            self.answer_position_embed = nn.Embedding(4, answer_position_embed_size, padding_idx=-1)
            self.ner_embed = nn.Embedding(13, ner_embed_size, padding_idx=-1)
            self.pos_embed = nn.Embedding(46, pos_embed_size, padding_idx=-1)
            self.case_embed = nn.Embedding(3, case_embed_size, padding_idx=-1)
            self.encoder.answer_position_embed = self.answer_position_embed
            self.encoder.ner_embed = self.ner_embed
            self.encoder.pos_embed = self.pos_embed
            self.encoder.case_embed = self.case_embed
        else:
            self.answer_position_embed = None
            self.ner_embed = None
            self.pos_embed = None
            self.case_embed = None
        if use_focus:
            self.focus_embed = nn.Embedding(3, focus_embed_size, padding_idx=-1)
            self.encoder.focus_embed = self.focus_embed
        if model == 'NQG':
            self.bridge = LinearBridge(enc_hidden_size, dec_hidden_size, rnn, 'tanh')
        elif model == 'PG':
            self.bridge = LinearBridge(enc_hidden_size, dec_hidden_size, rnn, 'ReLU')
        if model == 'NQG':
            self.decoder = NQGDecoder(word_embed_size, enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size, attention_size=700, vocab_size=vocab_size, dropout_p=dropout_p, rnn=rnn, tie=tie, n_mixture=n_mixture)
        elif model == 'PG':
            self.decoder = PGDecoder(word_embed_size, enc_hidden_size=enc_hidden_size, dec_hidden_size=dec_hidden_size, vocab_size=vocab_size, dropout_p=0, rnn=rnn, tie=tie, n_mixture=n_mixture)
        self.decoder.word_embed = self.word_embed
        if tie:
            self.decoder.readout.Wo.weight = self.word_embed.weight

    def forward(self, source_WORD_encoding, answer_WORD_encoding=None, answer_position_BIO_encoding=None, ner_encoding=None, pos_encoding=None, case_encoding=None, focus_mask=None, mixture_id=None, target_WORD_encoding=None, source_WORD_encoding_extended=None, train=True, decoding_type='beam', beam_k=12, max_dec_len=30, temperature=1.0, diversity_lambda=0.5):
        enc_outputs, h = self.encoder(source_WORD_encoding, answer_position_BIO_encoding=answer_position_BIO_encoding, ner_encoding=ner_encoding, pos_encoding=pos_encoding, case_encoding=case_encoding, focus_mask=focus_mask)
        B = enc_outputs.size(0)
        s = self.bridge(h)
        if self.n_mixture:
            if mixture_id is None:
                enc_outputs = repeat(enc_outputs, self.n_mixture)
                if self.rnn_type == 'GRU':
                    s = repeat(s, self.n_mixture)
                    device = s.device
                elif self.rnn_type == 'LSTM':
                    s = repeat(s[0], self.n_mixture), repeat(s[1], self.n_mixture)
                    device = s[0].device
                source_WORD_encoding = repeat(source_WORD_encoding, self.n_mixture)
                if self.model == 'PG':
                    source_WORD_encoding_extended = repeat(source_WORD_encoding_extended, self.n_mixture)
                mixture_id = torch.arange(self.n_mixture, dtype=torch.long, device=device).unsqueeze(0).repeat(B, 1).flatten()
                if train:
                    target_WORD_encoding = repeat(target_WORD_encoding, self.n_mixture)
            else:
                assert mixture_id.size(0) == B
        dec_output = self.decoder(enc_outputs, s, source_WORD_encoding, answer_WORD_encoding=answer_WORD_encoding, mixture_id=mixture_id, target_WORD_encoding=target_WORD_encoding, source_WORD_encoding_extended=source_WORD_encoding_extended, train=train, decoding_type=decoding_type, K=beam_k, max_dec_len=max_dec_len, temperature=temperature, diversity_lambda=diversity_lambda)
        if train:
            return dec_output
        else:
            output_sentence, score = dec_output
            return output_sentence, score


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LinearBridge,
     lambda: ([], {}),
     lambda: ([torch.rand([512, 512])], {}),
     False),
    (Maxout,
     lambda: ([], {'pool_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_clovaai_FocusSeq2Seq(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

