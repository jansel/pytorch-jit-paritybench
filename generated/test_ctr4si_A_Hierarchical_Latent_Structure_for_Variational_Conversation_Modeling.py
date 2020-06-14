import sys
_module = sys.modules[__name__]
del sys
cornell_preprocess = _module
model = _module
configs = _module
data_loader = _module
eval = _module
eval_embed = _module
layers = _module
beam_search = _module
decoder = _module
encoder = _module
feedforward = _module
loss = _module
rnncells = _module
models = _module
solver = _module
train = _module
utils = _module
bow = _module
embedding_metric = _module
mask = _module
pad = _module
probability = _module
tensorboard = _module
time_track = _module
tokenizer = _module
vocab = _module
ubuntu_preprocess = _module

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


from collections import defaultdict


from torch import optim


import torch.nn as nn


import random


import torch


from torch import nn


from torch.nn import functional as F


import math


import torch.nn.functional as F


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import PackedSequence


import copy


import numpy as np


from itertools import cycle


from math import isnan


import re


from collections import Counter


PAD_ID, UNK_ID, SOS_ID, EOS_ID = [0, 1, 2, 3]


class Beam(object):

    def __init__(self, batch_size, hidden_size, vocab_size, beam_size,
        max_unroll, batch_position):
        """Beam class for beam search"""
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.beam_size = beam_size
        self.max_unroll = max_unroll
        self.batch_position = batch_position
        self.log_probs = list()
        self.scores = list()
        self.back_pointers = list()
        self.token_ids = list()
        self.metadata = {'inputs': None, 'output': None, 'scores': None,
            'length': None, 'sequence': None}

    def update(self, score, back_pointer, token_id):
        """Append intermediate top-k candidates to beam at each step"""
        self.scores.append(score)
        self.back_pointers.append(back_pointer)
        self.token_ids.append(token_id)

    def backtrack(self):
        """Backtracks over batch to generate optimal k-sequences

        Returns:
            prediction ([batch, k, max_unroll])
                A list of Tensors containing predicted sequence
            final_score [batch, k]
                A list containing the final scores for all top-k sequences
            length [batch, k]
                A list specifying the length of each sequence in the top-k candidates
        """
        prediction = list()
        length = [([self.max_unroll] * self.beam_size) for _ in range(self.
            batch_size)]
        top_k_score, top_k_idx = self.scores[-1].topk(self.beam_size, dim=1)
        top_k_score = top_k_score.clone()
        n_eos_in_batch = [0] * self.batch_size
        back_pointer = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)
        for t in reversed(range(self.max_unroll)):
            token_id = self.token_ids[t].index_select(0, back_pointer)
            back_pointer = self.back_pointers[t].index_select(0, back_pointer)
            eos_indices = self.token_ids[t].data.eq(EOS_ID).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0) - 1, -1, -1):
                    eos_idx = eos_indices[i, 0].item()
                    batch_idx = eos_idx // self.beam_size
                    batch_start_idx = batch_idx * self.beam_size
                    _n_eos_in_batch = n_eos_in_batch[batch_idx
                        ] % self.beam_size
                    beam_idx_to_be_replaced = (self.beam_size -
                        _n_eos_in_batch - 1)
                    idx_to_be_replaced = (batch_start_idx +
                        beam_idx_to_be_replaced)
                    back_pointer[idx_to_be_replaced] = self.back_pointers[t][
                        eos_idx].item()
                    token_id[idx_to_be_replaced] = self.token_ids[t][eos_idx
                        ].item()
                    top_k_score[batch_idx, beam_idx_to_be_replaced
                        ] = self.scores[t].view(-1)[eos_idx].item()
                    length[batch_idx][beam_idx_to_be_replaced] = t + 1
                    n_eos_in_batch[batch_idx] += 1
            prediction.append(token_id)
        top_k_score, top_k_idx = top_k_score.topk(self.beam_size, dim=1)
        final_score = top_k_score.data
        for batch_idx in range(self.batch_size):
            length[batch_idx] = [length[batch_idx][beam_idx.item()] for
                beam_idx in top_k_idx[batch_idx]]
        top_k_idx = (top_k_idx + self.batch_position.unsqueeze(1)).view(-1)
        prediction = [step.index_select(0, top_k_idx).view(self.batch_size,
            self.beam_size) for step in reversed(prediction)]
        prediction = torch.stack(prediction, 2)
        return prediction, final_score, length


class BaseRNNDecoder(nn.Module):

    def __init__(self):
        """Base Decoder Class"""
        super(BaseRNNDecoder, self).__init__()

    @property
    def use_lstm(self):
        return isinstance(self.rnncell, StackedLSTMCell)

    def init_token(self, batch_size, SOS_ID=SOS_ID):
        """Get Variable of <SOS> Index (batch_size)"""
        x = to_var(torch.LongTensor([SOS_ID] * batch_size))
        return x

    def init_h(self, batch_size=None, zero=True, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden
        if self.use_lstm:
            return to_var(torch.zeros(self.num_layers, batch_size, self.
                hidden_size)), to_var(torch.zeros(self.num_layers,
                batch_size, self.hidden_size))
        else:
            return to_var(torch.zeros(self.num_layers, batch_size, self.
                hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTMCell)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size
        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def decode(self, out):
        """
        Args:
            out: unnormalized word distribution [batch_size, vocab_size]
        Return:
            x: word_index [batch_size]
        """
        if self.sample:
            x = torch.multinomial(self.softmax(out / self.temperature), 1
                ).view(-1)
        else:
            _, x = out.max(dim=1)
        return x

    def forward(self):
        """Base forward function to inherit"""
        raise NotImplementedError

    def forward_step(self):
        """Run RNN single step"""
        raise NotImplementedError

    def embed(self, x):
        """word index: [batch_size] => word vectors: [batch_size, hidden_size]"""
        if self.training and self.word_drop > 0.0:
            if random.random() < self.word_drop:
                embed = self.embedding(to_var(x.data.new([UNK_ID] * x.size(0)))
                    )
            else:
                embed = self.embedding(x)
        else:
            embed = self.embedding(x)
        return embed

    def beam_decode(self, init_h=None, encoder_outputs=None,
        input_valid_length=None, decode=False):
        """
        Args:
            encoder_outputs (Variable, FloatTensor): [batch_size, source_length, hidden_size]
            input_valid_length (Variable, LongTensor): [batch_size] (optional)
            init_h (variable, FloatTensor): [batch_size, hidden_size] (optional)
        Return:
            out   : [batch_size, seq_len]
        """
        batch_size = self.batch_size(h=init_h)
        x = self.init_token(batch_size * self.beam_size, SOS_ID)
        h = self.init_h(batch_size, hidden=init_h).repeat(1, self.beam_size, 1)
        batch_position = to_var(torch.arange(0, batch_size).long() * self.
            beam_size)
        score = torch.ones(batch_size * self.beam_size) * -float('inf')
        score.index_fill_(0, torch.arange(0, batch_size).long() * self.
            beam_size, 0.0)
        score = to_var(score)
        beam = Beam(batch_size, self.hidden_size, self.vocab_size, self.
            beam_size, self.max_unroll, batch_position)
        for i in range(self.max_unroll):
            out, h = self.forward_step(x, h, encoder_outputs=
                encoder_outputs, input_valid_length=input_valid_length)
            log_prob = F.log_softmax(out, dim=1)
            score = score.view(-1, 1) + log_prob
            score, top_k_idx = score.view(batch_size, -1).topk(self.
                beam_size, dim=1)
            x = (top_k_idx % self.vocab_size).view(-1)
            beam_idx = top_k_idx / self.vocab_size
            top_k_pointer = (beam_idx + batch_position.unsqueeze(1)).view(-1)
            h = h.index_select(1, top_k_pointer)
            beam.update(score.clone(), top_k_pointer, x)
            eos_idx = x.data.eq(EOS_ID).view(batch_size, self.beam_size)
            if eos_idx.nonzero().dim() > 0:
                score.data.masked_fill_(eos_idx, -float('inf'))
        prediction, final_score, length = beam.backtrack()
        return prediction, final_score, length


class BaseRNNEncoder(nn.Module):

    def __init__(self):
        """Base RNN Encoder Class"""
        super(BaseRNNEncoder, self).__init__()

    @property
    def use_lstm(self):
        if hasattr(self, 'rnn'):
            return isinstance(self.rnn, nn.LSTM)
        else:
            raise AttributeError('no rnn selected')

    def init_h(self, batch_size=None, hidden=None):
        """Return RNN initial state"""
        if hidden is not None:
            return hidden
        if self.use_lstm:
            return to_var(torch.zeros(self.num_layers * self.num_directions,
                batch_size, self.hidden_size)), to_var(torch.zeros(self.
                num_layers * self.num_directions, batch_size, self.hidden_size)
                )
        else:
            return to_var(torch.zeros(self.num_layers * self.num_directions,
                batch_size, self.hidden_size))

    def batch_size(self, inputs=None, h=None):
        """
        inputs: [batch_size, seq_len]
        h: [num_layers, batch_size, hidden_size] (RNN/GRU)
        h_c: [2, num_layers, batch_size, hidden_size] (LSTM)
        """
        if inputs is not None:
            batch_size = inputs.size(0)
            return batch_size
        else:
            if self.use_lstm:
                batch_size = h[0].size(1)
            else:
                batch_size = h.size(1)
            return batch_size

    def forward(self):
        raise NotImplementedError


class FeedForward(nn.Module):

    def __init__(self, input_size, output_size, num_layers=1, hidden_size=
        None, activation='Tanh', bias=True):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.activation = getattr(nn, activation)()
        n_inputs = [input_size] + [hidden_size] * (num_layers - 1)
        n_outputs = [hidden_size] * (num_layers - 1) + [output_size]
        self.linears = nn.ModuleList([nn.Linear(n_in, n_out, bias=bias) for
            n_in, n_out in zip(n_inputs, n_outputs)])

    def forward(self, input):
        x = input
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)
        return x


class StackedLSTMCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(x, (h_0[i], c_0[i]))
            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]
        last_h_c = h_list[-1], c_list[-1]
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = h_list, c_list
        return last_h_c, h_c_list


class StackedGRUCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRUCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h):
        """
        Args:
            x: [batch_size, input_size]
            h: [num_layers, batch_size, hidden_size]
        Return:
            last_h: [batch_size, hidden_size] (h from last layer)
            h_list: [num_layers, batch_size, hidden_size] (h from all layers)
        """
        h_list = []
        for i, layer in enumerate(self.layers):
            h_i = layer(x, h[i])
            x = h_i
            if i + 1 is not self.num_layers:
                x = self.dropout(x)
            h_list.append(h_i)
        last_h = h_list[-1]
        h_list = torch.stack(h_list)
        return last_h, h_list


def pad(tensor, length):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            return torch.cat([var, torch.zeros(length - var.size(0), *var.
                size()[1:]).cuda()])
        else:
            return var
    elif length > tensor.size(0):
        return torch.cat([tensor, torch.zeros(length - tensor.size(0), *
            tensor.size()[1:]).cuda()])
    else:
        return tensor


class HRED(nn.Module):

    def __init__(self, config):
        super(HRED, self).__init__()
        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size, config.
            embedding_size, config.encoder_hidden_size, config.rnn, config.
            num_layers, config.bidirectional, config.dropout)
        context_input_size = (config.num_layers * config.
            encoder_hidden_size * self.encoder.num_directions)
        self.context_encoder = layers.ContextRNN(context_input_size, config
            .context_size, config.rnn, config.num_layers, config.dropout)
        self.decoder = layers.DecoderRNN(config.vocab_size, config.
            embedding_size, config.decoder_hidden_size, config.rnncell,
            config.num_layers, config.dropout, config.word_drop, config.
            max_unroll, config.sample, config.temperature, config.beam_size)
        self.context2decoder = layers.FeedForward(config.context_size, 
            config.num_layers * config.decoder_hidden_size, num_layers=1,
            activation=config.activation)
        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def forward(self, input_sentences, input_sentence_length,
        input_conversation_length, target_sentences, decode=False):
        """
        Args:
            input_sentences: (Variable, LongTensor) [num_sentences, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        num_sentences = input_sentences.size(0)
        max_len = input_conversation_length.data.max().item()
        encoder_outputs, encoder_hidden = self.encoder(input_sentences,
            input_sentence_length)
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(
            num_sentences, -1)
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.
            data.new(1).zero_()), input_conversation_length[:-1])), 0)
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l),
            max_len) for s, l in zip(start.data.tolist(),
            input_conversation_length.data.tolist())], 0)
        context_outputs, context_last_hidden = self.context_encoder(
            encoder_hidden, input_conversation_length)
        context_outputs = torch.cat([context_outputs[(i), :l, :] for i, l in
            enumerate(input_conversation_length.data)])
        decoder_init = self.context2decoder(context_outputs)
        decoder_init = decoder_init.view(self.decoder.num_layers, -1, self.
            decoder.hidden_size)
        if not decode:
            decoder_outputs = self.decoder(target_sentences, init_h=
                decoder_init, decode=decode)
            return decoder_outputs
        else:
            prediction, final_score, length = self.decoder.beam_decode(init_h
                =decoder_init)
            return prediction

    def generate(self, context, sentence_length, n_context):
        batch_size = context.size(0)
        samples = []
        context_hidden = None
        for i in range(n_context):
            encoder_outputs, encoder_hidden = self.encoder(context[:, (i),
                :], sentence_length[:, (i)])
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(
                batch_size, -1)
            context_outputs, context_hidden = self.context_encoder.step(
                encoder_hidden, context_hidden)
        for j in range(self.config.n_sample_step):
            context_outputs = context_outputs.squeeze(1)
            decoder_init = self.context2decoder(context_outputs)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1,
                self.decoder.hidden_size)
            prediction, final_score, length = self.decoder.beam_decode(init_h
                =decoder_init)
            prediction = prediction[:, (0), :]
            length = [l[0] for l in length]
            length = to_var(torch.LongTensor(length))
            samples.append(prediction)
            encoder_outputs, encoder_hidden = self.encoder(prediction, length)
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(
                batch_size, -1)
            context_outputs, context_hidden = self.context_encoder.step(
                encoder_hidden, context_hidden)
        samples = torch.stack(samples, 1)
        return samples


def normal_logpdf(x, mean, var):
    """
    Args:
        x: (Variable, FloatTensor) [batch_size, dim]
        mean: (Variable, FloatTensor) [batch_size, dim] or [batch_size] or [1]
        var: (Variable, FloatTensor) [batch_size, dim]: positive value
    Return:
        log_p: (Variable, FloatTensor) [batch_size]
    """
    pi = to_var(torch.FloatTensor([np.pi]))
    return 0.5 * torch.sum(-torch.log(2.0 * pi) - torch.log(var) - (x -
        mean).pow(2) / var, dim=1)


class VHCR(nn.Module):

    def __init__(self, config):
        super(VHCR, self).__init__()
        self.config = config
        self.encoder = layers.EncoderRNN(config.vocab_size, config.
            embedding_size, config.encoder_hidden_size, config.rnn, config.
            num_layers, config.bidirectional, config.dropout)
        context_input_size = (config.num_layers * config.
            encoder_hidden_size * self.encoder.num_directions + config.
            z_conv_size)
        self.context_encoder = layers.ContextRNN(context_input_size, config
            .context_size, config.rnn, config.num_layers, config.dropout)
        self.unk_sent = nn.Parameter(torch.randn(context_input_size -
            config.z_conv_size))
        self.z_conv2context = layers.FeedForward(config.z_conv_size, config
            .num_layers * config.context_size, num_layers=1, activation=
            config.activation)
        context_input_size = (config.num_layers * config.
            encoder_hidden_size * self.encoder.num_directions)
        self.context_inference = layers.ContextRNN(context_input_size,
            config.context_size, config.rnn, config.num_layers, config.
            dropout, bidirectional=True)
        self.decoder = layers.DecoderRNN(config.vocab_size, config.
            embedding_size, config.decoder_hidden_size, config.rnncell,
            config.num_layers, config.dropout, config.word_drop, config.
            max_unroll, config.sample, config.temperature, config.beam_size)
        self.context2decoder = layers.FeedForward(config.context_size +
            config.z_sent_size + config.z_conv_size, config.num_layers *
            config.decoder_hidden_size, num_layers=1, activation=config.
            activation)
        self.softplus = nn.Softplus()
        self.conv_posterior_h = layers.FeedForward(config.num_layers * self
            .context_inference.num_directions * config.context_size, config
            .context_size, num_layers=2, hidden_size=config.context_size,
            activation=config.activation)
        self.conv_posterior_mu = nn.Linear(config.context_size, config.
            z_conv_size)
        self.conv_posterior_var = nn.Linear(config.context_size, config.
            z_conv_size)
        self.sent_prior_h = layers.FeedForward(config.context_size + config
            .z_conv_size, config.context_size, num_layers=1, hidden_size=
            config.z_sent_size, activation=config.activation)
        self.sent_prior_mu = nn.Linear(config.context_size, config.z_sent_size)
        self.sent_prior_var = nn.Linear(config.context_size, config.z_sent_size
            )
        self.sent_posterior_h = layers.FeedForward(config.z_conv_size + 
            config.encoder_hidden_size * self.encoder.num_directions *
            config.num_layers + config.context_size, config.context_size,
            num_layers=2, hidden_size=config.context_size, activation=
            config.activation)
        self.sent_posterior_mu = nn.Linear(config.context_size, config.
            z_sent_size)
        self.sent_posterior_var = nn.Linear(config.context_size, config.
            z_sent_size)
        if config.tie_embedding:
            self.decoder.embedding = self.encoder.embedding

    def conv_prior(self):
        return to_var(torch.FloatTensor([0.0])), to_var(torch.FloatTensor([
            1.0]))

    def conv_posterior(self, context_inference_hidden):
        h_posterior = self.conv_posterior_h(context_inference_hidden)
        mu_posterior = self.conv_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.conv_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def sent_prior(self, context_outputs, z_conv):
        h_prior = self.sent_prior_h(torch.cat([context_outputs, z_conv], dim=1)
            )
        mu_prior = self.sent_prior_mu(h_prior)
        var_prior = self.softplus(self.sent_prior_var(h_prior))
        return mu_prior, var_prior

    def sent_posterior(self, context_outputs, encoder_hidden, z_conv):
        h_posterior = self.sent_posterior_h(torch.cat([context_outputs,
            encoder_hidden, z_conv], 1))
        mu_posterior = self.sent_posterior_mu(h_posterior)
        var_posterior = self.softplus(self.sent_posterior_var(h_posterior))
        return mu_posterior, var_posterior

    def forward(self, sentences, sentence_length, input_conversation_length,
        target_sentences, decode=False):
        """
        Args:
            sentences: (Variable, LongTensor) [num_sentences + batch_size, seq_len]
            target_sentences: (Variable, LongTensor) [num_sentences, seq_len]
        Return:
            decoder_outputs: (Variable, FloatTensor)
                - train: [batch_size, seq_len, vocab_size]
                - eval: [batch_size, seq_len]
        """
        batch_size = input_conversation_length.size(0)
        num_sentences = sentences.size(0) - batch_size
        max_len = input_conversation_length.data.max().item()
        encoder_outputs, encoder_hidden = self.encoder(sentences,
            sentence_length)
        encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(
            num_sentences + batch_size, -1)
        start = torch.cumsum(torch.cat((to_var(input_conversation_length.
            data.new(1).zero_()), input_conversation_length[:-1] + 1)), 0)
        encoder_hidden = torch.stack([pad(encoder_hidden.narrow(0, s, l + 1
            ), max_len + 1) for s, l in zip(start.data.tolist(),
            input_conversation_length.data.tolist())], 0)
        encoder_hidden_inference = encoder_hidden[:, 1:, :]
        encoder_hidden_inference_flat = torch.cat([encoder_hidden_inference
            [(i), :l, :] for i, l in enumerate(input_conversation_length.data)]
            )
        encoder_hidden_input = encoder_hidden[:, :-1, :]
        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))
        conv_mu_prior, conv_var_prior = self.conv_prior()
        if not decode:
            if self.config.sentence_drop > 0.0:
                indices = np.where(np.random.rand(max_len) < self.config.
                    sentence_drop)[0]
                if len(indices) > 0:
                    encoder_hidden_input[:, (indices), :] = self.unk_sent
            context_inference_outputs, context_inference_hidden = (self.
                context_inference(encoder_hidden, input_conversation_length +
                1))
            context_inference_hidden = context_inference_hidden.transpose(1, 0
                ).contiguous().view(batch_size, -1)
            conv_mu_posterior, conv_var_posterior = self.conv_posterior(
                context_inference_hidden)
            z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior
                ) * conv_eps
            log_q_zx_conv = normal_logpdf(z_conv, conv_mu_posterior,
                conv_var_posterior).sum()
            log_p_z_conv = normal_logpdf(z_conv, conv_mu_prior, conv_var_prior
                ).sum()
            kl_div_conv = normal_kl_div(conv_mu_posterior,
                conv_var_posterior, conv_mu_prior, conv_var_prior).sum()
            context_init = self.z_conv2context(z_conv).view(self.config.
                num_layers, batch_size, self.config.context_size)
            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(1)
                ).expand(z_conv.size(0), max_len, z_conv.size(1))
            context_outputs, context_last_hidden = self.context_encoder(torch
                .cat([encoder_hidden_input, z_conv_expand], 2),
                input_conversation_length, hidden=context_init)
            context_outputs = torch.cat([context_outputs[(i), :l, :] for i,
                l in enumerate(input_conversation_length.data)])
            z_conv_flat = torch.cat([z_conv_expand[(i), :l, :] for i, l in
                enumerate(input_conversation_length.data)])
            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs,
                z_conv_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))
            sent_mu_posterior, sent_var_posterior = self.sent_posterior(
                context_outputs, encoder_hidden_inference_flat, z_conv_flat)
            z_sent = sent_mu_posterior + torch.sqrt(sent_var_posterior) * eps
            log_q_zx_sent = normal_logpdf(z_sent, sent_mu_posterior,
                sent_var_posterior).sum()
            log_p_z_sent = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior
                ).sum()
            kl_div_sent = normal_kl_div(sent_mu_posterior,
                sent_var_posterior, sent_mu_prior, sent_var_prior).sum()
            kl_div = kl_div_conv + kl_div_sent
            log_q_zx = log_q_zx_conv + log_q_zx_sent
            log_p_z = log_p_z_conv + log_p_z_sent
        else:
            z_conv = conv_mu_prior + torch.sqrt(conv_var_prior) * conv_eps
            context_init = self.z_conv2context(z_conv).view(self.config.
                num_layers, batch_size, self.config.context_size)
            z_conv_expand = z_conv.view(z_conv.size(0), 1, z_conv.size(1)
                ).expand(z_conv.size(0), max_len, z_conv.size(1))
            context_outputs, context_last_hidden = self.context_encoder(torch
                .cat([encoder_hidden_input, z_conv_expand], 2),
                input_conversation_length, hidden=context_init)
            context_outputs = torch.cat([context_outputs[(i), :l, :] for i,
                l in enumerate(input_conversation_length.data)])
            z_conv_flat = torch.cat([z_conv_expand[(i), :l, :] for i, l in
                enumerate(input_conversation_length.data)])
            sent_mu_prior, sent_var_prior = self.sent_prior(context_outputs,
                z_conv_flat)
            eps = to_var(torch.randn((num_sentences, self.config.z_sent_size)))
            z_sent = sent_mu_prior + torch.sqrt(sent_var_prior) * eps
            kl_div = None
            log_p_z = normal_logpdf(z_sent, sent_mu_prior, sent_var_prior).sum(
                )
            log_p_z += normal_logpdf(z_conv, conv_mu_prior, conv_var_prior
                ).sum()
            log_q_zx = None
        z_conv = torch.cat([z.view(1, -1).expand(m.item(), self.config.
            z_conv_size) for z, m in zip(z_conv, input_conversation_length)])
        latent_context = torch.cat([context_outputs, z_sent, z_conv], 1)
        decoder_init = self.context2decoder(latent_context)
        decoder_init = decoder_init.view(-1, self.decoder.num_layers, self.
            decoder.hidden_size)
        decoder_init = decoder_init.transpose(1, 0).contiguous()
        if not decode:
            decoder_outputs = self.decoder(target_sentences, init_h=
                decoder_init, decode=decode)
            return decoder_outputs, kl_div, log_p_z, log_q_zx
        else:
            prediction, final_score, length = self.decoder.beam_decode(init_h
                =decoder_init)
            return prediction, kl_div, log_p_z, log_q_zx

    def generate(self, context, sentence_length, n_context):
        batch_size = context.size(0)
        samples = []
        conv_eps = to_var(torch.randn([batch_size, self.config.z_conv_size]))
        encoder_hidden_list = []
        for i in range(n_context):
            encoder_outputs, encoder_hidden = self.encoder(context[:, (i),
                :], sentence_length[:, (i)])
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(
                batch_size, -1)
            encoder_hidden_list.append(encoder_hidden)
        encoder_hidden = torch.stack(encoder_hidden_list, 1)
        context_inference_outputs, context_inference_hidden = (self.
            context_inference(encoder_hidden, to_var(torch.LongTensor([
            n_context] * batch_size))))
        context_inference_hidden = context_inference_hidden.transpose(1, 0
            ).contiguous().view(batch_size, -1)
        conv_mu_posterior, conv_var_posterior = self.conv_posterior(
            context_inference_hidden)
        z_conv = conv_mu_posterior + torch.sqrt(conv_var_posterior) * conv_eps
        context_init = self.z_conv2context(z_conv).view(self.config.
            num_layers, batch_size, self.config.context_size)
        context_hidden = context_init
        for i in range(n_context):
            encoder_outputs, encoder_hidden = self.encoder(context[:, (i),
                :], sentence_length[:, (i)])
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(
                batch_size, -1)
            encoder_hidden_list.append(encoder_hidden)
            context_outputs, context_hidden = self.context_encoder.step(torch
                .cat([encoder_hidden, z_conv], 1), context_hidden)
        for j in range(self.config.n_sample_step):
            context_outputs = context_outputs.squeeze(1)
            mu_prior, var_prior = self.sent_prior(context_outputs, z_conv)
            eps = to_var(torch.randn((batch_size, self.config.z_sent_size)))
            z_sent = mu_prior + torch.sqrt(var_prior) * eps
            latent_context = torch.cat([context_outputs, z_sent, z_conv], 1)
            decoder_init = self.context2decoder(latent_context)
            decoder_init = decoder_init.view(self.decoder.num_layers, -1,
                self.decoder.hidden_size)
            if self.config.sample:
                prediction = self.decoder(None, decoder_init, decode=True)
                p = prediction.data.cpu().numpy()
                length = torch.from_numpy(np.where(p == EOS_ID)[1])
            else:
                prediction, final_score, length = self.decoder.beam_decode(
                    init_h=decoder_init)
                prediction = prediction[:, (0), :]
                length = [l[0] for l in length]
                length = to_var(torch.LongTensor(length))
            samples.append(prediction)
            encoder_outputs, encoder_hidden = self.encoder(prediction, length)
            encoder_hidden = encoder_hidden.transpose(1, 0).contiguous().view(
                batch_size, -1)
            context_outputs, context_hidden = self.context_encoder.step(torch
                .cat([encoder_hidden, z_conv], 1), context_hidden)
        samples = torch.stack(samples, 1)
        return samples


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ctr4si_A_Hierarchical_Latent_Structure_for_Variational_Conversation_Modeling(_paritybench_base):
    pass
    def test_000(self):
        self._check(FeedForward(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

