import sys
_module = sys.modules[__name__]
del sys
get_single = _module
conf = _module
mwptoolkit = _module
config = _module
configuration = _module
data = _module
dataloader = _module
abstract_dataloader = _module
dataloader_ept = _module
dataloader_gpt2 = _module
dataloader_hms = _module
dataloader_multiencdec = _module
multi_equation_dataloader = _module
pretrain_dataloader = _module
single_equation_dataloader = _module
template_dataloader = _module
dataset = _module
abstract_dataset = _module
dataset_ept = _module
dataset_gpt2 = _module
dataset_hms = _module
dataset_multiencdec = _module
multi_equation_dataset = _module
pretrain_dataset = _module
single_equation_dataset = _module
template_dataset = _module
utils = _module
evaluate = _module
evaluator = _module
hyper_search = _module
loss = _module
abstract_loss = _module
binary_cross_entropy_loss = _module
cross_entropy_loss = _module
masked_cross_entropy_loss = _module
mse_loss = _module
nll_loss = _module
smoothed_cross_entropy_loss = _module
GAN = _module
seqgan = _module
Graph2Tree = _module
graph2tree = _module
graph2treeibm = _module
multiencdec = _module
PreTrain = _module
albertgen = _module
bertgen = _module
gpt2 = _module
robertagen = _module
Seq2Seq = _module
dns = _module
ept = _module
groupatt = _module
lstm = _module
mathen = _module
rnnencdec = _module
rnnvae = _module
saligned = _module
transformer = _module
Seq2Tree = _module
berttd = _module
gts = _module
hms = _module
mathdqn = _module
mwpbert = _module
sausolver = _module
treelstm = _module
trnn = _module
tsn = _module
VAE = _module
model = _module
abstract_model = _module
Attention = _module
group_attention = _module
hierarchical_attention = _module
multi_head_attention = _module
self_attention = _module
separate_attention = _module
seq_attention = _module
tree_attention = _module
Decoder = _module
ept_decoder = _module
rnn_decoder = _module
transformer_decoder = _module
tree_decoder = _module
Discriminator = _module
seqgan_discriminator = _module
Embedder = _module
basic_embedder = _module
bert_embedder = _module
position_embedder = _module
roberta_embedder = _module
Encoder = _module
graph_based_encoder = _module
rnn_encoder = _module
transformer_encoder = _module
Environment = _module
env = _module
stack_machine = _module
Generator = _module
seqgan_generator = _module
Graph = _module
gcn = _module
graph_module = _module
Layer = _module
graph_layers = _module
layers = _module
transformer_layer = _module
tree_layers = _module
Optimizer = _module
optim = _module
Strategy = _module
beam_search = _module
greedy = _module
sampling = _module
weakly_supervising = _module
module = _module
transformer_layer = _module
quick_start = _module
trainer = _module
abstract_trainer = _module
supervised_trainer = _module
template_trainer = _module
weakly_supervised_trainer = _module
data_structure = _module
enum_type = _module
logger = _module
preprocess_tool = _module
dataset_operator = _module
equation_operator = _module
number_operator = _module
number_transfer = _module
sentence_operator = _module
preprocess_tools = _module
utils = _module
run_hyper_search = _module
run_mwptoolkit = _module
setup = _module

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


import copy


import re


import warnings


from logging import getLogger


from enum import Enum


import torch


from typing import List


import math


import numpy as np


import random


from collections import Counter


from torch import nn


from torch.nn import functional as F


import itertools


from typing import Tuple


from typing import Dict


from typing import Any


from torch.nn import functional


from collections import deque


from torch.nn.functional import cross_entropy


from typing import Union


from functools import singledispatch


import torch.nn.functional as F


from copy import deepcopy


import time


import queue as Q


from itertools import groupby


from collections import OrderedDict


class SmoothedCrossEntropyLoss(nn.Module):
    """
    Computes cross entropy loss with uniformly smoothed targets.
    """

    def __init__(self, smoothing: float=0.1, ignore_index: int=-1, reduction: str='batchmean'):
        """
        Cross entropy loss with uniformly smoothed targets.

        :param float smoothing: Label smoothing factor, between 0 and 1 (exclusive; default is 0.1)
        :param int ignore_index: Index to be ignored. (PAD_ID by default)
        :param str reduction: Style of reduction to be done. One of 'batchmean'(default), 'none', or 'sum'.
        """
        assert 0 < smoothing < 1, 'Smoothing factor should be in (0.0, 1.0)'
        assert reduction in {'batchmean', 'none', 'sum'}
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.LongTensor) ->torch.Tensor:
        """
        Computes cross entropy loss with uniformly smoothed targets.
        Since the entropy of smoothed target distribution is always same, we can compute this with KL-divergence.

        :param torch.Tensor input: Log probability for each class. This is a Tensor with shape [B, C]
        :param torch.LongTensor target: List of target classes. This is a LongTensor with shape [B]
        :rtype: torch.Tensor
        :return: Computed loss
        """
        target = target.view(-1, 1)
        smoothed_target = torch.zeros(input.shape, requires_grad=False, device=target.device)
        for r, row in enumerate(input):
            tgt = target[r].item()
            if tgt == self.ignore_index:
                continue
            finites = torch.isfinite(row)
            n_cls = finites.sum().item()
            assert n_cls > 0
            smoothing_prob = self.smoothing / n_cls
            smoothed_target[r].masked_fill_(finites, smoothing_prob)
            smoothed_target[r, tgt] = 1.0 - self.smoothing
        loss = -smoothed_target * input.masked_fill(~torch.isfinite(input), 0.0)
        if self.reduction == 'batchmean':
            return loss.sum() / input.shape[0]
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SeqGANDiscriminator(nn.Module):
    """The discriminator of SeqGAN.
    """

    def __init__(self, config):
        super(SeqGANDiscriminator, self).__init__()
        self.embedding_size = config['discriminator_embedding_size']
        self.l2_reg_lambda = config['l2_reg_lambda']
        self.dropout_rate = config['dropout_ratio']
        self.filter_sizes = config['filter_sizes']
        self.filter_nums = config['filter_nums']
        self.max_length = config['max_seq_length']
        self.filter_sum = sum(self.filter_nums)
        self.pad_idx = config['out_pad_token']
        self.vocab_size = config['symbol_size']
        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.pad_idx)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.filters = nn.ModuleList([])
        for filter_size, filter_num in zip(self.filter_sizes, self.filter_nums):
            self.filters.append(nn.Sequential(nn.Conv2d(1, filter_num, (filter_size, self.embedding_size)), nn.ReLU(), nn.MaxPool2d((self.max_length - filter_size + 1, 1))))
        self.W_T = nn.Linear(self.filter_sum, self.filter_sum)
        self.W_H = nn.Linear(self.filter_sum, self.filter_sum, bias=False)
        self.W_O = nn.Linear(self.filter_sum, 1)

    def _highway(self, data):
        """Apply the highway net to data.

        Args:
            data (torch.Tensor): The original data, shape: [batch_size, total_filter_num].

        Returns:
            torch.Tensor: The data processed after highway net, shape: [batch_size, total_filter_num].
        """
        tau = torch.sigmoid(self.W_T(data))
        non_linear = F.relu(self.W_H(data))
        return self.dropout(tau * non_linear + (1 - tau) * data)

    def forward(self, data):
        """Calculate the probability that the data is realistic.

        Args:
            data (torch.Tensor): The sentence data, shape: [batch_size, max_seq_len].

        Returns:
            torch.Tensor: The probability that each sentence is realistic, shape: [batch_size].
        """
        data = self.word_embedding(data).unsqueeze(1)
        combined_outputs = []
        for CNN_filter in self.filters:
            output = CNN_filter(data).squeeze(-1).squeeze(-1)
            combined_outputs.append(output)
        combined_outputs = torch.cat(combined_outputs, 1)
        C_tilde = self._highway(combined_outputs)
        y_hat = torch.sigmoid(self.W_O(C_tilde)).squeeze(1)
        return y_hat

    def calculate_loss(self, real_data, fake_data):
        """Calculate the loss for real data and fake data.

        Args:
            real_data (torch.Tensor): The realistic sentence data, shape: [batch_size, max_seq_len].
            fake_data (torch.Tensor): The generated sentence data, shape: [batch_size, max_seq_len].

        Returns:
            torch.Tensor: The calculated loss of real data and fake data, shape: [].
        """
        real_y = self.forward(real_data)
        fake_y = self.forward(fake_data)
        real_label = torch.ones_like(real_y)
        fake_label = torch.zeros_like(fake_y)
        real_loss = F.binary_cross_entropy(real_y, real_label)
        fake_loss = F.binary_cross_entropy(fake_y, fake_label)
        loss = (real_loss + fake_loss) / 2 + self.l2_reg_lambda * (self.W_O.weight.norm() + self.W_O.bias.norm())
        return loss


class SpecialTokens:
    """special tokens
    """
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    SOS_TOKEN = '<SOS>'
    EOS_TOKEN = '<EOS>'
    NON_TOKEN = '<NON>'
    BRG_TOKEN = '<BRG>'
    OPT_TOKEN = '<OPT>'


class BasicEmbedder(nn.Module):
    """
    Basic embedding layer
    """

    def __init__(self, input_size, embedding_size, dropout_ratio, padding_idx=0):
        super(BasicEmbedder, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.embedder = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_seq):
        """Implement the embedding process
        Args:
            input_seq (torch.Tensor): source sequence, shape [batch_size, sequence_length].
        
        Retruns:
            torch.Tensor: embedding output, shape [batch_size, sequence_length, embedding_size].
        """
        embedding_output = self.embedder(input_seq)
        embedding_output = self.dropout(embedding_output)
        return embedding_output

    def init_embedding_params(self, sentences, vocab):
        import numpy as np
        model = word2vec.Word2Vec(sentences, vector_size=self.embedding_size, min_count=1)
        emb_vectors = []
        pad_idx = vocab.index(SpecialTokens.PAD_TOKEN)
        for idx in range(len(vocab)):
            if idx != pad_idx:
                try:
                    emb_vectors.append(np.array(model.wv[vocab[idx]]))
                except:
                    emb_vectors.append(np.random.randn(self.embedding_size))
            else:
                emb_vectors.append(np.zeros(self.embedding_size))
        emb_vectors = np.array(emb_vectors)
        self.embedder.weight.data.copy_(torch.from_numpy(emb_vectors))


class BasicRNNDecoder(nn.Module):
    """
    Basic Recurrent Neural Network (RNN) decoder.
    """

    def __init__(self, embedding_size, hidden_size, num_layers, rnn_cell_type, dropout_ratio=0.0):
        super(BasicRNNDecoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        if rnn_cell_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == 'gru':
            self.decoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == 'rnn':
            self.decoder = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("The RNN type in decoder must in ['lstm', 'gru', 'rnn'].")

    def init_hidden(self, input_embeddings):
        """ Initialize initial hidden states of RNN.

        Args:
            input_embeddings (torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_cell_type == 'lstm':
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            hidden_states = h_0, c_0
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            raise NotImplementedError('No such rnn type {} for initializing decoder states.'.format(self.rnn_type))

    def forward(self, input_embeddings, hidden_states=None):
        """ Implement the decoding process.

        Args:
            input_embeddings (torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        outputs, hidden_states = self.decoder(input_embeddings, hidden_states)
        return outputs, hidden_states


class BasicRNNEncoder(nn.Module):
    """
    Basic Recurrent Neural Network (RNN) encoder.
    """

    def __init__(self, embedding_size, hidden_size, num_layers, rnn_cell_type, dropout_ratio, bidirectional=True, batch_first=True):
        super(BasicRNNEncoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        self.batch_first = batch_first
        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")

    def init_hidden(self, input_embeddings):
        """ Initialize initial hidden states of RNN.

        Args:
            input_embeddings (torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            torch.Tensor: the initial hidden states.
        """
        if self.batch_first:
            batch_size = input_embeddings.size(0)
        else:
            batch_size = input_embeddings.size(1)
        device = input_embeddings.device
        if self.rnn_cell_type == 'lstm':
            h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            hidden_states = h_0, c_0
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            tp_vec = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            return tp_vec
        else:
            raise NotImplementedError('No such rnn type {} for initializing encoder states.'.format(self.rnn_type))

    def forward(self, input_embeddings, input_length, hidden_states=None):
        """ Implement the encoding process.

        Args:
            input_embeddings (torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            input_length (torch.Tensor): length of input sequence, shape: [batch_size].
            hidden_states (torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length, batch_first=self.batch_first, enforce_sorted=True)
        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)
        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        return outputs, hidden_states


class SeqGANGenerator(nn.Module):
    """The generator of SeqGAN.
    """

    def __init__(self, config):
        super(SeqGANGenerator, self).__init__()
        self.bidirectional = config['bidirectional']
        self.rnn_cell_type = config['rnn_cell_type']
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['generator_embedding_size']
        self.max_gen_len = 30
        self.monte_carlo_num = config['Monte_Carlo_num']
        self.eval_generate_num = config['eval_generate_num']
        self.share_vocab = config['share vocab']
        self.teacher_force_ratio = 0.9
        self.batch_size = 64
        self.device = config['device']
        self.num_layers = config['num_layers']
        self.out_sos_token = config['out_sos_token']
        self.out_eos_token = config['out_eos_token']
        self.out_pad_token = config['out_pad_token']
        self.vocab_size = config['vocab_size']
        self.in_embedder = BasicEmbedder(config['vocab_size'], config['generator_embedding_size'], config['dropout_ratio'])
        if config['share_vocab']:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BasicEmbedder(config['symbol_size'], config['generator_embedding_size'], config['dropout_ratio'])
        self.encoder = BasicRNNEncoder(config['generator_embedding_size'], config['hidden_size'], config['num_layers'], config['rnn_cell_type'], config['dropout_ratio'])
        self.decoder = BasicRNNDecoder(config['generator_embedding_size'], config['hidden_size'], config['num_layers'], config['rnn_cell_type'], config['dropout_ratio'])
        self.dropout = nn.Dropout(config['dropout_ratio'])
        self.generate_linear = nn.Linear(config['hidden_size'], config['symbol_size'])

    def forward(self, seq, seq_length, target=None):
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.in_embedder(seq)
        encoder_outputs, encoder_hidden = self.encoder(seq_emb, seq_length)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if self.rnn_cell_type == 'lstm':
                encoder_hidden = encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous()
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        decoder_inputs = self.init_decoder_inputs(target, device, batch_size)
        if target != None:
            all_output, token_logits, outputs, P = self.generate_t(encoder_outputs, encoder_hidden, decoder_inputs)
            return all_output, token_logits, outputs, P
        else:
            all_outputs = self.generate_without_t(encoder_outputs, encoder_hidden, decoder_inputs)
            return all_outputs, None, None, None

    def generate_t(self, encoder_outputs, encoder_hidden, decoder_inputs):
        batch_size = encoder_outputs.size(0)
        fake_samples = self.sample(batch_size)
        with_t = random.random()
        seq_len = decoder_inputs.size(1)
        decoder_hidden = encoder_hidden
        tokens = decoder_inputs[:, 0].unsqueeze(1)
        monte_carlo_outputs = []
        token_logits = []
        P = []
        all_output = []
        for idx in range(seq_len):
            if with_t < self.teacher_force_ratio:
                tokens = decoder_inputs[:, idx].unsqueeze(1)
            decoder_input = self.out_embedder(tokens)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logit = self.generate_linear(step_output)
            predict = torch.nn.functional.log_softmax(token_logit, dim=1)
            tokens = fake_samples[:, idx].unsqueeze(1)
            if self.share_vocab:
                tokens = self.decode(tokens)
            P_t = torch.gather(predict, 1, tokens).squeeze(1)
            monte_carlo_output = self.Monte_Carlo_search(tokens, decoder_hidden, fake_samples, idx, seq_len)
            monte_carlo_outputs.append(monte_carlo_output)
            P.append(P_t)
            all_output.append(tokens)
            token_logits.append(predict)
        all_output = torch.cat(all_output, dim=1)
        token_logits = torch.stack(token_logits, dim=1)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        return all_output, token_logits, monte_carlo_outputs, P

    def generate_without_t(self, encoder_outputs, encoder_hidden, decoder_inputs):
        decoder_hidden = encoder_hidden
        tokens = decoder_inputs[:, 0].unsqueeze(1)
        all_output = []
        for idx in range(self.max_gen_len):
            decoder_input = self.out_embedder(tokens)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logit = self.generate_linear(step_output)
            predict = torch.nn.functional.log_softmax(token_logit, dim=1)
            tokens = predict.topk(1, dim=-1)[1]
            if self.share_vocab:
                tokens = self.decode(tokens)
            all_output.append(tokens)
        all_output = torch.cat(all_output, dim=1)
        return all_output

    def Monte_Carlo_search(self, tokens, decoder_hidden, fake_samples, idx, max_length):
        h_prev, o_prev = decoder_hidden
        self.eval()
        with torch.no_grad():
            monte_carlo_X = tokens.repeat_interleave(self.monte_carlo_num)
            monte_carlo_X = self.out_embedder(monte_carlo_X).unsqueeze(1)
            monte_carlo_h_prev = h_prev.clone().detach().repeat_interleave(self.monte_carlo_num, dim=1)
            monte_carlo_o_prev = o_prev.clone().detach().repeat_interleave(self.monte_carlo_num, dim=1)
            monte_carlo_output = torch.zeros(max_length, self.batch_size * self.monte_carlo_num, dtype=torch.long, device=self.device)
            for i in range(max_length - idx - 1):
                output, (monte_carlo_h_prev, monte_carlo_o_prev) = self.decoder(monte_carlo_X, (monte_carlo_h_prev, monte_carlo_o_prev))
                P = F.softmax(self.generate_linear(output), dim=-1).squeeze(0)
                for j in range(P.shape[0]):
                    monte_carlo_output[i + idx + 1][j] = torch.multinomial(P[j], 1)[0]
                monte_carlo_X = self.out_embedder(monte_carlo_output[i + idx + 1]).unsqueeze(1)
            monte_carlo_output = monte_carlo_output.permute(1, 0)
            monte_carlo_output[:, :idx + 1] = fake_samples[:, :idx + 1].repeat_interleave(self.monte_carlo_num, dim=0)
        self.train()
        return monte_carlo_output

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.out_sos_token] * batch_size).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        return decoder_inputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def _sample_batch(self, batch_size):
        """Sample a batch of generated sentence indice.

        Returns:
            torch.Tensor: The generated sentence indice, shape: [batch_size, max_seq_length].
        """
        self.eval()
        sentences = []
        with torch.no_grad():
            h_prev = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            o_prev = torch.zeros(self.num_layers, batch_size, self.hidden_size)
            prev_state = h_prev, o_prev
            tokens = self.init_decoder_inputs(None, self.device, batch_size)
            X = self.out_embedder(tokens)
            sentences = torch.zeros((batch_size, self.max_gen_len), dtype=torch.long)
            for i in range(0, self.max_gen_len):
                output, prev_state = self.decoder(X, prev_state)
                P = F.softmax(self.generate_linear(output), dim=-1).squeeze(0)
                for j in range(batch_size):
                    sentences[j][i] = torch.multinomial(P[j], 1)[0]
                X = self.out_embedder(sentences[:, i]).unsqueeze(1)
            for i in range(batch_size):
                end_pos = (sentences[i] == self.out_eos_token).nonzero(as_tuple=False)
                if end_pos.shape[0]:
                    sentences[i][end_pos[0][0] + 1:] = self.out_pad_token
        self.train()
        return sentences

    def sample(self, sample_num):
        """Sample sample_num generated sentence indice.

        Args:
            sample_num (int): The number to generate.

        Returns:
            torch.Tensor: The generated sentence indice, shape: [sample_num, max_seq_length].
        """
        samples = []
        batch_num = math.ceil(sample_num / self.batch_size)
        for _ in range(batch_num):
            samples.append(self._sample_batch(sample_num))
        samples = torch.cat(samples, dim=0)
        return samples[:sample_num, :]

    def pre_train(self, seq, seq_length, target):
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.in_embedder(seq)
        encoder_outputs, encoder_hidden = self.encoder(seq_emb, seq_length)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if self.rnn_cell_type == 'lstm':
                encoder_hidden = encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous()
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        decoder_inputs = self.init_decoder_inputs(target, device, batch_size)
        batch_size = encoder_outputs.size(0)
        with_t = random.random()
        seq_len = decoder_inputs.size(1)
        decoder_hidden = encoder_hidden
        tokens = decoder_inputs[:, 0].unsqueeze(1)
        monte_carlo_outputs = []
        token_logits = []
        P = []
        all_output = []
        for idx in range(seq_len):
            if with_t < self.teacher_force_ratio:
                tokens = decoder_inputs[:, idx].unsqueeze(1)
            decoder_input = self.out_embedder(tokens)
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logit = self.generate_linear(step_output)
            predict = torch.nn.functional.log_softmax(token_logit, dim=1)
            tokens = predict.topk(1, dim=1)[1]
            if self.share_vocab:
                tokens = self.decode(tokens)
            token_logits.append(predict)
        token_logits = torch.stack(token_logits, dim=1)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        return token_logits


class SeqGAN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.generator = SeqGANGenerator(config)
        self.discriminator = SeqGANDiscriminator(config)

    def forward(self, seq, seq_length, target=None):
        all_output, token_logits, _, _ = self.generator.forward(seq, seq_length, target)
        if target != None:
            return token_logits
        else:
            return all_output


class BertEmbedder(nn.Module):

    def __init__(self, input_size, pretrained_model_path):
        super(BertEmbedder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)

    def forward(self, input_seq):
        output = self.bert(input_seq)[0]
        return output

    def token_resize(self, input_size):
        self.bert.resize_token_embeddings(input_size)


class GenerateNode(nn.Module):

    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        """
        Args:
            node_embedding (torch.Tensor): node embedding, shape [batch_size, hidden_size].
            node_label (torch.Tensor): representation of node label, shape [batch_size, embedding_size].
            current_context (torch.Tensor): current context, shape [batch_size, hidden_size].
        
        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor):
                l_child, representation of left child, shape [batch_size, hidden_size].
                r_child, representation of right child, shape [batch_size, hidden_size].
                node_label_, representation of node label, shape [batch_size, embedding_size].
        """
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)
        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Args:
            x (torch.Tensor): input features, shape [batch_size, node_num, in_feat_dim]
            adj (torch.Tensor): adjacency matrix, shape [batch_size, node_num, node_num]
        
        Returns:
            torch.Tensor: gcn_enhance_feature, shape [batch_size, node_num, out_feat_dim]
        """
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Graph_Module(nn.Module):

    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        super(Graph_Module, self).__init__()
        """
        Args:
            indim: dimensionality of input node features
            hiddim: dimensionality of the joint hidden embedding
            outdim: dimensionality of the output node features
            combined_feature_dim: dimensionality of the joint hidden embedding for graph
            K: number of graph nodes/objects on the image
        """
        self.in_dim = indim
        self.h = 4
        self.d_k = outdim // self.h
        self.graph = nn.ModuleList()
        for _ in range(self.h):
            self.graph.append(GCN(indim, hiddim, self.d_k, dropout))
        self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
        self.norm = LayerNorm(outdim)

    def get_adj(self, graph_nodes):
        """
        Args:
            graph_nodes (torch.Tensor): input features, shape [batch_size, node_num, in_feat_dim]
        
        Returns:
            torch.Tensor: adjacency matrix, shape [batch_size, node_num, node_num]
        """
        self.K = graph_nodes.size(1)
        graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)
        h = self.edge_layer_2(h)
        h = F.relu(h)
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
        adjacency_matrix = self.b_normal(adjacency_matrix)
        return adjacency_matrix

    def normalize(self, A, symmetric=True):
        """
        Args:
            A (torch.Tensor): adjacency matrix (node_num, node_num)
        
        Returns:
            adjacency matrix (node_num, node_num) 
        """
        A = A + torch.eye(A.size(0)).float()
        d = A.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else:
            D = torch.diag(torch.pow(d, -1))
            return D.mm(A)

    def b_normal(self, adj):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i])
        return adj

    def forward(self, graph_nodes, graph):
        """
        Args:
            graph_nodes (torch.Tensor):input features, shape [batch_size, node_num, in_feat_dim]
        
        Returns:
            torch.Tensor: graph_encode_features, shape [batch_size, node_num, out_feat_dim]
        """
        nbatches = graph_nodes.size(0)
        mbatches = graph.size(0)
        if nbatches != mbatches:
            graph_nodes = graph_nodes.transpose(0, 1)
        if not bool(graph.numel()):
            adj = self.get_adj(graph_nodes)
            adj_list = [adj, adj, adj, adj]
        else:
            adj = graph.float()
            adj_list = [adj[:, 1, :], adj[:, 1, :], adj[:, 4, :], adj[:, 4, :]]
        g_feature = tuple([l(graph_nodes, x) for l, x in zip(self.graph, adj_list)])
        g_feature = self.norm(torch.cat(g_feature, 2)) + graph_nodes
        graph_encode_features = self.feed_foward(g_feature) + g_feature
        return adj, graph_encode_features


class GraphBasedEncoder(nn.Module):

    def __init__(self, embedding_size, hidden_size, rnn_cell_type, bidirectional, num_layers=2, dropout_ratio=0.5):
        super(GraphBasedEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_ratio = dropout_ratio
        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")
        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, input_embedding, input_lengths, batch_graph, hidden=None):
        """
        Args:
            input_embedding (torch.Tensor): input variable, shape [sequence_length, batch_size, embedding_size].
            input_lengths (torch.Tensor): length of input sequence, shape: [batch_size].
            batch_graph (torch.Tensor): graph input variable, shape [batch_size, 5, sequence_length, sequence_length].
        
        Returns:
            tuple(torch.Tensor, torch.Tensor):
                pade_outputs, encoded variable, shape [sequence_length, batch_size, hidden_size].
                problem_output, vector representation of problem, shape [batch_size, hidden_size]. 
        """
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_embedding, input_lengths, enforce_sorted=True)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.encoder(packed, pade_hidden)
        pade_outputs, hidden_states = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, problem_output


class AbstractLoss(object):

    def __init__(self, name, criterion):
        self.name = name
        self.criterion = criterion
        self.acc_loss = 0
        self.norm_term = 0

    def reset(self):
        """reset loss
        """
        self.acc_loss = 0
        self.norm_term = 0

    def get_loss(self):
        """return loss
        """
        raise NotImplementedError

    def eval_batch(self, outputs, target):
        """calculate loss
        """
        raise NotImplementedError

    def backward(self):
        """loss backward
        """
        if type(self.acc_loss) is int:
            raise ValueError('No loss to back propagate.')
        self.acc_loss.backward()


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss


class MaskedCrossEntropyLoss(AbstractLoss):
    _Name = 'avg masked cross entopy loss'

    def __init__(self):
        super().__init__(self._Name, masked_cross_entropy)

    def get_loss(self):
        """return loss

        Returns:
            loss (float)
        """
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()
        return loss

    def eval_batch(self, outputs, target, length):
        """calculate loss

        Args:
            outputs (Tensor): output distribution of model.

            target (Tensor): target classes. 

            length (Tensor): length of target.
        """
        self.acc_loss += self.criterion(outputs, target, length)
        self.norm_term += 1


class Merge(nn.Module):

    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        """
        Args:
            node_embedding (torch.Tensor): node embedding, shape [1, embedding_size].
            sub_tree_1 (torch.Tensor): representation of sub tree 1, shape [1, hidden_size].
            sub_tree_2 (torch.Tensor): representation of sub tree 2, shape [1, hidden_size].
        
        Returns:
            torch.Tensor: representation of merged tree, shape [1, hidden_size].
        """
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)
        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class NumMask:
    """number mask symbol list
    """
    NUM = ['NUM'] * 100
    alphabet = ['NUM_a', 'NUM_b', 'NUM_c', 'NUM_d', 'NUM_e', 'NUM_f', 'NUM_g', 'NUM_h', 'NUM_i', 'NUM_j', 'NUM_k', 'NUM_l', 'NUM_m', 'NUM_n', 'NUM_o', 'NUM_p', 'NUM_q', 'NUM_r', 'NUM_s', 'NUM_t', 'NUM_u', 'NUM_v', 'NUM_w', 'NUM_x', 'NUM_y', 'NUM_z']
    number = [('NUM_' + str(i)) for i in range(100)]


class Score(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        """
        Args:
            hidden (torch.Tensor): hidden representation, shape [batch_size, 1, hidden_size + input_size].
            num_embeddings (torch.Tensor): number embedding, shape [batch_size, number_size, hidden_size].
            num_mask (torch.BoolTensor): number mask, shape [batch_size, number_size].
        
        Returns:
            score (torch.Tensor): shape [batch_size, number_size].
        """
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1000000000000.0)
        return score


class TreeAttention(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(TreeAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        """
        Args:
            hidden (torch.Tensor): hidden representation, shape [1, batch_size, hidden_size]
            encoder_outputs (torch.Tensor): output from encoder, shape [sequence_length, batch_size, hidden_size]. 
            seq_mask (torch.Tensor): sequence mask, shape [batch_size, sequence_length].
        
        Returns:
            attn_energies (torch.Tensor): attention energies, shape [batch_size, 1, sequence_length].
        """
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)
        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1000000000000.0)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)
        return attn_energies.unsqueeze(1)


class Prediction(nn.Module):

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums
        self.dropout = nn.Dropout(dropout)
        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)
        self.ops = nn.Linear(hidden_size * 2, op_nums)
        self.attn = TreeAttention(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        """
        Args:
            node_stacks (list): node stacks.
            left_childs (list): representation of left childs.
            encoder_outputs (torch.Tensor): output from encoder, shape [sequence_length, batch_size, hidden_size].
            num_pades (torch.Tensor): number representation, shape [batch_size, number_size, hidden_size].
            padding_hidden (torch.Tensor): padding hidden, shape [1,hidden_size].
            seq_mask (torch.BoolTensor): sequence mask, shape [batch_size, sequence_length].
            mask_nums (torch.BoolTensor): number mask, shape [batch_size, number_size].
        
        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                num_score, number score, shape [batch_size, number_size].
                op, operator score, shape [batch_size, operator_size].
                current_node, current node representation, shape [batch_size, 1, hidden_size].
                current_context, current context representation, shape [batch_size, 1, hidden_size].
                embedding_weight, embedding weight, shape [batch_size, number_size, hidden_size].
        """
        current_embeddings = []
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)
        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)
        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node)
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))
        batch_size = current_embeddings.size(0)
        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)
        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)
        op = self.ops(leaf_input)
        return num_score, op, current_node, current_context, embedding_weight


class RobertaEmbedder(nn.Module):

    def __init__(self, input_size, pretrained_model_path):
        super(RobertaEmbedder, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_path)

    def forward(self, input_seq, attn_mask):
        output = self.roberta(input_seq, attention_mask=attn_mask)[0]
        return output

    def token_resize(self, input_size):
        self.roberta.resize_token_embeddings(input_size)


def copy_list(l):
    r = []
    for i in l:
        if isinstance(i, list):
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:

    def __init__(self, score, node_stack, embedding_stack, left_childs, out, token_logit=None):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)
        self.token_logit = token_logit


class TreeEmbedding:

    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal


class TreeNode:

    def __init__(self, embedding, left_flag=False, terminal=False):
        self.embedding = embedding
        self.left_flag = left_flag


def str2float(v):
    """convert string to float.
    """
    if not isinstance(v, str):
        return v
    else:
        if '%' in v:
            v = v[:-1]
            return float(v) / 100
        if '(' in v:
            try:
                return eval(v)
            except:
                if re.match('^\\d+\\(', v):
                    idx = v.index('(')
                    a = v[:idx]
                    b = v[idx:]
                    return eval(a) + eval(b)
                if re.match('.*\\)\\d+$', v):
                    l = len(v)
                    temp_v = v[::-1]
                    idx = temp_v.index(')')
                    a = v[:l - idx]
                    b = v[l - idx:]
                    return eval(a) + eval(b)
            return float(v)
        elif '/' in v:
            return eval(v)
        else:
            if v == '<UNK>':
                return float('inf')
            return float(v)


class Graph2Tree(nn.Module):
    """
    Reference:
        Zhang et al."Graph-to-Tree Learning for Solving Math Word Problems" in ACL 2020.
    """

    def __init__(self, config, dataset):
        super(Graph2Tree, self).__init__()
        self.device = config['device']
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['embedding_size']
        self.dropout_ratio = config['dropout_ratio']
        self.num_layers = config['num_layers']
        self.rnn_cell_type = config['rnn_cell_type']
        self.bidirectional = config['bidirectional']
        self.language = config['language']
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding = config['embedding']
        self.vocab_size = len(dataset.in_idx2word)
        self.num_start = dataset.num_start
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.operator_nums = dataset.operator_nums
        self.generate_size = len(generate_list)
        self.mask_list = NumMask.number
        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        try:
            self.in_pad_token = dataset.in_word2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.in_pad_token = None
        if config['embedding'] == 'roberta':
            self.embedder = RobertaEmbedder(self.vocab_size, config['pretrained_model_path'])
        elif config['embedding'] == 'bert':
            self.embedder = BertEmbedder(self.vocab_size, config['pretrained_model_path'])
        else:
            self.embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.encoder = GraphBasedEncoder(self.embedding_size, self.hidden_size, self.rnn_cell_type, self.bidirectional, self.num_layers, self.dropout_ratio)
        self.decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.loss = MaskedCrossEntropyLoss()

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, num_list, group_nums, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param list num_list: numbers of input sequence, length:[batch_size].
        :param list group_nums: group numbers of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)

        """
        seq_mask = torch.eq(seq, self.in_pad_token)
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        batch_size = len(seq_length)
        graph = self.build_graph(seq_length, num_list, num_pos, group_nums)
        seq_emb = self.embedder(seq)
        problem_output, encoder_outputs, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, graph, output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, self.hidden_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs['number_representation'] = all_nums_encoder_outputs
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos', 'num list', 'group nums'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.LongTensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        group_nums = batch_data['group nums']
        token_logits, _, all_layer_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, num_list, group_nums, target, output_all_layers=True)
        target = all_layer_outputs['target']
        loss = masked_cross_entropy(token_logits, target, target_length)
        loss.backward()
        return loss.item()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.
        
        :param batch_data: one batch data.
        :return: predicted equation, target equation.
        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list', 'num size', 'group nums'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        group_nums = batch_data['group nums']
        _, symbol_outputs, all_layer_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, num_list, group_nums, output_all_layers=True)
        all_output = self.convert_idx2symbol(symbol_outputs[0], num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num_size']
        group_nums = batch_data['group nums']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, num_list, group_nums, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, input_length, graph, output_all_layers=False):
        encoder_inputs = seq_emb.transpose(0, 1)
        encoder_outputs, problem_output = self.encoder(encoder_inputs, input_length, graph)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(1)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            target = target.transpose(0, 1)
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            target = target.transpose(0, 1)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['target'] = target
        return token_logits, outputs, all_layer_outputs

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [(1) for _ in range(hidden_size)]
        temp_0 = [(0) for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [(0) for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices
            masked_index = masked_index
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def build_graph(self, seq_length, num_list, num_pos, group_nums):
        max_len = seq_length.max()
        batch_size = len(seq_length)
        batch_graph = []
        for b_i in range(batch_size):
            x = torch.zeros((max_len, max_len))
            for idx in range(seq_length[b_i]):
                x[idx, idx] = 1
            quantity_cell_graph = torch.clone(x)
            graph_greater = torch.clone(x)
            graph_lower = torch.clone(x)
            graph_quanbet = torch.clone(x)
            graph_attbet = torch.clone(x)
            for idx, n_pos in enumerate(num_pos[b_i]):
                for pos in group_nums[b_i][idx]:
                    quantity_cell_graph[n_pos, pos] = 1
                    quantity_cell_graph[pos, n_pos] = 1
                    graph_quanbet[n_pos, pos] = 1
                    graph_quanbet[pos, n_pos] = 1
                    graph_attbet[n_pos, pos] = 1
                    graph_attbet[pos, n_pos] = 1
            for idx_i in range(len(num_pos[b_i])):
                for idx_j in range(len(num_pos[b_i])):
                    num_i = str2float(num_list[b_i][idx_i])
                    num_j = str2float(num_list[b_i][idx_j])
                    if num_i > num_j:
                        graph_greater[num_pos[b_i][idx_i]][num_pos[b_i][idx_j]] = 1
                        graph_lower[num_pos[b_i][idx_j]][num_pos[b_i][idx_i]] = 1
                    else:
                        graph_greater[num_pos[b_i][idx_j]][num_pos[b_i][idx_i]] = 1
                        graph_lower[num_pos[b_i][idx_i]][num_pos[b_i][idx_j]] = 1
            group_num_ = itertools.chain.from_iterable(group_nums[b_i])
            combn = itertools.permutations(group_num_, 2)
            for idx in combn:
                graph_quanbet[idx] = 1
                graph_quanbet[idx] = 1
                graph_attbet[idx] = 1
                graph_attbet[idx] = 1
            quantity_cell_graph = quantity_cell_graph
            graph_greater = graph_greater
            graph_lower = graph_lower
            graph_quanbet = graph_quanbet
            graph_attbet = graph_attbet
            graph = torch.stack([quantity_cell_graph, graph_greater, graph_lower, graph_quanbet, graph_attbet], dim=0)
            batch_graph.append(graph)
        batch_graph = torch.stack(batch_graph)
        return batch_graph

    def convert_idx2symbol(self, output, num_list, num_stack):
        """batch_size=1"""
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if 'NUM' in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list


class MeanAggregator(nn.Module):

    def __init__(self, input_dim, output_dim, activation=F.relu, concat=False):
        super(MeanAggregator, self).__init__()
        self.concat = concat
        self.fc_x = nn.Linear(input_dim, output_dim, bias=True)
        self.activation = activation

    def forward(self, inputs):
        x, neibs, _ = inputs
        agg_neib = neibs.mean(dim=1)
        if self.concat:
            out_tmp = torch.cat([x, agg_neib], dim=1)
            out = self.fc_x(out_tmp)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib)
        if self.activation:
            out = self.activation(out)
        return out


class GraphEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, hidden_size, sample_size, sample_layer, bidirectional, dropout_ratio):
        super(GraphEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.sample_size = sample_size
        self.sample_layer = sample_layer
        self.bidirectional = bidirectional
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout_ratio)
        self.embedding = BasicEmbedder(vocab_size, self.embedding_size, dropout_ratio, padding_idx=0)
        self.fw_aggregators = nn.ModuleList()
        self.bw_aggregators = nn.ModuleList()
        for layer in range(7):
            self.fw_aggregators.append(MeanAggregator(2 * self.hidden_size, self.hidden_size, concat=True))
            self.bw_aggregators.append(MeanAggregator(2 * self.hidden_size, self.hidden_size, concat=True))
        self.Linear_hidden = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.embedding_bilstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size // 2, bidirectional=True, bias=True, batch_first=True, dropout=dropout_ratio, num_layers=1)
        self.padding_vector = torch.randn(1, self.hidden_size, dtype=torch.float, requires_grad=True)

    def forward(self, fw_adj_info, bw_adj_info, feature_info, batch_nodes):
        device = batch_nodes.device
        feature_sentence_vector = self.embedding(feature_info)
        output_vector, (ht, _) = self.embedding_bilstm(feature_sentence_vector)
        feature_vector = output_vector.contiguous().view(-1, self.hidden_size)
        feature_embedded = feature_vector
        batch_size = feature_embedded.size()[0]
        node_repres = feature_embedded.view(batch_size, -1)
        nodes = batch_nodes.long().view(-1)
        fw_hidden = F.embedding(nodes, node_repres)
        bw_hidden = F.embedding(nodes, node_repres)
        fw_sampled_neighbors_len = torch.tensor(0)
        bw_sampled_neighbors_len = torch.tensor(0)
        fw_tmp = fw_adj_info[nodes]
        fw_perm = torch.randperm(fw_tmp.size(1))
        fw_tmp = fw_tmp[:, fw_perm]
        fw_sampled_neighbors = fw_tmp[:, :self.sample_size]
        bw_tmp = bw_adj_info[nodes]
        bw_perm = torch.randperm(bw_tmp.size(1))
        bw_tmp = bw_tmp[:, bw_perm]
        bw_sampled_neighbors = bw_tmp[:, :self.sample_size]
        for layer in range(self.sample_layer):
            if layer == 0:
                dim_mul = 1
            else:
                dim_mul = 1
            if layer == 0:
                neigh_vec_hidden = F.embedding(fw_sampled_neighbors, node_repres)
                tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                tmp_mask = torch.sign(tmp_sum)
                fw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
            else:
                neigh_vec_hidden = F.embedding(fw_sampled_neighbors, torch.cat([fw_hidden, torch.zeros([1, dim_mul * self.hidden_size])], dim=0))
            if layer > 6:
                fw_hidden = self.fw_aggregators[6]((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            else:
                fw_hidden = self.fw_aggregators[layer]((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))
            if self.bidirectional:
                if layer == 0:
                    neigh_vec_hidden = F.embedding(bw_sampled_neighbors, node_repres)
                    tmp_sum = torch.sum(F.relu(neigh_vec_hidden), 2)
                    tmp_mask = torch.sign(tmp_sum)
                    bw_sampled_neighbors_len = torch.sum(tmp_mask, 1)
                else:
                    neigh_vec_hidden = F.embedding(bw_sampled_neighbors, torch.cat([bw_hidden, torch.zeros([1, dim_mul * self.hidden_size])], dim=0))
                bw_hidden = self.dropout(bw_hidden)
                neigh_vec_hidden = self.dropout(neigh_vec_hidden)
                if layer > 6:
                    bw_hidden = self.bw_aggregators[6]((bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
                else:
                    bw_hidden = self.bw_aggregators[layer]((bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))
        fw_hidden = fw_hidden.view(-1, batch_nodes.size()[1], self.hidden_size)
        if self.bidirectional:
            bw_hidden = bw_hidden.view(-1, batch_nodes.size()[1], self.hidden_size)
            hidden = torch.cat([fw_hidden, bw_hidden], 2)
        else:
            hidden = fw_hidden
        pooled = torch.max(hidden, 1)[0]
        graph_embedding = pooled.view(-1, self.hidden_size)
        return hidden, graph_embedding, output_vector


class Dec_LSTM(nn.Module):

    def __init__(self, embedding_size, hidden_size, dropout_ratio):
        super(Dec_LSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.i2h = nn.Linear(self.embedding_size + 2 * self.hidden_size, 4 * self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, 4 * self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, x, prev_c, prev_h, parent_h, sibling_state):
        input_cat = torch.cat((x, parent_h, sibling_state), 1)
        gates = self.i2h(input_cat) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        cellgate = self.dropout(cellgate)
        cy = forgetgate * prev_c + ingate * cellgate
        hy = outgate * torch.tanh(cy)
        return cy, hy


class RNNBasedTreeDecoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, dropout_ratio):
        super(RNNBasedTreeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = BasicEmbedder(input_size, embedding_size, dropout_ratio, padding_idx=0)
        self.lstm = Dec_LSTM(embedding_size, hidden_size, dropout_ratio)

    def forward(self, input_src, prev_c, prev_h, parent_h, sibling_state):
        src_emb = self.embedding(input_src)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h, parent_h, sibling_state)
        return prev_cy, prev_hy


class SeparateAttention(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_ratio):
        super(SeparateAttention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_ratio = dropout_ratio
        self.separate_attention = True
        if self.separate_attention:
            self.linear_att = nn.Linear(3 * self.hidden_size, self.hidden_size)
        else:
            self.linear_att = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_ratio)
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, enc_s_top, dec_s_top, enc_2):
        dot = torch.bmm(enc_s_top, dec_s_top.unsqueeze(2))
        attention = self.softmax(dot.squeeze(2)).unsqueeze(2)
        enc_attention = torch.bmm(enc_s_top.permute(0, 2, 1), attention)
        if self.separate_attention:
            dot_2 = torch.bmm(enc_2, dec_s_top.unsqueeze(2))
            attention_2 = self.softmax(dot_2.squeeze(2)).unsqueeze(2)
            enc_attention_2 = torch.bmm(enc_2.permute(0, 2, 1), attention_2)
        if self.separate_attention:
            hid = torch.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2), enc_attention_2.squeeze(2), dec_s_top), 1)))
        else:
            hid = torch.tanh(self.linear_att(torch.cat((enc_attention.squeeze(2), dec_s_top), 1)))
        h2y_in = hid
        h2y_in = self.dropout(h2y_in)
        h2y = self.linear_out(h2y_in)
        pred = self.logsoftmax(h2y)
        return pred


class Tree:

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = []

    def __str__(self, level=0):
        ret = ''
        for child in self.children:
            if isinstance(child, type(self)):
                ret += child.__str__(level + 1)
            else:
                ret += '\t' * level + str(child) + '\n'
        return ret

    def add_child(self, c):
        if isinstance(c, type(self)):
            c.parent = self
        self.children.append(c)
        self.num_children = self.num_children + 1

    def to_string(self):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], Tree):
                r_list.append('( ' + self.children[i].to_string() + ' )')
            else:
                r_list.append(str(self.children[i]))
        return ''.join(r_list)

    def to_list(self, out_idx2symbol):
        r_list = []
        for i in range(self.num_children):
            if isinstance(self.children[i], type(self)):
                cl = self.children[i].to_list(out_idx2symbol)
                r_list.append(cl)
            elif self.children[i] == out_idx2symbol.index(SpecialTokens.NON_TOKEN):
                continue
            elif self.children[i] == out_idx2symbol.index(SpecialTokens.EOS_TOKEN):
                continue
            else:
                r_list.append(self.children[i])
        return r_list


class Graph2TreeIBM(nn.Module):

    def __init__(self, config):
        super(Graph2TreeIBM, self).__init__()
        self.hidden_size = config['hidden_size']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.max_length = config['max_output_len']
        self.out_idx2symbol = config['out_idx2symbol']
        self.encoder = GraphEncoder(config['vocab_size'], config['embedding_size'], config['hidden_size'], config['sample_size'], config['sample_layer'], config['bidirectional'], config['encoder_dropout_ratio'])
        self.attention = SeparateAttention(config['hidden_size'], config['symbol_size'], config['attention_dropout_ratio'])
        self.decoder = RNNBasedTreeDecoder(config['vocab_size'], config['embedding_size'], config['hidden_size'], config['decoder_dropout_ratio'])

    def forward(self, seq, seq_length, group_nums, target=None):
        enc_max_len = seq_length.max()
        batch_size = seq_length.size(0)
        device = seq.device
        enc_outputs = torch.zeros((batch_size, enc_max_len, self.hidden_size), requires_grad=True)
        fw_adj_info, bw_adj_info, nodes = self.build_graph(group_nums, seq_length)
        fw_adj_info = fw_adj_info
        bw_adj_info = bw_adj_info
        nodes = nodes
        node_embedding, graph_embedding, structural_info = self.encoder(fw_adj_info, bw_adj_info, seq, nodes)
        if target != None:
            predict, label = self.generate_t(node_embedding, graph_embedding, structural_info, target)
            return predict, label
        else:
            outputs = self.generate_without_t(node_embedding, graph_embedding, structural_info)
            return outputs

    def generate_t(self, node_embedding, graph_embedding, structural_info, target):
        device = node_embedding.device
        batch_size = node_embedding.size(0)
        enc_outputs = node_embedding
        graph_cell_state = graph_embedding
        graph_hidden_state = graph_embedding
        tree_batch = []
        for tar_equ in target:
            tree_batch.append(self.equ2tree(tar_equ))
        dec_batch, queue_tree, max_index = self.get_dec_batch(tree_batch, batch_size)
        predict = []
        label = []
        dec_s = {}
        for i in range(1, self.max_length + 1):
            dec_s[i] = {}
            for j in range(self.max_length + 1):
                dec_s[i][j] = {}
        cur_index = 1
        while cur_index <= max_index:
            for j in range(1, 3):
                dec_s[cur_index][0][j] = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True)
            sibling_state = torch.zeros((batch_size, self.hidden_size), dtype=torch.float, requires_grad=True)
            if cur_index == 1:
                for b_i in range(batch_size):
                    dec_s[1][0][1][b_i, :] = graph_cell_state[b_i]
                    dec_s[1][0][2][b_i, :] = graph_hidden_state[b_i]
            else:
                for b_i in range(1, batch_size + 1):
                    if cur_index <= len(queue_tree[b_i]):
                        par_index = queue_tree[b_i][cur_index - 1]['parent']
                        child_index = queue_tree[b_i][cur_index - 1]['child_index']
                        dec_s[cur_index][0][1][b_i - 1, :] = dec_s[par_index][child_index][1][b_i - 1, :]
                        dec_s[cur_index][0][2][b_i - 1, :] = dec_s[par_index][child_index][2][b_i - 1, :]
                    flag_sibling = False
                    for q_index in range(len(queue_tree[b_i])):
                        if cur_index <= len(queue_tree[b_i]) and q_index < cur_index - 1 and queue_tree[b_i][q_index]['parent'] == queue_tree[b_i][cur_index - 1]['parent'] and queue_tree[b_i][q_index]['child_index'] < queue_tree[b_i][cur_index - 1]['child_index']:
                            flag_sibling = True
                            sibling_index = q_index
                    if flag_sibling:
                        sibling_state[b_i - 1, :] = dec_s[sibling_index][dec_batch[sibling_index].size(1) - 1][2][b_i - 1, :]
            parent_h = dec_s[cur_index][0][2]
            for i in range(dec_batch[cur_index].size(1) - 1):
                teacher_force = random.random() < self.teacher_force_ratio
                if teacher_force != True and i > 0:
                    input_word = pred.argmax(1)
                else:
                    input_word = dec_batch[cur_index][:, i]
                dec_s[cur_index][i + 1][1], dec_s[cur_index][i + 1][2] = self.decoder(input_word, dec_s[cur_index][i][1], dec_s[cur_index][i][2], parent_h, sibling_state)
                pred = self.attention(enc_outputs, dec_s[cur_index][i + 1][2], structural_info)
                predict.append(pred)
                label.append(dec_batch[cur_index][:, i + 1])
            cur_index = cur_index + 1
        predict = torch.stack(predict, dim=1)
        label = torch.stack(label, dim=1)
        predict = predict.view(-1, predict.size(2))
        label = label.view(-1, label.size(1))
        return predict, label

    def generate_without_t(self, node_embedding, graph_embedding, structural_info):
        batch_size = node_embedding.size(0)
        device = node_embedding.device
        enc_outputs = node_embedding
        prev_c = graph_embedding
        prev_h = graph_embedding
        outputs = []
        for b_i in range(batch_size):
            queue_decode = []
            queue_decode.append({'s': (prev_c[b_i].unsqueeze(0), prev_h[b_i].unsqueeze(0)), 'parent': 0, 'child_index': 1, 't': Tree()})
            head = 1
            while head <= len(queue_decode) and head <= 100:
                s = queue_decode[head - 1]['s']
                parent_h = s[1]
                t = queue_decode[head - 1]['t']
                sibling_state = torch.zeros((1, self.encoder.hidden_size), dtype=torch.float, requires_grad=False)
                flag_sibling = False
                for q_index in range(len(queue_decode)):
                    if head <= len(queue_decode) and q_index < head - 1 and queue_decode[q_index]['parent'] == queue_decode[head - 1]['parent'] and queue_decode[q_index]['child_index'] < queue_decode[head - 1]['child_index']:
                        flag_sibling = True
                        sibling_index = q_index
                if flag_sibling:
                    sibling_state = queue_decode[sibling_index]['s'][1]
                if head == 1:
                    prev_word = torch.tensor([self.out_idx2symbol.index(SpecialTokens.SOS_TOKEN)], dtype=torch.long)
                else:
                    prev_word = torch.tensor([self.out_idx2symbol.index(SpecialTokens.NON_TOKEN)], dtype=torch.long)
                i_child = 1
                while True:
                    curr_c, curr_h = self.decoder(prev_word, s[0], s[1], parent_h, sibling_state)
                    prediction = self.attention(enc_outputs[b_i].unsqueeze(0), curr_h, structural_info[b_i].unsqueeze(0))
                    s = curr_c, curr_h
                    _, _prev_word = prediction.max(1)
                    prev_word = _prev_word
                    if int(prev_word[0]) == self.out_idx2symbol.index(SpecialTokens.EOS_TOKEN) or t.num_children >= self.max_length:
                        break
                    elif int(prev_word[0]) == self.out_idx2symbol.index(SpecialTokens.NON_TOKEN):
                        queue_decode.append({'s': (s[0].clone(), s[1].clone()), 'parent': head, 'child_index': i_child, 't': Tree()})
                        t.add_child(int(prev_word[0]))
                    else:
                        t.add_child(int(prev_word[0]))
                    i_child = i_child + 1
                head = head + 1
            for i in range(len(queue_decode) - 1, 0, -1):
                cur = queue_decode[i]
                queue_decode[cur['parent'] - 1]['t'].children[cur['child_index'] - 1] = cur['t']
            output = queue_decode[0]['t'].to_list(self.out_idx2symbol)
            outputs.append(output)
        return outputs

    def build_graph(self, group_nums, seq_length):
        max_length = seq_length.max()
        batch_size = len(seq_length)
        max_degree = 6
        fw_adj_info_batch = []
        bw_adj_info_batch = []
        slide = 0
        for b_i in range(batch_size):
            x = torch.zeros((max_length, max_degree)).long()
            fw_adj_info = torch.clone(x)
            bw_adj_info = torch.clone(x)
            fw_idx = torch.zeros(max_length).long()
            bw_idx = torch.zeros(max_length).long()
            for idx in group_nums[b_i]:
                if fw_idx[idx[0]] < max_degree:
                    fw_adj_info[idx[0], fw_idx[idx[0]]] = idx[1] + slide
                    fw_idx[idx[0]] += 1
                if bw_idx[idx[1]] < max_degree:
                    bw_adj_info[idx[1], bw_idx[idx[1]]] = idx[0] + slide
                    bw_idx[idx[1]] += 1
            for row_idx, col_idx in enumerate(fw_idx):
                for idx_slide in range(max_degree - col_idx):
                    fw_adj_info[row_idx, col_idx + idx_slide] = max_length - 1 + slide
            for row_idx, col_idx in enumerate(bw_idx):
                for idx_slide in range(max_degree - col_idx):
                    bw_adj_info[row_idx, col_idx + idx_slide] = max_length - 1 + slide
            fw_adj_info_batch.append(fw_adj_info)
            bw_adj_info_batch.append(bw_adj_info)
            slide += max_length
        fw_adj_info_batch = torch.cat(fw_adj_info_batch, dim=0)
        bw_adj_info_batch = torch.cat(bw_adj_info_batch, dim=0)
        nodes_batch = torch.arange(0, fw_adj_info_batch.size(0)).view(batch_size, max_length)
        return fw_adj_info_batch, bw_adj_info_batch, nodes_batch

    def get_dec_batch(self, dec_tree_batch, batch_size):
        queue_tree = {}
        for i in range(1, batch_size + 1):
            queue_tree[i] = []
            queue_tree[i].append({'tree': dec_tree_batch[i - 1], 'parent': 0, 'child_index': 1})
        cur_index, max_index = 1, 1
        dec_batch = {}
        while cur_index <= max_index:
            max_w_len = -1
            batch_w_list = []
            for i in range(1, batch_size + 1):
                w_list = []
                counts = 0
                if cur_index <= len(queue_tree[i]):
                    t = queue_tree[i][cur_index - 1]['tree']
                    for ic in range(t.num_children):
                        if isinstance(t.children[ic], Tree):
                            queue_tree[i].append({'tree': t.children[ic], 'parent': cur_index, 'child_index': ic + 1 - counts})
                            counts += 1
                        else:
                            w_list.append(t.children[ic])
                    if len(queue_tree[i]) > max_index:
                        max_index = len(queue_tree[i])
                if len(w_list) > max_w_len:
                    max_w_len = len(w_list)
                batch_w_list.append(w_list)
            dec_batch[cur_index] = torch.zeros((batch_size, max_w_len + 1), dtype=torch.long)
            for i in range(batch_size):
                w_list = batch_w_list[i]
                if len(w_list) > 0:
                    for j in range(len(w_list)):
                        dec_batch[cur_index][i][j + 1] = w_list[j]
                if cur_index == 1:
                    dec_batch[cur_index][i][0] = self.out_idx2symbol.index(SpecialTokens.SOS_TOKEN)
                else:
                    dec_batch[cur_index][i][0] = self.out_idx2symbol.index(SpecialTokens.NON_TOKEN)
            cur_index += 1
        return dec_batch, queue_tree, max_index

    def equ2tree(self, equ):
        t = Tree()
        for symbol in equ:
            if isinstance(symbol, list):
                sub_tree = self.equ2tree(symbol)
                t.add_child(self.out_idx2symbol.index(SpecialTokens.NON_TOKEN))
                t.add_child(sub_tree)
            else:
                t.add_child(symbol)
        t.add_child(self.out_idx2symbol.index(SpecialTokens.EOS_TOKEN))
        return t


class Beam:

    def __init__(self, score, input_var, hidden, token_logits, outputs, all_output=None):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output
        self.token_logits = token_logits
        self.outputs = outputs


class Parse_Graph_Module(nn.Module):

    def __init__(self, hidden_size):
        super(Parse_Graph_Module, self).__init__()
        self.hidden_size = hidden_size
        self.node_fc1 = nn.Linear(hidden_size, hidden_size)
        self.node_fc2 = nn.Linear(hidden_size, hidden_size)
        self.node_out = nn.Linear(hidden_size * 2, hidden_size)

    def normalize(self, graph, symmetric=True):
        d = graph.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(graph).mm(D)
        else:
            D = torch.diag(torch.pow(d, -1))
            return D.mm(graph)

    def forward(self, node, graph):
        graph = graph.float()
        batch_size = node.size(0)
        for i in range(batch_size):
            graph[i] = self.normalize(graph[i])
        node_info = torch.relu(self.node_fc1(torch.matmul(graph, node)))
        node_info = torch.relu(self.node_fc2(torch.matmul(graph, node_info)))
        agg_node_info = torch.cat((node, node_info), dim=2)
        agg_node_info = torch.relu(self.node_out(agg_node_info))
        return agg_node_info


def clones(module, N):
    """Produce N identical layers.
    """
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GraphBasedMultiEncoder(nn.Module):

    def __init__(self, input1_size, input2_size, embed_model, embedding1_size, embedding2_size, hidden_size, n_layers=2, hop_size=2, dropout=0.5):
        super(GraphBasedMultiEncoder, self).__init__()
        self.input1_size = input1_size
        self.input2_size = input2_size
        self.embedding1_size = embedding1_size
        self.embedding2_size = embedding2_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.hop_size = hop_size
        self.embedding1 = embed_model
        self.embedding2 = nn.Embedding(input2_size, embedding2_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding1_size + embedding2_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.parse_gnn = clones(Parse_Graph_Module(hidden_size), hop_size)

    def forward(self, input1_var, input2_var, input_length, parse_graph, hidden=None):
        """
        """
        embedded1 = self.embedding1(input1_var)
        embedded2 = self.embedding2(input2_var)
        embedded = torch.cat((embedded1, embedded2), dim=2)
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_length)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.gru(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        pade_outputs = pade_outputs.transpose(0, 1)
        for i in range(self.hop_size):
            pade_outputs = self.parse_gnn[i](pade_outputs, parse_graph[:, 2])
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs, pade_hidden


class Num_Graph_Module(nn.Module):

    def __init__(self, node_dim):
        super(Num_Graph_Module, self).__init__()
        self.node_dim = node_dim
        self.node1_fc1 = nn.Linear(node_dim, node_dim)
        self.node1_fc2 = nn.Linear(node_dim, node_dim)
        self.node2_fc1 = nn.Linear(node_dim, node_dim)
        self.node2_fc2 = nn.Linear(node_dim, node_dim)
        self.graph_weight = nn.Linear(node_dim * 4, node_dim)
        self.node_out = nn.Linear(node_dim * 2, node_dim)

    def normalize(self, graph, symmetric=True):
        d = graph.sum(1)
        if symmetric:
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(graph).mm(D)
        else:
            D = torch.diag(torch.pow(d, -1))
            return D.mm(graph)

    def forward(self, node, graph1, graph2):
        graph1 = graph1.float()
        graph2 = graph2.float()
        batch_size = node.size(0)
        for i in range(batch_size):
            graph1[i] = self.normalize(graph1[i], False)
            graph2[i] = self.normalize(graph2[i], False)
        node_info1 = torch.relu(self.node1_fc1(torch.matmul(graph1, node)))
        node_info1 = torch.relu(self.node1_fc2(torch.matmul(graph1, node_info1)))
        node_info2 = torch.relu(self.node2_fc1(torch.matmul(graph2, node)))
        node_info2 = torch.relu(self.node2_fc2(torch.matmul(graph2, node_info2)))
        gate = torch.cat((node_info1, node_info2, node_info1 + node_info2, node_info1 - node_info2), dim=2)
        gate = torch.sigmoid(self.graph_weight(gate))
        node_info = gate * node_info1 + (1 - gate) * node_info2
        agg_node_info = torch.cat((node, node_info), dim=2)
        agg_node_info = torch.relu(self.node_out(agg_node_info))
        return agg_node_info


def replace_masked_values(tensor, mask, replace_with):
    return tensor.masked_fill((1 - mask).bool(), replace_with)


class NumEncoder(nn.Module):

    def __init__(self, node_dim, hop_size=2):
        super(NumEncoder, self).__init__()
        self.node_dim = node_dim
        self.hop_size = hop_size
        self.num_gnn = clones(Num_Graph_Module(node_dim), hop_size)

    def forward(self, encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad):
        num_embedding = num_encoder_outputs.clone()
        batch_size = num_embedding.size(0)
        num_mask = (num_pos_pad > -1).long()
        node_mask = (num_order_pad > 0).long()
        greater_graph_mask = num_order_pad.unsqueeze(-1).expand(batch_size, -1, num_order_pad.size(-1)) > num_order_pad.unsqueeze(1).expand(batch_size, num_order_pad.size(-1), -1)
        lower_graph_mask = num_order_pad.unsqueeze(-1).expand(batch_size, -1, num_order_pad.size(-1)) <= num_order_pad.unsqueeze(1).expand(batch_size, num_order_pad.size(-1), -1)
        greater_graph_mask = greater_graph_mask.long()
        lower_graph_mask = lower_graph_mask.long()
        diagmat = torch.diagflat(torch.ones(num_embedding.size(1), dtype=torch.long, device=num_embedding.device))
        diagmat = diagmat.unsqueeze(0).expand(num_embedding.size(0), -1, -1)
        graph_ = node_mask.unsqueeze(1) * node_mask.unsqueeze(-1) * (1 - diagmat)
        graph_greater = graph_ * greater_graph_mask + diagmat
        graph_lower = graph_ * lower_graph_mask + diagmat
        for i in range(self.hop_size):
            num_embedding = self.num_gnn[i](num_embedding, graph_greater, graph_lower)
        gnn_info_vec = torch.zeros((batch_size, encoder_outputs.size(0) + 1, encoder_outputs.size(-1)), dtype=torch.float, device=num_embedding.device)
        clamped_number_indices = replace_masked_values(num_pos_pad, num_mask, gnn_info_vec.size(1) - 1)
        gnn_info_vec.scatter_(1, clamped_number_indices.unsqueeze(-1).expand(-1, -1, num_embedding.size(-1)), num_embedding)
        gnn_info_vec = gnn_info_vec[:, :-1, :]
        gnn_info_vec = gnn_info_vec.transpose(0, 1)
        gnn_info_vec = encoder_outputs + gnn_info_vec
        num_embedding = num_encoder_outputs + num_embedding
        problem_output = torch.max(gnn_info_vec, 0).values
        return gnn_info_vec, num_embedding, problem_output


class TreeAttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(TreeAttnDecoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(hidden_size, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))
        return output, hidden


class MultiEncDec(nn.Module):
    """
    Reference:
        Shen et al. "Solving Math Word Problems with Multi-Encoders and Multi-Decoders" in COLING 2020.
    """

    def __init__(self, config, dataset):
        super(MultiEncDec, self).__init__()
        self.device = config['device']
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.rnn_cell_type = config['rnn_cell_type']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.n_layers = config['num_layers']
        self.hop_size = config['hop_size']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.dropout_ratio = config['dropout_ratio']
        self.operator_nums = dataset.operator_nums
        self.generate_nums = len(dataset.generate_list)
        self.num_start1 = dataset.num_start1
        self.num_start2 = dataset.num_start2
        self.input1_size = len(dataset.in_idx2word_1)
        self.input2_size = len(dataset.in_idx2word_2)
        self.output2_size = len(dataset.out_idx2symbol_2)
        self.unk1 = dataset.out_symbol2idx_1[SpecialTokens.UNK_TOKEN]
        self.unk2 = dataset.out_symbol2idx_2[SpecialTokens.UNK_TOKEN]
        self.sos2 = dataset.out_symbol2idx_2[SpecialTokens.SOS_TOKEN]
        self.eos2 = dataset.out_symbol2idx_2[SpecialTokens.EOS_TOKEN]
        self.out_symbol2idx1 = dataset.out_symbol2idx_1
        self.out_idx2symbol1 = dataset.out_idx2symbol_1
        self.out_symbol2idx2 = dataset.out_symbol2idx_2
        self.out_idx2symbol2 = dataset.out_idx2symbol_2
        generate_list = dataset.generate_list
        self.generate_list = [self.out_symbol2idx1[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number
        try:
            self.out_sos_token1 = self.out_symbol2idx1[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token1 = None
        try:
            self.out_eos_token1 = self.out_symbol2idx1[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token1 = None
        try:
            self.out_pad_token1 = self.out_symbol2idx1[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token1 = None
        try:
            self.out_sos_token2 = self.out_symbol2idx2[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token2 = None
        try:
            self.out_eos_token2 = self.out_symbol2idx2[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token2 = None
        try:
            self.out_pad_token2 = self.out_symbol2idx2[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token2 = None
        embedder = nn.Embedding(self.input1_size, self.embedding_size)
        in_embedder = self._init_embedding_params(dataset.trainset, dataset.in_idx2word_1, config['embedding_size'], embedder)
        self.encoder = GraphBasedMultiEncoder(input1_size=self.input1_size, input2_size=self.input2_size, embed_model=in_embedder, embedding1_size=self.embedding_size, embedding2_size=self.embedding_size // 4, hidden_size=self.hidden_size, n_layers=self.n_layers, hop_size=self.hop_size)
        self.numencoder = NumEncoder(node_dim=self.hidden_size, hop_size=self.hop_size)
        self.tree_decoder = Prediction(hidden_size=self.hidden_size, op_nums=self.operator_nums, input_size=self.generate_nums)
        self.generate = GenerateNode(hidden_size=self.hidden_size, op_nums=self.operator_nums, embedding_size=self.embedding_size)
        self.merge = Merge(hidden_size=self.hidden_size, embedding_size=self.embedding_size)
        self.attn_decoder = TreeAttnDecoderRNN(self.hidden_size, self.embedding_size, self.output2_size, self.output2_size, self.n_layers, self.dropout_ratio)
        self.loss = MaskedCrossEntropyLoss()

    def forward(self, input1, input2, input_length, num_size, num_pos, num_order, parse_graph, num_stack, target1=None, target2=None, output_all_layers=False):
        """

        :param torch.Tensor input1:
        :param torch.Tensor input2:
        :param torch.Tensor input_length:
        :param list num_size:
        :param list num_pos:
        :param list num_order:
        :param torch.Tensor parse_graph:
        :param list num_stack:
        :param torch.Tensor | None target1:
        :param torch.Tensor | None target2:
        :param bool output_all_layers:
        :return:
        """
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([(0) for _ in range(i)] + [(1) for _ in range(i, max_len)])
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_list)
        for i in num_size:
            d = i + len(self.generate_list)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_pos_pad = []
        max_num_pos_size = max(num_size)
        for i in range(len(num_pos)):
            temp = num_pos[i] + [-1] * (max_num_pos_size - len(num_pos[i]))
            num_pos_pad.append(temp)
        num_order_pad = []
        max_num_order_size = max(num_size)
        for i in range(len(num_order)):
            temp = num_order[i] + [0] * (max_num_order_size - len(num_order[i]))
            num_order_pad.append(temp)
        seq_mask = torch.ByteTensor(seq_mask)
        num_mask = torch.ByteTensor(num_mask)
        num_pos_pad = torch.LongTensor(num_pos_pad)
        num_order_pad = torch.LongTensor(num_order_pad)
        encoder_outputs, num_outputs, encoder_hidden, problem_output, encoder_layer_outputs = self.encoder_forward(input1, input2, input_length, parse_graph, num_pos, num_pos_pad, num_order_pad, output_all_layers)
        attn_decoder_hidden = encoder_hidden[:self.n_layers]
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, problem_output, attn_decoder_hidden, num_outputs, seq_mask, num_mask, num_stack, target1, target2, output_all_layers)
        model_layer_outputs = {}
        if output_all_layers:
            model_layer_outputs.update(encoder_layer_outputs)
            model_layer_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_layer_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'input1', 'input2', 'output1', 'output2',
        'input1 len', 'parse graph', 'num stack', 'output1 len', 'output2 len',
        'num size', 'num pos', 'num order'
        """
        input1_var = torch.tensor(batch_data['input1'])
        input2_var = torch.tensor(batch_data['input2'])
        target1 = torch.tensor(batch_data['output1'])
        target2 = torch.tensor(batch_data['output2'])
        input_length = torch.tensor(batch_data['input1 len'])
        parse_graph = torch.tensor(batch_data['parse graph'])
        num_stack_batch = copy.deepcopy(batch_data['num stack'])
        target1_length = torch.LongTensor(batch_data['output1 len'])
        target2_length = torch.LongTensor(batch_data['output2 len'])
        num_size_batch = batch_data['num size']
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']
        token_logits, _, all_layer_outputs = self.forward(input1_var, input2_var, input_length, num_size_batch, num_pos_batch, num_order_batch, parse_graph, num_stack_batch, target1, target2, output_all_layers=True)
        target1 = all_layer_outputs['target1']
        target2 = all_layer_outputs['target2']
        tree_token_logits, attn_token_logits = token_logits
        loss1 = masked_cross_entropy(tree_token_logits, target1, target1_length)
        loss2 = masked_cross_entropy(attn_token_logits.contiguous(), target2.contiguous(), target2_length)
        loss = loss1 + loss2
        loss.backward()
        if self.USE_CUDA:
            torch.cuda.empty_cache()
        return loss

    def model_test(self, batch_data: dict) ->Tuple[str, list, list]:
        """Model test.

        :param batch_data: one batch data.
        :return: result_type, predicted equation, target equation.

        batch_data should include keywords 'input1', 'input2', 'output1', 'output2',
        'input1 len', 'parse graph', 'num stack', 'num pos', 'num order', 'num list'
        """
        input1_var = torch.tensor(batch_data['input1'])
        input2_var = torch.tensor(batch_data['input2'])
        target1 = torch.tensor(batch_data['output1'])
        target2 = torch.tensor(batch_data['output2'])
        input_length = torch.tensor(batch_data['input1 len'])
        parse_graph = torch.tensor(batch_data['parse graph'])
        num_stack_batch = copy.deepcopy(batch_data['num stack'])
        num_size_batch = batch_data['num size']
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']
        num_list = batch_data['num list']
        _, outputs, all_layer_outputs = self.forward(input1_var, input2_var, input_length, num_size_batch, num_pos_batch, num_order_batch, parse_graph, num_stack_batch, output_all_layers=True)
        tree_outputs, attn_outputs = outputs
        tree_score = all_layer_outputs['tree_score']
        attn_score = all_layer_outputs['attn_score']
        if tree_score < attn_score:
            output1 = self.convert_idx2symbol1(tree_outputs[0], num_list[0], copy_list(num_stack_batch[0]))
            targets1 = self.convert_idx2symbol1(target1[0], num_list[0], copy_list(num_stack_batch[0]))
            result_type = 'tree'
            if self.USE_CUDA:
                torch.cuda.empty_cache()
            return result_type, output1, targets1
        else:
            output2 = self.convert_idx2symbol2(attn_outputs, num_list, copy_list(num_stack_batch))
            targets2 = self.convert_idx2symbol2(target2, num_list, copy_list(num_stack_batch))
            result_type = 'attn'
            if self.USE_CUDA:
                torch.cuda.empty_cache()
            return result_type, output2, targets2

    def predict(self, batch_data, output_all_layers=False):
        input1_var = torch.tensor(batch_data['input1'])
        input2_var = torch.tensor(batch_data['input2'])
        input_length = torch.tensor(batch_data['input1 len'])
        parse_graph = torch.tensor(batch_data['parse graph'])
        num_stack_batch = copy.deepcopy(batch_data['num stack'])
        num_size_batch = batch_data['num size']
        num_pos_batch = batch_data['num pos']
        num_order_batch = batch_data['num order']
        token_logits, symbol_outputs, model_all_outputs = self.forward(input1_var, input2_var, input_length, num_size_batch, num_pos_batch, num_order_batch, parse_graph, num_stack_batch, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [(1) for _ in range(hidden_size)]
        temp_0 = [(0) for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [(0) for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.ByteTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices
            masked_index = masked_index
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index.bool(), 0.0), masked_index

    def generate_decoder_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        if self.USE_CUDA:
            decoder_output = decoder_output.cpu()
        target = torch.LongTensor(target)
        for i in range(target.size(0)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
        return target

    def encoder_forward(self, input1, input2, input_length, parse_graph, num_pos, num_pos_pad, num_order_pad, output_all_layers=False):
        input1 = input1.transpose(0, 1)
        input2 = input2.transpose(0, 1)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        batch_size = input1.size(1)
        encoder_outputs, encoder_hidden = self.encoder(input1, input2, input_length, parse_graph)
        num_encoder_outputs, masked_index = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, self.hidden_size)
        encoder_outputs, num_outputs, problem_output = self.numencoder(encoder_outputs, num_encoder_outputs, num_pos_pad, num_order_pad)
        num_outputs = num_outputs.masked_fill_(masked_index.bool(), 0.0)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
            all_layer_outputs['inputs_representation'] = problem_output
            all_layer_outputs['number_representation'] = num_encoder_outputs
            all_layer_outputs['num_encoder_outputs'] = num_outputs
        return encoder_outputs, num_outputs, encoder_hidden, problem_output, all_layer_outputs

    def decoder_forward(self, encoder_outputs, problem_output, attn_decoder_hidden, all_nums_encoder_outputs, seq_mask, num_mask, num_stack, target1, target2, output_all_layers):
        num_stack1 = copy.deepcopy(num_stack)
        num_stack2 = copy.deepcopy(num_stack)
        tree_token_logits, tree_outputs, tree_layer_outputs = self.tree_decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, num_stack1, seq_mask, num_mask, target1, output_all_layers)
        attn_token_logits, attn_outputs, attn_layer_outputs = self.attn_decoder_forward(encoder_outputs, seq_mask, attn_decoder_hidden, num_stack2, target2, output_all_layers)
        all_layer_output = {}
        if output_all_layers:
            all_layer_output.update(tree_layer_outputs)
            all_layer_output.update(attn_layer_outputs)
        return (tree_token_logits, attn_token_logits), (tree_outputs, attn_outputs), all_layer_output

    def tree_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(1)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            target = target.transpose(0, 1)
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.tree_decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start1, self.unk1)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start1:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start1].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            target = target.transpose(0, 1)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.tree_decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start1:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start1].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
            score = beams[0].score
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['tree_token_logits'] = token_logits
            all_layer_outputs['tree_outputs'] = outputs
            all_layer_outputs['target1'] = target
            if target is None:
                all_layer_outputs['tree_score'] = score
        return token_logits, outputs, all_layer_outputs

    def attn_decoder_forward(self, encoder_outputs, seq_mask, decoder_hidden, num_stack, target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(1)
        decoder_input = torch.LongTensor([self.sos2] * batch_size)
        if target is not None:
            target = target.transpose(0, 1)
        max_output_length = target.size(0) if target is not None else self.max_out_len
        token_logits = torch.zeros(max_output_length, batch_size, self.attn_decoder.output_size)
        outputs = torch.zeros(max_output_length, batch_size)
        if self.USE_CUDA:
            token_logits = token_logits
        if target is not None and random.random() < self.teacher_force_ratio:
            for t in range(max_output_length):
                if self.USE_CUDA:
                    decoder_input = decoder_input
                token_logit, decoder_hidden = self.attn_decoder(decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits[t] = token_logit
                outputs[t] = output.squeeze(-1)
                decoder_input = self.generate_decoder_input(target[t].cpu().tolist(), token_logit, num_stack, self.num_start2, self.unk2)
                target[t] = decoder_input
        else:
            beam_list = list()
            score = torch.zeros(batch_size)
            if self.USE_CUDA:
                score = score
            beam_list.append(Beam(score, decoder_input, decoder_hidden, token_logits, outputs))
            for t in range(max_output_length):
                beam_len = len(beam_list)
                beam_scores = torch.zeros(batch_size, self.attn_decoder.output_size * beam_len)
                all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
                all_token_logits = torch.zeros(max_output_length, batch_size * beam_len, self.attn_decoder.output_size)
                all_outputs = torch.zeros(max_output_length, batch_size * beam_len)
                if self.USE_CUDA:
                    beam_scores = beam_scores
                    all_hidden = all_hidden
                    all_token_logits = all_token_logits
                    all_outputs = all_outputs
                for b_idx in range(len(beam_list)):
                    decoder_input = beam_list[b_idx].input_var
                    decoder_hidden = beam_list[b_idx].hidden
                    if self.USE_CUDA:
                        decoder_input = decoder_input
                    token_logit, decoder_hidden = self.attn_decoder(decoder_input, decoder_hidden, encoder_outputs, seq_mask)
                    score = F.log_softmax(token_logit, dim=1)
                    beam_score = beam_list[b_idx].score
                    beam_score = beam_score.unsqueeze(1)
                    repeat_dims = [1] * beam_score.dim()
                    repeat_dims[1] = score.size(1)
                    beam_score = beam_score.repeat(*repeat_dims)
                    score += beam_score
                    beam_scores[:, b_idx * self.attn_decoder.output_size:(b_idx + 1) * self.attn_decoder.output_size] = score
                    all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden
                    beam_list[b_idx].token_logits[t] = token_logit
                    all_token_logits[:, batch_size * b_idx:batch_size * (b_idx + 1), :] = beam_list[b_idx].token_logits
                topv, topi = beam_scores.topk(self.beam_size, dim=1)
                beam_list = list()
                for k in range(self.beam_size):
                    temp_topk = topi[:, k]
                    temp_input = temp_topk % self.attn_decoder.output_size
                    temp_input = temp_input.data
                    if self.USE_CUDA:
                        temp_input = temp_input.cpu()
                    temp_beam_pos = temp_topk / self.attn_decoder.output_size
                    temp_beam_pos = torch.floor(temp_beam_pos).long()
                    indices = torch.LongTensor(range(batch_size))
                    if self.USE_CUDA:
                        indices = indices
                    indices += temp_beam_pos * batch_size
                    temp_hidden = all_hidden.index_select(1, indices)
                    temp_token_logits = all_token_logits.index_select(1, indices)
                    temp_output = all_outputs.index_select(1, indices)
                    temp_output[t] = temp_input
                    beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_token_logits, temp_output))
            token_logits = beam_list[0].token_logits
            outputs = beam_list[0].outputs
            score = beam_list[0].score
            if target is not None:
                for t in range(max_output_length):
                    target[t] = self.generate_decoder_input(target[t].cpu().tolist(), token_logits[t], num_stack, self.num_start2, self.unk2)
        token_logits = token_logits.transpose(0, 1)
        outputs = outputs.transpose(0, 1)
        if target is not None:
            target = target.transpose(0, 1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['attn_token_logits'] = token_logits
            all_layer_outputs['attn_outputs'] = outputs
            all_layer_outputs['target2'] = target
            if target is None:
                all_layer_outputs['attn_score'] = score
        return token_logits, outputs, all_layer_outputs

    def _init_embedding_params(self, train_data, vocab, embedding_size, embedder):
        sentences = []
        for data in train_data:
            sentence = []
            for word in data['question']:
                if word in vocab:
                    sentence.append(word)
                else:
                    sentence.append(SpecialTokens.UNK_TOKEN)
            sentences.append(sentence)
        model = word2vec.Word2Vec(sentences, vector_size=embedding_size, min_count=1)
        emb_vectors = []
        pad_idx = vocab.index(SpecialTokens.PAD_TOKEN)
        for idx in range(len(vocab)):
            if idx != pad_idx:
                emb_vectors.append(np.array(model.wv[vocab[idx]]))
            else:
                emb_vectors.append(np.zeros(embedding_size))
        emb_vectors = np.array(emb_vectors)
        embedder.weight.data.copy_(torch.from_numpy(emb_vectors))
        return embedder

    def convert_idx2symbol1(self, output, num_list, num_stack):
        """batch_size=1"""
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token1, self.out_eos_token1, self.out_pad_token1]:
                break
            symbol = self.out_idx2symbol1[idx]
            if 'NUM' in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list

    def convert_idx2symbol2(self, output, num_list, num_stack):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token2, self.out_eos_token2, self.out_pad_token2]:
                    break
                symbol = self.out_idx2symbol2[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                elif symbol == SpecialTokens.UNK_TOKEN:
                    try:
                        pos_list = num_stack[b_i].pop()
                        c = num_list[b_i][pos_list[0]]
                        res.append(c)
                    except:
                        res.append(symbol)
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list


class DatasetName:
    """dataset name
    """
    math23k = 'math23k'
    hmwp = 'hmwp'
    mawps = 'mawps'
    ape200k = 'ape200k'
    alg514 = 'alg514'
    draw = 'draw'
    SVAMP = 'SVAMP'
    asdiv_a = 'asdiv-a'
    mawps_single = 'mawps-single'
    mawps_asdiv_a_svamp = 'mawps_asdiv-a_svamp'


class NLLLoss(AbstractLoss):
    _NAME = 'Avg NLLLoss'

    def __init__(self, weight=None, mask=None, size_average=True):
        """
        Args:
            weight (Tensor, optional): a manual rescaling weight given to each class.
            
            mask (Tensor, optional): index of classes to rescale weight
        """
        self.mask = mask
        self.size_average = size_average
        if mask is not None:
            if weight is None:
                raise ValueError('Must provide weight with a mask.')
            weight[mask] = 0
        super(NLLLoss, self).__init__(self._NAME, nn.NLLLoss(weight=weight, reduction='mean'))

    def get_loss(self):
        """return loss

        Returns:
            loss (float)
        """
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()
        if self.size_average:
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        """calculate loss

        Args:
            outputs (Tensor): output distribution of model.

            target (Tensor): target classes. 
        """
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1


class PositionEmbedder(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.
    """

    def __init__(self, embedding_size, max_length=512):
        super(PositionEmbedder, self).__init__()
        self.embedding_size = embedding_size
        self.weights = self.get_embedding(max_length, embedding_size)

    def get_embedding(self, max_length, embedding_size):
        """Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_length, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(max_length, -1)
        if embedding_size % 2 == 1:
            emb = torch.cat([emb, torch.zeros(max_length, 1)], dim=1)
        return emb

    def forward(self, input_seq, offset=0):
        """
        Args:
            input_seq (torch.Tensor): input sequence, shape [batch_size, sequence_length].
        
        Returns:
            torch.Tensor: position embedding, shape [batch_size, sequence_length, embedding_size].
        """
        batch_size, seq_len = input_seq.size()
        max_position = seq_len + offset
        if self.weights is None or max_position > self.weights.size(0):
            self.weights = self.get_embedding(max_position, self.embedding_size)
        positions = offset + torch.arange(seq_len)
        pos_embeddings = self.weights.index_select(0, positions).unsqueeze(0).expand(batch_size, -1, -1).detach()
        return pos_embeddings


class SelfAttentionMask(nn.Module):

    def __init__(self, init_size=100):
        super(SelfAttentionMask, self).__init__()
        self.weights = SelfAttentionMask.get_mask(init_size)

    @staticmethod
    def get_mask(size):
        weights = torch.ones((size, size), dtype=torch.uint8).triu_(1)
        return weights

    def forward(self, size):
        if self.weights is None or size > self.weights.size(0):
            self.weights = SelfAttentionMask.get_mask(size)
        masks = self.weights[:size, :size].detach()
        return masks


class AveragePooling(nn.Module):
    """
    Layer class for computing mean of a sequence
    """

    def __init__(self, dim: int=-1, keepdim: bool=False):
        """
        Layer class for computing mean of a sequence

        :param int dim: Dimension to be averaged. -1 by default.
        :param bool keepdim: True if you want to keep averaged dimensions. False by default.
        """
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, tensor: torch.Tensor):
        """
        Do average pooling over a sequence

        Args:
            tensor (torch.Tensor): FloatTensor to be averaged.
        
        Returns:
            torch.FloatTensor: Averaged result.
        """
        return tensor.mean(dim=self.dim, keepdim=self.keepdim)

    def extra_repr(self):
        return 'dim={dim}, keepdim={keepdim}'.format(**self.__dict__)


def apply_across_dim(function, dim=1, shared_keys=None, **tensors):
    """
    Apply a function repeatedly for each tensor slice through the given dimension.
    For example, we have tensor [batch_size, X, input_sequence_length] and dim = 1, then we will concatenate the following matrices on dim=1.
    - function([:, 0, :])
    - function([:, 1, :])
    - ...
    - function([:, X-1, :]).

    Args:
        function (function): Function to apply.
        dim (int): Dimension through which we'll apply function. (1 by default)
        shared_keys (set): Set of keys representing tensors to be shared. (None by default)
        tensors (torch.Tensor): Keyword arguments of tensors to compute. Dimension should >= `dim`.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary of tensors, whose keys are corresponding to the output of the function.
    """
    shared_arguments = {}
    repeat_targets = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or shared_keys and key in shared_keys:
            shared_arguments[key] = tensor
        else:
            repeat_targets[key] = tensor
    size = {key: tensor.shape[dim] for key, tensor in repeat_targets.items()}
    assert len(set(size.values())) == 1, 'Tensors does not have same size on dimension %s: We found %s' % (dim, size)
    size = list(size.values())[0]
    output = {}
    for i in range(size):
        kwargs = {key: tensor.select(dim=dim, index=i).contiguous() for key, tensor in repeat_targets.items()}
        kwargs.update(shared_arguments)
        for key, tensor in function(**kwargs).items():
            if key in shared_keys:
                continue
            if key not in output:
                output[key] = []
            output[key].append(tensor.unsqueeze(dim=dim))
    assert all(len(t) == size for t in output.values())
    return {key: torch.cat(tensor, dim=dim).contiguous() for key, tensor in output.items()}


class DecoderModel(nn.Module):
    """
    Base model for equation generation/classification (Abstract class)
    """

    def __init__(self, config):
        """
        Initiate Equation Builder instance

        :param ModelConfig config: Configuration of this model
        """
        super().__init__()
        self.config = config
        self.embedding_dim = 128
        self.hidden_dim = 768
        self.intermediate_dim = 3072
        self.num_decoder_layers = self.config['num_decoder_layers']
        self.layernorm_eps = 1e-12
        self.num_decoder_heads = 12
        self.num_pointer_heads = self.config['num_pointer_heads']
        self.num_hidden_layers = 6
        self.max_arity = 2
        self.training = True

    def init_factor(self):
        """
        Returns:
            float: Standard deviation of normal distribution that will be used for initializing weights.
        """
        return 0.02

    @property
    def required_field(self) ->str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        raise NotImplementedError()

    @property
    def is_expression_type(self) ->bool:
        """
        :rtype: bool
        :return: True if this model requires Expression type sequence
        """
        return self.required_field in ['ptr', 'gen']

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights

        :param nn.Module module: Module to be initialized.
        """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.MultiheadAttention)):
            for name, param in module.named_parameters():
                if param is None:
                    continue
                if 'weight' in name:
                    param.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in name:
                    param.data.zero_()
                else:
                    raise NotImplementedError('This case is not considered!')
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _forward_single(self, **kwargs) ->Dict[str, torch.Tensor]:
        """
        Forward computation of a single beam

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of computed values
        """
        raise NotImplementedError()

    def _build_target_dict(self, **kwargs) ->Dict[str, torch.Tensor]:
        """
        Build dictionary of target matrices.

        :rtype: Dict[str, torch.Tensor]
        :return: Dictionary of target values
        """
        raise NotImplementedError()

    def forward(self, text: torch.Tensor=None, text_pad: torch.Tensor=None, text_num: torch.Tensor=None, text_numpad: torch.Tensor=None, equation: torch.Tensor=None, beam: int=1, max_len: int=128, function_arities: Dict[int, int]=None):
        """
        Forward computation of decoder model

        Returns:
            Dict[str, torch.Tensor]: Dictionary of tensors.
                If this model is currently on training phase, values will be accuracy or loss tensors
                Otherwise, values will be tensors representing predicted distribution of output
        """
        if equation is not None:
            output = self._forward_single(text, text_pad, text_num, text_numpad, equation)
            with torch.no_grad():
                targets = self._build_target_dict(equation, text_numpad)
            return output, targets
        else:
            self.training = False
            if 'expr' in self.config['decoder']:
                batch_sz = text.shape[0]
                batch_range = range(batch_sz)
                device = text.device
                arity = self.max_arity
                if 'gen' in self.config['decoder']:
                    num_range = lambda n: 1 <= n < 1 + EPT.NUM_MAX
                    con_range = lambda n: n == 0 or 1 + EPT.NUM_MAX + EPT.MEM_MAX <= n
                    num_offset = mem_offset = con_offset = 0
                else:
                    con_offset = 0
                    num_offset = self.constant_vocab_size
                    mem_offset = num_offset + text_num.shape[1]
                    con_range = lambda n: n < num_offset
                    num_range = lambda n: num_offset <= n < mem_offset
                function_arities = {} if self.function_arities is None else self.function_arities
                init = [EPT.FUN_NEW_EQN_ID] + [EPT.PAD_ID] * (2 * arity)
                result = torch.tensor([[[init]] for _ in batch_range], dtype=torch.long)
                beamscores = torch.zeros(batch_sz, 1)
                all_exit = False
                seq_len = 1
                while seq_len < max_len and not all_exit:
                    kwargs = {'text': text, 'text_pad': text_pad, 'text_num': text_num, 'text_numpad': text_numpad, 'equation': result}
                    scores = apply_across_dim(self._forward_single, dim=1, shared_keys={'text', 'text_pad', 'text_num', 'text_numpad'}, **kwargs)
                    scores = {key: score[:, :, -1].cpu().detach() for key, score in scores.items()}
                    beam_function_score = scores['operator'] + beamscores.unsqueeze(-1)
                    next_beamscores = torch.zeros(batch_sz, beam)
                    next_result = torch.full((batch_sz, beam, seq_len + 1, 1 + 2 * arity), fill_value=-1, dtype=torch.long)
                    beam_range = range(beam_function_score.shape[1])
                    operator_range = range(beam_function_score.shape[2])
                    for i in batch_range:
                        score_i = []
                        for m in beam_range:
                            last_item = result[i, m, -1, 0].item()
                            after_last = last_item in {EPT.PAD_ID, EPT.FUN_END_EQN_ID}
                            if after_last:
                                score_i.append((beamscores[i, m].item(), m, EPT.PAD_ID, []))
                                continue
                            operator_scores = {}
                            for f in operator_range:
                                operator_score = beam_function_score[i, m, f].item()
                                if f < len(EPT.FUN_TOKENS):
                                    if f == EPT.FUN_END_EQN_ID and last_item == EPT.FUN_NEW_EQN_ID:
                                        continue
                                    score_i.append((operator_score, m, f, []))
                                else:
                                    operator_scores[f] = operator_score
                            operand_beams = [(0.0, [])]
                            for a in range(arity):
                                score_ia, index_ia = scores['operand_%s' % a][i, m].topk(beam)
                                score_ia = score_ia.tolist()
                                index_ia = index_ia.tolist()
                                operand_beams = [(s_prev + s_a, arg_prev + [arg_a]) for s_prev, arg_prev in operand_beams for s_a, arg_a in zip(score_ia, index_ia)]
                                operand_beams = sorted(operand_beams, key=lambda t: t[0], reverse=True)[:beam]
                                for f, s_f in operator_scores.items():
                                    if function_arities.get(f, arity) == a + 1:
                                        score_i += [(s_f + s_args, m, f, args) for s_args, args in operand_beams]
                        beam_registered = set()
                        for score, prevbeam, operator, operands in sorted(score_i, key=lambda t: t[0], reverse=True):
                            if len(beam_registered) == beam:
                                break
                            beam_signature = prevbeam, operator, *operands
                            if beam_signature in beam_registered:
                                continue
                            newbeam = len(beam_registered)
                            next_beamscores[i, newbeam] = score
                            next_result[i, newbeam, :-1] = result[i, prevbeam]
                            new_tokens = [operator]
                            for j, a in enumerate(operands):
                                if con_range(a):
                                    new_tokens += [EPT.ARG_CON_ID, a - con_offset]
                                elif num_range(a):
                                    new_tokens += [EPT.ARG_NUM_ID, a - num_offset]
                                else:
                                    new_tokens += [EPT.ARG_MEM_ID, a - mem_offset]
                            new_tokens = torch.as_tensor(new_tokens, dtype=torch.long, device=device)
                            next_result[i, newbeam, -1, :new_tokens.shape[0]] = new_tokens
                            beam_registered.add(beam_signature)
                    beamscores = next_beamscores
                    last_tokens = next_result[:, :, -1, 0]
                    all_exit = ((last_tokens == EPT.PAD_ID) | (last_tokens == EPT.FUN_END_EQN_ID)).all().item()
                    result = next_result
                    seq_len += 1
            else:
                batch_sz = text.shape[0]
                batch_range = range(batch_sz)
                device = text.device
                result = torch.tensor([[[EPT.SEQ_NEW_EQN_ID]] for _ in batch_range], dtype=torch.long)
                beamscores = torch.zeros(batch_sz, 1)
                all_exit = False
                seq_len = 1
                while seq_len < max_len and not all_exit:
                    scores = self._forward_single(text, text_pad, text_num, text_numpad, equation=result)
                    scores = scores['op'][:, :, -1].cpu().detach()
                    beam_token_score = scores + beamscores.unsqueeze(-1)
                    next_beamscores = torch.zeros(batch_sz, beam)
                    next_result = torch.full((batch_sz, beam, seq_len + 1), fill_value=EPT.PAD_ID, dtype=torch.long)
                    beam_range = range(beam_token_score.shape[1])
                    token_range = range(beam_token_score.shape[2])
                    for i in batch_range:
                        score_i = []
                        for m in beam_range:
                            last_item = result[i, m, -1].item()
                            after_last = last_item == EPT.PAD_ID or last_item == EPT.SEQ_END_EQN_ID
                            if after_last:
                                score_i.append((beamscores[i, m].item(), m, EPT.PAD_ID))
                                continue
                            for v in token_range:
                                if v == EPT.SEQ_END_EQN_ID and last_item == EPT.SEQ_NEW_EQN_ID:
                                    continue
                                token_score = beam_token_score[i, m, v].item()
                                score_i.append((token_score, m, v))
                        beam_registered = set()
                        for score, prevbeam, token in sorted(score_i, key=lambda t: t[0], reverse=True):
                            if len(beam_registered) == beam:
                                break
                            if (prevbeam, token, token) in beam_registered:
                                continue
                            newbeam = len(beam_registered)
                            next_beamscores[i, newbeam] = score
                            next_result[i, newbeam, :-1] = result[i, prevbeam]
                            next_result[i, newbeam, -1] = token
                            beam_registered.add((prevbeam, token, token))
                    beamscores = next_beamscores
                    last_token_ids = next_result[:, :, -1]
                    all_exit = ((last_token_ids == EPT.PAD_ID) | (last_token_ids == EPT.SEQ_END_EQN_ID)).all().item()
                    result = next_result
                    seq_len += 1
            return result, None


class EPTPositionalEncoding(nn.Module):
    """
    Positional encoding that extends trigonometric embedding proposed in 'Attention is all you need'
    """

    def __init__(self, embedding_dim):
        """
        Instantiate positional encoding instance.

        :param int embedding_dim:
            Dimension of embedding vector
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        div_term = torch.arange(0, embedding_dim) // 2 * 2
        div_term = torch.exp(div_term.float() * (-math.log(10000.0) / embedding_dim))
        multiplier = torch.zeros(2, embedding_dim, dtype=torch.float)
        multiplier[0, 1::2] = 1.0
        multiplier[1, 0::2] = 1.0
        self.register_buffer('_div_term', div_term)
        self.register_buffer('multiplier', multiplier)

    @property
    def device(self) ->torch.device:
        """
        Get the device where weights are currently put.
        :rtype: torch.device
        :return: Device instance
        """
        return self._div_term.device

    def before_trigonometric(self, indices: torch.Tensor) ->torch.Tensor:
        """
        Compute a_p * t + b_p for each index t.
        :param torch.Tensor indices: A Long tensor to compute indices.
        :rtype: torch.Tensor
        :return: Tensor whose values are a_p * t + b_p for each (t, p) entry.
        """
        indices = indices.float()
        return indices * self._div_term

    def forward(self, index_or_range, ignored_index=-1) ->torch.Tensor:
        """
        Compute positional encoding. If this encoding is not learnable, the result cannot have any gradient vector.

        .. math::
            P_{t, p} = c_p * \\cos(a_p * t + b_p) + d_p * \\sin(a_p * t + b_p).

        :param Union[torch.Tensor,int,range] index_or_range:
            Value that represents positional encodings to be built.
            - A Tensor value indicates indices itself.
            - A integer value indicates indices from 0 to the value
            - A range value indicates indices within the range.
        :param int ignored_index: The index to be ignored. `PAD_ID` by default.
        :rtype: torch.Tensor
        :return:
            Positional encoding of given value.
            - If torch.Tensor of shape [*, L] is given, this will have shape [*, L, E] if L is not 1, otherwise [*, E].
            - If integer or range is given, this will have shape [T, E], where T is the length of range.
        """
        with torch.no_grad():
            return self._forward(index_or_range, ignored_index)

    def _forward(self, index_or_range, ignored_index=-1) ->torch.Tensor:
        """
        Compute positional encoding

        .. math::
            P_{t, p} = c_p * \\cos(a_p * t + b_p) + d_p * \\sin(a_p * t + b_p).

        :param Union[torch.Tensor,int,range] index_or_range:
            Value that represents positional encodings to be built.
            - A Tensor value indicates indices itself.
            - A integer value indicates indices from 0 to the value
            - A range value indicates indices within the range.
        :param int ignored_index: The index to be ignored. `PAD_ID` by default.
        :rtype: torch.Tensor
        :return:
            Positional encoding of given value.
            - If torch.Tensor of shape [*, L] is given, this will have shape [*, L, E] if L is not 1, otherwise [*, E].
            - If integer or range is given, this will have shape [T, E], where T is the length of range.
        """
        if type(index_or_range) is int:
            indices = torch.arange(0, index_or_range)
        elif type(index_or_range) is range:
            indices = torch.as_tensor(list(index_or_range))
        else:
            indices = index_or_range
        indices = indices.unsqueeze(-1)
        indices = indices
        phase = self.before_trigonometric(indices)
        cos_value = phase.cos()
        sin_value = phase.sin()
        cos_multiplier = self.multiplier[0]
        sin_multiplier = self.multiplier[1]
        result_shape = [1] * (phase.dim() - 1) + [-1]
        cos_multiplier = cos_multiplier.view(*result_shape)
        sin_multiplier = sin_multiplier.view(*result_shape)
        result = cos_value * cos_multiplier + sin_value * sin_multiplier
        ignored_indices = indices == ignored_index
        if ignored_indices.any():
            result.masked_fill_(ignored_indices, 0.0)
        return result.contiguous()


class EPTTransformerLayer(nn.Module):
    """
    Class for Transformer Encoder/Decoder layer (follows the paper, 'Attention is all you need')
    """

    def __init__(self, hidden_dim=None, num_decoder_heads=None, layernorm_eps=None, intermediate_dim=None):
        """
        Initialize TransformerLayer class

        :param ModelConfig config: Configuration of this Encoder/Decoder layer
        """
        super().__init__()
        self.attn = EPTMultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_decoder_heads, layernorm_eps=layernorm_eps, dropout=0.0)
        self.mem = EPTMultiHeadAttention(hidden_dim=hidden_dim, num_heads=num_decoder_heads, layernorm_eps=layernorm_eps, dropout=0.0)
        self.dropout_attn = nn.Dropout(0.0)
        self.dropout_mem = nn.Dropout(0.0)
        self.dropout_expand = nn.Dropout(0.0)
        self.dropout_out = nn.Dropout(0.0)
        self.lin_expand = nn.Linear(hidden_dim, intermediate_dim)
        self.lin_collapse = nn.Linear(intermediate_dim, hidden_dim)
        self.norm_attn = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        self.norm_mem = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        self.norm_out = nn.LayerNorm(hidden_dim, eps=layernorm_eps)

    def forward(self, target, target_ignorance_mask=None, target_attention_mask=None, memory=None, memory_ignorance_mask=None):
        """
        Forward-computation of Transformer Encoder/Decoder layers

        :param torch.Tensor target:
            FloatTensor indicating Sequence of target vectors. Shape [B, T, H]
            where B = batch size, T = length of target sequence, H = vector dimension of hidden state
        :param torch.Tensor target_ignorance_mask:
            BoolTensor indicating Mask for target tokens that should be ignored. Shape [B, T].
        :param torch.Tensor target_attention_mask:
            BoolTensor indicating Target-to-target Attention mask for target tokens. Shape [T, T].
        :param torch.Tensor memory:
            FloatTensor indicating Sequence of source vectors. Shape [B, S, H]
            where S = length of source sequence
            This can be None when you want to use this layer as an encoder layer.
        :param torch.Tensor memory_ignorance_mask:
            BoolTensor indicating Mask for source tokens that should be ignored. Shape [B, S].
        :rtype: torch.FloatTensor
        :return: Decoder hidden states per each target token, shape [B, S, H].
        """
        attented = self.attn(query=target, attention_mask=target_attention_mask, key_ignorance_mask=target_ignorance_mask)
        target = target + self.dropout_attn(attented)
        target = self.norm_attn(target)
        if memory is not None:
            attented = self.mem(query=target, key_value=memory, key_ignorance_mask=memory_ignorance_mask)
            target = target + self.dropout_mem(attented)
            target = self.norm_mem(target)
        output = self.lin_collapse(self.dropout_expand(gelu_bert(self.lin_expand(target))))
        target = target + self.dropout_out(output)
        target = self.norm_out(target)
        return target


class LogSoftmax(nn.LogSoftmax):
    """
    LogSoftmax layer that can handle infinity values.
    """

    def forward(self, tensor: torch.Tensor):
        """
        Compute log(softmax(tensor))

        Args:
            tensor torch.Tensor: FloatTensor whose log-softmax value will be computed
        
        Returns:
            torch.FloatTensor: LogSoftmax result.
        """
        max_t = tensor.max(dim=self.dim, keepdim=True).values
        tensor = tensor - max_t.masked_fill(~torch.isfinite(max_t), 0.0)
        all_inf_mask = torch.isinf(tensor).all(dim=self.dim, keepdim=True)
        if all_inf_mask.any().item():
            tensor = tensor.masked_fill(all_inf_mask, 0.0)
        return super().forward(tensor)


def get_embedding_without_pad(embedding: Union[nn.Embedding, torch.Tensor], tokens: torch.Tensor, ignore_index=-1):
    """
    Get embedding vectors of given token tensor with ignored indices are zero-filled.

    Args:
        embedding (nn.Embedding): An embedding instance
        tokens (torch.Tensor): A Long Tensor to build embedding vectors.
        ignore_index (int): Index to be ignored. `PAD_ID` by default.
    
    Returns:
        torch.Tensor: Embedding vector of given token tensor.
    """
    tokens = tokens.clone()
    ignore_positions = tokens == ignore_index
    if ignore_positions.any():
        tokens.masked_fill_(ignore_positions, 0)
    if isinstance(embedding, nn.Embedding):
        embedding = embedding(tokens)
    else:
        embedding = F.embedding(tokens, embedding)
    if ignore_positions.any():
        embedding.masked_fill_(ignore_positions.unsqueeze(-1), 0.0)
    return embedding.contiguous()


def mask_forward(sz: int, diagonal: int=1):
    """
    Generate a mask that ignores future words. Each (i, j)-entry will be True if j >= i + diagonal

    Args:
        sz (int): Length of the sequence.
        diagonal (int): Amount of shift for diagonal entries.
    
    Returns: 
        torch.Tensor: Mask tensor with shape [sz, sz].
    """
    return torch.ones(sz, sz, dtype=torch.bool, requires_grad=False).triu(diagonal=diagonal).contiguous()


class ExpressionDecoderModel(DecoderModel):
    """
    Decoding model that generates expression sequences (Abstract class)
    """

    def __init__(self, config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym):
        super().__init__(config)
        self.operator_vocab_size = len(out_idx2opsym)
        None
        self.operand_vocab_size = len(out_idx2consym)
        self.constant_vocab_size = len(out_idx2consym)
        self.max_arity = max([op['arity'] for op in EPT.OPERATORS.values()], default=2)
        self.function_arities = {i: EPT.OPERATORS[f]['arity'] for i, f in enumerate(out_idx2opsym) if i >= len(EPT.FUN_TOKENS)}
        """ Embedding layers """
        self.operator_word_embedding = nn.Embedding(self.operator_vocab_size, self.hidden_dim)
        self.operator_pos_embedding = EPTPositionalEncoding(self.hidden_dim)
        self.operand_source_embedding = nn.Embedding(3, self.hidden_dim)
        """ Scalar parameters """
        degrade_factor = self.embedding_dim ** 0.5
        self.operator_pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)
        self.operand_source_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)
        """ Layer Normalizations """
        self.operator_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)
        self.operand_norm = nn.LayerNorm(self.hidden_dim, eps=self.layernorm_eps)
        """ Linear Transformation """
        self.embed_to_hidden = nn.Linear(self.hidden_dim * (self.max_arity + 1), self.hidden_dim)
        """ Transformer layer """
        self.shared_decoder_layer = EPTTransformerLayer(hidden_dim=self.hidden_dim, num_decoder_heads=self.num_decoder_heads, layernorm_eps=self.layernorm_eps, intermediate_dim=self.intermediate_dim)
        """ Output layer """
        self.operator_out = nn.Linear(self.hidden_dim, self.operator_vocab_size)
        self.softmax = LogSoftmax(dim=-1)

    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor) ->torch.Tensor:
        """
        Build operand embedding a_ij in the paper.

        :param torch.Tensor ids:
            LongTensor containing index-type information of operands. (This corresponds to a_ij in the paper)
        :param torch.Tensor mem_pos:
            FloatTensor containing positional encoding used so far. (i.e. PE(.) in the paper)
        :param torch.Tensor nums:
            FloatTensor containing encoder's hidden states corresponding to numbers in the text.
            (i.e. e_{a_ij} in the paper)
        :rtype: torch.Tensor
        :return: A FloatTensor representing operand embedding vector a_ij in Equation 3, 4, 5
        """
        raise NotImplementedError()

    def _build_decoder_input(self, ids: torch.Tensor, nums: torch.Tensor):
        """
        Compute input of the decoder

        Args:
            ids (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length, 1+2*arity_size]
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, hidden_size].
        
        Returns:
            torch.Tensor: A FloatTensor representing input vector. Shape [batch_size, equation_length, hidden_size].
        """
        operator = get_embedding_without_pad(self.operator_word_embedding, ids.select(dim=-1, index=0))
        operator_pos = self.operator_pos_embedding(ids.shape[1])
        operator = self.operator_norm(operator * self.operator_pos_factor + operator_pos.unsqueeze(0)).unsqueeze(2)
        operand = get_embedding_without_pad(self.operand_source_embedding, ids[:, :, 1::2]) * self.operand_source_factor
        operand += self._build_operand_embed(ids, operator_pos, nums)
        operand = self.operand_norm(operand)
        operator_operands = torch.cat([operator, operand], dim=2).contiguous().flatten(start_dim=2)
        return self.embed_to_hidden(operator_operands)

    def _build_decoder_context(self, embedding: torch.Tensor, embedding_pad: torch.Tensor=None, text: torch.Tensor=None, text_pad: torch.Tensor=None):
        """
        Compute decoder's hidden state vectors

        Args:
            embedding (torch.Tensor): FloatTensor containing input vectors. Shape [batch_size, equation_length, hidden_size],
            embedding_pad (torch.Tensor):BoolTensor, whose values are True if corresponding position is PAD in the decoding sequence, Shape [batch_size, equation_length]
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, hidden_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
        
        Returns: 
            torch.Tensor: A FloatTensor of shape [batch_size, equation_length, hidden_size], which contains decoder's hidden states.
        """
        mask = mask_forward(embedding.shape[1])
        output = embedding
        for _ in range(self.num_hidden_layers):
            output = self.shared_decoder_layer(target=output, memory=text, target_attention_mask=mask, target_ignorance_mask=embedding_pad, memory_ignorance_mask=text_pad)
        return output

    def _forward_single(self, text: torch.Tensor=None, text_pad: torch.Tensor=None, text_num: torch.Tensor=None, text_numpad: torch.Tensor=None, equation: torch.Tensor=None):
        """
        Forward computation of a single beam

        Args:
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, hidden_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, hidden_size].
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands.
                Shape: [batch_size, equation_length, 1+2*arity_size].
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of followings
                'operator': Log probability of next operators. FloatTensor with shape [batch_size, equation_length, operator_size].
                '_out': Decoder's hidden states. FloatTensor with shape [batch_size, equation_length, hidden_size].
                '_not_usable': Indicating positions that corresponding output values are not usable in the operands. BoolTensor with Shape [batch_size, equation_length].
        """
        operator_ids = equation.select(dim=2, index=0)
        output = self._build_decoder_input(ids=equation, nums=text_num)
        output_pad = operator_ids == EPT.PAD_ID
        output_not_usable = output_pad.clone()
        output_not_usable[:, :-1].masked_fill_(operator_ids[:, 1:] == EPT.FUN_EQ_SGN_ID, True)
        output = self._build_decoder_context(embedding=output, embedding_pad=output_pad, text=text, text_pad=text_pad)
        operator_out = self.operator_out(output)
        if not self.training:
            operator_out[:, :, EPT.FUN_NEW_EQN_ID] = EPT.NEG_INF
            operator_out[:, :, EPT.FUN_END_EQN_ID].masked_fill_(operator_ids != EPT.FUN_EQ_SGN_ID, EPT.NEG_INF)
        result = {'operator': self.softmax(operator_out), '_out': output, '_not_usable': output_not_usable}
        return result


class Squeeze(nn.Module):
    """
    Layer class for squeezing a dimension
    """

    def __init__(self, dim: int=-1):
        """
        Layer class for squeezing a dimension

        :param int dim: Dimension to be squeezed, -1 by default.
        """
        super().__init__()
        self.dim = dim

    def forward(self, tensor: torch.Tensor):
        """
        Do squeezing

        Args:
            tensor (torch.Tensor): FloatTensor to be squeezed.
        
        Returns: 
            torch.FloatTensor: Squeezed result.
        """
        return tensor.squeeze(dim=self.dim)

    def extra_repr(self):
        return 'dim={dim}'.format(**self.__dict__)


def apply_module_dict(modules: nn.ModuleDict, encoded: torch.Tensor, **kwargs):
    """
    Predict next entry using given module and equation.

    Args:
        modules (nn.ModuleDict): Dictionary of modules to be applied. Modules will be applied with ascending order of keys.
            We expect three types of modules: nn.Linear, nn.LayerNorm and MultiheadAttention.
        
        encoded (torch.Tensor): Float Tensor that represents encoded vectors. Shape [batch_size, equation_length, hidden_size].
        key_value (torch.Tensor): Float Tensor that represents key and value vectors when computing attention. Shape [batch_size, key_size, hidden_size].

        key_ignorance_mask (torch.Tensor):Bool Tensor whose True values at (b, k) make attention layer ignore k-th key on b-th item in the batch. Shape [batch_size, key_size].
        
        attention_mask (torch.BoolTensor): Bool Tensor whose True values at (t, k) make attention layer ignore k-th key when computing t-th query. Shape [equation_length, key_size].
    
    Returns:
        torch.Tensor: Float Tensor that indicates the scores under given information. Shape will be [batch_size, equation_length, ?]
    """
    output = encoded
    keys = sorted(modules.keys())
    for key in keys:
        layer = modules[key]
        if isinstance(layer, (EPTMultiHeadAttention, EPTMultiHeadAttentionWeights)):
            output = layer(query=output, **kwargs)
        else:
            output = layer(output)
    return output


class ExpressionPointerTransformer(ExpressionDecoderModel):
    """
    The EPT model
    """

    def __init__(self, config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym):
        super().__init__(config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym)
        """ Operand embedding """
        self.constant_word_embedding = nn.Embedding(self.constant_vocab_size, self.hidden_dim)
        """ Output layer """
        self.operand_out = nn.ModuleList([nn.ModuleDict({'0_attn': EPTMultiHeadAttentionWeights(hidden_dim=self.hidden_dim, num_heads=self.num_pointer_heads), '1_mean': Squeeze(dim=-1) if self.num_pointer_heads == 1 else AveragePooling(dim=-1)}) for _ in range(self.max_arity)])
        """ Initialize weights """
        with torch.no_grad():
            self.apply(self._init_weights)

    @property
    def required_field(self) ->str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return 'ptr'

    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor):
        """
        Build operand embedding.

        Args: 
            ids (torch.Tensor): LongTensor containing source-content information of operands. Shape [batch_size, equation_length, 1+2*arity_size].
            mem_pos (torch.Tensor): FloatTensor containing positional encoding used so far. Shape [batch_size, equation_length, hidden_size].
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
        
        Returns: 
            torch.Tensor: A FloatTensor representing operand embedding vector. Shape [batch_size, equation_length, arity_size, hidden_size]
        """
        operand_source = ids[:, :, 1::2]
        operand_value = ids[:, :, 2::2]
        number_operand = operand_value.masked_fill(operand_source != EPT.ARG_NUM_ID, EPT.PAD_ID)
        operand = torch.stack([get_embedding_without_pad(nums[b], number_operand[b]) for b in range(ids.shape[0])], dim=0).contiguous()
        operand += get_embedding_without_pad(self.constant_word_embedding, operand_value.masked_fill(operand_source != EPT.ARG_CON_ID, EPT.PAD_ID))
        prior_result_operand = operand_value.masked_fill(operand_source != EPT.ARG_MEM_ID, EPT.PAD_ID)
        operand += get_embedding_without_pad(mem_pos, prior_result_operand)
        return operand

    def _build_attention_keys(self, num: torch.Tensor, mem: torch.Tensor, num_pad: torch.Tensor=None, mem_pad: torch.Tensor=None):
        """
        Generate Attention Keys by concatenating all items.

        Args: 
            num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
            mem (torch.Tensor): FloatTensor containing decoder's hidden states corresponding to prior expression outputs. Shape [batch_size, equation_length, hidden_size].
            num_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the number sequence. Shape [batch_size, num_size]
            mem_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the target expression sequence. Shape [batch_size, equation_length]
        
        Returns: 
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Triple of Tensors
                - [0] Keys (A_ij in the paper). Shape [batch_size, constant_size+num_size+equation_length, hidden_size], where C = size of constant vocabulary.
                - [1] Mask for positions that should be ignored in keys. Shape [batch_size, C+num_size+equation_length]
                - [2] Forward Attention Mask to ignore future tokens in the expression sequence. Shape [equation_length, C+num_size+equation_length]
        """
        batch_sz = num.shape[0]
        const_sz = self.constant_vocab_size
        const_num_sz = const_sz + num.shape[1]
        const_key = self.constant_word_embedding.weight.unsqueeze(0).expand(batch_sz, const_sz, self.hidden_dim)
        key = torch.cat([const_key, num, mem], dim=1).contiguous()
        key_ignorance_mask = torch.zeros(key.shape[:2], dtype=torch.bool, device=key.device)
        if num_pad is not None:
            key_ignorance_mask[:, const_sz:const_num_sz] = num_pad
        if mem_pad is not None:
            key_ignorance_mask[:, const_num_sz:] = mem_pad
        attention_mask = torch.zeros(mem.shape[1], key.shape[1], dtype=torch.bool, device=key.device)
        attention_mask[:, const_num_sz:] = mask_forward(mem.shape[1], diagonal=0)
        return key, key_ignorance_mask, attention_mask

    def _forward_single(self, text: torch.Tensor=None, text_pad: torch.Tensor=None, text_num: torch.Tensor=None, text_numpad: torch.Tensor=None, equation: torch.Tensor=None):
        """
        Forward computation of a single beam

        Args: 
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, hidden_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, hidden_size].
            text_numpad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the number sequence. Shape [batch_size, num_size]
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length, 1+2*arity_size].
        
        Returns: 
            Dict[str, torch.Tensor]: Dictionary of followings
                'operator': Log probability of next operators.FloatTensor with shape [batch_size, equation_length, operator_size].
                'operand_J': Log probability of next J-th operands. FloatTensor with shape [batch_size, equation_length, operand_size].
        """
        result = super()._forward_single(text, text_pad, text_num, text_numpad, equation)
        output = result.pop('_out')
        output_not_usable = result.pop('_not_usable')
        key, key_ign_msk, attn_msk = self._build_attention_keys(num=text_num, mem=output, num_pad=text_numpad, mem_pad=output_not_usable)
        for j, layer in enumerate(self.operand_out):
            score = apply_module_dict(layer, encoded=output, key=key, key_ignorance_mask=key_ign_msk, attention_mask=attn_msk)
            result['operand_%s' % j] = self.softmax(score)
        return result

    def _build_target_dict(self, equation, num_pad=None):
        """
        Build dictionary of target matrices.

        Returns: 
            Dict[str, torch.Tensor]: Dictionary of target values
                'operator': Index of next operators. LongTensor with shape [batch_size, equation_length].
                'operand_J': Index of next J-th operands. LongTensor with shape [batch_size, equation_length].
        """
        num_offset = self.constant_vocab_size
        mem_offset = num_offset + num_pad.shape[1]
        targets = {'operator': equation.select(dim=-1, index=0)}
        for i in range(self.max_arity):
            operand_source = equation[:, :, i * 2 + 1]
            operand_value = equation[:, :, i * 2 + 2].clamp_min(0)
            operand_value += operand_source.masked_fill(operand_source == EPT.ARG_NUM_ID, num_offset).masked_fill_(operand_source == EPT.ARG_MEM_ID, mem_offset)
            targets['operand_%s' % i] = operand_value
        return targets


class ExpressionTransformer(ExpressionDecoderModel):
    """
    Vanilla Transformer + Expression (The second ablated model)
    """

    def __init__(self, config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym):
        super().__init__(config, out_opsym2idx, out_idx2opsym, out_consym2idx, out_idx2consym)
        """ Operand embedding """
        self.operand_word_embedding = nn.Embedding(self.operand_vocab_size, self.hidden_dim)
        """ Output layer """
        self.operand_out = nn.ModuleList([nn.ModuleDict({'0_out': nn.Linear(self.hidden_dim, self.operand_vocab_size)}) for _ in range(self.max_arity)])
        """ Initialize weights """
        with torch.no_grad():
            self.apply(self._init_weights)

    @property
    def required_field(self) ->str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return 'gen'

    def _build_operand_embed(self, ids: torch.Tensor, mem_pos: torch.Tensor, nums: torch.Tensor):
        """
        Build operand embedding.

        Args:
            ids (torch.Tensor): LongTensor containing source-content information of operands. Shape [batch_size, equation_length, 1+2*arity_size].
            mem_pos (torch.Tensor): FloatTensor containing positional encoding used so far. Shape [batch_size, equation_length, hidden_size], where hidden_size = dimension of hidden state
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
        
        Returns: 
            torch.Tensor: A FloatTensor representing operand embedding vector. Shape [batch_size, equation_length, arity_size, hidden_size]
        """
        return get_embedding_without_pad(self.operand_word_embedding, ids[:, :, 2::2])

    def _forward_single(self, text: torch.Tensor=None, text_pad: torch.Tensor=None, text_num: torch.Tensor=None, text_numpad: torch.Tensor=None, equation: torch.Tensor=None):
        """
        Forward computation of a single beam

        Args:
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, hidden_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text.Shape: [batch_size, num_size, hidden_size].
            text_numpad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the number sequence. Shape [batch_size, num_size]
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length, 1+2*arity_size].
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary of followings
                'operator': Log probability of next operators. FloatTensor with shape [batch_size, equation_length, operator_size], where operator_size = size of operator vocabulary.
                'operand_J': Log probability of next J-th operands.FloatTensor with shape [batch_size, equation_length, operand_size].
        """
        result = super()._forward_single(text, text_pad, text_num, text_numpad, equation)
        output = result.pop('_out')
        output_not_usable = result.pop('_not_usable').unsqueeze(1)
        forward_mask = mask_forward(output.shape[1], diagonal=0).unsqueeze(0)
        num_begin = 1
        num_used = num_begin + min(text_num.shape[1], EPT.NUM_MAX)
        num_end = num_begin + EPT.NUM_MAX
        mem_used = num_end + min(output.shape[1], EPT.MEM_MAX)
        mem_end = num_end + EPT.MEM_MAX
        for j, layer in enumerate(self.operand_out):
            word_output = apply_module_dict(layer, encoded=output)
            if not self.training:
                word_output[:, :, num_begin:num_used].masked_fill_(text_numpad.unsqueeze(1), EPT.NEG_INF)
                word_output[:, :, num_used:num_end] = EPT.NEG_INF
                word_output[:, :, num_end:mem_used].masked_fill_(output_not_usable, EPT.NEG_INF)
                word_output[:, :, num_end:mem_used].masked_fill_(forward_mask, EPT.NEG_INF)
                word_output[:, :, mem_used:mem_end] = EPT.NEG_INF
            result['operand_%s' % j] = self.softmax(word_output)
        return result

    def _build_target_dict(self, equation, num_pad=None):
        """
        Build dictionary of target matrices.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of target values
                'operator': Index of next operators. LongTensor with shape [batch_size, equation_length].
                'operand_J': Index of next J-th operands. LongTensor with shape [batch_size, equation_length].
        """
        targets = {'operator': equation.select(dim=-1, index=0)}
        for j in range(2):
            targets['operand_%s' % j] = equation[:, :, j * 2 + 2]
        return targets


class SmoothCrossEntropyLoss(AbstractLoss):
    """
    Computes cross entropy loss with uniformly smoothed targets.
    """
    _NAME = 'SmoothCrossEntropyLoss'

    def __init__(self, weight=None, mask=None, size_average=True):
        """
        Cross entropy loss with uniformly smoothed targets.

        :param float smoothing: Label smoothing factor, between 0 and 1 (exclusive; default is 0.1)
        :param int ignore_index: Index to be ignored. (PAD_ID by default)
        :param str reduction: Style of reduction to be done. One of 'batchmean'(default), 'none', or 'sum'.
        """
        super(SmoothCrossEntropyLoss, self).__init__(self._NAME, SmoothedCrossEntropyLoss())
        self.norm_term = 1

    def get_loss(self):
        """return loss

        Returns:
            loss (float)
        """
        if isinstance(self.acc_loss, int):
            return 0
        loss = self.acc_loss.item()
        loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        """calculate loss

        Args:
            outputs (Tensor): output distribution of model.

            target (Tensor): target classes. 
        """
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term = 1


class OpDecoderModel(DecoderModel):
    """
    Decoding model that generates Op(Operator/Operand) sequences (Abstract class)
    """

    def __init__(self, config):
        super().__init__(config)
        """ Embedding look-up tables """
        self.word_embedding = nn.Embedding(config['op_vocab_size'], config['hidden_dim'])
        self.pos_embedding = EPTPositionalEncoding(config['hidden_dim'])
        self.word_hidden_norm = nn.LayerNorm(config['hidden_dim'], eps=self.layernorm_eps)
        degrade_factor = config['hidden_dim'] ** 0.5
        self.pos_factor = nn.Parameter(torch.tensor(degrade_factor), requires_grad=True)
        """ Decoding layer """
        self.shared_layer = EPTTransformerLayer(hidden_dim=self.hidden_dim, num_decoder_heads=self.num_decoder_heads, layernorm_eps=self.layernorm_eps, intermediate_dim=self.intermediate_dim)

    def _build_word_embed(self, ids: torch.Tensor, nums: torch.Tensor):
        """
        Build Op embedding

        Args:
            ids (torch.Tensor): LongTensor containing source-content information of operands. Shape [batch_size, equation_length].
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
        
        Returns: 
            torch.Tensor: A FloatTensor representing op embedding vector. Shape [batch_size, equation_length, hidden_size]
        """
        raise NotImplementedError()

    def _build_decoder_input(self, ids: torch.Tensor, nums: torch.Tensor):
        """
        Compute input of the decoder.

        Args:
            ids (torch.Tensor): LongTensor containing op tokens. Shape: [batch_size, equation_length]
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, hidden_size],
        
        Returns: 
            torch.Tensor: A FloatTensor representing input vector. Shape [batch_size, equation_length, hidden_size].
        """
        pos = self.pos_embedding(ids.shape[1])
        word = self._build_word_embed(ids, nums)
        return self.word_hidden_norm(word * self.pos_factor + pos.unsqueeze(0))

    def _build_decoder_context(self, embedding: torch.Tensor, embedding_pad: torch.Tensor=None, text: torch.Tensor=None, text_pad: torch.Tensor=None):
        """
        Compute decoder's hidden state vectors.

        Args: 
            embedding (torch.Tensor): FloatTensor containing input vectors. Shape [batch_size, decoding_sequence, input_embedding_size].
            embedding_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the decoding sequence. Shape [batch_size, decoding_sequence]
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, input_embedding_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
        
        Returns: 
        torch.Tensor: A FloatTensor of shape [batch_size, decoding_sequence, hidden_size], which contains decoder's hidden states.
        """
        mask = mask_forward(embedding.shape[1])
        output = embedding
        for _ in range(self.num_hidden_layers):
            output = self.shared_layer(target=output, memory=text, target_attention_mask=mask, target_ignorance_mask=embedding_pad, memory_ignorance_mask=text_pad)
        return output

    def _forward_single(self, text: torch.Tensor=None, text_pad: torch.Tensor=None, text_num: torch.Tensor=None, text_numpad: torch.Tensor=None, equation: torch.Tensor=None):
        """
        Forward computation of a single beam

        Args:
            text (torch.Tensor): FloatTensor containing encoder's hidden states e_i. Shape [batch_size, input_sequence_length, input_embedding_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, input_embedding_size].
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length, 1+2*arity_size].
        
        Returns: 
            Dict[str, torch.Tensor]: Dictionary of followings
                '_out': Decoder's hidden states. FloatTensor with shape [batch_size, equation_length, hidden_size].
        """
        output = self._build_decoder_input(ids=equation, nums=text_num.relu())
        output_pad = equation == EPT.PAD_ID
        output = self._build_decoder_context(embedding=output, embedding_pad=output_pad, text=text, text_pad=text_pad)
        result = {'_out': output}
        return result


class VanillaOpTransformer(OpDecoderModel):
    """
    The vanilla Transformer model
    """

    def __init__(self, config):
        super().__init__(config)
        """ Op token Generator """
        self.op_out = nn.Linear(config['hidden_dim'], config['op_vocab_size'])
        self.softmax = LogSoftmax(dim=-1)
        """ Initialize weights """
        with torch.no_grad():
            self.apply(self._init_weights)

    @property
    def required_field(self) ->str:
        """
        :rtype: str
        :return: Name of required field type to process
        """
        return 'vallina'

    def _build_word_embed(self, ids: torch.Tensor, nums: torch.Tensor):
        """
        Build Op embedding

        Args:
            ids (torch.Tensor): LongTensor containing source-content information of operands. Shape [batch_size, equation_length].
            nums (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape [batch_size, num_size, hidden_size].
        
        Returns: 
            torch.Tensor:A FloatTensor representing op embedding vector. Shape [batch_size, equation_length, hidden_size].
        """
        return get_embedding_without_pad(self.word_embedding, ids)

    def _forward_single(self, text: torch.Tensor=None, text_pad: torch.Tensor=None, text_num: torch.Tensor=None, text_numpad: torch.Tensor=None, equation: torch.Tensor=None):
        """
        Forward computation of a single beam

        Args:
            text (torch.Tensor): FloatTensor containing encoder's hidden states. Shape [batch_size, input_sequence_length, input_embedding_size].
            text_pad (torch.Tensor): BoolTensor, whose values are True if corresponding position is PAD in the input sequence. Shape [batch_size, input_sequence_length]
            text_num (torch.Tensor): FloatTensor containing encoder's hidden states corresponding to numbers in the text. Shape: [batch_size, num_size, input_embedding_size].
            equation (torch.Tensor): LongTensor containing index-type information of an operator and its operands. Shape: [batch_size, equation_length].
        
        Returns: 
            Dict[str, torch.Tensor]: Dictionary of followings
                'op': Log probability of next op tokens. FloatTensor with shape [batch_size, equation_length, operator_size].
        """
        result = super()._forward_single(text, text_pad, text_num, equation)
        output = result.pop('_out')
        op_out = self.op_out(output)
        result['op'] = self.softmax(op_out)
        return result

    def _build_target_dict(self, equation, num_pad=None):
        """
        Build dictionary of target matrices.

        Returns: 
            Dict[str, torch.Tensor]: Dictionary of target values
                'op': Index of next op tokens. LongTensor with shape [batch_size, equation_length].
        """
        return {'op': equation}


class EPT(nn.Module):
    """
    Reference:
        Kim et al. "Point to the Expression: Solving Algebraic Word Problems using the Expression-Pointer Transformer Model" in EMNLP 2020.
    """

    def __init__(self, config, dataset):
        super(EPT, self).__init__()
        self.device = config['device']
        self.max_output_len = config['max_output_len']
        self.share_vocab = config['share_vocab']
        self.decoding_strategy = config['decoding_strategy']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.task_type = config['task_type']
        try:
            self.in_pad_idx = dataset.in_word2idx['<pad>']
        except:
            self.in_pad_idx = None
        self.in_word2idx = dataset.in_word2idx
        self.in_idx2word = dataset.in_idx2word
        self.mode = config['decoder']
        if 'vall' in config['decoder']:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.decoder = VanillaOpTransformer(config, self.out_symbol2idx, self.out_idx2symbol)
        else:
            self.out_opsym2idx = dataset.out_opsym2idx
            self.out_idx2opsym = dataset.out_idx2opsymbol
            self.out_consym2idx = dataset.out_consym2idx
            self.out_idx2consym = dataset.out_idx2consymbol
            if 'gen' in config['decoder']:
                self.decoder = ExpressionTransformer(config, self.out_opsym2idx, self.out_idx2opsym, self.out_consym2idx, self.out_idx2consym)
            elif 'ptr' in config['decoder']:
                self.decoder = ExpressionPointerTransformer(config, self.out_opsym2idx, self.out_idx2opsym, self.out_consym2idx, self.out_idx2consym)
        pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config['transformers_pretrained_model']
        self.encoder = AutoModel.from_pretrained(pretrained_model_path)
        self.loss = SmoothCrossEntropyLoss()

    def forward(self, src, src_mask, num_pos, num_size, target=None, output_all_layers=False):
        """

        :param torch.Tensor src: input sequence.
        :param list src_mask: mask of input sequence.
        :param list num_pos: number position of input sequence.
        :param list num_size: number of numbers of input sequence.
        :param torch.Tensor target: target, default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return: token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        """
        encoder_output, encoder_layer_outputs = self.encoder_forward(src, src_mask, output_all_layers)
        max_numbers = max(num_size)
        if num_pos is not None:
            text_num, text_numpad = self.gather_vectors(encoder_output, num_pos, max_len=max_numbers)
        else:
            text_num = text_numpad = None
        token_logits, outputs, decoder_layer_outputs = self.decoder_forward(encoder_output, text_num, text_numpad, src_mask, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation','ques mask', 'num pos',
        'num size' and 'max numbers'.
        """
        src = torch.tensor(batch_data['question'])
        src_mask = torch.BoolTensor(batch_data['ques mask'])
        num_pos = batch_data['num pos']
        target = torch.tensor(batch_data['equation'])
        num_size = batch_data['num size']
        token_logits, _, all_layers = self.forward(src, src_mask, num_pos, num_size, target, output_all_layers=True)
        targets = all_layers['targets']
        self.loss.reset()
        for key, result in targets.items():
            predicted = token_logits[key].flatten(0, -2)
            result = self.shift_target(result)
            target = result.flatten()
            self.loss.eval_batch(predicted, target)
        self.loss.backward()
        batch_loss = self.loss.get_loss()
        return batch_loss

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.
        
        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'equation','ques mask', 'num pos',
        'num size'.
        """
        src = torch.tensor(batch_data['question'])
        src_mask = torch.BoolTensor(batch_data['ques mask'])
        num_pos = batch_data['num pos']
        num_size = batch_data['num size']
        _, symbol_outputs, _ = self.forward(src, src_mask, num_pos, num_size)
        all_outputs = self.convert_idx2symbol(symbol_outputs, batch_data['num list'])
        targets = self.convert_idx2symbol(batch_data['equation'], batch_data['num list'])
        return all_outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        raise NotImplementedError

    def encoder_forward(self, src, src_mask, output_all_layers=False):
        encoder_outputs = self.encoder(input_ids=src, attention_mask=(~src_mask).float())
        encoder_output = encoder_outputs[0]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_output
            all_layer_outputs['inputs_representation'] = encoder_output
        return encoder_output, all_layer_outputs

    def decoder_forward(self, encoder_output, text_num, text_numpad, src_mask, target=None, output_all_layers=False):
        if target is not None:
            token_logits, targets = self.decoder(text=encoder_output, text_num=text_num, text_numpad=text_numpad, text_pad=src_mask, equation=target)
            outputs = None
        else:
            max_len = self.max_output_len
            outputs, _ = self.decoder(text=encoder_output, text_num=text_num, text_numpad=text_numpad, text_pad=src_mask, beam=1, max_len=max_len)
            token_logits = None
            shape = list(outputs.shape)
            seq_len = shape[2]
            if seq_len < max_len:
                shape[2] = max_len
                tensor = torch.full(shape, fill_value=-1, dtype=torch.long)
                tensor[:, :, :seq_len] = outputs.cpu()
                outputs = tensor
            outputs.squeeze(1)
            targets = None
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['targets'] = targets
        return token_logits, outputs, all_layer_outputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def gather_vectors(self, hidden: torch.Tensor, mask: torch.Tensor, max_len: int=1):
        """
        Gather hidden states of indicated positions.

        :param torch.Tensor hidden:
            Float Tensor of hidden states.
            Shape [B, S, H], where B = batch size, S = length of sequence, and H = hidden dimension
        :param torch.Tensor mask:
            Long Tensor which indicates number indices that we're interested in. Shape [B, S].
        :param int max_len:
            Expected maximum length of vectors per batch. 1 by default.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        :return:
            Tuple of Tensors:
            - [0]:  Float Tensor of indicated hidden states.
                    Shape [B, N, H], where N = max(number of interested positions, max_len)
            - [1]:  Bool Tensor of padded positions.
                    Shape [B, N].
        """
        max_len = max(mask.max().item(), max_len)
        batch_size, seq_len, hidden_size = hidden.shape
        gathered = torch.zeros(batch_size, max_len, hidden_size, dtype=hidden.dtype, device=hidden.device)
        pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=hidden.device)
        for row in range(batch_size):
            for i in range(max_len):
                indices = (mask[row] == i).nonzero().view(-1).tolist()
                if len(indices) > 0:
                    begin = min(indices)
                    end = max(indices) + 1
                    gathered[row, i] = hidden[row, begin:end].mean(dim=0)
                    pad_mask[row, i] = False
        return gathered, pad_mask

    def shift_target(self, target: torch.Tensor, fill_value=-1) ->torch.Tensor:
        """
        Shift matrix to build generation targets.

        :param torch.Tensor target: Target tensor to build generation targets. Shape [B, T]
        :param fill_value: Value to be filled at the padded positions.
        :rtype: torch.Tensor
        :return: Tensor with shape [B, T], where (i, j)-entries are (i, j+1) entry of target tensor.
        """
        with torch.no_grad():
            pad_at_end = torch.full((target.shape[0], 1), fill_value=fill_value, dtype=target.dtype, device=target.device)
            return torch.cat([target[:, 1:], pad_at_end], dim=-1).contiguous()

    def convert_idx2symbol(self, output, num_list):
        """batch_size=1"""
        output_list = []
        if 'vall' in self.mode:
            for id, single in enumerate(output):
                output_list.append(self.out_expression_op(single, num_list[id]))
        else:
            for id, single in enumerate(output):
                output_list.append(self.out_expression_expr(single, num_list[id]))
        return output_list

    def out_expression_op(self, item, num_list):
        equation = []
        for i, token in enumerate(item.tolist()):
            if token != EPT_CON.PAD_ID:
                token = self.out_idx2sym[token]
                if token == EPT_CON.SEQ_NEW_EQN:
                    equation.clear()
                    continue
                elif token == EPT_CON.SEQ_END_EQN:
                    break
            else:
                break
            equation.append(token)
        return equation

    def out_expression_expr(self, item, num_list):
        expressions = []
        for token in item:
            operator = self.out_idx2opsym[token[0]]
            if operator == EPT_CON.FUN_NEW_EQN:
                expressions.clear()
                continue
            if operator == EPT_CON.FUN_END_EQN:
                break
            operands = []
            for i in range(1, len(token), 2):
                src = token[i]
                if src != EPT_CON.PAD_ID:
                    src = EPT_CON.ARG_TOKENS[src]
                    operand = token[i + 1]
                    if src == EPT_CON.ARG_CON or 'gen' in self.mode:
                        operand = self.out_idx2consym[operand]
                    if type(operand) is str and operand.startswith(EPT_CON.MEM_PREFIX):
                        operands.append((EPT_CON.ARG_MEM, int(operand[2:])))
                    else:
                        operands.append((src, operand))
            expressions.append((operator, operands))
        computation_history = []
        expression_used = []
        for operator, operands in expressions:
            computation = []
            if operator == EPT_CON.FUN_NEW_VAR:
                computation.append(EPT_CON.FORMAT_VAR % len(computation_history))
            else:
                for src, operand in operands:
                    if src == EPT_CON.ARG_NUM and 'ptr' in self.mode:
                        computation.append(EPT_CON.FORMAT_NUM % operand)
                    elif src == EPT_CON.ARG_MEM:
                        if operand < len(computation_history):
                            computation += computation_history[operand]
                            expression_used[operand] = True
                        else:
                            computation.append(EPT_CON.ARG_UNK)
                    else:
                        computation.append(operand)
                computation.append(operator)
            computation_history.append(computation)
            expression_used.append(False)
        computation_history = [equation for used, equation in zip(expression_used, computation_history) if not used]
        result = sum(computation_history, [])
        replace_result = []
        for word in result:
            if 'N_' in word:
                replace_result.append(str(num_list[int(word[2:])]['value']))
            elif 'C_' in word:
                replace_result.append(str(word[2:].replace('_', '.')))
            else:
                replace_result.append(word)
        if '=' in replace_result[:-1]:
            replace_result.append('<BRG>')
        return replace_result

    def __str__(self) ->str:
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class EPTMultiHeadAttentionWeights(nn.Module):
    """
    Class for computing multi-head attention weights (follows the paper, 'Attention is all you need')

    This class computes dot-product between query Q and key K, i.e.
    """

    def __init__(self, **config):
        """
        Initialize MultiHeadAttentionWeights class

        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default.
        :keyword int num_heads: Number of attention heads (N). 12 by default.
        """
        super().__init__()
        self.config = config
        assert self.hidden_dim % self.num_heads == 0, 'Hidden dimension %s is not divisible by the number of heads %s.' % (self.hidden_dim, self.num_heads)
        self.linear_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dim_head = self.hidden_dim // self.num_heads
        self.sqrt_dim = self.dim_head ** 0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor=None, key_ignorance_mask: torch.Tensor=None, attention_mask: torch.Tensor=None, head_at_last: bool=True) ->torch.Tensor:
        """
        Compute multi-head attention weights
        
        Args:
            query (torch.Tensor): FloatTensor representing the query matrix with shape [batch_size, query_sequence_length, hidden_size].
            key (torch.Tensor): FloatTensor representing the key matrix with shape [batch_size, key_sequence_length, hidden_size] or [1, key_sequence_length, hidden_size]. By default, this is `None` (Use query matrix as a key matrix)
            key_ignorance_mask (torch.Tensor): BoolTensor representing the mask for ignoring column vector in key matrix, with shape [batch_size, key_sequence_length]. 
                If an element at (b, t) is `True,` then all return elements at batch_size=b, key_sequence_length=t will set to be -Infinity. By default, this is `None` (There's no mask to apply).
            attention_mask (torch.Tensor): BoolTensor representing Attention mask for ignoring a key for each query item, with shape [query_sequence_length, key_sequence_length].
                If an element at (s, t) is `True,` then all return elements at sequence_length=s, T=t will set to be -Infinity. By default, this is `None` (There's no mask to apply).
            head_at_last (bool): Use `True` to make shape of return value be [batch_size, query_sequence_length, key_sequence_length, head_nums].
                If `False,` this method will return [batch_size, head_nums, sequence_length, key_sequence_length]. By default, this is `True`
        
        Returns:
            torch.FloatTensor: FloatTensor of Multi-head Attention weights.
        """
        if key is None:
            key = query
        assert query.shape[0] == key.shape[0] or key.shape[0] == 1 or query.shape[0] == 1
        assert key_ignorance_mask is None or key.shape[:2] == key_ignorance_mask.shape and key_ignorance_mask.dtype == torch.bool
        assert attention_mask is None or query.shape[1] == attention_mask.shape[0] and key.shape[1] == attention_mask.shape[1] and attention_mask.dtype == torch.bool
        query_len = query.shape[1]
        key_len = key.shape[1]
        batch_size = max(key.shape[0], query.shape[0])
        query = self.linear_q(query)
        key = self.linear_k(key)
        query = query / self.sqrt_dim
        if query.shape[0] == 1:
            query = query.expand(batch_size, -1, -1)
        if key.shape[0] == 1:
            key = key.expand(batch_size, -1, -1)
        query = query.view(batch_size, query_len, self.num_heads, self.dim_head).transpose(1, 2).flatten(0, 1).contiguous()
        key = key.view(batch_size, key_len, self.num_heads, self.dim_head).permute(0, 2, 3, 1).flatten(0, 1).contiguous()
        attention_weights = torch.bmm(query, key).view(batch_size, self.num_heads, query_len, key_len).contiguous()
        if attention_mask is not None:
            attention_weights.masked_fill_(attention_mask, EPT.NEG_INF)
        if key_ignorance_mask is not None:
            attention_weights.masked_fill_(key_ignorance_mask.unsqueeze(1).unsqueeze(1), EPT.NEG_INF)
        if head_at_last:
            return attention_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return attention_weights

    @property
    def hidden_dim(self) ->int:
        """
        :rtype: int
        :return: Vector dimension of hidden states (H)
        """
        return self.config.get('hidden_dim', 768)

    @property
    def num_heads(self) ->int:
        """
        :rtype: int
        :return: Number of attention heads (N)
        """
        return self.config.get('num_heads', 12)


class EPTMultiHeadAttention(nn.Module):
    """
    Class for computing multi-head attention (follows the paper, 'Attention is all you need')

    This class computes attention over K-V pairs with query Q, i.e.

    """

    def __init__(self, **config):
        """
        Initialize MultiHeadAttention class

        :keyword int hidden_dim: Vector dimension of hidden states (H). 768 by default
        :keyword int num_heads: Number of attention heads (N). 12 by default
        :keyword float dropout_p: Probability of dropout. 0 by default
        """
        super().__init__()
        self.attn = EPTMultiHeadAttentionWeights(**config)
        self.dropout_p = 0.0
        self.dropout_attn = nn.Dropout(self.dropout_p)
        self.linear_v = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)
        self.linear_out = nn.Linear(self.attn.hidden_dim, self.attn.hidden_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor=None, key_ignorance_mask: torch.Tensor=None, attention_mask: torch.Tensor=None, return_weights: bool=False, **kwargs):
        """
        Compute multi-head attention

        Args:
            query (torch.Tensor): FloatTensor representing the query matrix with shape [batch_size, query_sequence_length, hidden_size].
            key_value (torch.Tensor): FloatTensor representing the key matrix or value matrix with shape [batch_size, key_sequence_length, hidden_size] or [1, key_sequence_length, hidden_size].
                By default, this is `None` (Use query matrix as a key matrix).
            key_ignorance_mask (torch.Tensor): BoolTensor representing the mask for ignoring column vector in key matrix, with shape [batch_size, key_sequence_length].
                If an element at (b, t) is `True,` then all return elements at batch_size=b, key_sequence_length=t will set to be -Infinity. By default, this is `None` (There's no mask to apply).
            attention_mask (torch.Tensor): BoolTensor representing Attention mask for ignoring a key for each query item, with shape [query_sequence_length, key_sequence_length].
                If an element at (s, t) is `True,` then all return elements at query_sequence_length=s, key_sequence_length=t will set to be -Infinity. By default, this is `None` (There's no mask to apply).
            return_weights (bool): Use `True` to return attention weights. By default, this is `True.`
        
        Returns:
            Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
                If head_at_last is True, return (Attention Output, Attention Weights). Otherwise, return only the Attention Output.
                Attention Output: Shape [batch_size, query_sequence_length, hidden_size].
                Attention Weights: Shape [batch_size, query_sequence_length, key_sequence_length, head_nums].
        """
        if key_value is None:
            key_value = query
        attn_weights = self.attn(query=query, key=key_value, key_ignorance_mask=key_ignorance_mask, attention_mask=attention_mask, head_at_last=False)
        batch_size, _, query_len, key_len = attn_weights.shape
        attn = attn_weights.softmax(dim=-1)
        attn = self.dropout_attn(attn)
        attn = attn.masked_fill(torch.isnan(attn), 0.0).view(-1, query_len, key_len)
        value_size = key_value.shape[0]
        value = self.linear_v(key_value).view(value_size, key_len, self.attn.num_heads, self.attn.dim_head).transpose(1, 2)
        if value_size == 1:
            value = value.expand(batch_size, -1, -1, -1)
        value = value.flatten(0, 1).contiguous()
        output = torch.bmm(attn, value).view(batch_size, self.attn.num_heads, query_len, self.attn.dim_head).transpose(1, 2).flatten(2, 3).contiguous()
        output = self.linear_out(output)
        if return_weights:
            return output, attn_weights.permute(0, 2, 3, 1).contiguous()
        else:
            return output


class TransformerLayer(nn.Module):
    """Transformer Layer, including
        a multi-head self-attention,
        a external multi-head self-attention layer (only for conditional decoder) and
        a point-wise feed-forward layer.

    Args:
        self_padding_mask (torch.bool): the padding mask for the multi head attention sublayer.
        self_attn_mask (torch.bool): the attention mask for the multi head attention sublayer.
        external_states (torch.Tensor): the external context for decoder, e.g., hidden states from encoder.
        external_padding_mask (torch.bool): the padding mask for the external states.

    Returns:
        feedforward_output (torch.Tensor): the output of the point-wise feed-forward sublayer, is the output of the transformer layer
    """

    def __init__(self, embedding_size, ffn_size, num_heads, attn_dropout_ratio=0.0, attn_weight_dropout_ratio=0.0, ffn_dropout_ratio=0.0, with_external=False):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = EPTMultiHeadAttention(embedding_size, num_heads, attn_weight_dropout_ratio)
        self.feed_forward_1 = nn.Linear(embedding_size, ffn_size)
        self.feed_forward_2 = nn.Linear(ffn_size, embedding_size)
        self.attn_layer_norm = nn.LayerNorm(embedding_size, eps=1e-06)
        self.ffn_layer_norm = nn.LayerNorm(embedding_size, eps=1e-06)
        self.attn_dropout = nn.Dropout(attn_dropout_ratio)
        self.ffn_dropout = nn.Dropout(ffn_dropout_ratio)
        self.with_external = with_external
        if self.with_external:
            self.external_multi_head_attention = EPTMultiHeadAttention(embedding_size, num_heads, attn_weight_dropout_ratio)
            self.external_layer_norm = nn.LayerNorm(embedding_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.feed_forward_1.weight, std=0.02)
        nn.init.normal_(self.feed_forward_2.weight, std=0.02)
        nn.init.constant_(self.feed_forward_1.bias, 0.0)
        nn.init.constant_(self.feed_forward_2.bias, 0.0)

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x, kv=None, self_padding_mask=None, self_attn_mask=None, external_states=None, external_padding_mask=None):
        residual = x
        if kv is None:
            x, self_attn_weights = self.multi_head_attention(query=x, key=x, value=x, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask)
        else:
            x, self_attn_weights = self.multi_head_attention(query=x, key=kv, value=kv, key_padding_mask=self_padding_mask, attn_mask=self_attn_mask)
        x = self.attn_dropout(x)
        x = self.attn_layer_norm(residual + x)
        if self.with_external:
            residual = x
            x, external_attn_weights = self.external_multi_head_attention(query=x, key=external_states, value=external_states, key_padding_mask=external_padding_mask)
            x = self.attn_dropout(x)
            x = self.external_layer_norm(residual + x)
        else:
            external_attn_weights = None
        residual = x
        x = self.feed_forward_2(self.gelu(self.feed_forward_1(x)))
        x = self.ffn_dropout(x)
        x = self.ffn_layer_norm(residual + x)
        return x, self_attn_weights, external_attn_weights


class TransformerDecoder(nn.Module):
    """
    The stacked Transformer decoder layers.
    """

    def __init__(self, embedding_size, ffn_size, num_decoder_layers, num_heads, attn_dropout_ratio=0.0, attn_weight_dropout_ratio=0.0, ffn_dropout_ratio=0.0, with_external=True):
        super(TransformerDecoder, self).__init__()
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_decoder_layers):
            self.transformer_layers.append(TransformerLayer(embedding_size, ffn_size, num_heads, attn_dropout_ratio, attn_weight_dropout_ratio, ffn_dropout_ratio, with_external))

    def forward(self, x, kv=None, self_padding_mask=None, self_attn_mask=None, external_states=None, external_padding_mask=None):
        """ Implement the decoding process step by step.

        Args:
            x (torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            kv (torch.Tensor): the cached history latent vector, shape: [batch_size, sequence_length, embedding_size], default: None.
            self_padding_mask (torch.Tensor): padding mask of target sequence, shape: [batch_size, sequence_length], default: None.
            self_attn_mask (torch.Tensor): diagonal attention mask matrix of target sequence, shape: [batch_size, sequence_length, sequence_length], default: None.
            external_states (torch.Tensor): output features of encoder, shape: [batch_size, sequence_length, feature_size], default: None.
            external_padding_mask (torch.Tensor): padding mask of source sequence, shape: [batch_size, sequence_length], default: None.

        Returns:
            torch.Tensor: output features, shape: [batch_size, sequence_length, ffn_size].
        """
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask, self_attn_mask, external_states, external_padding_mask)
        return x


def greedy_search(logits):
    """Find the index of max logits

    Args:
        logits (torch.Tensor): logits distribution

    Return:
        torch.Tensor: the chosen index of token
    """
    return logits.argmax(dim=-1)


def topk_sampling(logits, temperature=1.0, top_k=0, top_p=0.9):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits (torch.Tensor): logits distribution
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).

    Return:
        torch.Tensor: the chosen index of token.
    """
    logits = logits / temperature
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        values = torch.topk(logits, top_k)[0]
        batch_mins = values[:, :, -1].expand_as(logits.squeeze(1)).unsqueeze(1)
        logits = torch.where(logits < batch_mins, torch.ones_like(logits) * -10000000000.0, logits)
    if 0.0 < top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, _ = torch.sort(probs, descending=True, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        mask = cumprobs < top_p
        mask = F.pad(mask[:, :, :-1], (1, 0, 0, 0), value=1)
        masked_probs = torch.where(mask, sorted_probs, torch.tensor(float('inf')))
        batch_mins = masked_probs.min(dim=-1, keepdim=True)[0].expand_as(logits)
        logits = torch.where(probs < batch_mins, torch.tensor(float('-inf')), logits)
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities.squeeze(1)
    token_idx = torch.multinomial(probabilities, 1)
    return token_idx


class AlbertGen(nn.Module):

    def __init__(self, config, dataset):
        super(AlbertGen, self).__init__()
        self.device = config['device']
        self.pretrained_model_path = config['pretrained_model_path']
        self.max_input_len = config['max_len']
        self.dataset = dataset
        if config['dataset'] in [DatasetName.math23k, DatasetName.hmwp, DatasetName.ape200k]:
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
            self.encoder = BertModel.from_pretrained(self.pretrained_model_path)
        else:
            self.tokenizer = AlbertTokenizer.from_pretrained(self.pretrained_model_path)
            self.encoder = AlbertModel.from_pretrained(self.pretrained_model_path)
        self.eos_token_id = self.tokenizer.sep_token_id
        self.eos_token = self.tokenizer.sep_token
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.max_output_len = config['max_output_len']
        self.share_vocab = config['share_vocab']
        self.decoding_strategy = config['decoding_strategy']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.out_pad_idx = self.out_symbol2idx['<PAD>']
        self.out_sos_idx = self.out_symbol2idx['<SOS>']
        self.out_eos_idx = self.out_symbol2idx['<EOS>']
        self.out_unk_idx = self.out_symbol2idx['<UNK>']
        config['vocab_size'] = len(self.tokenizer)
        config['symbol_size'] = len(self.out_symbol2idx)
        config['in_word2idx'] = self.tokenizer.get_vocab()
        config['in_idx2word'] = list(self.tokenizer.get_vocab().keys())
        self.in_embedder = BasicEmbedder(config['vocab_size'], config['embedding_size'], config['embedding_dropout_ratio'])
        self.out_embedder = BasicEmbedder(config['symbol_size'], config['embedding_size'], config['embedding_dropout_ratio'])
        self.pos_embedder = PositionEmbedder(config['embedding_size'], config['max_len'])
        self.self_attentioner = SelfAttentionMask()
        self.decoder = TransformerDecoder(config['embedding_size'], config['ffn_size'], config['num_decoder_layers'], config['num_heads'], config['attn_dropout_ratio'], config['attn_weight_dropout_ratio'], config['ffn_dropout_ratio'])
        self.out = nn.Linear(config['embedding_size'], config['symbol_size'])
        self.loss = NLLLoss()

    def calculate_loss(self, batch_data):
        seq, target = batch_data['ques_source'], batch_data['equ_source']
        outputs, target = self.forward(seq, target)
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)
        self.loss.reset()
        self.loss.eval_batch(outputs, target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data):
        seq = batch_data['ques_source']
        num_list = batch_data['num list']
        target = batch_data['equ_source']
        outputs, _ = self.forward(seq)
        batch_size = len(target)
        outputs = self.convert_idx2symbol(outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return outputs, targets

    def forward(self, seq, target=None):
        srcs = []
        for idx, s in enumerate(seq):
            if self.max_input_len is not None:
                src = self.tokenizer.encode(seq[idx], max_length=self.max_input_len - 1)
            else:
                src = self.tokenizer.encode(seq[idx])
            srcs.append(src)
        src_length = max([len(_) for _ in srcs])
        for i in range(len(srcs)):
            srcs[i] = [self.tokenizer.cls_token_id] + srcs[i] + (src_length - len(srcs[i])) * [self.tokenizer.pad_token_id]
        src_length = src_length + 1
        srcs_tensor = torch.LongTensor(srcs)
        src_feat = self.encoder(srcs_tensor)[0]
        source_padding_mask = torch.eq(srcs_tensor, self.tokenizer.pad_token_id)
        if target != None:
            tgts = []
            for idx, t in enumerate(target):
                tgt = []
                if isinstance(t, str):
                    t = t.split()
                for _ in t:
                    if _ not in self.out_symbol2idx:
                        tgt.append(self.out_symbol2idx['<UNK>'])
                    else:
                        tgt.append(self.out_symbol2idx[_])
                if self.max_output_len is not None:
                    tgts.append(tgt[:self.max_output_len - 1])
                else:
                    tgts.append(tgt)
            target_length = max([len(_) for _ in tgts])
            for i in range(len(tgts)):
                tgts[i] = tgts[i] + [self.out_eos_idx] + (target_length - len(tgts[i])) * [self.out_pad_idx]
            tgts = torch.LongTensor(tgts)
            token_logits = self.generate_t(src_feat, tgts, source_padding_mask)
            return token_logits, tgts
        else:
            all_output = self.generate_without_t(src_feat, source_padding_mask)
            return all_output, None

    def generate_t(self, encoder_outputs, target, source_padding_mask):
        with_t = random.random()
        seq_len = target.size(1)
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        if with_t < self.teacher_force_ratio:
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1)
            target = torch.cat((input_seq, target), dim=1)[:, :-1]
            decoder_inputs = self.pos_embedder(self.out_embedder(target))
            self_padding_mask = torch.eq(target, self.out_pad_idx)
            self_attn_mask = self.self_attentioner(target.size(-1)).bool()
            decoder_outputs = self.decoder(decoder_inputs, self_padding_mask=self_padding_mask, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)
            token_logits = self.out(decoder_outputs)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        else:
            token_logits = []
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1)
            pre_tokens = [input_seq]
            for idx in range(seq_len):
                self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
                decoder_input = self.pos_embedder(self.out_embedder(input_seq))
                decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)
                token_logit = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
                token_logits.append(token_logit)
                if self.decoding_strategy == 'topk_sampling':
                    output = topk_sampling(token_logit, top_k=5)
                elif self.decoding_strategy == 'greedy_search':
                    output = greedy_search(token_logit)
                else:
                    raise NotImplementedError
                if self.share_vocab:
                    pre_tokens.append(self.decode(output))
                else:
                    pre_tokens.append(output)
                input_seq = torch.cat(pre_tokens, dim=1)
            token_logits = torch.cat(token_logits, dim=1)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        return token_logits

    def generate_without_t(self, encoder_outputs, source_padding_mask):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1)
        pre_tokens = [input_seq]
        all_outputs = []
        for gen_idx in range(self.max_output_len):
            self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
            decoder_input = self.pos_embedder(self.out_embedder(input_seq))
            decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)
            token_logits = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
            if self.decoding_strategy == 'topk_sampling':
                output = topk_sampling(token_logits, top_k=5)
            elif self.decoding_strategy == 'greedy_search':
                output = greedy_search(token_logits)
            else:
                raise NotImplementedError
            all_outputs.append(output)
            if self.share_vocab:
                pre_tokens.append(self.decode(output))
            else:
                pre_tokens.append(output)
            input_seq = torch.cat(pre_tokens, dim=1)
        all_outputs = torch.cat(all_outputs, dim=1)
        all_outputs = self.decode_(all_outputs)
        return all_outputs

    def decode_(self, outputs):
        batch_size = outputs.size(0)
        all_outputs = []
        for b in range(batch_size):
            symbols = self.tokenizer.decode(outputs[b])
            symbols = self.tokenizer.tokenize(symbols)
            symbols = [self.out_idx2symbol[_] for _ in outputs[b]]
            symbols_ = []
            for token in symbols:
                if token == '<EOS>':
                    break
                else:
                    symbols_.append(token)
            symbols = symbols_[:]
            all_outputs.append(symbols)
        return all_outputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def convert_idx2symbol(self, outputs, num_lists):
        batch_size = len(outputs)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_lists[b_i])
            res = []
            if isinstance(outputs[b_i], str):
                output = outputs[b_i].split()
            else:
                output = outputs[b_i]
            for s_i in range(len(output)):
                symbol = output[s_i]
                if 'NUM' in symbol:
                    num_idx = NumMask.number.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_lists[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class BERTGen(nn.Module):
    """
    Reference:
        Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding".
    """

    def __init__(self, config, dataset):
        super(BERTGen, self).__init__()
        self.device = config['device']
        self.pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config['transformers_pretrained_model']
        self.max_input_len = config['max_len']
        self.tokenizer = dataset.tokenizer
        self.eos_token_id = self.tokenizer.sep_token_id
        self.eos_token = self.tokenizer.sep_token
        self.encoder = BertModel.from_pretrained(self.pretrained_model_path)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.max_output_len = config['max_output_len']
        self.share_vocab = config['share_vocab']
        self.decoding_strategy = config['decoding_strategy']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.out_pad_idx = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        self.out_sos_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        self.out_eos_idx = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        self.out_unk_idx = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        config['vocab_size'] = len(self.tokenizer)
        config['symbol_size'] = len(self.out_symbol2idx)
        config['in_word2idx'] = self.tokenizer.get_vocab()
        config['in_idx2word'] = list(self.tokenizer.get_vocab().keys())
        self.in_embedder = BasicEmbedder(config['vocab_size'], config['embedding_size'], config['embedding_dropout_ratio'])
        self.out_embedder = BasicEmbedder(config['symbol_size'], config['embedding_size'], config['embedding_dropout_ratio'])
        self.pos_embedder = PositionEmbedder(config['embedding_size'], config['max_len'])
        self.self_attentioner = SelfAttentionMask()
        self.decoder = TransformerDecoder(config['embedding_size'], config['ffn_size'], config['num_decoder_layers'], config['num_heads'], config['attn_dropout_ratio'], config['attn_weight_dropout_ratio'], config['ffn_dropout_ratio'])
        self.out = nn.Linear(config['embedding_size'], config['symbol_size'])
        self.loss = NLLLoss()
        self._pretrained_model_resize()

    def _pretrained_model_resize(self):
        self.encoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, seq, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor | None target: target, shape: [batch_size,target_length].
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return: token_logits: [batch_size, output_length, output_size], symbol_outputs: [batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        src_feat, encoder_layer_outputs = self.encoder_forward(seq, output_all_layers)
        source_padding_mask = torch.eq(seq, self.tokenizer.pad_token_id)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(src_feat, source_padding_mask, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.
        
        Args:
            batch_data (dict): one batch data.
        
        Returns:
            float: loss value.
        """
        seq, target = batch_data['question'], batch_data['equation']
        seq = torch.LongTensor(seq)
        target = torch.LongTensor(target)
        token_logits, _, _ = self.forward(seq, target)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        outputs = torch.nn.functional.log_softmax(token_logits, dim=1)
        self.loss.reset()
        self.loss.eval_batch(outputs, target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.
        
        Args:
            batch_data (dict): one batch data.
        
        Returns:
            tuple(list,list): predicted equation, target equation.
        """
        seq = batch_data['question']
        num_list = batch_data['num list']
        target = batch_data['equation']
        seq = torch.LongTensor(seq)
        target = torch.LongTensor(target)
        _, outputs, _ = self.forward(seq)
        outputs = self.decode_(outputs)
        target = self.decode_(target)
        outputs = self.convert_idx2symbol(outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq, output_all_layers=False):
        encoder_outputs = self.encoder(seq)
        src_feat = encoder_outputs[0]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
        return src_feat, all_layer_outputs

    def decoder_forward(self, encoder_outputs, source_padding_mask, target=None, output_all_layers=None):
        with_t = random.random()
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        if target is not None and with_t < self.teacher_force_ratio:
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1)
            target = torch.cat((input_seq, target), dim=1)[:, :-1]
            decoder_inputs = self.pos_embedder(self.out_embedder(target))
            self_padding_mask = torch.eq(target, self.out_pad_idx)
            self_attn_mask = self.self_attentioner(target.size(-1)).bool()
            decoder_outputs = self.decoder(decoder_inputs, self_padding_mask=self_padding_mask, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)
            token_logits = self.out(decoder_outputs)
            outputs = torch.topk(token_logits, 1, dim=-1)[1].squeeze(-1)
        else:
            token_logits = []
            outputs = []
            seq_len = target.size(1) if target is not None else self.max_output_len
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1)
            pre_tokens = [input_seq]
            for idx in range(seq_len):
                self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
                decoder_input = self.pos_embedder(self.out_embedder(input_seq))
                decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)
                token_logit = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
                token_logits.append(token_logit)
                if self.decoding_strategy == 'topk_sampling':
                    output = topk_sampling(token_logit, top_k=5)
                elif self.decoding_strategy == 'greedy_search':
                    output = greedy_search(token_logit)
                else:
                    raise NotImplementedError
                outputs.append(output)
                if self.share_vocab:
                    pre_tokens.append(self.decode(output))
                else:
                    pre_tokens.append(output)
                input_seq = torch.cat(pre_tokens, dim=1)
            token_logits = torch.cat(token_logits, dim=1)
            outputs = torch.cat(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def decode_(self, outputs):
        batch_size = outputs.size(0)
        all_outputs = []
        for b in range(batch_size):
            symbols = [self.out_idx2symbol[_] for _ in outputs[b]]
            symbols_ = []
            for token in symbols:
                if token == SpecialTokens.EOS_TOKEN or token == SpecialTokens.PAD_TOKEN:
                    break
                else:
                    symbols_.append(token)
            symbols = symbols_[:]
            all_outputs.append(symbols)
        return all_outputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def convert_idx2symbol(self, outputs, num_lists):
        batch_size = len(outputs)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_lists[b_i])
            res = []
            if isinstance(outputs[b_i], str):
                output = outputs[b_i].split()
            else:
                output = outputs[b_i]
            for s_i in range(len(output)):
                symbol = output[s_i]
                if 'NUM' in symbol:
                    num_idx = NumMask.number.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_lists[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class GPT2(nn.Module):
    """
    Reference:
        Radford et al. "Language Models are Unsupervised Multitask Learners".
    """

    def __init__(self, config, dataset):
        super(GPT2, self).__init__()
        self.device = config['device']
        self.max_out_len = config['max_output_len']
        self.max_input_len = config['max_len']
        self.pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config['transformers_pretrained_model']
        self.tokenizer = dataset.tokenizer
        if config['dataset'] in [DatasetName.math23k, DatasetName.hmwp, DatasetName.ape200k]:
            self.eos_token_id = self.tokenizer.sep_token_id
            self.eos_token = self.tokenizer.sep_token
            self.start_token = self.tokenizer.cls_token
        else:
            self.eos_token_id = self.tokenizer.eos_token_id
            self.eos_token = self.tokenizer.eos_token
            self.start_token = ''
        self.configuration = GPT2Config.from_pretrained(self.pretrained_model_path)
        self.decoder = GPT2LMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self._pretrained_model_resize()
        self.loss = NLLLoss()

    def _pretrained_model_resize(self):
        self.decoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, seq, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor | None target: target, shape: [batch_size,target_length].
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return: token_logits: [batch_size, output_length, output_size], symbol_outputs: [batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(seq, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        Args:
            batch_data (dict): one batch data.

        Returns:
            float: loss value.
        """
        seq, target = batch_data['question'], batch_data['equation']
        seq = torch.LongTensor(seq)
        target = torch.LongTensor(target)
        token_logits, _, _ = self.forward(seq, target)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        outputs = torch.nn.functional.log_softmax(token_logits, dim=1)
        self.loss.reset()
        self.loss.eval_batch(outputs, target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        Args:
            batch_data (dict): one batch data.

        Returns:
            tuple(list,list): predicted equation, target equation.
        """
        seq = batch_data['question']
        num_list = batch_data['num list']
        target = batch_data['equation']
        seq = torch.LongTensor(seq)
        target = torch.LongTensor(target)
        _, outputs, _ = self.forward(seq)
        outputs = self.decode_(outputs)
        target = self.decode_(target)
        outputs = self.convert_idx2symbol(outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def list2str(self, x):
        y = ''.join(x)
        return y

    def decoder_forward(self, seq, target=None, output_all_layers=False):
        if target is not None:
            tgts_inputs_tensor = target[:, :-1]
            tgts_outputs_tensor = target
            seq_mask = (tgts_inputs_tensor != self.eos_token_id).float()
            seq_mask = torch.cat([torch.FloatTensor(seq_mask.shape[0], 1).fill_(1.0), seq_mask], 1)
            inputs = torch.cat([seq, tgts_inputs_tensor], 1)
            logits = self.decoder(inputs)[0]
            logits = logits[:, -tgts_outputs_tensor.shape[1]:, :].contiguous()
            outputs = torch.topk(logits, 1, dim=-1)[1]
        else:
            outputs = []
            logits = []
            inputs = seq
            for idx in range(self.max_out_len):
                decoder_outputs = self.decoder(inputs)
                token_logit = decoder_outputs[0][:, -1, :]
                tokens = token_logit.topk(1, dim=1)[1]
                logits.append(token_logit)
                outputs.append(tokens)
                inputs = torch.cat((inputs, tokens), dim=1)
            logits = torch.stack(logits, dim=1)
            outputs = torch.cat(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = logits
            all_layer_outputs['outputs'] = outputs
        return logits, outputs, all_layer_outputs

    def decode_(self, outputs):
        batch_size = outputs.size(0)
        all_outputs = []
        for b in range(batch_size):
            symbols = self.tokenizer.decode(outputs[b])
            symbols = self.tokenizer.tokenize(symbols)
            symbols_ = []
            for token in symbols:
                if token == self.start_token:
                    continue
                if 'Ġ' in token:
                    symbols_.append(token[1:])
                elif token == self.eos_token:
                    break
                else:
                    symbols_.append(token)
            symbols = symbols_[:]
            all_outputs.append(symbols)
        return all_outputs

    def encode_(self, inputs):
        outputs = []
        for idx, s in enumerate(inputs):
            out = self.tokenizer.encode(inputs[idx])
            outputs.append(out)
        output_length = max([len(_) for _ in outputs]) + 1
        for i in range(len(outputs)):
            outputs[i] += (output_length - len(outputs[i])) * [self.eos_token_id]
        outputs_tensor = torch.LongTensor(outputs)
        return outputs_tensor

    def convert_idx2symbol(self, outputs, num_lists):
        batch_size = len(outputs)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_lists[b_i])
            res = []
            if isinstance(outputs[b_i], str):
                output = outputs[b_i].split()
            else:
                output = outputs[b_i]
            for s_i in range(len(output)):
                symbol = output[s_i]
                if 'NUM' in symbol:
                    num_idx = NumMask.number.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_lists[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list


class RobertaGen(nn.Module):
    """
    Reference:
        Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach".
    """

    def __init__(self, config, dataset):
        super(RobertaGen, self).__init__()
        self.device = config['device']
        self.pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config['transformers_pretrained_model']
        self.max_input_len = config['max_len']
        self.max_output_len = config['max_output_len']
        self.tokenizer = dataset.tokenizer
        self.eos_token_id = self.tokenizer.sep_token_id
        self.eos_token = self.tokenizer.sep_token
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.max_output_len = config['max_output_len']
        self.share_vocab = config['share_vocab']
        self.decoding_strategy = config['decoding_strategy']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.out_pad_idx = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        self.out_sos_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        self.out_eos_idx = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        self.out_unk_idx = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        config['vocab_size'] = len(self.tokenizer)
        config['symbol_size'] = len(self.out_symbol2idx)
        config['in_word2idx'] = self.tokenizer.get_vocab()
        config['in_idx2word'] = list(self.tokenizer.get_vocab().keys())
        if config['dataset'] in [DatasetName.math23k, DatasetName.hmwp, DatasetName.ape200k]:
            self.encoder = BertModel.from_pretrained(self.pretrained_model_path)
        else:
            self.encoder = RobertaModel.from_pretrained(self.pretrained_model_path)
        self.in_embedder = BasicEmbedder(config['vocab_size'], config['embedding_size'], config['embedding_dropout_ratio'])
        self.out_embedder = BasicEmbedder(config['symbol_size'], config['embedding_size'], config['embedding_dropout_ratio'])
        self.pos_embedder = PositionEmbedder(config['embedding_size'], config['max_len'])
        self.self_attentioner = SelfAttentionMask()
        self.decoder = TransformerDecoder(config['embedding_size'], config['ffn_size'], config['num_decoder_layers'], config['num_heads'], config['attn_dropout_ratio'], config['attn_weight_dropout_ratio'], config['ffn_dropout_ratio'])
        self.out = nn.Linear(config['embedding_size'], config['symbol_size'])
        self.loss = NLLLoss()
        self._pretrained_model_resize()

    def _pretrained_model_resize(self):
        self.encoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, seq, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor | None target: target, shape: [batch_size,target_length].
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return: token_logits: [batch_size, output_length, output_size], symbol_outputs: [batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        seq_feat, encoder_layer_outputs = self.encoder_forward(seq, output_all_layers)
        source_padding_mask = torch.eq(seq, self.tokenizer.pad_token_id)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(seq_feat, source_padding_mask, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        Args:
            batch_data (dict): one batch data.

        Returns:
            float: loss value.
        """
        seq, target = batch_data['question'], batch_data['equation']
        seq = torch.LongTensor(seq)
        target = torch.LongTensor(target)
        token_logits, _, _ = self.forward(seq, target)
        token_logits = token_logits.view(-1, token_logits.size(-1))
        outputs = torch.nn.functional.log_softmax(token_logits, dim=1)
        self.loss.reset()
        self.loss.eval_batch(outputs, target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        Args:
            batch_data (dict): one batch data.

        Returns:
            tuple(list,list): predicted equation, target equation.
        """
        seq = batch_data['question']
        num_list = batch_data['num list']
        target = batch_data['equation']
        seq = torch.LongTensor(seq)
        target = torch.LongTensor(target)
        _, outputs, _ = self.forward(seq)
        outputs = self.decode_(outputs)
        target = self.decode_(target)
        outputs = self.convert_idx2symbol(outputs, num_list)
        target = self.convert_idx2symbol(target, num_list)
        return outputs, target

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq, output_all_layers=False):
        encoder_outputs = self.encoder(seq, return_dict=True)
        src_feat = encoder_outputs[0]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
        return src_feat, all_layer_outputs

    def decoder_forward(self, encoder_outputs, source_padding_mask, target=None, output_all_layers=None):
        with_t = random.random()
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        if target is not None and with_t < self.teacher_force_ratio:
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1)
            target = torch.cat((input_seq, target), dim=1)[:, :-1]
            decoder_inputs = self.pos_embedder(self.out_embedder(target))
            self_padding_mask = torch.eq(target, self.out_pad_idx)
            self_attn_mask = self.self_attentioner(target.size(-1)).bool()
            decoder_outputs = self.decoder(decoder_inputs, self_padding_mask=self_padding_mask, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)
            token_logits = self.out(decoder_outputs)
            outputs = torch.topk(token_logits, 1, dim=-1)[1].squeeze(-1)
        else:
            token_logits = []
            outputs = []
            seq_len = target.size(1) if target is not None else self.max_output_len
            input_seq = torch.LongTensor([self.out_sos_idx] * batch_size).view(batch_size, -1)
            pre_tokens = [input_seq]
            for idx in range(seq_len):
                self_attn_mask = self.self_attentioner(input_seq.size(-1)).bool()
                decoder_input = self.pos_embedder(self.out_embedder(input_seq))
                decoder_outputs = self.decoder(decoder_input, self_attn_mask=self_attn_mask, external_states=encoder_outputs, external_padding_mask=source_padding_mask)
                token_logit = self.out(decoder_outputs[:, -1, :].unsqueeze(1))
                token_logits.append(token_logit)
                if self.decoding_strategy == 'topk_sampling':
                    output = topk_sampling(token_logit, top_k=5)
                elif self.decoding_strategy == 'greedy_search':
                    output = greedy_search(token_logit)
                else:
                    raise NotImplementedError
                outputs.append(output)
                if self.share_vocab:
                    pre_tokens.append(self.decode(output))
                else:
                    pre_tokens.append(output)
                input_seq = torch.cat(pre_tokens, dim=1)
            token_logits = torch.cat(token_logits, dim=1)
            outputs = torch.cat(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def decode_(self, outputs):
        batch_size = outputs.size(0)
        all_outputs = []
        for b in range(batch_size):
            symbols = [self.out_idx2symbol[_] for _ in outputs[b]]
            symbols_ = []
            for token in symbols:
                if token == SpecialTokens.EOS_TOKEN or token == SpecialTokens.PAD_TOKEN:
                    break
                else:
                    symbols_.append(token)
            symbols = symbols_[:]
            all_outputs.append(symbols)
        return all_outputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def convert_idx2symbol(self, outputs, num_lists):
        batch_size = len(outputs)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_lists[b_i])
            res = []
            if isinstance(outputs[b_i], str):
                output = outputs[b_i].split()
            else:
                output = outputs[b_i]
            for s_i in range(len(output)):
                symbol = output[s_i]
                if 'NUM' in symbol:
                    num_idx = NumMask.number.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_lists[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class SeqAttention(nn.Module):

    def __init__(self, hidden_size, context_size):
        super(SeqAttention, self).__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.linear_out = nn.Linear(hidden_size * 2, context_size)

    def forward(self, inputs, encoder_outputs, mask):
        """
        Args:
            inputs (torch.Tensor): shape [batch_size, 1, hidden_size].
            encoder_outputs (torch.Tensor): shape [batch_size, sequence_length, hidden_size].

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output, shape [batch_size, 1, context_size].
                attention, shape [batch_size, 1, sequence_length].
        """
        batch_size = inputs.size(0)
        seq_length = encoder_outputs.size(1)
        attn = torch.bmm(inputs, encoder_outputs.transpose(1, 2))
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn.view(-1, seq_length), dim=1).view(batch_size, -1, seq_length)
        mix = torch.bmm(attn, encoder_outputs)
        combined = torch.cat((mix, inputs), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * self.hidden_size))).view(batch_size, -1, self.context_size)
        return output, attn


class AttentionalRNNDecoder(nn.Module):
    """
    Attention-based Recurrent Neural Network (RNN) decoder.
    """

    def __init__(self, embedding_size, hidden_size, context_size, num_dec_layers, rnn_cell_type, dropout_ratio=0.0):
        super(AttentionalRNNDecoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_dec_layers = num_dec_layers
        self.rnn_cell_type = rnn_cell_type
        self.attentioner = SeqAttention(hidden_size, hidden_size)
        if rnn_cell_type == 'lstm':
            self.decoder = nn.LSTM(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == 'gru':
            self.decoder = nn.GRU(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        elif rnn_cell_type == 'rnn':
            self.decoder = nn.RNN(embedding_size, hidden_size, num_dec_layers, batch_first=True, dropout=dropout_ratio)
        else:
            raise ValueError("RNN type in attentional decoder must be in ['lstm', 'gru', 'rnn'].")
        self.attention_dense = nn.Linear(hidden_size, hidden_size)

    def init_hidden(self, input_embeddings):
        """ Initialize initial hidden states of RNN.

        Args:
            input_embeddings (torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_cell_type == 'lstm':
            h_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size)
            c_0 = torch.zeros(self.num_dec_layers, batch_size, self.hidden_size)
            hidden_states = h_0, c_0
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            return torch.zeros(self.num_dec_layers, batch_size, self.hidden_size)
        else:
            raise NotImplementedError('No such rnn type {} for initializing decoder states.'.format(self.rnn_cell_type))

    def forward(self, input_embeddings, hidden_states=None, encoder_outputs=None, encoder_masks=None):
        """ Implement the attention-based decoding process.

        Args:
            input_embeddings (torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            hidden_states (torch.Tensor): initial hidden states, default: None.
            encoder_outputs (torch.Tensor): encoder output features, shape: [batch_size, sequence_length, hidden_size], default: None.
            encoder_masks (torch.Tensor): encoder state masks, shape: [batch_size, sequence_length], default: None.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        decode_length = input_embeddings.size(1)
        all_outputs = []
        for step in range(decode_length):
            output, hidden_states = self.decoder(input_embeddings[:, step, :].unsqueeze(1), hidden_states)
            output, attn = self.attentioner(output, encoder_outputs, encoder_masks)
            output = self.attention_dense(output.view(-1, self.hidden_size))
            output = output.view(-1, 1, self.hidden_size)
            all_outputs.append(output)
        outputs = torch.cat(all_outputs, dim=1)
        return outputs, hidden_states


class DNS(nn.Module):
    """
    Reference:
        Wang et al. "Deep Neural Solver for Math Word Problems" in EMNLP 2017.
    """

    def __init__(self, config, dataset):
        super(DNS, self).__init__()
        self.device = config['device']
        self.embedding_size = config['embedding_size']
        self.bidirectional = config['bidirectional']
        self.hidden_size = config['hidden_size']
        self.decode_hidden_size = config['decode_hidden_size']
        self.encoder_rnn_cell_type = config['encoder_rnn_cell_type']
        self.decoder_rnn_cell_type = config['decoder_rnn_cell_type']
        self.dropout_ratio = config['dropout_ratio']
        self.num_layers = config['num_layers']
        self.attention = config['attention']
        self.share_vocab = config['share_vocab']
        self.max_gen_len = config['max_output_len']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.num_start = dataset.num_start
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.mask_list = NumMask.number
        if config['share_vocab']:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.in_word2idx = dataset.in_word2idx
            self.in_idx2word = dataset.in_idx2word
            self.sos_token_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.in_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        if self.share_vocab:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BasicEmbedder(self.symbol_size, self.embedding_size, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.encoder_rnn_cell_type, self.dropout_ratio, self.bidirectional)
        if self.attention:
            self.decoder = AttentionalRNNDecoder(self.embedding_size, self.decode_hidden_size, self.hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size, self.decode_hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.generate_linear = nn.Linear(self.hidden_size, self.symbol_size)
        weight = torch.ones(self.symbol_size)
        pad = self.out_pad_token
        self.loss = NLLLoss(weight, pad)

    def forward(self, seq, seq_length, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.in_embedder(seq)
        encoder_outputs, encoder_hidden, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, output_all_layers)
        decoder_inputs = self.init_decoder_inputs(target, device, batch_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, encoder_hidden, decoder_inputs, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len' and 'equation'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        token_logits, _, _ = self.forward(seq, seq_length, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.loss.reset()
        self.loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation' and 'num list'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        num_list = batch_data['num list']
        target = torch.tensor(batch_data['equation'])
        _, symbol_outputs, _ = self.forward(seq, seq_length)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(symbol_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.
        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_outputs, encoder_hidden = self.encoder(seq_emb, seq_length)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if self.encoder_rnn_cell_type == 'lstm':
                encoder_hidden = encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous()
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        if self.encoder.rnn_cell_type == self.decoder.rnn_cell_type:
            pass
        elif self.encoder.rnn_cell_type == 'gru' and self.decoder.rnn_cell_type == 'lstm':
            encoder_hidden = encoder_hidden, encoder_hidden
        elif self.encoder.rnn_cell_type == 'rnn' and self.decoder.rnn_cell_type == 'lstm':
            encoder_hidden = encoder_hidden, encoder_hidden
        elif self.encoder.rnn_cell_type == 'lstm' and (self.decoder.rnn_cell_type == 'gru' or self.decoder.rnn_cell_type == 'rnn'):
            encoder_hidden = encoder_hidden[0]
        else:
            pass
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, all_layer_outputs

    def decoder_forward(self, encoder_outputs, encoder_hidden, decoder_inputs, target=None, output_all_layers=False):
        with_t = random.random()
        seq_len = decoder_inputs.size(1) if target is not None else self.max_gen_len
        decoder_hidden = encoder_hidden
        decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
        decoder_outputs = []
        token_logits = []
        outputs = []
        output = []
        for idx in range(seq_len):
            if target is not None and with_t < self.teacher_force_ratio:
                decoder_input = decoder_inputs[:, idx, :].unsqueeze(1)
            if self.attention:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logit = self.generate_linear(step_output)
            output = self.rule_filter_(output, token_logit)
            decoder_outputs.append(decoder_output)
            token_logits.append(token_logit)
            outputs.append(output)
            if self.share_vocab:
                output_ = self.convert_out_idx_2_in_idx(output)
                decoder_input = self.out_embedder(output_)
            else:
                decoder_input = self.out_embedder(output)
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.out_embedder(decoder_inputs)
        return decoder_inputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def rule1_filter(self):
        """if r_t−1 in {+, −, ∗, /}, then rt will not in {+, −, ∗, /,), =}.
        """
        filters = []
        filters.append(self.out_symbol2idx['+'])
        filters.append(self.out_symbol2idx['-'])
        filters.append(self.out_symbol2idx['*'])
        filters.append(self.out_symbol2idx['/'])
        filters.append(self.out_symbol2idx['^'])
        try:
            filters.append(self.out_symbol2idx[')'])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx['='])
        except:
            pass
        filters.append(self.out_symbol2idx['<EOS>'])
        return torch.tensor(filters).long()

    def rule2_filter(self):
        """if r_t-1 is a number, then r_t will not be a number and not in {(, =)}.
        """
        filters = []
        try:
            filters.append(self.out_symbol2idx['('])
        except:
            pass
        for idx in range(self.num_start, len(self.out_idx2symbol)):
            filters.append(idx)
        return torch.tensor(filters).long()

    def rule3_filter(self):
        """if rt−1 is '=', then rt will not in {+, −, ∗, /, =,)}.
        """
        filters = []
        filters.append(self.out_symbol2idx['+'])
        filters.append(self.out_symbol2idx['-'])
        filters.append(self.out_symbol2idx['*'])
        filters.append(self.out_symbol2idx['/'])
        filters.append(self.out_symbol2idx['^'])
        try:
            filters.append(self.out_symbol2idx['='])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx[')'])
        except:
            pass
        return torch.tensor(filters).long()

    def rule4_filter(self):
        """if r_t-1 is '(' , then r_t will not in {(,), +, -, *, /, =}).
        """
        filters = []
        try:
            filters.append(self.out_symbol2idx[')'])
        except:
            pass
        try:
            filters.append(self.out_symbol2idx['='])
        except:
            pass
        filters.append(self.out_symbol2idx['+'])
        filters.append(self.out_symbol2idx['-'])
        filters.append(self.out_symbol2idx['*'])
        filters.append(self.out_symbol2idx['/'])
        filters.append(self.out_symbol2idx['^'])
        filters.append(self.out_symbol2idx['<EOS>'])
        return torch.tensor(filters).long()

    def rule5_filter(self):
        """if r_t−1 is ')', then r_t will not be a number and not in {(,)};
        """
        filters = []
        try:
            filters.append(self.out_symbol2idx['('])
        except:
            pass
        for idx in range(self.num_start, len(self.out_idx2symbol)):
            filters.append(idx)
        return torch.tensor(filters).long()

    def filter_op(self):
        filters = []
        filters.append(self.out_symbol2idx['+'])
        filters.append(self.out_symbol2idx['-'])
        filters.append(self.out_symbol2idx['*'])
        filters.append(self.out_symbol2idx['/'])
        filters.append(self.out_symbol2idx['^'])
        return torch.tensor(filters).long()

    def filter_END(self):
        filters = []
        filters.append(self.out_symbol2idx['<EOS>'])
        return torch.tensor(filters).long()

    def rule_filter_(self, symbols, token_logit):
        """
        Args:
            symbols (torch.Tensor): [batch_size]
            token_logit (torch.Tensor): [batch_size, symbol_size]
        return:
            symbols of next step (torch.Tensor): [batch_size]
        """
        device = token_logit.device
        next_symbols = []
        current_logit = token_logit.clone().detach()
        if symbols == []:
            filters = torch.cat([self.filter_op(), self.filter_END()])
            for b in range(current_logit.size(0)):
                current_logit[b][filters] = -float('inf')
        else:
            for b, symbol in enumerate(symbols.split(1)):
                if self.out_idx2symbol[symbol] in ['+', '-', '*', '/', '^']:
                    filters = self.rule1_filter()
                    current_logit[b][filters] = -float('inf')
                elif symbol >= self.num_start:
                    filters = self.rule2_filter()
                    current_logit[b][filters] = -float('inf')
                elif self.out_idx2symbol[symbol] in ['=']:
                    filters = self.rule3_filter()
                    current_logit[b][filters] = -float('inf')
                elif self.out_idx2symbol[symbol] in ['(']:
                    filters = self.rule4_filter()
                    current_logit[b][filters] = -float('inf')
                elif self.out_idx2symbol[symbol] in [')']:
                    filters = self.rule5_filter()
                    current_logit[b][filters] = -float('inf')
        next_symbols = current_logit.topk(1, dim=1)[1]
        return next_symbols

    def convert_out_idx_2_in_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_list[b_i])
            res = []
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def __str__(self) ->str:
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size, eps=1e-06)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class GAEncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(GAEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class GroupATTEncoder(nn.Module):
    """Group attentional encoder, N layers of group attentional encoder layer.
    """

    def __init__(self, layer, N):
        super(GroupATTEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, inputs, mask):
        """Pass the input (and mask) through each layer in turn.

        Args:
            inputs (torch.Tensor): input variavle, shape [batch_size, sequence_length, hidden_size].
        
        Returns:
            torch.Tensor: encoded variavle, shape [batch_size, sequence_length, hidden_size].
        """
        for layer in self.layers:
            inputs = layer(inputs, mask)
        return self.norm(inputs)


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention
    
    Args:
        query (torch.Tensor): shape [batch_size, sequence_length, hidden_size].
        key (torch.Tensor): shape [batch_size, sequence_length, hidden_size].
        value (torch.Tensor): shape [batch_size, sequence_length, hidden_size].
        mask (torch.Tensor): group attention mask, shape [batch_size, 4, sequence_length, sequence_length].
    
    Returns:
        tuple(torch.Tensor, torch.Tensor):

    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1000000000.0)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def group_mask(batch, type='self', pad=0):
    length = batch.shape[1]
    lis = []
    if type == 'self':
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask, -1)
            for ele in tok:
                if ele == pad:
                    copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    if ele != 1000:
                        copy[copy == 1000] = 0
                    copy[copy != ele] = 0
                    copy[copy == ele] = 1
                """
                if ele == 1000:
                    copy[copy != ele] = 1
                    copy[copy == ele] = 0
                """
                copy = np.expand_dims(copy, -1)
                mask = np.concatenate((mask, copy), axis=1)
            mask = mask[:, 1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask, 0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type == 'between':
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask, -1)
            for ele in tok:
                if ele == pad:
                    copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy == 1000] = 0
                    copy[copy == ele] = 0
                    copy[copy != 0] = 1
                    """
                    copy[copy != ele and copy != 1000] = 1
                    copy[copy == ele or copy == 1000] = 0
                    """
                copy = np.expand_dims(copy, -1)
                mask = np.concatenate((mask, copy), axis=1)
            mask = mask[:, 1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask, 0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    elif type == 'question':
        for tok in batch:
            mask = np.zeros(tok.shape)
            mask = np.expand_dims(mask, -1)
            for ele in tok:
                if ele == pad:
                    copy = np.zeros(length)
                else:
                    copy = tok.copy()
                    copy[copy != 1000] = 0
                    copy[copy == 1000] = 1
                if ele == 1000:
                    copy[copy == 0] = -1
                    copy[copy == 1] = 0
                    copy[copy == -1] = 1
                copy = np.expand_dims(copy, -1)
                mask = np.concatenate((mask, copy), axis=1)
            mask = mask[:, 1:]
            mask = mask.transpose()
            mask = np.expand_dims(mask, 0)
            lis.append(mask)
        res = np.concatenate(tuple(lis))
    else:
        return 'error'
    return res


class GroupAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(GroupAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def get_mask(self, src, split_list, pad=0):
        """
        Args:
            src (torch.Tensor): source sequence, shape [batch_size, sequence_length].
            split_list (list): group split index.
            pad (int): pad token index.
        
        Returns:
            torch.Tensor: group attention mask, shape [batch_size, 4, sequence_length, sequence_length].
        """
        device = src.device
        mask = self.src_to_mask(src, split_list)
        self.src_mask_self = torch.from_numpy(group_mask(mask, 'self', pad).astype('uint8')).unsqueeze(1)
        self.src_mask_between = torch.from_numpy(group_mask(mask, 'between', pad).astype('uint8')).unsqueeze(1)
        self.src_mask_question = torch.from_numpy(group_mask(mask, 'question', pad).astype('uint8')).unsqueeze(1)
        self.src_mask_global = (src != pad).unsqueeze(-2).unsqueeze(1)
        self.src_mask_global = self.src_mask_global.expand(self.src_mask_self.shape)
        self.final = torch.cat((self.src_mask_between.bool(), self.src_mask_self.bool(), self.src_mask_global, self.src_mask_question.bool()), 1)
        return self.final

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (torch.Tensor): shape [batch_size, head_nums, sequence_length, dim_k].
            key (torch.Tensor): shape [batch_size, head_nums, sequence_length, dim_k].
            value (torch.Tensor): shape [batch_size, head_nums, sequence_length, dim_k].
            mask (torch.Tensor): group attention mask, shape [batch_size, head_nums, sequence_length, sequence_length].
        
        Returns:
            torch.Tensor: shape [batch_size, sequence_length, hidden_size].
        """
        if mask is not None and len(mask.shape) < 4:
            mask = mask.unsqueeze(1)
        else:
            mask = torch.cat((mask, mask), 1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def src_to_mask(self, src, split_list):
        src = src.cpu().numpy()
        batch_data_mask_tok = []
        for encode_sen_idx in src:
            token = 1
            mask = [0] * len(encode_sen_idx)
            for num in range(len(encode_sen_idx)):
                mask[num] = token
                if encode_sen_idx[num] in split_list and num != len(encode_sen_idx) - 1:
                    token += 1
                if encode_sen_idx[num] == 0:
                    mask[num] = 0
            for num in range(len(encode_sen_idx)):
                if mask[num] == token and token != 1:
                    mask[num] = 1000
            batch_data_mask_tok.append(mask)
        return np.array(batch_data_mask_tok)


class GroupAttentionRNNEncoder(nn.Module):
    """Group Attentional Recurrent Neural Network (RNN) encoder.
    """

    def __init__(self, emb_size=100, hidden_size=128, n_layers=1, bidirectional=False, rnn_cell=None, rnn_cell_name='gru', variable_lengths=True, d_ff=2048, dropout=0.3, N=1):
        super(GroupAttentionRNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        self.dropout = dropout
        if bidirectional:
            self.d_model = 2 * hidden_size
        else:
            self.d_model = hidden_size
        ff = PositionwiseFeedForward(self.d_model, d_ff, dropout)
        if rnn_cell_name.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell_name.lower() == 'gru':
            self.rnn_cell = nn.GRU
        if rnn_cell is None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional, dropout=self.dropout)
        else:
            self.rnn = rnn_cell
        self.group_attention = GroupAttention(8, self.d_model)
        self.onelayer = GroupATTEncoder(GAEncoderLayer(self.d_model, deepcopy(self.group_attention), deepcopy(ff), dropout), N)

    def forward(self, embedded, input_var, split_list, input_lengths=None):
        """
        Args:
            embedded (torch.Tensor): embedded inputs, shape [batch_size, sequence_length, embedding_size].
            input_var (torch.Tensor): source sequence, shape [batch_size, sequence_length].
            split_list (list): group split index.
            input_lengths (torch.Tensor): length of input sequence, shape: [batch_size].
        
        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        
        """
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=True)
        else:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        src_mask = self.group_attention.get_mask(input_var, split_list)
        output = self.onelayer(output, src_mask)
        return output, hidden


class GroupATT(nn.Module):
    """
    Reference:
        Li et al. "Modeling Intra-Relation in Math Word Problems with Different Functional Multi-Head Attentions" in ACL 2019.
    """

    def __init__(self, config, dataset):
        super(GroupATT, self).__init__()
        self.device = config['device']
        self.bidirectional = config['bidirectional']
        self.hidden_size = config['hidden_size']
        self.decode_hidden_size = config['decode_hidden_size']
        self.encoder_rnn_cell_type = config['encoder_rnn_cell_type']
        self.decoder_rnn_cell_type = config['decoder_rnn_cell_type']
        self.attention = config['attention']
        self.share_vocab = config['share_vocab']
        self.max_gen_len = config['max_output_len']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.embedding_size = config['embedding_size']
        self.num_layers = config['num_layers']
        self.dropout_ratio = config['dropout_ratio']
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.mask_list = NumMask.number
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.in_word2idx = dataset.in_word2idx
        self.in_idx2word = dataset.in_idx2word
        if self.share_vocab:
            self.sos_token_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.split_list = []
        try:
            self.split_list.append(self.in_word2idx['．'])
        except:
            pass
        try:
            self.split_list.append(self.in_word2idx['，'])
        except:
            pass
        try:
            self.split_list.append(self.in_word2idx['.'])
        except:
            pass
        try:
            self.split_list.append(self.in_word2idx[','])
        except:
            pass
        self.in_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        if self.share_vocab:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BasicEmbedder(self.symbol_size, self.embedding_size, self.dropout_ratio)
        self.encoder = GroupAttentionRNNEncoder(emb_size=self.embedding_size, hidden_size=self.hidden_size, n_layers=self.num_layers, bidirectional=self.bidirectional, rnn_cell=None, rnn_cell_name=self.encoder_rnn_cell_type, variable_lengths=False, d_ff=2048, dropout=self.dropout_ratio, N=1)
        self.decoder = AttentionalRNNDecoder(self.embedding_size, self.decode_hidden_size, self.hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.generate_linear = nn.Linear(self.decode_hidden_size, self.symbol_size)
        weight = torch.ones(self.symbol_size)
        pad = self.out_pad_token
        self.loss = NLLLoss(weight, pad)

    def process_gap_encoder_decoder(self, encoder_hidden):
        if self.encoder_rnn_cell_type == 'lstm' and self.decoder_rnn_cell_type == 'lstm':
            """ lstm -> lstm """
            encoder_hidden = self._init_state(encoder_hidden)
        elif self.encoder_rnn_cell_type == 'gru' and self.decoder_rnn_cell_type == 'gru':
            """ gru -> gru """
            encoder_hidden = self._init_state(encoder_hidden)
        elif self.encoder_rnn_cell_type == 'gru' and self.decoder_rnn_cell_type == 'lstm':
            """ gru -> lstm """
            encoder_hidden = encoder_hidden, encoder_hidden
            encoder_hidden = self._init_state(encoder_hidden)
        elif self.encoder_rnn_cell_type == 'lstm' and self.decoder_rnn_cell_type == 'gru':
            """ lstm -> gru """
            encoder_hidden = encoder_hidden[0]
            encoder_hidden = self._init_state(encoder_hidden)
        return encoder_hidden

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.encoder.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def forward(self, seq, seq_length, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.in_embedder(seq)
        encoder_outputs, encoder_hidden, encoder_layer_outputs = self.encoder_forward(seq_emb, seq, seq_length, output_all_layers)
        decoder_inputs = self.init_decoder_inputs(target, device, batch_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, encoder_hidden, decoder_inputs, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.
        
        :param batch_data: one batch data. batch_data should include keywords 'question', 'ques len', 'equation'.
        :return: loss value.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        token_logits, _, _ = self.forward(seq, seq_length, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.loss.reset()
        self.loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.
        
        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation' and 'num list'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        num_list = batch_data['num list']
        _, symbol_outputs, _ = self.forward(seq, seq_length)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(symbol_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, seq, seq_length, output_all_layers=False):
        encoder_outputs, encoder_hidden = self.encoder.forward(seq_emb, seq, self.split_list, seq_length)
        encoder_hidden = self.process_gap_encoder_decoder(encoder_hidden)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, all_layer_outputs

    def decoder_forward(self, encoder_outputs, encoder_hidden, decoder_inputs, target=None, output_all_layers=False):
        if target is not None and random.random() < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden)
            token_logits = self.generate_linear(decoder_outputs)
            outputs = token_logits.topk(1, dim=-1)[1]
        else:
            seq_len = decoder_inputs.size(1) if target is not None else self.max_gen_len
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            decoder_outputs = []
            token_logits = []
            outputs = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                step_output = decoder_output.squeeze(1)
                token_logit = self.generate_linear(step_output)
                output = token_logit.topk(1, dim=-1)[1]
                decoder_outputs.append(step_output)
                token_logits.append(token_logit)
                outputs.append(output)
                if self.share_vocab:
                    output = self.convert_out_idx_2_in_idx(output)
                    decoder_input = self.out_embedder(output)
                else:
                    decoder_input = self.out_embedder(output)
            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            token_logits = torch.stack(token_logits, dim=1)
            outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.out_embedder(decoder_inputs)
        return decoder_inputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def convert_out_idx_2_in_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_list[b_i])
            res = []
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class LSTM(nn.Module):

    def __init__(self, config, dataset):
        super(LSTM, self).__init__()
        self.device = config['device']
        self.bidirectional = config['bidirectional']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.decode_hidden_size = config['decode_hidden_size']
        self.dropout_ratio = config['dropout_ratio']
        self.attention = config['attention']
        self.num_layers = config['num_layers']
        self.share_vocab = config['share_vocab']
        self.max_gen_len = config['max_output_len']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.encoder_rnn_cell_type = 'lstm'
        self.decoder_rnn_cell_type = 'lstm'
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.mask_list = NumMask.number
        if self.share_vocab:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.in_word2idx = dataset.in_word2idx
            self.in_idx2word = dataset.in_idx2word
            self.sos_token_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.in_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        if self.share_vocab:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BasicEmbedder(self.symbol_size, self.embedding_size, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.encoder_rnn_cell_type, self.dropout_ratio, self.bidirectional)
        if self.attention:
            self.decoder = AttentionalRNNDecoder(self.embedding_size, self.decode_hidden_size, self.hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size, self.decode_hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.generate_linear = nn.Linear(self.hidden_size, self.symbol_size)
        weight = torch.ones(self.symbol_size)
        pad = self.out_pad_token
        self.loss = NLLLoss(weight, pad)

    def forward(self, seq, seq_length, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param seq: torch.Tensor, shape of [batch_size, seq_length]
        :param seq_length: torch.Tensor, the length of seq, shape of [batch_size]
        :param target: torch.Tensor | None, shape should be [batch_size, target_length] if target is not None.
        :param output_all_layers: bool, default False, return output of all layers if output_all_layers is True
        :return: token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.in_embedder(seq)
        encoder_outputs, encoder_hidden, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, output_all_layers)
        decoder_inputs = self.init_decoder_inputs(target, device, batch_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, encoder_hidden, decoder_inputs, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        token_logits, _, _ = self.forward(seq, seq_length, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.loss.reset()
        self.loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation' and 'num list'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        num_list = batch_data['num list']
        _, symbol_outputs, _ = self.forward(seq, seq_length, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(symbol_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_outputs, encoder_hidden = self.encoder(seq_emb, seq_length)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            encoder_hidden = encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous()
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, all_layer_outputs

    def decoder_forward(self, encoder_outputs, encoder_hidden, decoder_inputs, target=None, output_all_layers=False):
        with_t = random.random()
        if target is not None and with_t < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden)
            token_logits = self.generate_linear(decoder_outputs)
            outputs = token_logits.topk(1, dim=-1)[1]
        else:
            seq_len = decoder_inputs.size(1) if target is not None else self.max_gen_len
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            decoder_outputs = []
            token_logits = []
            outputs = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                step_output = decoder_output.squeeze(1)
                token_logit = self.generate_linear(step_output)
                output = token_logit.topk(1, dim=-1)[1]
                decoder_outputs.append(step_output)
                token_logits.append(token_logit)
                outputs.append(output)
                if self.share_vocab:
                    output = self.convert_out_idx_2_in_idx(output)
                    decoder_input = self.out_embedder(output)
                else:
                    decoder_input = self.out_embedder(output)
            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            token_logits = torch.stack(token_logits, dim=1)
            outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.out_embedder(decoder_inputs)
        return decoder_inputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def convert_out_idx_2_in_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_list[b_i])
            res = []
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def __str__(self) ->str:
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class SelfAttentionRNNEncoder(nn.Module):
    """
    Self Attentional Recurrent Neural Network (RNN) encoder.
    """

    def __init__(self, embedding_size, hidden_size, context_size, num_layers, rnn_cell_type, dropout_ratio, bidirectional=True):
        super(SelfAttentionRNNEncoder, self).__init__()
        self.rnn_cell_type = rnn_cell_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        if rnn_cell_type == 'lstm':
            self.encoder = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'gru':
            self.encoder = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        elif rnn_cell_type == 'rnn':
            self.encoder = nn.RNN(embedding_size, hidden_size, num_layers, batch_first=True, dropout=dropout_ratio, bidirectional=bidirectional)
        else:
            raise ValueError("The RNN type of encoder must be in ['lstm', 'gru', 'rnn'].")
        self.attention = SeqAttention(hidden_size, context_size)

    def init_hidden(self, input_embeddings):
        """ Initialize initial hidden states of RNN.

        Args:
            input_embeddings (torch.Tensor): input sequence embedding, shape: [batch_size, sequence_length, embedding_size].

        Returns:
            torch.Tensor: the initial hidden states.
        """
        batch_size = input_embeddings.size(0)
        device = input_embeddings.device
        if self.rnn_cell_type == 'lstm':
            h_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            c_0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            hidden_states = h_0, c_0
            return hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            tp_vec = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size)
            return tp_vec
        else:
            raise NotImplementedError('No such rnn type {} for initializing encoder states.'.format(self.rnn_type))

    def forward(self, input_embeddings, input_length, hidden_states=None):
        """ Implement the encoding process.

        Args:
            input_embeddings (torch.Tensor): source sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            input_length (torch.Tensor): length of input sequence, shape: [batch_size].
            hidden_states (torch.Tensor): initial hidden states, default: None.

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                output features, shape: [batch_size, sequence_length, num_directions * hidden_size].
                hidden states, shape: [batch_size, num_layers * num_directions, hidden_size].
        """
        if hidden_states is None:
            hidden_states = self.init_hidden(input_embeddings)
        packed_input_embeddings = torch.nn.utils.rnn.pack_padded_sequence(input_embeddings, input_length, batch_first=True, enforce_sorted=True)
        outputs, hidden_states = self.encoder(packed_input_embeddings, hidden_states)
        outputs, outputs_length = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if self.bidirectional:
            encoder_outputs = outputs[:, :, self.hidden_size:] + outputs[:, :, :self.hidden_size]
            if self.rnn_cell_type == 'lstm':
                encoder_hidden = hidden_states[0][::2].contiguous(), hidden_states[1][::2].contiguous()
            else:
                encoder_hidden = hidden_states[::2].contiguous()
        outputs, attn = self.attention.forward(encoder_outputs, encoder_outputs, mask=None)
        return outputs, hidden_states


class MathEN(nn.Module):
    """
    Reference:
        Wang et al. "Translating a Math Word Problem to a Expression Tree" in EMNLP 2018.
    """

    def __init__(self, config, dataset):
        super(MathEN, self).__init__()
        self.device = config['device']
        self.bidirectional = config['bidirectional']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.decode_hidden_size = config['decode_hidden_size']
        self.dropout_ratio = config['dropout_ratio']
        self.attention = config['attention']
        self.num_layers = config['num_layers']
        self.share_vocab = config['share_vocab']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.encoder_rnn_cell_type = config['encoder_rnn_cell_type']
        self.decoder_rnn_cell_type = config['decoder_rnn_cell_type']
        self.self_attention = config['self_attention']
        self.max_gen_len = config['max_output_len']
        self.embedding = config['embedding']
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.mask_list = NumMask.number
        if self.share_vocab:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.in_word2idx = dataset.in_word2idx
            self.in_idx2word = dataset.in_idx2word
            self.sos_token_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        if config['embedding'] == 'roberta':
            self.in_embedder = RobertaEmbedder(self.vocab_size, config['pretrained_model_path'])
        elif config['embedding'] == 'bert':
            self.in_embedder = BertEmbedder(self.vocab_size, config['pretrained_model_path'])
        else:
            self.in_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        if self.share_vocab:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BasicEmbedder(self.symbol_size, self.embedding_size, self.dropout_ratio)
        if self.self_attention:
            self.encoder = SelfAttentionRNNEncoder(self.embedding_size, self.hidden_size, self.hidden_size, self.num_layers, self.encoder_rnn_cell_type, self.dropout_ratio, self.bidirectional)
        else:
            self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.encoder_rnn_cell_type, self.dropout_ratio, self.bidirectional)
        if self.attention:
            self.decoder = AttentionalRNNDecoder(self.embedding_size, self.decode_hidden_size, self.hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size, self.decode_hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.generate_linear = nn.Linear(self.hidden_size, self.symbol_size)
        weight = torch.ones(self.symbol_size)
        pad = self.out_pad_token
        self.loss = NLLLoss(weight, pad)

    def forward(self, seq, seq_length, seq_mask=None, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param torch.Tensor | None seq_mask: mask of sequence, shape: [batch_size, seq_length], default None.
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        batch_size = seq.size(0)
        device = seq.device
        if self.embedding == 'roberta':
            seq_emb = self.in_embedder(seq, seq_mask)
        else:
            seq_emb = self.in_embedder(seq)
        encoder_outputs, encoder_hidden, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, output_all_layers)
        decoder_inputs = self.init_decoder_inputs(target, device, batch_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, encoder_hidden, decoder_inputs, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.
        
        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'ques mask'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        seq_mask = torch.BoolTensor(batch_data['ques mask'])
        token_logits, _, _ = self.forward(seq, seq_length, seq_mask, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.loss.reset()
        self.loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation', 'num list', 'ques mask'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        num_list = batch_data['num list']
        seq_mask = torch.BoolTensor(batch_data['ques mask'])
        _, symbol_outputs, _ = self.forward(seq, seq_length, seq_mask)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(symbol_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_outputs, encoder_hidden = self.encoder(seq_emb, seq_length)
        if self.self_attention:
            pass
        elif self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
        if self.bidirectional:
            if self.encoder_rnn_cell_type == 'lstm':
                encoder_hidden = encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous()
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        if self.encoder.rnn_cell_type == self.decoder.rnn_cell_type:
            pass
        elif self.encoder.rnn_cell_type == 'gru' and self.decoder.rnn_cell_type == 'lstm':
            encoder_hidden = encoder_hidden, encoder_hidden
        elif self.encoder.rnn_cell_type == 'rnn' and self.decoder.rnn_cell_type == 'lstm':
            encoder_hidden = encoder_hidden, encoder_hidden
        elif self.encoder.rnn_cell_type == 'lstm' and (self.decoder.rnn_cell_type == 'gru' or self.decoder.rnn_cell_type == 'rnn'):
            encoder_hidden = encoder_hidden[0]
        else:
            pass
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, all_layer_outputs

    def decoder_forward(self, encoder_outputs, encoder_hidden, decoder_inputs, target=None, output_all_layers=False):
        if target is not None and random.random() < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden)
            token_logits = self.generate_linear(decoder_outputs)
            outputs = token_logits.topk(1, dim=-1)[1]
        else:
            seq_len = decoder_inputs.size(1) if target is not None else self.max_gen_len
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            decoder_outputs = []
            token_logits = []
            outputs = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                step_output = decoder_output.squeeze(1)
                token_logit = self.generate_linear(step_output)
                output = token_logit.topk(1, dim=-1)[1]
                decoder_outputs.append(step_output)
                token_logits.append(token_logit)
                outputs.append(output)
                if self.share_vocab:
                    output = self.convert_out_idx_2_in_idx(output)
                    decoder_input = self.out_embedder(output)
                else:
                    decoder_input = self.out_embedder(output)
            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            token_logits = torch.stack(token_logits, dim=1)
            outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.out_embedder(decoder_inputs)
        return decoder_inputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def convert_out_idx_2_in_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            num_len = len(num_list[b_i])
            res = []
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def __str__(self) ->str:
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class RNNEncDec(nn.Module):
    """
    Reference:
        Sutskever et al. "Sequence to Sequence Learning with Neural Networks".
    """

    def __init__(self, config, dataset):
        super(RNNEncDec, self).__init__()
        self.device = config['device']
        self.bidirectional = config['bidirectional']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.decode_hidden_size = config['decode_hidden_size']
        self.dropout_ratio = config['dropout_ratio']
        self.attention = config['attention']
        self.num_layers = config['num_layers']
        self.share_vocab = config['share_vocab']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.encoder_rnn_cell_type = config['encoder_rnn_cell_type']
        self.decoder_rnn_cell_type = config['decoder_rnn_cell_type']
        self.self_attention = config['self_attention']
        self.max_gen_len = config['max_output_len']
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.mask_list = NumMask.number
        if self.share_vocab:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.in_word2idx = dataset.in_word2idx
            self.in_idx2word = dataset.in_idx2word
            self.sos_token_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.in_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        if self.share_vocab:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BasicEmbedder(self.symbol_size, self.embedding_size, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.encoder_rnn_cell_type, self.dropout_ratio)
        if self.attention:
            self.decoder = AttentionalRNNDecoder(self.embedding_size, self.decode_hidden_size, self.hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size, self.decode_hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.dropout_ratio)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.generate_linear = nn.Linear(self.hidden_size, self.symbol_size)
        weight = torch.ones(self.symbol_size)
        pad = self.out_pad_token
        self.loss = NLLLoss(weight, pad)

    def forward(self, seq, seq_length, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.in_embedder(seq)
        encoder_outputs, encoder_hidden, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, output_all_layers)
        decoder_inputs = self.init_decoder_inputs(target, device, batch_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, encoder_hidden, decoder_inputs, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.
        
        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        token_logits, _, _ = self.forward(seq, seq_length, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.loss.reset()
        self.loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation' and 'num list'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        num_list = batch_data['num list']
        _, symbol_outputs, _ = self.forward(seq, seq_length)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(symbol_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.out_embedder(decoder_inputs)
        return decoder_inputs

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output

    def convert_out_idx_2_in_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_outputs, encoder_hidden = self.encoder(seq_emb, seq_length)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if self.encoder_rnn_cell_type == 'lstm':
                encoder_hidden = encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous()
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        if self.encoder.rnn_cell_type == self.decoder.rnn_cell_type:
            pass
        elif self.encoder.rnn_cell_type == 'gru' and self.decoder.rnn_cell_type == 'lstm':
            encoder_hidden = encoder_hidden, encoder_hidden
        elif self.encoder.rnn_cell_type == 'rnn' and self.decoder.rnn_cell_type == 'lstm':
            encoder_hidden = encoder_hidden, encoder_hidden
        elif self.encoder.rnn_cell_type == 'lstm' and (self.decoder.rnn_cell_type == 'gru' or self.decoder.rnn_cell_type == 'rnn'):
            encoder_hidden = encoder_hidden[0]
        else:
            pass
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, all_layer_outputs

    def decoder_forward(self, encoder_outputs, encoder_hidden, decoder_inputs, target=None, output_all_layers=False):
        if target is not None and random.random() < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.decoder(decoder_inputs, encoder_hidden)
            token_logits = self.generate_linear(decoder_outputs)
            outputs = token_logits.topk(1, dim=-1)[1]
        else:
            seq_len = decoder_inputs.size(1) if target is not None else self.max_gen_len
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            decoder_outputs = []
            token_logits = []
            outputs = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                step_output = decoder_output.squeeze(1)
                token_logit = self.generate_linear(step_output)
                output = token_logit.topk(1, dim=-1)[1]
                decoder_outputs.append(step_output)
                token_logits.append(token_logit)
                outputs.append(output)
                if self.share_vocab:
                    output = self.convert_out_idx_2_in_idx(output)
                    decoder_input = self.out_embedder(output)
                else:
                    decoder_input = self.out_embedder(output)
            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            token_logits = torch.stack(token_logits, dim=1)
            outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def __str__(self) ->str:
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class RNNVAE(nn.Module):
    """
    Reference:
        Zhang et al. "Variational Neural Machine Translation".
    
    We apply translation machine based rnnvae to math word problem task.
    """

    def __init__(self, config, dataset):
        super(RNNVAE, self).__init__()
        self.device = config['device']
        self.max_length = config['max_output_len']
        self.max_gen_len = config['max_output_len']
        self.share_vocab = config['share_vocab']
        self.num_directions = 2 if config['bidirectional'] else 1
        self.rnn_cell_type = config['rnn_cell_type']
        self.bidirectional = config['bidirectional']
        self.attention = config['attention']
        self.embedding_size = config['embedding_size']
        self.latent_size = config['latent_size']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.hidden_size = config['hidden_size']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.dropout_ratio = config['dropout_ratio']
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.padding_token_idx = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        self.mask_list = NumMask.number
        if self.share_vocab:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.in_word2idx = dataset.in_word2idx
            self.in_idx2word = dataset.in_idx2word
            self.sos_token_idx = self.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.out_symbol2idx = dataset.out_symbol2idx
            self.out_idx2symbol = dataset.out_idx2symbol
            self.sos_token_idx = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.in_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        if self.share_vocab:
            self.out_embedder = self.in_embedder
        else:
            self.out_embedder = BasicEmbedder(self.symbol_size, self.embedding_size, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_encoder_layers, self.rnn_cell_type, self.dropout_ratio, self.bidirectional)
        if self.attention:
            self.decoder = AttentionalRNNDecoder(self.embedding_size + self.latent_size, self.hidden_size, self.hidden_size, self.num_decoder_layers, self.rnn_cell_type, self.dropout_ratio)
        else:
            self.decoder = BasicRNNDecoder(self.embedding_size + self.latent_size, self.hidden_size, self.num_decoder_layers, self.rnn_cell_type, self.dropout_ratio)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.out = nn.Linear(self.hidden_size, self.symbol_size)
        if self.rnn_cell_type == 'lstm':
            self.hidden_to_mean = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.latent_to_hidden = nn.Linear(self.latent_size, 2 * self.hidden_size)
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            self.hidden_to_mean = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.hidden_to_logvar = nn.Linear(self.num_directions * self.hidden_size, self.latent_size)
            self.latent_to_hidden = nn.Linear(self.latent_size, 2 * self.hidden_size)
        else:
            raise ValueError('No such rnn type {} for RNNVAE.'.format(self.rnn_cell_type))
        weight = torch.ones(self.symbol_size)
        pad = self.out_pad_token
        self.loss = NLLLoss(weight, pad)

    def forward(self, seq, seq_length, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.in_embedder(seq)
        encoder_outputs, encoder_hidden, z, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, output_all_layers)
        decoder_inputs = self.init_decoder_inputs(target, device, batch_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, encoder_hidden, decoder_inputs, z, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.
        
        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        token_logits, _, _ = self.forward(seq, seq_length, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.loss.reset()
        self.loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation' and 'num list'.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        num_list = batch_data['num list']
        _, symbol_outputs, _ = self.forward(seq, seq_length)
        if self.share_vocab:
            target = self.convert_in_idx_2_out_idx(target)
        all_outputs = self.convert_idx2symbol(symbol_outputs, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        return all_outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        batch_size = seq_emb.size(0)
        device = seq_emb.device
        encoder_outputs, hidden_states = self.encoder(seq_emb, seq_length)
        if self.rnn_cell_type == 'lstm':
            h_n, c_n = hidden_states
        elif self.rnn_cell_type == 'gru' or self.rnn_cell_type == 'rnn':
            h_n = hidden_states
        else:
            raise NotImplementedError('No such rnn type {} for RNNVAE.'.format(self.rnn_cell_type))
        if self.bidirectional:
            h_n = h_n.view(self.num_encoder_layers, 2, batch_size, self.hidden_size)
            h_n = h_n[-1]
            h_n = torch.cat([h_n[0], h_n[1]], dim=1)
        else:
            h_n = h_n[-1]
        mean = self.hidden_to_mean(h_n)
        logvar = self.hidden_to_logvar(h_n)
        z = torch.randn([batch_size, self.latent_size])
        z = mean + z * torch.exp(0.5 * logvar)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.hidden_size:] + encoder_outputs[:, :, :self.hidden_size]
            if self.rnn_cell_type == 'lstm':
                hidden_states = hidden_states[0][::2].contiguous(), hidden_states[1][::2].contiguous()
            else:
                hidden_states = hidden_states[::2].contiguous()
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = hidden_states
            all_layer_outputs['mean'] = mean
            all_layer_outputs['logvar'] = logvar
            all_layer_outputs['z'] = z
        return encoder_outputs, hidden_states, z, all_layer_outputs

    def decoder_forward(self, encoder_outputs, encoder_hidden, decoder_inputs, z, target=None, output_all_layers=False):
        decoder_hidden = encoder_hidden
        if target is not None and random.random() < self.teacher_force_ratio:
            decoder_inputs = torch.cat((decoder_inputs, z.unsqueeze(1).repeat(1, decoder_inputs.size(1), 1)), dim=2)
            decoder_outputs, decoder_hidden = self.decoder(input_embeddings=decoder_inputs, hidden_states=decoder_hidden, encoder_outputs=encoder_outputs)
            token_logits = self.out(decoder_outputs)
            outputs = token_logits.topk(1, dim=-1)[1]
        else:
            seq_len = decoder_inputs.size(1) if target is not None else self.max_gen_len
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            decoder_input = torch.cat((decoder_input, z.unsqueeze(1).repeat(1, decoder_input.size(1), 1)), dim=2)
            decoder_outputs = []
            token_logits = []
            outputs = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                step_output = decoder_output.squeeze(1)
                token_logit = self.out(step_output)
                output = token_logit.topk(1, dim=-1)[1]
                decoder_outputs.append(step_output)
                token_logits.append(token_logit)
                outputs.append(output)
                if self.share_vocab:
                    output = self.convert_out_idx_2_in_idx(output)
                    decoder_input = self.out_embedder(output)
                else:
                    decoder_input = self.out_embedder(output)
                decoder_input = torch.cat((decoder_input, z.unsqueeze(1).repeat(1, decoder_input.size(1), 1)), dim=2)
            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            token_logits = torch.stack(token_logits, dim=1)
            outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def init_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).view(batch_size, 1)
        if target != None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.out_embedder(decoder_inputs)
        return decoder_inputs

    def convert_out_idx_2_in_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def decode(self, output):
        device = output.device
        batch_size = output.size(0)
        decoded_output = []
        for idx in range(batch_size):
            decoded_output.append(self.in_word2idx[self.out_idx2symbol[output[idx]]])
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return output


class OPERATIONS:

    def __init__(self, out_symbol2idx):
        self.NOOP = -1
        self.GEN_VAR = -2
        self.PAD = out_symbol2idx['<PAD>']
        self.ADD = out_symbol2idx['+']
        self.SUB = out_symbol2idx['-']
        self.MUL = out_symbol2idx['*']
        self.DIV = out_symbol2idx['/']
        self.POWER = out_symbol2idx['^']
        self.RAW_EQL = out_symbol2idx['='] if '=' in out_symbol2idx else -1
        self.BRG = out_symbol2idx['<BRG>'] if '<BRG>' in out_symbol2idx else -1
        self.EQL = out_symbol2idx['<EOS>']
        self.N_OPS = out_symbol2idx['NUM_0']


class RelevantScore(nn.Module):

    def __init__(self, dim_value, dim_query, hidden1, dropout_rate=0):
        super(RelevantScore, self).__init__()
        self.lW1 = nn.Linear(dim_value, hidden1, bias=False)
        self.lW2 = nn.Linear(dim_query, hidden1, bias=False)
        self.b = nn.Parameter(torch.normal(torch.zeros(1, 1, hidden1), 0.01))
        self.tanh = nn.Tanh()
        self.lw = nn.Linear(hidden1, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, value, query):
        """
        Args:
            value (torch.FloatTensor): shape [batch, seq_len, dim_value].
            query (torch.FloatTensor): shape [batch, dim_query].
        """
        u = self.tanh(self.dropout(self.lW1(value) + self.lW2(query).unsqueeze(1) + self.b))
        return self.lw(u).squeeze(-1)


class MaskedRelevantScore(nn.Module):
    """ Relevant score masked by sequence lengths.

    Args:
        dim_value (int): Dimension of value.
        dim_query (int): Dimension of query.
        dim_hidden (int): Dimension of hidden layer in attention calculation.
    """

    def __init__(self, dim_value, dim_query, dim_hidden=256, dropout_rate=0.0):
        super(MaskedRelevantScore, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.relevant_score = RelevantScore(dim_value, dim_query, dim_hidden, dropout_rate)

    def forward(self, value, query, lens):
        """ Choose candidate from candidates.

        Args:
            query (torch.FloatTensor): Current hidden state, with size [batch_size, dim_query].
            value (torch.FloatTensor): Sequence to be attented, with size [batch_size, seq_len, dim_value].
            lens (list of int): Lengths of values in a batch.

        Return:
            torch.Tensor: Activation for each operand, with size [batch, max([len(os) for os in operands])].
        """
        relevant_scores = self.relevant_score(value, query)
        mask = torch.zeros_like(relevant_scores)
        for b, n_c in enumerate(lens):
            mask[b, n_c:] = -math.inf
        relevant_scores += mask
        return relevant_scores


class Attention(nn.Module):
    """ Calculate attention

    Args:
        dim_value (int): Dimension of value.
        dim_query (int): Dimension of query.
        dim_hidden (int): Dimension of hidden layer in attention calculation.
    """

    def __init__(self, dim_value, dim_query, dim_hidden=256, dropout_rate=0.5):
        super(Attention, self).__init__()
        self.relevant_score = MaskedRelevantScore(dim_value, dim_query, dim_hidden)

    def forward(self, value, query, lens):
        """ Generate variable embedding with attention.

        Args:
            query (FloatTensor): Current hidden state, with size [batch_size, dim_query].
            value (FloatTensor): Sequence to be attented, with size [batch_size, seq_len, dim_value].
            lens (list of int): Lengths of values in a batch.

        Return:
            FloatTensor: Calculated attention, with size [batch_size, dim_value].
        """
        relevant_scores = self.relevant_score(value, query, lens)
        e_relevant_scores = torch.exp(relevant_scores)
        weights = e_relevant_scores / e_relevant_scores.sum(-1, keepdim=True)
        attention = (weights.unsqueeze(-1) * value).sum(1)
        return attention


class Transformer(nn.Module):

    def __init__(self, dim_hidden):
        super(Transformer, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim_hidden, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_hidden), nn.Tanh())
        self.ret = nn.Parameter(torch.zeros(dim_hidden))
        nn.init.normal_(self.ret.data)

    def forward(self, top2):
        return self.mlp(top2)


class SalignedDecoder(nn.Module):

    def __init__(self, operations, dim_hidden=300, dropout_rate=0.5, device=None):
        super(SalignedDecoder, self).__init__()
        self.NOOP = operations.NOOP
        self.GEN_VAR = operations.GEN_VAR
        self.ADD = operations.ADD
        self.SUB = operations.SUB
        self.MUL = operations.MUL
        self.DIV = operations.DIV
        self.POWER = operations.POWER
        self.EQL = operations.EQL
        self.N_OPS = operations.N_OPS
        self.PAD = operations.PAD
        self.RAW_EQL = operations.RAW_EQL
        self.BRG = operations.BRG
        self._device = device
        self.transformer_add = Transformer(2 * dim_hidden)
        self.transformer_sub = Transformer(2 * dim_hidden)
        self.transformer_mul = Transformer(2 * dim_hidden)
        self.transformer_div = Transformer(2 * dim_hidden)
        self.transformer_power = Transformer(2 * dim_hidden)
        self.transformers = {self.ADD: self.transformer_add, self.SUB: self.transformer_sub, self.MUL: self.transformer_mul, self.DIV: self.transformer_div, self.POWER: self.transformer_power, self.RAW_EQL: None, self.BRG: None}
        self.gen_var = Attention(2 * dim_hidden, dim_hidden, dropout_rate=0.0)
        self.attention = Attention(2 * dim_hidden, dim_hidden, dropout_rate=dropout_rate)
        self.choose_arg = MaskedRelevantScore(dim_hidden * 2, dim_hidden * 7, dropout_rate=dropout_rate)
        self.arg_gate = torch.nn.Linear(dim_hidden * 7, 3, torch.nn.Sigmoid())
        self.rnn = torch.nn.LSTM(2 * dim_hidden, dim_hidden, 1, batch_first=True)
        self.op_selector = torch.nn.Sequential(torch.nn.Linear(dim_hidden * 7, 256), torch.nn.ReLU(), torch.nn.Dropout(dropout_rate), torch.nn.Linear(256, self.N_OPS + 1))
        self.op_gate = torch.nn.Linear(dim_hidden * 7, 3, torch.nn.Sigmoid())
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.register_buffer('noop_padding_return', torch.zeros(dim_hidden * 2))
        self.register_buffer('padding_embedding', torch.zeros(dim_hidden * 2))

    def forward(self, context, text_len, operands, stacks, prev_op, prev_output, prev_state, number_emb, N_OPS):
        """
        Args:
            context (torch.Tensor): Encoded context, with size [batch_size, text_len, dim_hidden].
            text_len (torch.Tensor): Text length for each problem in the batch.
            operands (list of torch.Tensor): List of operands embeddings for each problem in the batch. Each element in the list is of size [n_operands, dim_hidden].
            stacks (list of StackMachine): List of stack machines used for each problem.
            prev_op (torch.LongTensor): Previous operation, with size [batch, 1].
            prev_arg (torch.LongTensor): Previous argument indices, with size [batch, 1]. Can be None for the first step.
            prev_output (torch.Tensor): Previous decoder RNN outputs, with size [batch, dim_hidden]. Can be None for the first step.
            prev_state (torch.Tensor): Previous decoder RNN state, with size [batch, dim_hidden]. Can be None for the first step.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
            op_logits: Logits of operation selection.
            arg_logits: Logits of argument choosing.
            outputs: Outputs of decoder RNN.
            state: Hidden state of decoder RNN.
        """
        batch_size = context.size(0)
        stack_states = torch.stack([stack.get_top2().view(-1) for stack in stacks], dim=0)
        if prev_output is not None:
            batch_result = {self.ADD: self.transformer_add(stack_states), self.SUB: self.transformer_sub(stack_states), self.MUL: self.transformer_mul(stack_states), self.DIV: self.transformer_div(stack_states), self.POWER: self.transformer_power(stack_states)}
        prev_returns = []
        for b in range(batch_size):
            if prev_op[b].item() == self.NOOP:
                ret = self.noop_padding_return
            elif prev_op[b].item() == self.PAD:
                ret = self.noop_padding_return
            elif prev_op[b].item() == self.GEN_VAR:
                variable = batch_result[self.GEN_VAR][b]
                operands[b].append(variable)
                stacks[b].add_variable(variable)
                ret = variable
            elif prev_op[b].item() in [self.ADD, self.SUB, self.MUL, self.DIV, self.POWER]:
                transformed = batch_result[prev_op[b].item()][b]
                ret = stacks[b].apply_embed_only(prev_op[b].item(), transformed)
            elif prev_op[b].item() == self.EQL:
                ret = stacks[b].apply_eql(prev_op[b].item())
            else:
                stacks[b].push(prev_op[b].item() - N_OPS)
                ret = number_emb[b][prev_op[b].item() - N_OPS]
            prev_returns.append(ret)
        stack_states = torch.stack([stack.get_top2().view(-1) for stack in stacks], dim=0)
        prev_returns = torch.stack(prev_returns)
        prev_returns = self.dropout(prev_returns)
        outputs, hidden_state = self.rnn(prev_returns.unsqueeze(1), prev_state)
        outputs = outputs.squeeze(1)
        attention = self.attention(context, outputs, text_len)
        gate_in = torch.cat([outputs, stack_states, attention], -1)
        op_gate_in = self.dropout(gate_in)
        op_gate = self.op_gate(op_gate_in)
        arg_gate_in = self.dropout(gate_in)
        arg_gate = self.arg_gate(arg_gate_in)
        op_in = torch.cat([op_gate[:, 0:1] * outputs, op_gate[:, 1:2] * stack_states, op_gate[:, 2:3] * attention], -1)
        arg_in = torch.cat([arg_gate[:, 0:1] * outputs, arg_gate[:, 1:2] * stack_states, arg_gate[:, 2:3] * attention], -1)
        op_logits = self.op_selector(op_in)
        n_operands, cated_operands = self.pad_and_cat(operands, self.padding_embedding)
        arg_logits = self.choose_arg(cated_operands, arg_in, n_operands)
        return op_logits, arg_logits, outputs, hidden_state

    def pad_and_cat(self, tensors, padding):
        """ Pad lists to have same number of elements, and concatenate
        those elements to a 3d tensor.

        Args:
            tensors (list of list of Tensors): Each list contains
                list of operand embeddings. Each operand embedding is of
                size (dim_element,).
            padding (Tensor):
                Element used to pad lists, with size (dim_element,).

        Return:
            n_tensors (list of int): Length of lists in tensors.
            tensors (Tensor): Concatenated tensor after padding the list.
        """
        n_tensors = [len(ts) for ts in tensors]
        pad_size = max(n_tensors)
        tensors = [(ts + (pad_size - len(ts)) * [padding]) for ts in tensors]
        tensors = torch.stack([torch.stack(t) for t in tensors], dim=0)
        return n_tensors, tensors


class SalignedEncoder(nn.Module):
    """ Simple RNN encoder with attention which also extract variable embedding.
    """

    def __init__(self, dim_embed, dim_hidden, dim_last, dropout_rate, dim_attn_hidden=256):
        """
        Args:
            dim_embed (int): Dimension of input embedding.
            dim_hidden (int): Dimension of encoder RNN.
            dim_last (int): Dimension of the last state will be transformed to.
            dropout_rate (float): Dropout rate.
        """
        super(SalignedEncoder, self).__init__()
        self.rnn = torch.nn.LSTM(dim_embed, dim_hidden, 1, bidirectional=True, batch_first=True)
        self.mlp1 = torch.nn.Sequential(torch.nn.Linear(dim_hidden * 2, dim_last), torch.nn.Dropout(dropout_rate), torch.nn.Tanh())
        self.mlp2 = torch.nn.Sequential(torch.nn.Linear(dim_hidden * 2, dim_last), torch.nn.Dropout(dropout_rate), torch.nn.Tanh())
        self.attn = Attention_x(dim_hidden * 2, dim_hidden * 2, dim_attn_hidden)
        self.register_buffer('padding', torch.zeros(dim_hidden * 2))
        self.embeddings = torch.nn.Parameter(torch.normal(torch.zeros(20, 2 * dim_hidden), 0.01))
        self.dim_hidden = dim_hidden

    def initialize_fix_constant(self, con_len, device):
        self.embedding_con = [torch.nn.Parameter(torch.normal(torch.zeros(2 * self.dim_hidden), 0.01)) for c in range(con_len)]

    def get_fix_constant(self):
        return self.embedding_con

    def forward(self, inputs, lengths, constant_indices):
        """

        Args:
            inputs (torch.Tensor): Indices of words, shape [batch_size, sequence_length].
            length (torch.Tensor): Length of inputs, shape [batch_size].
            constant_indices (list of int): Each list contains list.

        Return:
            torch.Tensor: Encoded sequence, shape [batch_size, sequence_length, hidden_size].
        """
        packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True)
        hidden_state = None
        outputs, hidden_state = self.rnn(packed, hidden_state)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden_state = hidden_state[0].transpose(1, 0).contiguous().view(hidden_state[0].size(1), -1), hidden_state[1].transpose(1, 0).contiguous().view(hidden_state[1].size(1), -1)
        hidden_state = self.mlp1(hidden_state[0]).unsqueeze(0), self.mlp2(hidden_state[1]).unsqueeze(0)
        batch_size = outputs.size(0)
        operands = [[outputs[b][i] for i in constant_indices[b]] for b in range(batch_size)]
        return outputs, hidden_state, operands


class StackMachine:

    def __init__(self, operations, constants, embeddings, bottom_embedding, dry_run=False):
        """
        Args:
            constants (list): Value of numbers.
            embeddings (tensor): Tensor of shape [len(constants), dim_embedding].
                Embedding of the constants.
            bottom_embedding (teonsor): Tensor of shape (dim_embedding,). The
                embeding to return when stack is empty.
        """
        self._operands = list(constants)
        self._embeddings = [embedding for embedding in embeddings]
        self.operations = operations
        self._n_nuknown = 0
        self._stack = []
        self._equations = []
        self.stack_log = []
        self.stack_log_index = []
        self._val_funcs = {self.operations.ADD: sympy.Add, self.operations.SUB: lambda a, b: sympy.Add(-a, b), self.operations.MUL: sympy.Mul, self.operations.DIV: lambda a, b: sympy.Mul(1 / a, b), self.operations.POWER: lambda a, b: sympy.POW(a, b)}
        self._op_chars = {self.operations.ADD: '+', self.operations.SUB: '-', self.operations.MUL: '*', self.operations.DIV: '/', self.operations.POWER: '^', self.operations.RAW_EQL: '=', self.operations.BRG: '<BRG>', self.operations.EQL: '<EOS>'}
        self._bottom_embed = bottom_embedding
        if dry_run:
            self.apply = self.apply_embed_only

    def add_variable(self, embedding):
        """ Tell the stack machine to increase the number of nuknown variables
            by 1.

        Args:
            embedding (torch.Tensor): Tensor of shape (dim_embedding). Embedding
                of the unknown varialbe.
        """
        var = sympy.Symbol('x{}'.format(self._n_nuknown))
        self._operands.append(var)
        self._embeddings.append(embedding)
        self._n_nuknown += 1

    def push(self, operand_index):
        """ Push var to stack.

        Args:
            operand_index (int): Index of the operand. If index >= number of constants, then it implies a variable is pushed.
        
        Returns:
            torch.Tensor: Simply return the pushed embedding.
        """
        self._stack.append((self._operands[operand_index], self._embeddings[operand_index]))
        self.stack_log.append(self._operands[operand_index])
        self.stack_log_index.append(operand_index + self.operations.N_OPS)
        return self._embeddings[operand_index]

    def apply_embed_only(self, operation, embed_res):
        """ Apply operator on stack with embedding operation only.

        Args:
            operator (mwptoolkit.module.Environment.stack_machine.OPERATION): One of
                - OPERATIONS.ADD
                - OPERATIONS.SUB
                - OPERATIONS.MUL
                - OPERATIONS.DIV
                - OPERATIONS.EQL
            embed_res (torch.FloatTensor): Resulted embedding after transformation, with size (dim_embedding,).
        
        Returns:
            torch.Tensor: embedding on the top of the stack.
        """
        if len(self._stack) < 2:
            return self._bottom_embed
        val1, embed1 = self._stack.pop()
        val2, embed2 = self._stack.pop()
        if operation not in [self.operations.RAW_EQL, self.operations.BRG]:
            val_res = None
            self._stack.append((val_res, embed_res))
        self.stack_log.append(self._op_chars[operation])
        self.stack_log_index.append(operation)
        if len(self._stack) > 0:
            return self._stack[-1][1]
        else:
            return self._bottom_embed

    def apply_eql(self, operation):
        self.stack_log.append(self._op_chars[operation])
        self.stack_log_index.append(operation)
        return self._bottom_embed

    def get_solution(self):
        """ Get solution. If the problem has not been solved, return None.

        Returns:
            list: If the problem has been solved, return result from sympy.solve. If not, return None.
        """
        if self._n_nuknown == 0:
            return None
        try:
            root = sympy.solve(self._equations)
            for i in range(self._n_nuknown):
                if self._operands[-i - 1] not in root:
                    return None
            return root
        except:
            return None

    def get_top2(self):
        """ Get the top 2 embeddings of the stack.

        Return:
            torch.Tensor: Return tensor of shape (2, embed_dim).
        """
        if len(self._stack) >= 2:
            return torch.stack([self._stack[-1][1], self._stack[-2][1]], dim=0)
        elif len(self._stack) == 1:
            return torch.stack([self._stack[-1][1], self._bottom_embed], dim=0)
        else:
            return torch.stack([self._bottom_embed, self._bottom_embed], dim=0)

    def get_height(self):
        """ Get the height of the stack.

        Return:
            int: height.
        """
        return len(self._stack)

    def get_stack(self):
        return [self._bottom_embed] + [s[1] for s in self._stack]


class Saligned(nn.Module):
    """
    Reference:
        Chiang et al. "Semantically-Aligned Equation Generation for Solving and Reasoning Math Word Problems".
    """

    def __init__(self, config, dataset):
        super(Saligned, self).__init__()
        self.device = config['device']
        self.operations = operations = OPERATIONS(dataset.out_symbol2idx)
        self._vocab_size = vocab_size = len(dataset.in_idx2word)
        self._dim_embed = dim_embed = config['embedding_size']
        self._dim_hidden = dim_hidden = config['hidden_size']
        self._dropout_rate = dropout_rate = config['dropout_ratio']
        self.max_gen_len = 40
        self.NOOP = operations.NOOP
        self.GEN_VAR = operations.GEN_VAR
        self.ADD = operations.ADD
        self.SUB = operations.SUB
        self.MUL = operations.MUL
        self.DIV = operations.DIV
        self.POWER = operations.POWER
        self.EQL = operations.EQL
        self.N_OPS = operations.N_OPS
        self.PAD = operations.PAD
        self._device = device = config['device']
        self.min_NUM = dataset.out_symbol2idx['NUM_0']
        self.POWER = dataset.out_symbol2idx['^']
        self.min_CON = self.N_OPS_out = self.POWER + 1
        self.fix_constants = list(dataset.out_symbol2idx.keys())[self.min_CON:self.min_NUM]
        self.mask_list = NumMask.number
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.embedder = BasicEmbedder(vocab_size, dim_embed, dropout_rate)
        self.encoder = SalignedEncoder(dim_embed, dim_hidden, dim_hidden, dropout_rate)
        self.decoder = SalignedDecoder(operations, dim_hidden, dropout_rate, device)
        self.embedding_one = torch.nn.Parameter(torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.embedding_pi = torch.nn.Parameter(torch.normal(torch.zeros(2 * dim_hidden), 0.01))
        self.encoder.initialize_fix_constant(len(self.fix_constants), self._device)
        class_weights = torch.ones(operations.N_OPS + 1)
        self._op_loss = torch.nn.CrossEntropyLoss(class_weights, size_average=False, reduce=False, ignore_index=-1)
        self._arg_loss = torch.nn.CrossEntropyLoss()

    def forward(self, seq, seq_length, number_list, number_position, number_size, target=None, target_length=None, output_all_layers=False) ->Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq:
        :param torch.Tensor seq_length:
        :param list number_list:
        :param list number_position:
        :param list number_size:
        :param torch.Tensor | None target:
        :param torch.Tensor | None target_length:
        :param bool output_all_layers:
        :return: token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        constant_indices = number_position
        constants = number_list
        num_len = number_size
        seq_length = seq_length.long()
        batch_size = seq.size(0)
        bottom = torch.zeros(self._dim_hidden * 2)
        bottom.requires_grad = False
        seq_emb = self.embedder(seq)
        encoder_outputs, encoder_hidden, operands, number_emb, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, constant_indices, output_all_layers)
        stacks = [StackMachine(self.operations, constants[b] + self.fix_constants, number_emb[b], bottom, dry_run=True) for b in range(batch_size)]
        if target is not None:
            operands_len = torch.LongTensor(self.N_OPS + np.array(num_len))
            operands_len = operands_len.unsqueeze(1).repeat(1, target.size(1))
            target[target >= operands_len] = self.N_OPS
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, encoder_hidden, seq_length, operands, stacks, number_emb, target, target_length, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.
        
        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num pos', 'num list', 'num size'.
        """
        text = torch.tensor(batch_data['question'])
        ops = torch.tensor(batch_data['equation'])
        text_len = torch.tensor(batch_data['ques len']).long()
        ops_len = torch.tensor(batch_data['equ len']).long()
        constant_indices = batch_data['num pos']
        constants = batch_data['num list']
        num_len = batch_data['num size']
        logits, _, all_layers = self.forward(text, text_len, constants, constant_indices, num_len, ops, ops_len, output_all_layers=True)
        op_logits, arg_logits = logits
        op_targets, arg_targets = all_layers['op_targets'], all_layers['arg_targets']
        batch_size = ops.size(0)
        loss = torch.zeros(batch_size)
        for t in range(max(ops_len)):
            loss += self._op_loss(op_logits[:, t, :], op_targets[:, t])
            for b in range(batch_size):
                if self.NOOP <= arg_targets[b, t] < self.N_OPS:
                    continue
                loss[b] += self._arg_loss(arg_logits[b, t].unsqueeze(0), arg_targets[b, t].unsqueeze(0) - self.N_OPS)
        loss = (loss / max(ops_len)).mean()
        loss.backward()
        return loss.item()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num pos', 'num list', 'num size'.
        """
        text = torch.tensor(batch_data['question'])
        text_len = torch.tensor(batch_data['ques len']).long()
        constant_indices = batch_data['num pos']
        constants = batch_data['num list']
        num_len = batch_data['num size']
        target = torch.tensor(batch_data['equation'])
        _, outputs, _ = self.forward(text, text_len, constants, constant_indices, num_len)
        predicts = self.convert_idx2symbol(outputs, constants)
        targets = self.convert_idx2symbol(target, constants)
        return predicts, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_len = torch.tensor(batch_data['ques len']).long()
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_len, num_list, num_pos, num_size, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, seq_length, constant_indices, output_all_layers=False):
        batch_size = seq_emb.size(0)
        encoder_outputs, encoder_hidden, operands = self.encoder.forward(seq_emb, seq_length, constant_indices)
        number_emb = [(operands[b_i] + self.encoder.get_fix_constant()) for b_i in range(batch_size)]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, operands, number_emb, all_layer_outputs

    def decoder_forward(self, encoder_outputs, encoder_hidden, inputs_length, operands, stacks, number_emb, target=None, target_length=None, output_all_layers=False):
        batch_size = encoder_outputs.size(0)
        prev_op = (torch.zeros(batch_size) - 1).type(torch.LongTensor)
        prev_output = None
        prev_state = encoder_hidden
        decoder_outputs = []
        token_logits = []
        arg_logits = []
        outputs = []
        op_targets = []
        arg_targets = []
        if target is not None:
            for t in range(max(target_length)):
                op_logit, arg_logit, prev_output, prev_state = self.decoder(encoder_outputs, inputs_length, operands, stacks, prev_op, prev_output, prev_state, number_emb, self.N_OPS)
                prev_op = target[:, t]
                decoder_outputs.append(prev_output)
                token_logits.append(op_logit)
                arg_logits.append(arg_logit)
                op_target = target[:, t].clone().detach()
                op_target[np.array(target_length) <= t] = self.NOOP
                op_target[op_target >= self.N_OPS] = self.N_OPS
                op_target.require_grad = False
                op_targets.append(op_target)
                _, pred_op = torch.log(torch.nn.functional.softmax(op_logit, -1)).max(-1)
                _, pred_arg = torch.log(torch.nn.functional.softmax(arg_logit, -1)).max(-1)
                for b in range(batch_size):
                    if pred_op[b] == self.N_OPS:
                        pred_op[b] += pred_arg[b]
                outputs.append(pred_op)
        else:
            finished = [False] * batch_size
            for t in range(self.max_gen_len):
                op_logit, arg_logit, prev_output, prev_state = self.decoder(encoder_outputs, inputs_length, operands, stacks, prev_op, prev_output, prev_state, number_emb, self.N_OPS)
                n_finished = 0
                for b in range(batch_size):
                    if len(stacks[b].stack_log_index) and stacks[b].stack_log_index[-1] == self.EQL:
                        finished[b] = True
                    if finished[b]:
                        op_logit[b, self.PAD] = math.inf
                        n_finished += 1
                op_loss, prev_op = torch.log(torch.nn.functional.softmax(op_logit, -1)).max(-1)
                arg_loss, prev_arg = torch.log(torch.nn.functional.softmax(arg_logit, -1)).max(-1)
                for b in range(batch_size):
                    if prev_op[b] == self.N_OPS:
                        prev_op[b] += prev_arg[b]
                if n_finished == batch_size:
                    break
                decoder_outputs.append(prev_output)
                token_logits.append(op_logit)
                arg_logits.append(arg_logit)
                outputs.append(prev_op)
                if n_finished == batch_size:
                    break
        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        token_logits = torch.stack(token_logits, dim=1)
        arg_logits = torch.stack(arg_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        if target is not None:
            op_targets = torch.stack(op_targets, dim=1)
            arg_targets = target.clone()
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['decoder_outputs'] = decoder_outputs
            all_layer_outputs['op_logits'] = token_logits
            all_layer_outputs['arg_logits'] = arg_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['op_targets'] = op_targets
            all_layer_outputs['arg_targets'] = arg_targets
        return (token_logits, arg_logits), outputs, all_layer_outputs

    def convert_mask_num(self, batch_output, num_list):
        output_list = []
        for b_i, output in enumerate(batch_output):
            res = []
            num_len = len(num_list[b_i])
            for symbol in output:
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list


class BertEncoder(nn.Module):

    def __init__(self, hidden_size, dropout_ratio, pretrained_model_path):
        super(BertEncoder, self).__init__()
        self.embedding_size = 768
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.hidden_size = hidden_size
        self.dropout = dropout_ratio
        self.em_dropout = nn.Dropout(dropout_ratio)
        self.linear = nn.Linear(self.embedding_size, self.hidden_size)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embedded = self.em_dropout(output[0])
        pade_outputs = self.linear(embedded)
        pade_outputs = pade_outputs.transpose(0, 1)
        return pade_outputs

    def token_resize(self, input_size):
        self.bert.resize_token_embeddings(input_size)


class BertTD(nn.Module):
    """
    Reference:
    Li et al. Seeking Patterns, Not just Memorizing Procedures: Contrastive Learning for Solving Math Word Problems
    """

    def __init__(self, config, dataset):
        super(BertTD, self).__init__()
        self.hidden_size = config['hidden_size']
        self.device = config['device']
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config['embedding_size']
        self.dropout_ratio = config['dropout_ratio']
        self.num_layers = config['num_layers']
        self.rnn_cell_type = config['rnn_cell_type']
        self.embedding = config['embedding']
        self.pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config['transformers_pretrained_model']
        self.add_num_symbol = config['add_num_symbol']
        self.vocab_size = len(dataset.in_idx2word)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number
        self.num_start = dataset.num_start
        self.operator_nums = dataset.operator_nums
        self.generate_size = len(generate_list)
        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        try:
            self.in_pad_token = dataset.in_word2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.in_pad_token = None
        self.encoder = BertEncoder(self.hidden_size, self.dropout_ratio, self.pretrained_model_path)
        if self.add_num_symbol:
            self._pretrained_model_resize()
        self.decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.loss = MaskedCrossEntropyLoss()

    def _pretrained_model_resize(self):
        self.encoder.token_resize(self.vocab_size)

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        encoder_seq_mask = seq != self.in_pad_token
        decoder_seq_mask = torch.eq(seq, self.in_pad_token)
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        batch_size = len(seq_length)
        problem_output, encoder_outputs, encoder_layer_outputs = self.encoder_forward(seq, encoder_seq_mask, output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, self.hidden_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, decoder_seq_mask, num_mask, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs['number_representation'] = all_nums_encoder_outputs
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.tensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, _, all_layer_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, target, output_all_layers=True)
        target = all_layer_outputs['target']
        loss = masked_cross_entropy(token_logits, target, target_length)
        loss.backward()
        return loss.item()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.
        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        _, outputs, _ = self.forward(seq, seq_length, nums_stack, num_size, num_pos)
        all_output = self.convert_idx2symbol(outputs[0], num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq, seq_mask, output_all_layers=False):
        encoder_outputs = self.encoder(seq, seq_mask)
        problem_output = encoder_outputs[0]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = problem_output.size(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            target = target.transpose(0, 1)
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            target = target.transpose(0, 1)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['target'] = target
        return token_logits, outputs, all_layer_outputs

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [(1) for _ in range(hidden_size)]
        temp_0 = [(0) for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [(0) for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices
            masked_index = masked_index
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def convert_idx2symbol(self, output, num_list, num_stack):
        """batch_size=1"""
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if 'NUM' in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list


class GTS(nn.Module):
    """
    Reference:
        Xie et al. "A Goal-Driven Tree-Structured Neural Model for Math Word Problems" in IJCAI 2019.
    """

    def __init__(self, config, dataset):
        super(GTS, self).__init__()
        self.hidden_size = config['hidden_size']
        self.device = config['device']
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config['embedding_size']
        self.dropout_ratio = config['dropout_ratio']
        self.num_layers = config['num_layers']
        self.rnn_cell_type = config['rnn_cell_type']
        self.embedding = config['embedding']
        self.vocab_size = len(dataset.in_idx2word)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.in_word2idx = dataset.in_word2idx
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number
        self.num_start = dataset.num_start
        self.operator_nums = dataset.operator_nums
        self.generate_size = len(generate_list)
        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        try:
            self.in_pad_token = dataset.in_word2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.in_pad_token = None
        if config['embedding'] == 'roberta':
            self.embedder = RobertaEmbedder(self.vocab_size, config['pretrained_model_path'])
        elif config['embedding'] == 'bert':
            self.embedder = BertEmbedder(self.vocab_size, config['pretrained_model_path'])
        else:
            self.embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio, batch_first=False)
        self.decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.loss = MaskedCrossEntropyLoss()

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        seq_mask = torch.eq(seq, self.in_pad_token)
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        batch_size = len(seq_length)
        seq_emb = self.embedder(seq)
        problem_output, encoder_outputs, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, self.hidden_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs['number_representation'] = all_nums_encoder_outputs
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.LongTensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, _, all_layer_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, target, output_all_layers=True)
        target = all_layer_outputs['target']
        loss = masked_cross_entropy(token_logits, target, target_length)
        loss.backward()
        return loss.item()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list','num size'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        _, outputs, _ = self.forward(seq, seq_length, nums_stack, num_size, num_pos)
        all_output = self.convert_idx2symbol(outputs[0], num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [(1) for _ in range(hidden_size)]
        temp_0 = [(0) for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [(0) for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices
            masked_index = masked_index
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_inputs = seq_emb.transpose(0, 1)
        pade_outputs, hidden_states = self.encoder(encoder_inputs, seq_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = hidden_states
            all_layer_outputs['inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(1)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            target = target.transpose(0, 1)
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            target = target.transpose(0, 1)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['target'] = target
        return token_logits, outputs, all_layer_outputs

    def convert_idx2symbol(self, output, num_list, num_stack):
        """batch_size=1"""
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if 'NUM' in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list


class BeamNode:

    def __init__(self, score, nodes_hidden, node_stacks, tree_stacks, decoder_outputs_list, sequence_symbols_list):
        self.score = score
        self.nodes_hidden = nodes_hidden
        self.node_stacks = node_stacks
        self.tree_stacks = tree_stacks
        self.decoder_outputs_list = decoder_outputs_list
        self.sequence_symbols_list = sequence_symbols_list
        return

    def copy(self):
        node = BeamNode(self.score, self.nodes_hidden, copy_list(self.node_stacks), copy_list(self.tree_stacks), copy_list(self.decoder_outputs_list), copy_list(self.sequence_symbols_list))
        return node


class GateNN(nn.Module):

    def __init__(self, hidden_size, input1_size, input2_size=0, dropout=0.4, single_layer=False):
        super(GateNN, self).__init__()
        self.single_layer = single_layer
        self.hidden_l1 = nn.Linear(input1_size + hidden_size, hidden_size)
        self.gate_l1 = nn.Linear(input1_size + hidden_size, hidden_size)
        if not single_layer:
            self.dropout = nn.Dropout(p=dropout)
            self.hidden_l2 = nn.Linear(input2_size + hidden_size, hidden_size)
            self.gate_l2 = nn.Linear(input2_size + hidden_size, hidden_size)
        return

    def forward(self, hidden, input1, input2=None):
        input1 = torch.cat((hidden, input1), dim=-1)
        h = torch.tanh(self.hidden_l1(input1))
        g = torch.sigmoid(self.gate_l1(input1))
        h = h * g
        if not self.single_layer:
            h1 = self.dropout(h)
            if input2 is not None:
                input2 = torch.cat((h1, input2), dim=-1)
            else:
                input2 = h1
            h = torch.tanh(self.hidden_l2(input2))
            g = torch.sigmoid(self.gate_l2(input2))
            h = h * g
        return h


class NodeEmbeddingNode:

    def __init__(self, node_hidden, node_context=None, label_embedding=None):
        self.node_hidden = node_hidden
        self.node_context = node_context
        self.label_embedding = label_embedding
        return


class DecomposeModel(nn.Module):

    def __init__(self, hidden_size, dropout, device):
        super(DecomposeModel, self).__init__()
        self.pad_hidden = torch.zeros(hidden_size)
        self.pad_hidden = self.pad_hidden
        self.dropout = nn.Dropout(p=dropout)
        self.l_decompose = GateNN(hidden_size, hidden_size * 2, 0, dropout=dropout, single_layer=False)
        self.r_decompose = GateNN(hidden_size, hidden_size * 2, hidden_size, dropout=dropout, single_layer=False)
        return

    def forward(self, node_stacks, tree_stacks, nodes_context, labels_embedding, pad_node=True):
        children_hidden = []
        for node_stack, tree_stack, node_context, label_embedding in zip(node_stacks, tree_stacks, nodes_context, labels_embedding):
            if len(node_stack) > 0:
                if not tree_stack[-1].terminal:
                    node_hidden = node_stack[-1].node_hidden
                    node_stack[-1] = NodeEmbeddingNode(node_hidden, node_context, label_embedding)
                    l_input = torch.cat((node_context, label_embedding), dim=-1)
                    l_input = self.dropout(l_input)
                    node_hidden = self.dropout(node_hidden)
                    child_hidden = self.l_decompose(node_hidden, l_input, None)
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
                else:
                    node_stack.pop()
                    if len(node_stack) > 0:
                        parent_node = node_stack.pop()
                        node_hidden = parent_node.node_hidden
                        node_context = parent_node.node_context
                        label_embedding = parent_node.label_embedding
                        left_embedding = tree_stack[-1].embedding
                        left_embedding = self.dropout(left_embedding)
                        r_input = torch.cat((node_context, label_embedding), dim=-1)
                        r_input = self.dropout(r_input)
                        node_hidden = self.dropout(node_hidden)
                        child_hidden = self.r_decompose(node_hidden, r_input, left_embedding)
                        node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
            if len(node_stack) == 0:
                child_hidden = self.pad_hidden
                if pad_node:
                    node_stack.append(NodeEmbeddingNode(child_hidden, None, None))
            children_hidden.append(child_hidden)
        children_hidden = torch.stack(children_hidden, dim=0)
        return children_hidden


class HierarchicalAttention(nn.Module):

    def __init__(self, dim):
        super(HierarchicalAttention, self).__init__()
        self.span_attn = Attention(dim, mix=False, fn=False)
        self.word_attn = Attention(dim, mix=True, fn=False)
        return

    def forward(self, output, span_context, word_contexts, span_mask=None, word_masks=None):
        batch_size, output_size, _ = output.size()
        _, span_size, hidden_size = span_context.size()
        _, span_attn = self.span_attn(output, span_context, span_mask)
        word_outputs = []
        for word_context, word_mask in zip(word_contexts, word_masks):
            word_output, _ = self.word_attn(output, word_context, word_mask)
            word_outputs.append(word_output.unsqueeze(-2))
        word_output = torch.cat(word_outputs, dim=-2)
        word_output = word_output.view(-1, span_size, hidden_size)
        span_context = span_context.unsqueeze(1).expand(-1, output_size, -1, -1).view(-1, span_size, hidden_size)
        span_attn = span_attn.view(-1, 1, span_size)
        attn_output = torch.bmm(span_attn, span_context + word_output)
        attn_output = attn_output.view(batch_size, output_size, hidden_size)
        return attn_output


class ScoreModel(nn.Module):

    def __init__(self, hidden_size):
        super(ScoreModel, self).__init__()
        self.w = nn.Linear(hidden_size * 3, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, context, token_embeddings):
        batch_size, class_size, _ = token_embeddings.size()
        hc = torch.cat((hidden, context), dim=-1)
        hc = hc.unsqueeze(1).expand(-1, class_size, -1)
        hidden = torch.cat((hc, token_embeddings), dim=-1)
        hidden = F.leaky_relu(self.w(hidden))
        score = self.score(hidden).view(batch_size, class_size)
        return score


class PredictModel(nn.Module):

    def __init__(self, hidden_size, class_size, dropout=0.4):
        super(PredictModel, self).__init__()
        self.class_size = class_size
        self.dropout = nn.Dropout(p=dropout)
        self.attn = HierarchicalAttention(hidden_size)
        self.score_pointer = ScoreModel(hidden_size)
        self.score_generator = ScoreModel(hidden_size)
        self.score_span = ScoreModel(hidden_size)
        self.gen_prob = nn.Linear(hidden_size * 2, 1)
        return

    def score_pn(self, hidden, context, embedding_masks):
        device = hidden.device
        (pointer_embedding, pointer_mask), generator_embedding, _ = embedding_masks
        pointer_embedding = pointer_embedding
        pointer_mask = pointer_mask
        generator_embedding = generator_embedding
        hidden = self.dropout(hidden)
        context = self.dropout(context)
        pointer_embedding = self.dropout(pointer_embedding)
        pointer_score = self.score_pointer(hidden, context, pointer_embedding)
        pointer_score.data.masked_fill_(pointer_mask, -float('inf'))
        pointer_prob = F.softmax(pointer_score, dim=-1)
        generator_embedding = self.dropout(generator_embedding)
        generator_score = self.score_generator(hidden, context, generator_embedding)
        generator_prob = F.softmax(generator_score, dim=-1)
        return pointer_prob, generator_prob

    def forward(self, node_hidden, encoder_outputs, masks, embedding_masks):
        use_cuda = node_hidden.is_cuda
        node_hidden_dropout = self.dropout(node_hidden).unsqueeze(1)
        span_output, word_outputs = encoder_outputs
        span_mask, word_masks = masks
        if use_cuda:
            span_mask = span_mask
            word_masks = [mask for mask in word_masks]
        output_attn = self.attn(node_hidden_dropout, span_output, word_outputs, span_mask, word_masks)
        context = output_attn.squeeze(1)
        hc = torch.cat((node_hidden, context), dim=-1)
        pointer_prob, generator_prob = self.score_pn(node_hidden, context, embedding_masks)
        gen_prob = torch.sigmoid(self.gen_prob(hc))
        prob = torch.cat((gen_prob * generator_prob, (1 - gen_prob) * pointer_prob), dim=-1)
        pad_empty_pointer = torch.zeros(prob.size(0), self.class_size - prob.size(-1))
        if use_cuda:
            pad_empty_pointer = pad_empty_pointer
        prob = torch.cat((prob, pad_empty_pointer), dim=-1)
        output = torch.log(prob + 1e-30)
        return output, context


class TreeEmbeddingModel(nn.Module):

    def __init__(self, hidden_size, op_set, dropout=0.4):
        super(TreeEmbeddingModel, self).__init__()
        self.op_set = op_set
        self.dropout = nn.Dropout(p=dropout)
        self.combine = GateNN(hidden_size, hidden_size * 2, dropout=dropout, single_layer=True)
        return

    def merge(self, op_embedding, left_embedding, right_embedding):
        te_input = torch.cat((left_embedding, right_embedding), dim=-1)
        te_input = self.dropout(te_input)
        op_embedding = self.dropout(op_embedding)
        tree_embed = self.combine(op_embedding, te_input)
        return tree_embed

    def forward(self, class_embedding, tree_stacks, embed_node_index):
        use_cuda = embed_node_index.is_cuda
        batch_index = torch.arange(embed_node_index.size(0))
        if use_cuda:
            batch_index = batch_index
        labels_embedding = class_embedding[batch_index, embed_node_index]
        for node_label, tree_stack, label_embedding in zip(embed_node_index.cpu().tolist(), tree_stacks, labels_embedding):
            if node_label in self.op_set:
                tree_node = TreeEmbedding(label_embedding, terminal=False)
            else:
                right_embedding = label_embedding
                while len(tree_stack) >= 2 and tree_stack[-1].terminal and not tree_stack[-2].terminal:
                    left_embedding = tree_stack.pop().embedding
                    op_embedding = tree_stack.pop().embedding
                    right_embedding = self.merge(op_embedding, left_embedding, right_embedding)
                tree_node = TreeEmbedding(right_embedding, terminal=True)
            tree_stack.append(tree_node)
        return labels_embedding


class HMSDecoder(nn.Module):

    def __init__(self, embedding_model, hidden_size, dropout, op_set, vocab_dict, class_list, device):
        super(HMSDecoder, self).__init__()
        self.hidden_size = hidden_size
        embed_size = embedding_model.embedding_size
        class_size = len(class_list)
        self.get_predict_meta(class_list, vocab_dict, device)
        self.embed_model = embedding_model
        self.op_hidden = nn.Linear(embed_size, hidden_size)
        self.predict = PredictModel(hidden_size, class_size, dropout=dropout)
        op_set = set(i for i, symbol in enumerate(class_list) if symbol in op_set)
        self.tree_embedding = TreeEmbeddingModel(hidden_size, op_set, dropout=dropout)
        self.decompose = DecomposeModel(hidden_size, dropout, device)
        return

    def get_predict_meta(self, class_list, vocab_dict, device):
        pointer_list = [token for token in class_list if token in NumMask.number or token == SpecialTokens.UNK_TOKEN]
        generator_list = [token for token in class_list if token not in pointer_list]
        embed_list = generator_list + pointer_list
        self.pointer_index = torch.LongTensor([class_list.index(token) for token in pointer_list])
        self.generator_vocab = torch.LongTensor([vocab_dict[token] for token in generator_list])
        self.class_to_embed_index = torch.LongTensor([embed_list.index(token) for token in class_list])
        self.pointer_index = self.pointer_index
        self.generator_vocab = self.generator_vocab
        self.class_to_embed_index = self.class_to_embed_index
        return

    def get_pad_masks(self, encoder_outputs, input_lengths, span_length=None):
        span_output, word_outputs = encoder_outputs
        span_pad_length = span_output.size(1)
        word_pad_lengths = [word_output.size(1) for word_output in word_outputs]
        span_mask = self.get_mask(span_length, span_pad_length)
        word_masks = [self.get_mask(input_length, word_pad_length) for input_length, word_pad_length in zip(input_lengths, word_pad_lengths)]
        masks = span_mask, word_masks
        return masks

    def get_mask(self, encode_lengths, pad_length):
        device = encode_lengths.device
        batch_size = encode_lengths.size(0)
        index = torch.arange(pad_length)
        mask = (index.unsqueeze(0).expand(batch_size, -1) >= encode_lengths.unsqueeze(-1)).byte()
        mask[mask.sum(dim=-1) == pad_length, 0] = 0
        return mask

    def get_pointer_meta(self, num_pos, sub_num_poses=None):
        batch_size = num_pos.size(0)
        pointer_num_pos = num_pos.index_select(dim=1, index=self.pointer_index)
        num_pos_occupied = pointer_num_pos.sum(dim=0) == -batch_size
        occupied_len = num_pos_occupied.size(-1)
        for i, elem in enumerate(reversed(num_pos_occupied.cpu().tolist())):
            if not elem:
                occupied_len = occupied_len - i
                break
        pointer_num_pos = pointer_num_pos[:, :occupied_len]
        if sub_num_poses is not None:
            sub_pointer_poses = [sub_num_pos.index_select(dim=1, index=self.pointer_index)[:, :occupied_len] for sub_num_pos in sub_num_poses]
        else:
            sub_pointer_poses = None
        return pointer_num_pos, sub_pointer_poses

    def get_pointer_embedding(self, pointer_num_pos, encoder_outputs):
        device = encoder_outputs.device
        batch_size, pointer_size = pointer_num_pos.size()
        batch_index = torch.arange(batch_size)
        batch_index = batch_index
        batch_index = batch_index.unsqueeze(1).expand(-1, pointer_size)
        pointer_embedding = encoder_outputs[batch_index, pointer_num_pos]
        pointer_embedding = pointer_embedding * (pointer_num_pos != -1).unsqueeze(-1)
        return pointer_embedding

    def get_pointer_mask(self, pointer_num_pos):
        pointer_mask = pointer_num_pos == -1
        return pointer_mask

    def get_generator_embedding_mask(self, batch_size):
        generator_embedding = self.op_hidden(self.embed_model(self.generator_vocab))
        generator_embedding = generator_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        generator_mask = (self.generator_vocab == -1).unsqueeze(0).expand(batch_size, -1)
        return generator_embedding, generator_mask

    def get_class_embedding_mask(self, num_pos, encoder_outputs):
        _, word_outputs = encoder_outputs
        span_num_pos, word_num_poses = num_pos
        generator_embedding, generator_mask = self.get_generator_embedding_mask(span_num_pos.size(0))
        span_pointer_num_pos, word_pointer_num_poses = self.get_pointer_meta(span_num_pos, word_num_poses)
        pointer_mask = self.get_pointer_mask(span_pointer_num_pos)
        num_pointer_embeddings = []
        for word_output, word_pointer_num_pos in zip(word_outputs, word_pointer_num_poses):
            num_pointer_embedding = self.get_pointer_embedding(word_pointer_num_pos, word_output)
            num_pointer_embeddings.append(num_pointer_embedding)
        pointer_embedding = torch.cat([embedding.unsqueeze(0) for embedding in num_pointer_embeddings], dim=0).sum(dim=0)
        all_embedding = torch.cat((generator_embedding, pointer_embedding), dim=1)
        pointer_embedding_mask = pointer_embedding, pointer_mask
        return pointer_embedding_mask, generator_embedding, all_embedding

    def init_stacks(self, encoder_hidden):
        batch_size = encoder_hidden.size(0)
        node_stacks = [[NodeEmbeddingNode(hidden, None, None)] for hidden in encoder_hidden]
        tree_stacks = [[] for _ in range(batch_size)]
        return node_stacks, tree_stacks

    def forward_step(self, node_stacks, tree_stacks, nodes_hidden, encoder_outputs, masks, embedding_masks, decoder_nodes_class=None):
        nodes_output, nodes_context = self.predict(nodes_hidden, encoder_outputs, masks, embedding_masks)
        nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
        predict_nodes_class = nodes_output.topk(1)[1]
        if decoder_nodes_class is not None:
            nodes_class = decoder_nodes_class.view(-1)
        else:
            nodes_class = predict_nodes_class.view(-1)
        embed_nodes_index = self.class_to_embed_index[nodes_class]
        labels_embedding = self.tree_embedding(embedding_masks[-1], tree_stacks, embed_nodes_index)
        nodes_hidden = self.decompose(node_stacks, tree_stacks, nodes_context, labels_embedding)
        return nodes_output, predict_nodes_class, nodes_hidden

    def forward_teacher(self, decoder_nodes_label, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length=None):
        decoder_outputs_list = []
        sequence_symbols_list = []
        decoder_hidden = decoder_init_hidden
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        if decoder_nodes_label is not None:
            seq_len = decoder_nodes_label.size(1)
        else:
            seq_len = max_length
        for di in range(seq_len):
            if decoder_nodes_label is not None:
                decoder_node_class = decoder_nodes_label[:, di]
            else:
                decoder_node_class = None
            decoder_output, symbols, decoder_hidden = self.forward_step(node_stacks, tree_stacks, decoder_hidden, encoder_outputs, masks, embedding_masks, decoder_nodes_class=decoder_node_class)
            decoder_outputs_list.append(decoder_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list

    def forward_beam(self, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, beam_width=1):
        node_stacks, tree_stacks = self.init_stacks(decoder_init_hidden)
        beams = [BeamNode(0, decoder_init_hidden, node_stacks, tree_stacks, [], [])]
        for _ in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stacks) == 0:
                    current_beams.append(b)
                    continue
                nodes_output, nodes_context = self.predict(b.nodes_hidden, encoder_outputs, masks, embedding_masks)
                nodes_output = nodes_output.index_select(dim=-1, index=self.class_to_embed_index)
                top_value, top_index = nodes_output.topk(beam_width)
                top_value = torch.exp(top_value)
                for predict_score, predicted_symbol in zip(top_value.split(1, dim=-1), top_index.split(1, dim=-1)):
                    nb = b.copy()
                    embed_nodes_index = self.class_to_embed_index[predicted_symbol.view(-1)]
                    labels_embedding = self.tree_embedding(embedding_masks[-1], nb.tree_stacks, embed_nodes_index)
                    nodes_hidden = self.decompose(nb.node_stacks, nb.tree_stacks, nodes_context, labels_embedding, pad_node=False)
                    nb.score = b.score + predict_score.item()
                    nb.nodes_hidden = nodes_hidden
                    nb.decoder_outputs_list.append(nodes_output)
                    nb.sequence_symbols_list.append(predicted_symbol)
                    current_beams.append(nb)
            beams = sorted(current_beams, key=lambda b: b.score, reverse=True)
            beams = beams[:beam_width]
            all_finished = True
            for b in beams:
                if len(b.node_stacks[0]) != 0:
                    all_finished = False
                    break
            if all_finished:
                break
        output = beams[0]
        return output.decoder_outputs_list, output.nodes_hidden, output.sequence_symbols_list

    def forward(self, targets=None, encoder_hidden=None, encoder_outputs=None, input_lengths=None, span_length=None, num_pos=None, max_length=None, beam_width=None):
        masks = self.get_pad_masks(encoder_outputs, input_lengths, span_length)
        embedding_masks = self.get_class_embedding_mask(num_pos, encoder_outputs)
        if type(encoder_hidden) is tuple:
            encoder_hidden = encoder_hidden[0]
        decoder_init_hidden = encoder_hidden[-1, :, :]
        if max_length is None:
            if targets is not None:
                max_length = targets.size(1)
            else:
                max_length = 40
        if beam_width is not None:
            return self.forward_beam(decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length, beam_width)
        else:
            return self.forward_teacher(targets, decoder_init_hidden, encoder_outputs, masks, embedding_masks, max_length)


class PositionalEncoding(nn.Module):

    def __init__(self, pos_size, dim):
        super(PositionalEncoding, self).__init__()
        pe = torch.rand(pos_size, dim)
        pe = pe * 2 - 1
        self.pe = nn.Parameter(pe)

    def forward(self, input):
        output = input + self.pe[:input.size(1)]
        return output


class HWCPEncoder(nn.Module):
    """Hierarchical word-clause-problem encoder"""

    def __init__(self, embedding_model, embedding_size, hidden_size=512, span_size=0, dropout_ratio=0.4):
        super(HWCPEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = embedding_model
        self.word_rnn = nn.GRU(embedding_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=dropout_ratio)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.span_attn = Attention(self.hidden_size, mix=True, fn=True)
        self.pos_enc = PositionalEncoding(span_size, hidden_size)
        self.to_parent = Attention(self.hidden_size, mix=True, fn=True)
        return

    def forward(self, input_var, input_lengths, span_length, tree=None, output_all_layers=False):
        """Not implemented"""
        device = span_length.device
        word_outputs = []
        span_inputs = []
        input_vars = input_var
        trees = tree
        bi_word_hidden = None
        for span_index, input_var in enumerate(input_vars):
            input_length = input_lengths[span_index]
            embedded = self.embedding(input_var)
            word_output, bi_word_hidden = self.word_level_forward(embedded, input_length, bi_word_hidden)
            word_output, word_hidden = self.bi_combine(word_output, bi_word_hidden)
            tree_batch = trees[span_index]
            span_span_input = self.clause_level_forward(word_output, tree_batch)
            span_input = torch.cat(span_span_input, dim=0)
            span_inputs.append(span_input.unsqueeze(1))
            word_outputs.append(word_output)
        span_input = torch.cat(span_inputs, dim=1)
        span_mask = self.get_mask(span_length, span_input.size(1))
        span_output, _ = self.problem_level_forword(span_input, span_mask)
        span_output = span_output * (span_mask == 0).unsqueeze(-1)
        dim0 = torch.arange(span_output.size(0))
        span_hidden = span_output[dim0, span_length - 1].unsqueeze(0)
        return (span_output, word_outputs), span_hidden

    def word_level_forward(self, embedding_inputs, input_length, bi_word_hidden=None):
        pad_input_length = input_length.clone()
        pad_input_length[pad_input_length == 0] = 1
        embedded = nn.utils.rnn.pack_padded_sequence(embedding_inputs, pad_input_length, batch_first=True, enforce_sorted=False)
        word_output, bi_word_hidden = self.word_rnn(embedded, bi_word_hidden)
        word_output, _ = nn.utils.rnn.pad_packed_sequence(word_output, batch_first=True)
        return word_output, bi_word_hidden

    def clause_level_forward(self, word_output, tree_batch):
        device = word_output.device
        span_span_input = []
        for b_i, data_word_output in enumerate(word_output):
            data_word_output = data_word_output.unsqueeze(0)
            tree = tree_batch[b_i]
            if tree is not None:
                data_span_input = self.dependency_encode(data_word_output, tree.root)
            else:
                pad_hidden = torch.zeros(1, self.hidden_size)
                data_span_input = pad_hidden
            span_span_input.append(data_span_input)
        return span_span_input

    def problem_level_forword(self, span_input, span_mask):
        span_output = self.pos_enc(span_input)
        span_output = self.dropout(span_output)
        span_output, span_attn = self.span_attn(span_output, span_output, span_mask)
        return span_output, span_attn

    def bi_combine(self, output, hidden):
        hidden = hidden[0:hidden.size(0):2] + hidden[1:hidden.size(0):2]
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden

    def dependency_encode(self, word_output, node):
        pos = node.position
        word_vector = word_output[:, pos]
        if node.is_leaf:
            vector = word_vector
        else:
            children = node.left_nodes + node.right_nodes
            children_vector = [self.dependency_encode(word_output, child).unsqueeze(1) for child in children]
            children_vector = torch.cat(children_vector, dim=1)
            query = word_vector.unsqueeze(1)
            vector = self.to_parent(query, children_vector)[0].squeeze(1)
        return vector

    def get_mask(self, encode_lengths, pad_length):
        device = encode_lengths.device
        batch_size = encode_lengths.size(0)
        index = torch.arange(pad_length)
        mask = (index.unsqueeze(0).expand(batch_size, -1) >= encode_lengths.unsqueeze(-1)).byte()
        mask[mask.sum(dim=-1) == pad_length, 0] = 0
        return mask


class HMS(nn.Module):

    def __init__(self, config, dataset):
        super(HMS, self).__init__()
        self.device = config['device']
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['embedding_size']
        self.dropout_ratio = config['dropout_ratio']
        self.beam_size = config['beam_size']
        self.output_length = config['max_output_len']
        self.share_vacab = config['share_vocab']
        self.span_size = dataset.max_span_size
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.operator_list = dataset.operator_list
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.in_word2idx = dataset.in_word2idx
        self.in_idx2word = dataset.in_idx2word
        self.mask_list = NumMask.number
        self.out_pad_token = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        embedder = self._init_embedding_params(dataset.trainset, dataset.in_idx2word, embedder)
        self.encoder = HWCPEncoder(embedder, self.embedding_size, self.hidden_size, self.span_size, self.dropout_ratio)
        self.decoder = HMSDecoder(embedder, self.hidden_size, self.dropout_ratio, self.operator_list, self.in_word2idx, self.out_idx2symbol, self.device)
        weight = torch.ones(self.symbol_size)
        pad = self.out_pad_token
        self.loss = nn.NLLLoss(weight, pad, reduction='sum')

    def forward(self, input_variable, input_lengths, span_num_pos, word_num_poses, span_length=None, tree=None, target_variable=None, max_length=None, beam_width=None, output_all_layers=False):
        """

        :param input_variable:
        :param input_lengths:
        :param span_num_pos:
        :param word_num_poses:
        :param span_length:
        :param tree:
        :param target_variable:
        :param max_length:
        :param beam_width:
        :param output_all_layers:
        :return:
        """
        num_pos = span_num_pos, word_num_poses
        if beam_width != None:
            beam_width = self.beam_size
            max_length = self.output_length
        encoder_outputs, encoder_hidden = self.encoder(input_var=input_variable, input_lengths=input_lengths, span_length=span_length, tree=tree)
        output = self.decoder(targets=target_variable, encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs, input_lengths=input_lengths, span_length=span_length, num_pos=num_pos, max_length=max_length, beam_width=beam_width)
        token_logits, decoder_hidden, outputs = output
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['encoder_outputs'] = encoder_outputs
            model_all_outputs['encoder_hidden'] = encoder_hidden
            model_all_outputs['decoder_hidden'] = decoder_hidden
            model_all_outputs['token_logits'] = token_logits
            model_all_outputs['outputs'] = outputs
        return token_logits, outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'spans', 'spans len', 'span num pos', 'word num poses',
        'span nums', 'deprel tree', 'num pos', 'equation'
        """
        input_variable = [torch.tensor(span_i_batch) for span_i_batch in batch_data['spans']]
        input_lengths = torch.tensor(batch_data['spans len']).long()
        span_num_pos = torch.LongTensor(batch_data['span num pos'])
        word_num_poses = [torch.LongTensor(word_num_pos) for word_num_pos in batch_data['word num poses']]
        span_length = torch.tensor(batch_data['span nums'])
        target_variable = torch.tensor(batch_data['equation'])
        tree = batch_data['deprel tree']
        if self.share_vacab:
            target_variable = self.convert_in_idx_2_out_idx(target_variable)
        num_pos = span_num_pos, word_num_poses
        max_length = None
        beam_width = None
        encoder_outputs, encoder_hidden = self.encoder(input_var=input_variable, input_lengths=input_lengths, span_length=span_length, tree=tree)
        decoder_outputs, _, _ = self.decoder(targets=target_variable, encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs, input_lengths=input_lengths, span_length=span_length, num_pos=num_pos, max_length=max_length, beam_width=beam_width)
        batch_size = span_length.size(0)
        loss = 0
        for step, step_output in enumerate(decoder_outputs):
            loss += self.loss(step_output.contiguous().view(batch_size, -1), target_variable[:, step].view(-1))
        total_target_length = (target_variable != self.out_pad_token).sum().item()
        loss = loss / total_target_length
        loss.backward()
        return loss.item()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'spans', 'spans len', 'span num pos', 'word num poses',
        'span nums', 'deprel tree', 'num pos', 'equation', 'num list'
        """
        input_variable = [torch.tensor(span_i_batch) for span_i_batch in batch_data['spans']]
        input_lengths = torch.tensor(batch_data['spans len']).long()
        span_num_pos = torch.LongTensor(batch_data['span num pos'])
        word_num_poses = [torch.LongTensor(word_num_pos) for word_num_pos in batch_data['word num poses']]
        span_length = torch.tensor(batch_data['span nums'])
        target_variable = torch.tensor(batch_data['equation'])
        tree = batch_data['deprel tree']
        num_list = batch_data['num list']
        if self.share_vacab:
            target_variable = self.convert_in_idx_2_out_idx(target_variable)
        num_pos = span_num_pos, word_num_poses
        max_length = self.output_length
        beam_width = self.beam_size
        encoder_outputs, encoder_hidden = self.encoder(input_var=input_variable, input_lengths=input_lengths, span_length=span_length, tree=tree)
        _, _, sequence_symbols = self.decoder(targets=target_variable, encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs, input_lengths=input_lengths, span_length=span_length, num_pos=num_pos, max_length=max_length, beam_width=beam_width)
        targets = self.convert_idx2symbol(target_variable, num_list)
        outputs = torch.cat(sequence_symbols, dim=1)
        outputs = self.convert_idx2symbol(outputs, num_list)
        return outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        input_variable = [torch.tensor(span_i_batch) for span_i_batch in batch_data['spans']]
        input_lengths = torch.tensor(batch_data['spans len']).long()
        span_num_pos = torch.LongTensor(batch_data['span num pos'])
        word_num_poses = [torch.LongTensor(word_num_pos) for word_num_pos in batch_data['word num poses']]
        span_length = torch.tensor(batch_data['span nums'])
        tree = batch_data['deprel tree']
        token_logits, symbol_outputs, model_all_layers = self.forward(input_variable, input_lengths, span_num_pos, word_num_poses, span_length, tree, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_layers

    def _init_embedding_params(self, train_data, vocab, embedder):
        sentences = []
        for data in train_data:
            sentence = [SpecialTokens.SOS_TOKEN]
            for word in data['question']:
                if word in vocab:
                    sentence.append(word)
                else:
                    sentence.append(SpecialTokens.UNK_TOKEN)
            sentence += [SpecialTokens.EOS_TOKEN]
            sentences.append(sentence)
        embedder.init_embedding_params(sentences, vocab)
        return embedder

    def convert_out_idx_2_in_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.out_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_out_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.out_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list


class DQN(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, dropout_ratio):
        super(DQN, self).__init__()
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, embedding_size)
        self.action_pred = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        out_1 = self.hidden_layer_1(inputs)
        out_2 = self.hidden_layer_2(out_1)
        pred = self.action_pred(out_1)
        return pred, out_2

    def play_one(self, inputs):
        pred, obv = self.forward(inputs)
        act = pred.topk(1, dim=0)[1]
        return act, obv


class Node:

    def __init__(self, left, right, op):
        self.left = left
        self.right = right
        self.op = op
        self.parent = None
        self._res = None
        self.prob = None
        self.max_prob = None

    def res(self):
        if self._res != None:
            return self._res
        left_res = self.left.res()
        right_res = self.right.res()
        op_res = self.op.res()
        prob = left_res[1] + right_res[1] + op_res[1]
        max_prob = left_res[2] + right_res[2] + op_res[2]
        try:
            res = op_res[0](left_res[0], right_res[0])
        except:
            res = float('nan')
        self._res = [res, prob, max_prob]
        self.prob = prob
        self.max_prob = max_prob
        return self._res


class State:

    def __init__(self, num_list):
        self.nodes = self.get_nodes(num_list)
        self.fix_nodes = self.get_nodes(num_list)
        self.length = len(self.nodes)

    def str_2_quant(self, word):
        word = word.lower()
        l = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        return l.index(word) + 1

    def get_nodes(self, num_list):
        nodes = []
        for i in range(len(num_list)):
            node = Node()
            node.init_node(i, num_list[i])
            nodes.append(node)
        return nodes

    def get_node_via_index(self, index):
        for i in range(len(self.nodes)):
            if self.nodes[i].is_belong(index):
                return self.nodes[i]

    def is_lca_i_and_j(self, i, j):
        for node in self.nodes:
            if node.i_and_j_is_belong(i, j):
                return True
        return False

    def change(self, i, j, newnode):
        li = []
        for node in self.nodes:
            if node.is_belong(i) or node.is_belong(j):
                pass
            else:
                li.append(node)
        li.append(newnode)
        self.nodes = li

    def remove_node(self, i):
        li = []
        for node in self.nodes:
            if node.is_belong(i):
                pass
            else:
                li.append(node)
        self.nodes = li

    def print_state(self):
        None
        s = '['
        for i in range(len(self.nodes)):
            s += '['
            for ind in self.nodes[i].index:
                s += str(self.fix_nodes[ind].value) + ', '
            s += '], '
        s += ']'
        None


class Agent:

    def __init__(self, parse_obj, gold_tree, reject, pick, quantities_emb, look_up, op_list):
        self.parse_obj = parse_obj
        self.gold_tree = gold_tree
        self.reject = reject
        self.pick = pick
        self.quantities_emb = quantities_emb
        self.look_up = look_up
        self.op_list = op_list

    def print_agent(self):
        None
        None

    def get_feat_vector(self, index1, index2):
        emb1 = self.quantities_emb[index1]
        emb2 = self.quantities_emb[index2]
        self.feat_vector = torch.cat([emb1, emb2], dim=-1)
        return self.feat_vector

    def select_tuple(self):
        self.candidate_select = []
        if self.pick != []:
            self.candidate_select.append(self.pick)
        for i in range(self.state.length):
            for j in range(self.state.length):
                if i != j and i < j and not (i in self.pick and j in self.pick):
                    self.candidate_select.append([i, j])
        self.reject_select = self.reject

    def select_combine(self):
        for elem_pair in self.candidate_select:
            if elem_pair[0] in self.reject_select or elem_pair[1] in self.reject_select or self.state.is_lca_i_and_j(elem_pair[0], elem_pair[1]):
                continue
            else:
                return elem_pair
        return []

    def init_state_info(self):
        self.state = State(self.look_up)
        for index in self.reject:
            self.state.remove_node(index)
        self.select_tuple()
        self.breakout = 0
        elem_pair = self.select_combine()
        if not elem_pair:
            self.breakout = 1
            self.feat_vector = torch.randn(self.quantities_emb[0].size(-1) * 2)
            return
        self.node_1_index = elem_pair[0]
        self.node_2_index = elem_pair[1]
        self.get_feat_vector(self.node_1_index, self.node_2_index)

    def compound_two_nodes_predict(self, op):
        op_symbol = self.op_list[op]
        if self.breakout == 1:
            return None, 1, 0
        self.reward = 0
        node1 = self.state.get_node_via_index(self.node_1_index)
        node2 = self.state.get_node_via_index(self.node_2_index)
        newNode = Node()
        newNode.combine_node(node1, node2, op_symbol)
        self.state.change(self.node_1_index, self.node_2_index, newNode)
        if len(self.state.nodes) == 1:
            if abs(str2float(self.state.nodes[0].value) - str2float(self.gold_tree.gold_ans)) < 0.0001:
                return None, 1, 1
            else:
                return None, 1, 0
        elif len(self.state.nodes) == 0:
            return None, 1, 0
        else:
            elem_pair = self.select_combine()
            self.node_1_index = elem_pair[0]
            self.node_2_index = elem_pair[1]
            next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
            return next_states, 0, 0

    def compound_two_nodes(self, op):
        self.reward = 0
        if self.breakout == 1:
            return None, 0, 1, 0
        node1 = self.state.get_node_via_index(self.node_1_index)
        node2 = self.state.get_node_via_index(self.node_2_index)
        fix_node1 = self.state.fix_nodes[self.node_1_index]
        fix_node2 = self.state.fix_nodes[self.node_2_index]
        flag1 = False
        flag2 = False
        if node1.is_compound:
            flag1 = True
        elif self.gold_tree.is_in_rel_quants(fix_node1.value, self.look_up):
            flag1 = True
        else:
            flag1 = False
        if node2.is_compound:
            flag1 = True
        elif self.gold_tree.is_in_rel_quants(fix_node2.value, self.look_up):
            flag2 = True
        else:
            flag2 = False
        self.flag1 = flag1
        self.flag2 = flag2
        op_symbol = self.op_list[op]
        if op_symbol == '+':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '+':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) - str2float(self.gold_tree.gold_ans)) < 0.0001:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif op_symbol == '-':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '-':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    if newNode.value < 0:
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) - str2float(self.gold_tree.gold_ans)) < 0.0001:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif op_symbol == '*':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '*':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    if newNode.value < 0:
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) == str2float(self.gold_tree.gold_ans)) < 0.0001:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif op_symbol == '/':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '/':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    if newNode.value < 0:
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) == str2float(self.gold_tree.gold_ans)) < 0.0001:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif op_symbol == '^':
            if flag1 and flag2:
                if self.gold_tree.query(fix_node1.value, fix_node2.value) == '^':
                    newNode = Node()
                    newNode.combine_node(node1, node2, op_symbol)
                    if newNode.value < 0:
                        return None, -5, 1, 1
                    self.state.change(self.node_1_index, self.node_2_index, newNode)
                    if len(self.state.nodes) == 1:
                        if abs(str2float(self.state.nodes[0].value) == str2float(self.gold_tree.gold_ans)) < 0.0001:
                            return None, 5, 1, 1
                        else:
                            return None, -1, 1, 0
                    elif len(self.state.nodes) == 0:
                        return None, -1, 1, 0
                    else:
                        elem_pair = self.select_combine()
                        if len(elem_pair) == 0:
                            return None, -1, 2, 0
                        self.node_1_index = elem_pair[0]
                        self.node_2_index = elem_pair[1]
                        self.candidate_select.remove(elem_pair)
                        next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                        return next_states, 5, 0, 0
                else:
                    return None, -5, 3, 0
            else:
                return None, -5, 4, 0
        elif flag1 and flag2:
            if self.gold_tree.query(fix_node1.value, fix_node2.value) == '-':
                newNode = Node()
                newNode.combine_node(node1, node2, op)
                if newNode.value < 0:
                    return None, -5, 1, 1
                self.state.change(self.node_1_index, self.node_2_index, newNode)
                if len(self.state.nodes) == 1:
                    if self.state.nodes[0].node_value == self.gold_tree.gold_ans < 0.0001:
                        return None, 5, 1, 1
                    else:
                        return None, -1, 1, 0
                elif len(self.state.nodes) == 0:
                    return None, -1, 1, 0
                else:
                    elem_pair = self.select_combine()
                    if len(elem_pair) == 0:
                        return None, -1, 2, 0
                    self.node_1_index = elem_pair[0]
                    self.node_2_index = elem_pair[1]
                    self.candidate_select.remove(elem_pair)
                    next_states = self.get_feat_vector(self.node_1_index, self.node_2_index)
                    return next_states, 5, 0, 0
            else:
                return None, -5, 3, 0
        else:
            return None, -5, 4, 0

    def test_gate(self, flag):
        self.test_flag = flag

    def write_info(self, filename, op):
        with open(filename, 'a') as f:
            f.write('index: ' + str(self.parse_obj.parse_id) + '\n')
            f.write(self.parse_obj.word_problem_text + '\n')
            f.write('equations: ' + str(self.gold_tree.exp_str) + '\n')
            f.write('node_1: ' + str(self.state.fix_nodes[self.node_1_index].value) + ', node_2: ' + str(self.state.fix_nodes[self.node_2_index].value) + '\n')
            f.write('op: ' + ['+', '-', 'in-', '11', '22', '33'][op] + '\n')
            f.write('gold_ans: ' + str(self.gold_tree.gold_ans) + '\n')

    def write_single_info(self, filename, flag, prefix, content):
        if flag:
            with open(filename, 'a') as f:
                f.write(prefix + content + '\n\n')


class Env:

    def __init__(self):
        self.count = 0
        self.agents = []
        self.curr_agent = None

    def make_env(self, batch_tree, look_up, emb, op_list):
        self.count = 0
        self.agents = []
        self.curr_agent = None
        for b_i in range(len(batch_tree)):
            agent = Agent(parse_obj={}, gold_tree=batch_tree[b_i], reject=[], pick=[], quantities_emb=emb[b_i], look_up=look_up[b_i], op_list=op_list)
            self.agents.append(agent)

    def reset(self):
        num = self.count
        self.count += 1
        self.curr_agent = self.agents[num]
        self.curr_agent.init_state_info()
        return self.curr_agent.feat_vector

    def validate_reset(self, iteration):
        self.curr_agent = self.agents[iteration]
        self.curr_agent.init_state_info()
        return self.curr_agent.feat_vector

    def separate_data_set(self):
        train_set = []
        validate_set = []
        for ind in self.config.train_list:
            train_set.append(self.agents[ind])
        for ind in self.config.validate_list:
            validate_set.append(self.agents[ind])
        return train_set, validate_set

    def reset_inner_count(self):
        self.count = 0

    def step(self, action_op):
        next_states, reward, done, flag = self.curr_agent.compound_two_nodes(action_op)
        return next_states, reward, done

    def val_step(self, action_op):
        next_states, done, flag = self.curr_agent.compound_two_nodes_predict(action_op)
        return next_states, done, flag


class AbstractTree:

    def __init__(self):
        self.root = None

    def equ2tree():
        raise NotImplementedError

    def tree2equ():
        raise NotImplementedError


class GoldTree(AbstractTree):

    def __init__(self, root_node=None, gold_ans=None):
        super().__init__()
        self.root = root_node
        self.gold_ans = gold_ans

    def equ2tree(self, equ_list, out_idx2symbol, op_list, num_list, ans):
        stack = []
        for idx in equ_list:
            if idx == out_idx2symbol.index(SpecialTokens.PAD_TOKEN):
                break
            if idx == out_idx2symbol.index(SpecialTokens.EOS_TOKEN):
                break
            symbol = out_idx2symbol[idx]
            if symbol in op_list:
                node = Node(symbol, isleaf=False)
                node.set_right_node(stack.pop())
                node.set_left_node(stack.pop())
                stack.append(node)
            else:
                if symbol in NumMask.number:
                    i = NumMask.number.index(symbol)
                    value = num_list[i]
                    node = Node(value, isleaf=True)
                elif symbol == SpecialTokens.UNK_TOKEN:
                    node = Node('-inf', isleaf=True)
                else:
                    node = Node(symbol, isleaf=True)
                stack.append(node)
        self.root = stack.pop()
        self.gold_ans = ans

    def is_float(self, num_str, num_list):
        if num_str in num_list:
            return True
        else:
            return False

    def is_equal(self, v1, v2):
        if v1 == v2:
            return True
        else:
            return False

    def lca(self, root, va, vb, parent):
        left = False
        right = False
        if not self.result and root.left_node:
            left = self.lca(root.left_node, va, vb, root)
        if not self.result and root.right_node:
            right = self.lca(root.right_node, va, vb, root)
        mid = False
        if self.is_equal(root.node_value, va) or self.is_equal(root.node_value, vb):
            mid = True
        if not self.result and left + right + mid == 2:
            if mid:
                self.result = parent
            else:
                self.result = root
        return left or mid or right

    def is_in_rel_quants(self, value, rel_quants):
        if value in rel_quants:
            return True
        else:
            return False

    def query(self, va, vb):
        if self.root == None:
            return None
        self.result = None
        self.lca(self.root, va, vb, None)
        if self.result:
            return self.result.node_value
        else:
            return self.result


class MathDQN(nn.Module):

    def __init__(self, config):
        super(MathDQN, self).__init__()
        self.out_idx2symbol = config['out_idx2symbol']
        self.generate_list = config['generate_list']
        self.num_start = config['num_start']
        self.operator_list = config['operator_list']
        self.replay_size = config['replay_size']
        self.max_out_len = 30
        self.embedder = BaiscEmbedder(config['vocab_size'], config['embedding_size'], config['dropout_ratio'], padding_idx=0)
        self.encoder = SelfAttentionRNNEncoder(config['embedding_size'], config['hidden_size'], config['embedding_size'], config['num_layers'], config['rnn_cell_type'], config['dropout_ratio'], config['bidirectional'])
        self.dqn = DQN(config['embedding_size'] * 2, config['embedding_size'] * 2, config['hidden_size'], config['operator_nums'], config['dropout_ratio'])
        self.env = Env()

    def forward(self, seq, seq_length, num_pos, num_list, ans, target=None):
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.embedder(seq)
        encoder_output, encoder_hidden = self.encoder(seq_emb, seq_length)
        generate_num = [self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)] + [self.out_idx2symbol.index(num) for num in self.generate_list]
        generate_num = torch.tensor(generate_num)
        generate_emb = self.embedder(generate_num)
        tree = []
        look_ups = []
        embs = []
        for b_i in range(batch_size):
            tree.append(self.equ2tree(target[b_i], num_list[b_i], ans[b_i]))
            look_up = [SpecialTokens.UNK_TOKEN] + self.generate_list + num_list[b_i]
            num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
            num_list_, emb_ = self.get_num_list(target[b_i], num_list[b_i], look_up, num_embedding)
            look_ups.append(num_list_)
            embs.append(emb_)
        self.env.make_env(tree, look_ups, embs, self.operator_list)
        self.replay_memory = deque(maxlen=self.replay_size)
        for b_i in range(batch_size):
            obs = self.env.reset()
            for step in range(self.max_out_len):
                obs = obs
                action, next_obs = self.dqn.play_one(obs)
                n_o, reward, done = self.env.step(action)
                if n_o != None:
                    next_obs = n_o
                self.replay_memory.append((obs, action, reward, next_obs, done))
                obs = next_obs
                if done:
                    break
        states, actions, rewards, next_states, dones = self.sample_experiences(batch_size)
        dones = dones
        rewards = rewards
        self.dqn.eval()
        next_Q_values, _ = self.dqn(next_states)
        self.dqn.train()
        max_next_Q_values = torch.max(next_Q_values, dim=1)[0]
        discount_rate = 0.95
        target_Q_values = rewards + (1 - dones) * discount_rate * max_next_Q_values
        mask = torch.zeros(batch_size, len(self.operator_list))
        idxs = torch.arange(0, batch_size)
        mask[idxs, actions] = 1
        mask = mask
        all_Q_values, _ = self.dqn(states)
        Q_values = torch.sum(all_Q_values * mask, dim=1)
        return Q_values, target_Q_values

    def predict(self, seq, seq_length, num_pos, num_list, ans, target=None):
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.embedder(seq)
        encoder_output, encoder_hidden = self.encoder(seq_emb, seq_length)
        generate_num = [self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)] + [self.out_idx2symbol.index(num) for num in self.generate_list]
        generate_num = torch.tensor(generate_num)
        generate_emb = self.embedder(generate_num)
        tree = []
        look_ups = []
        embs = []
        for b_i in range(batch_size):
            tree.append(self.equ2tree(target[b_i], num_list[b_i], ans[b_i]))
            look_up = [SpecialTokens.UNK_TOKEN] + self.generate_list + num_list[b_i]
            num_embedding = torch.cat([generate_emb, encoder_output[b_i, num_pos[b_i]]], dim=0)
            num_list_, emb_ = self.get_num_list(target[b_i], num_list[b_i], look_up, num_embedding)
            look_ups.append(num_list_)
            embs.append(emb_)
        self.env.make_env(tree, look_ups, embs, self.operator_list)
        acc = 0
        for b_i in range(batch_size):
            obs = self.env.validate_reset(b_i)
            for step in range(self.max_out_len):
                obs = obs
                action, next_obs = self.dqn.play_one(obs)
                n_o, done, flag = self.env.val_step(action)
                if n_o != None:
                    next_obs = n_o
                obs = next_obs
                if done:
                    if flag:
                        acc += 1
                    break
        return acc

    def sample_experiences(self, batch_size):
        indices = torch.randint(len(self.replay_memory), size=(batch_size, 1))
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for experience in batch:
            states.append(experience[0])
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states.append(experience[3])
            dones.append(experience[4])
        states = torch.stack(states)
        actions = torch.cat(actions)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones)
        rewards = torch.tensor(rewards)
        return states, actions, rewards, next_states, dones

    def equ2tree(self, equation, num_list, ans):
        tree = GoldTree()
        tree.equ2tree(equation, self.out_idx2symbol, self.operator_list, num_list, ans)
        return tree

    def get_num_list(self, equation, num_list, look_up, emb):
        num_list_ = []
        emb_list = []
        for idx in equation:
            if idx > self.num_start:
                symbol = self.out_idx2symbol[idx]
                if symbol in NumMask.number:
                    i = NumMask.number.index(symbol)
                    num = num_list[i]
                else:
                    num = symbol
                if num in num_list_:
                    continue
                i = look_up.index(num)
                emb_list.append(emb[i])
                num_list_.append(num)
        return num_list_, emb_list


class MWPBert(nn.Module):

    def __init__(self, config, dataset):
        super(MWPBert, self).__init__()
        self.hidden_size = config['hidden_size']
        self.device = config['device']
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config['embedding_size']
        self.dropout_ratio = config['dropout_ratio']
        self.num_layers = config['num_layers']
        self.rnn_cell_type = config['rnn_cell_type']
        self.embedding = config['embedding']
        self.pretrained_model_path = config['pretrained_model'] if config['pretrained_model'] else config['transformers_pretrained_model']
        self.vocab_size = len(dataset.in_idx2word)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number
        self.num_start = dataset.num_start
        self.operator_nums = dataset.operator_nums
        self.generate_size = len(generate_list)
        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        try:
            self.in_pad_token = dataset.in_word2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.in_pad_token = None
        self.encoder = BertModel.from_pretrained(self.pretrained_model_path)
        self.encoder.resize_token_embeddings(self.vocab_size)
        self.decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.loss = MaskedCrossEntropyLoss()

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        encoder_seq_mask = seq != self.in_pad_token
        decoder_seq_mask = torch.eq(seq, self.in_pad_token)
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        batch_size = len(seq_length)
        problem_output, encoder_outputs, encoder_layer_outputs = self.encoder_forward(seq, encoder_seq_mask, seq_length, output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, self.hidden_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, decoder_seq_mask, num_mask, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs['number_representation'] = all_nums_encoder_outputs
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.LongTensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, _, all_layer_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, target, output_all_layers=True)
        target = all_layer_outputs['target']
        loss = masked_cross_entropy(token_logits, target, target_length)
        loss.backward()
        return loss.item()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list','num size'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        _, outputs, _ = self.forward(seq, seq_length, nums_stack, num_size, num_pos)
        all_output = self.convert_idx2symbol(outputs[0], num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq, seq_mask, seq_length, output_all_layers=False):
        encoder_outputs = self.encoder(seq, seq_mask)[0]
        encoder_outputs = encoder_outputs.transpose(0, 1)
        problem_output = []
        batch_size = seq.size(0)
        for b_i in range(batch_size):
            problem_output.append(torch.mean(encoder_outputs[:seq_length[b_i], b_i, :], dim=0))
        problem_output = torch.stack(problem_output, dim=0)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = problem_output.size(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            target = target.transpose(0, 1)
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            target = target.transpose(0, 1)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['target'] = target
        return token_logits, outputs, all_layer_outputs

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [(1) for _ in range(hidden_size)]
        temp_0 = [(0) for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [(0) for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices
            masked_index = masked_index
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def convert_idx2symbol(self, output, num_list, num_stack):
        """batch_size=1"""
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if 'NUM' in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list


class SemanticAlignmentModule(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size, hidden_size, batch_first=False):
        super(SemanticAlignmentModule, self).__init__()
        self.batch_first = batch_first
        self.attn = TreeAttention(encoder_hidden_size, decoder_hidden_size)
        self.encoder_linear1 = nn.Linear(encoder_hidden_size, hidden_size)
        self.encoder_linear2 = nn.Linear(hidden_size, hidden_size)
        self.decoder_linear1 = nn.Linear(decoder_hidden_size, hidden_size)
        self.decoder_linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        if self.batch_first:
            decoder_hidden = decoder_hidden.unsqueeze(0)
            encoder_outputs = encoder_outputs.unsqueeze(0)
        else:
            decoder_hidden = decoder_hidden.unsqueeze(0)
            encoder_outputs = encoder_outputs.unsqueeze(1)
        attn_weights = self.attn(decoder_hidden, encoder_outputs, None)
        if self.batch_first:
            align_context = attn_weights.bmm(encoder_outputs)
        else:
            align_context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
            align_context = align_context.transpose(0, 1)
        encoder_linear1 = torch.tanh(self.encoder_linear1(align_context))
        encoder_linear2 = self.encoder_linear2(encoder_linear1)
        decoder_linear1 = torch.tanh(self.decoder_linear1(decoder_hidden))
        decoder_linear2 = self.decoder_linear2(decoder_linear1)
        return encoder_linear2, decoder_linear2


class SAUSolver(nn.Module):
    """
    Reference:
        Qin et al. "Semantically-Aligned Universal Tree-Structured Solver for Math Word Problems" in EMNLP 2020.
    """

    def __init__(self, config, dataset):
        super(SAUSolver, self).__init__()
        self.hidden_size = config['hidden_size']
        self.device = config['device']
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config['embedding_size']
        self.dropout_ratio = config['dropout_ratio']
        self.num_layers = config['num_layers']
        self.rnn_cell_type = config['rnn_cell_type']
        self.loss_weight = config['loss_weight']
        self.batch_first = False
        self.vocab_size = len(dataset.in_idx2word)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number
        self.num_start = dataset.num_start
        self.operator_nums = dataset.operator_nums
        self.generate_size = len(generate_list)
        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio, batch_first=False)
        self.decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.sa = SemanticAlignmentModule(self.hidden_size, self.hidden_size, self.hidden_size)
        self.loss1 = MaskedCrossEntropyLoss()

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        seq_mask = torch.eq(seq, self.in_pad_token)
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        if not self.batch_first:
            target = target.transpose(0, 1)
        batch_size = len(seq_length)
        seq_emb = self.embedder(seq)
        problem_output, encoder_outputs, encoder_layer_outputs = self.encoder_forward(seq_emb, seq_length, output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, self.hidden_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs['number_representation'] = all_nums_encoder_outputs
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.LongTensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        generate_nums = self.generate_nums
        num_start = self.num_start
        unk = self.unk_token
        loss = self.train_tree(seq, seq_length, target, target_length, nums_stack, num_size, generate_nums, num_pos, unk, num_start)
        return loss

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.
        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        generate_nums = self.generate_nums
        num_start = self.num_start
        all_node_output = self.evaluate_tree(seq, seq_length, generate_nums, num_pos, num_start, self.beam_size, self.max_out_len)
        all_output = self.convert_idx2symbol(all_node_output, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def train_tree(self, input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums, num_pos, unk, num_start, english=False, var_nums=[], batch_first=False):
        seq_mask = []
        max_len = max(input_length)
        for i in input_length:
            seq_mask.append([(0) for _ in range(i)] + [(1) for _ in range(i, max_len)])
        seq_mask = torch.ByteTensor(seq_mask)
        num_mask = []
        max_num_size = max(num_size_batch) + len(generate_nums) + len(var_nums)
        for i in num_size_batch:
            d = i + len(generate_nums) + len(var_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.ByteTensor(num_mask)
        input_var = input_batch.transpose(0, 1)
        target = target_batch.transpose(0, 1)
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.decoder.hidden_size)]).unsqueeze(0)
        batch_size = len(input_length)
        if self.USE_CUDA:
            input_var = input_var
            seq_mask = seq_mask
            padding_hidden = padding_hidden
            num_mask = num_mask
        seq_emb = self.embedder(input_var)
        pade_outputs, _ = self.encoder(seq_emb, input_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        max_target_length = max(target_length)
        all_node_outputs = []
        all_sa_outputs = []
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, self.encoder.hidden_size)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        for t in range(max_target_length):
            num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
            outputs = torch.cat((op, num_score), 1)
            all_node_outputs.append(outputs)
            target_t, generate_input = self.generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
            target[t] = target_t
            if self.USE_CUDA:
                generate_input = generate_input
            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
            left_childs = []
            for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                if len(node_stack) != 0:
                    node = node_stack.pop()
                else:
                    left_childs.append(None)
                    continue
                if i < num_start:
                    node_stack.append(TreeNode(r))
                    node_stack.append(TreeNode(l, left_flag=True))
                    o.append(TreeEmbedding(node_label[idx].unsqueeze(0), terminal=False))
                else:
                    current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                    while len(o) > 0 and o[-1].terminal:
                        sub_stree = o.pop()
                        op = o.pop()
                        current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        if batch_first:
                            encoder_mapping, decoder_mapping = self.sa(current_num, encoder_outputs[idx])
                        else:
                            temp_encoder_outputs = encoder_outputs.transpose(0, 1)
                            encoder_mapping, decoder_mapping = self.sa(current_num, temp_encoder_outputs[idx])
                        all_sa_outputs.append((encoder_mapping, decoder_mapping))
                    o.append(TreeEmbedding(current_num, terminal=True))
                if len(o) > 0 and o[-1].terminal:
                    left_childs.append(o[-1].embedding)
                else:
                    left_childs.append(None)
        all_node_outputs = torch.stack(all_node_outputs, dim=1)
        target = target.transpose(0, 1).contiguous()
        if self.USE_CUDA:
            all_node_outputs = all_node_outputs
            target = target
            new_all_sa_outputs = []
            for sa_pair in all_sa_outputs:
                new_all_sa_outputs.append((sa_pair[0], sa_pair[1]))
            all_sa_outputs = new_all_sa_outputs
        else:
            pass
        semantic_alignment_loss = nn.MSELoss()
        total_semanti_alognment_loss = 0
        sa_len = len(all_sa_outputs)
        for sa_pair in all_sa_outputs:
            total_semanti_alognment_loss += semantic_alignment_loss(sa_pair[0], sa_pair[1])
        total_semanti_alognment_loss = total_semanti_alognment_loss / sa_len
        loss = masked_cross_entropy(all_node_outputs, target, target_length) + 0.01 * total_semanti_alognment_loss
        loss.backward()
        return loss.item()

    def evaluate_tree(self, input_batch, input_length, generate_nums, num_pos, num_start, beam_size=5, max_length=30):
        seq_mask = torch.BoolTensor(1, input_length).fill_(0)
        input_var = input_batch.transpose(0, 1)
        num_mask = torch.BoolTensor(1, len(num_pos[0]) + len(generate_nums)).fill_(0)
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        batch_size = 1
        if self.USE_CUDA:
            input_var = input_var
            seq_mask = seq_mask
            padding_hidden = padding_hidden
            num_mask = num_mask
        seq_emb = self.embedder(input_var)
        pade_outputs, _ = self.encoder(seq_emb, input_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        num_size = len(num_pos[0])
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, self.hidden_size)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
        for t in range(max_length):
            current_beams = []
            while len(beams) > 0:
                b = beams.pop()
                if len(b.node_stack[0]) == 0:
                    current_beams.append(b)
                    continue
                left_childs = b.left_childs
                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
                topv, topi = out_score.topk(beam_size)
                for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                    current_node_stack = copy_list(b.node_stack)
                    current_left_childs = []
                    current_embeddings_stacks = copy_list(b.embedding_stack)
                    current_out = copy.deepcopy(b.out)
                    out_token = int(ti)
                    current_out.append(out_token)
                    node = current_node_stack[0].pop()
                    if out_token < num_start:
                        generate_input = torch.LongTensor([out_token])
                        if self.USE_CUDA:
                            generate_input = generate_input
                        left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                        current_node_stack[0].append(TreeNode(right_child))
                        current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                        current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                        while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            sub_stree = current_embeddings_stacks[0].pop()
                            op = current_embeddings_stacks[0].pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    else:
                        current_left_childs.append(None)
                    current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out))
            beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
            beams = beams[:beam_size]
            flag = True
            for b in beams:
                if len(b.node_stack[0]) != 0:
                    flag = False
            if flag:
                break
        return beams[0].out

    def encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        if not self.batch_first:
            encoder_inputs = seq_emb.transpose(0, 1)
        else:
            encoder_inputs = seq_emb
        pade_outputs, hidden_states = self.encoder(encoder_inputs, seq_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = hidden_states
            all_layer_outputs['inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = problem_output.size(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        all_sa_outputs = []
        if target is not None:
            max_target_length = max(target.size(0))
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            if self.batch_first:
                                encoder_mapping, decoder_mapping = self.sa(current_num, encoder_outputs[idx])
                            else:
                                temp_encoder_outputs = encoder_outputs.transpose(0, 1)
                                encoder_mapping, decoder_mapping = self.sa(current_num, temp_encoder_outputs[idx])
                            all_sa_outputs.append((encoder_mapping, decoder_mapping))
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            if not self.batch_first:
                target = target.transpose(0, 1).contiguous()
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['target'] = target
            all_layer_outputs['semantic_alignment_pair'] = all_sa_outputs
        return token_logits, outputs, all_layer_outputs

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [(1) for _ in range(hidden_size)]
        temp_0 = [(0) for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [(0) for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices
            masked_index = masked_index
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def mse_loss(self, outputs, targets, mask=None):
        mask = mask
        x = torch.sqrt(torch.sum(torch.square(outputs - targets), dim=-1))
        y = torch.sum(x * mask, dim=-1) / torch.sum(mask, dim=-1)
        return torch.sum(y)

    def convert_idx2symbol(self, output, num_list, num_stack):
        """batch_size=1"""
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if 'NUM' in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list


class LSTMBasedTreeDecoder(nn.Module):
    """
    """

    def __init__(self, embedding_size, hidden_size, op_nums, generate_size, dropout=0.5):
        super(LSTMBasedTreeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.generate_size = generate_size
        self.op_nums = op_nums
        self.dropout = nn.Dropout(dropout)
        self.embedding_weight = nn.Parameter(torch.randn(1, generate_size, embedding_size))
        self.rnn = nn.LSTMCell(embedding_size * 2 + hidden_size, hidden_size)
        self.tree_rnn = nn.LSTMCell(embedding_size * 2 + hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        self.ops = nn.Linear(hidden_size, op_nums)
        self.trans = nn.Linear(hidden_size, embedding_size)
        self.attention = TreeAttention(hidden_size, hidden_size)
        self.score = Score(hidden_size, embedding_size)
        self.p_z = nn.Linear(hidden_size, 1)
        self.copy_attention = TreeAttention(hidden_size, hidden_size)

    def forward(self, parent_embed, left_embed, prev_embed, encoder_outputs, num_pades, padding_hidden, seq_mask, nums_mask, hidden, tree_hidden):
        """
        Args:
            parent_embed (list): parent embedding, length [batch_size], list of torch.Tensor with shape [1, 2 * hidden_size].
            left_embed (list): left embedding, length [batch_size], list of torch.Tensor with shape [1, embedding_size].
            prev_embed (list): previous embedding, length [batch_size], list of torch.Tensor with shape [1, embedding_size].
            encoder_outputs (torch.Tensor): output from encoder, shape [batch_size, sequence_length, hidden_size].
            num_pades (torch.Tensor): number representation, shape [batch_size, number_size, hidden_size].
            padding_hidden (torch.Tensor): padding hidden, shape [1,hidden_size].
            seq_mask (torch.BoolTensor): sequence mask, shape [batch_size, sequence_length].
            mask_nums (torch.BoolTensor): number mask, shape [batch_size, number_size].
            hidden (tuple(torch.Tensor, torch.Tensor)): hidden states, shape [batch_size, num_directions * hidden_size].
            tree_hidden (tuple(torch.Tensor, torch.Tensor)): tree hidden states, shape [batch_size, num_directions * hidden_size].
        
        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                num_score, number score, shape [batch_size, number_size].
                op, operator score, shape [batch_size, operator_size].
                current_embeddings, current node representation, shape [batch_size, 1, num_directions * hidden_size].
                current_context, current context representation, shape [batch_size, 1, num_directions * hidden_size].
                embedding_weight, embedding weight, shape [batch_size, number_size, embedding_size].
                hidden (tuple(torch.Tensor, torch.Tensor)): hidden states, shape [batch_size, num_directions * hidden_size].
                tree_hidden (tuple(torch.Tensor, torch.Tensor)): tree hidden states, shape [batch_size, num_directions * hidden_size].
        """
        parent_embed = torch.cat(parent_embed, dim=0)
        left_embed = torch.cat(left_embed, dim=0)
        prev_embed = torch.cat(prev_embed, dim=0)
        batch_size = parent_embed.size(0)
        embedded = torch.cat([parent_embed, left_embed, prev_embed], dim=1)
        if hidden[0].size(0) != batch_size:
            hidden = hidden[0].repeat(batch_size, 1), hidden[1].repeat(batch_size, 1)
        hidden_h, hidden_c = self.rnn(embedded, hidden)
        hidden = hidden_h, hidden_c
        if tree_hidden[0].size(0) != batch_size:
            tree_hidden = tree_hidden[0].repeat(batch_size, 1), tree_hidden[1].repeat(batch_size, 1)
        tree_hidden_h, tree_hidden_c = self.tree_rnn(embedded, tree_hidden)
        tree_hidden = tree_hidden_h, tree_hidden_c
        output = self.linear(torch.cat((hidden_h, tree_hidden_h), dim=-1)).unsqueeze(1)
        if encoder_outputs.size(0) != batch_size:
            repeat_dims = [1] * encoder_outputs.dim()
            repeat_dims[0] = batch_size
            encoder_outputs = encoder_outputs.repeat(*repeat_dims)
        current_attn = self.attention(output.transpose(0, 1), encoder_outputs.transpose(0, 1), seq_mask)
        output = current_attn.bmm(encoder_outputs)
        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)
        embedding_weight = torch.cat((embedding_weight, self.trans(num_pades)), dim=1)
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(output, embedding_weight_, nums_mask)
        op = self.ops(output.squeeze(1))
        return num_score, op, output, output, embedding_weight, hidden, tree_hidden


class NodeEmbeddingLayer(nn.Module):

    def __init__(self, op_nums, embedding_size):
        super(NodeEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.op_nums = op_nums
        self.embeddings = nn.Embedding(op_nums, embedding_size)

    def forward(self, node_embedding, node_label, current_context):
        """
        Args:
            node_embedding (torch.Tensor): node embedding, shape [batch_size, num_directions * hidden_size].
            node_label (torch.Tensor): shape [batch_size].
        
        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor):
                l_child, representation of left child, shape [batch_size, num_directions * hidden_size].
                r_child, representation of right child, shape [batch_size, num_directions * hidden_size].
                node_label_, representation of node label, shape [batch_size, embedding_size].
        """
        node_label_ = self.embeddings(node_label)
        return node_embedding, node_embedding, node_label_


class TreeLSTM(nn.Module):
    """
    Reference:
        Liu et al. "Tree-structured Decoding for Solving Math Word Problems" in EMNLP | IJCNLP 2019.
    """

    def __init__(self, config, dataset):
        super(TreeLSTM, self).__init__()
        self.hidden_size = config['hidden_size']
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.num_layers = config['num_layers']
        self.dropout_ratio = config['dropout_ratio']
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        self.mask_list = NumMask.number
        self.operator_nums = dataset.operator_nums
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.generate_size = len(generate_list)
        self.num_start = dataset.num_start
        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.encoder = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_ratio, bidirectional=True)
        self.decoder = LSTMBasedTreeDecoder(self.embedding_size, self.hidden_size * self.num_layers, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.node_generater = NodeEmbeddingLayer(self.operator_nums, self.embedding_size)
        self.root = nn.Parameter(torch.randn(1, self.embedding_size))
        self.loss = MaskedCrossEntropyLoss()

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        seq_mask = []
        max_len = max(seq_length)
        for i in seq_length:
            seq_mask.append([(0) for _ in range(i)] + [(1) for _ in range(i, max_len)])
        seq_mask = torch.BoolTensor(seq_mask)
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        seq_emb = self.embedder(seq)
        encoder_outputs, initial_hidden, problem_output, encoder_layer_outputs = self.encoder_forward(seq_emb, output_all_layers=output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, num_size, self.hidden_size * self.num_layers)
        token_logits, symbol_outputs, decoder_layer_outputs = self.decoder_forward(encoder_outputs, initial_hidden, problem_output, all_nums_encoder_outputs, seq_mask, num_mask, nums_stack, target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs['inputs_embedding'] = seq_emb
            model_all_outputs.update(encoder_layer_outputs)
            model_all_outputs['number_representation'] = all_nums_encoder_outputs
            model_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, model_all_outputs

    def calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.LongTensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, _, all_layer_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, target, output_all_layers=True)
        target = all_layer_outputs['target']
        self.loss.reset()
        self.loss.eval_batch(token_logits, target, target_length)
        self.loss.backward()
        return self.loss.get_loss()

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.
        
        :param batch_data: one batch data.
        :return: predicted equation, target equation.
        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', num size, 'num list'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        _, outputs, _ = self.forward(seq, seq_length, nums_stack, num_size, num_pos)
        all_outputs = self.convert_idx2symbol(outputs[0], num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_outputs, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, nums_stack, num_size, num_pos, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def encoder_forward(self, seq_emb, output_all_layers=False):
        pade_outputs, initial_hidden = self.encoder(seq_emb)
        problem_output = torch.cat([pade_outputs[:, -1, :self.hidden_size], pade_outputs[:, 0, self.hidden_size:]], dim=1)
        encoder_outputs = pade_outputs
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['encoder_outputs'] = encoder_outputs
            all_layer_outputs['encoder_hidden'] = initial_hidden
            all_layer_outputs['inputs_representation'] = problem_output
        return encoder_outputs, initial_hidden, problem_output, all_layer_outputs

    def decoder_forward(self, encoder_outputs, initial_hidden, problem_output, all_nums_encoder_outputs, seq_mask, num_mask, nums_stack, target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(0)
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.embedding_size)]).unsqueeze(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        all_node_outputs = []
        left_childs = [None for _ in range(batch_size)]
        embeddings_stacks = [[] for _ in range(batch_size)]
        nodes = [[] for _ in range(batch_size)]
        hidden = initial_hidden[0][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1), initial_hidden[1][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1)
        tree_hidden = initial_hidden[0][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1), initial_hidden[1][:self.num_layers].transpose(1, 0).contiguous().view(batch_size, -1)
        nodes_hiddens = [[] for _ in range(batch_size)]
        parent = [hidden[0][idx].unsqueeze(0) for idx in range(batch_size)]
        left = [self.root for _ in range(batch_size)]
        prev = [self.root for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            max_target_length = target.size(1)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings, hidden, tree_hidden = self.decoder(parent, left, prev, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask, hidden, tree_hidden)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[:, t].tolist(), outputs, nums_stack, self.num_start, self.unk_token)
                target[:, t] = target_t
                generate_input = generate_input
                left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                left, parent, prev, prev_idx = [], [], [], []
                left_childs = []
                for idx, l, r, node_stack, i, o, n, n_hidden in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[:, t].tolist(), embeddings_stacks, nodes, nodes_hiddens):
                    continue_flag = False
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue_flag = True
                    if not continue_flag:
                        if i < self.num_start:
                            n.append(i)
                            n_hidden.append(hidden[0][idx].unsqueeze(0))
                            node_stack.append(TreeNode(r))
                            node_stack.append(TreeNode(l, left_flag=True))
                            o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                            while len(o) > 0 and o[-1].terminal:
                                sub_stree = o.pop()
                                op = o.pop()
                                n.pop()
                                n.pop()
                                n_hidden.pop()
                                n_hidden.pop()
                                current_num = sub_stree.embedding
                            o.append(TreeEmbedding(current_num, True))
                            n.append(i)
                            n_hidden.append(hidden[0][idx].unsqueeze(0))
                        if len(o) > 0 and o[-1].terminal:
                            left_childs.append(o[-1].embedding)
                        else:
                            left_childs.append(None)
                    parent_flag = True
                    if len(node_stack) == 0:
                        left.append(self.root)
                        parent.append(hidden[0][idx].unsqueeze(0))
                        prev.append(self.root)
                        prev_idx.append(None)
                    elif n[-1] < self.num_start:
                        left.append(self.root)
                        parent.append(n_hidden[-1])
                        prev.append(self.node_generater.embeddings(torch.LongTensor([n[-1]])))
                        prev_idx.append(n[-1])
                    else:
                        left.append(current_nums_embeddings[idx, n[-1] - self.num_start].unsqueeze(0))
                        prev.append(current_nums_embeddings[idx, n[-1] - self.num_start].unsqueeze(0))
                        for i in range(len(n) - 1, -1, -1):
                            if n[i] < self.num_start:
                                parent.append(n_hidden[i])
                                parent_flag = False
                                break
                        if parent_flag:
                            parent.append(hidden[0][idx].unsqueeze(0))
                        prev_idx.append(n[-1])
        else:
            max_length = self.max_out_len
            beams = [([hidden[0]], [self.root], [self.root], nodes, nodes_hiddens, hidden, tree_hidden, TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], []))]
            for t in range(max_length):
                current_beams = []
                while len(beams) > 0:
                    parent, left, prev, nodes, nodes_hiddens, hidden, tree_hidden, b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append((parent, left, prev, nodes, nodes_hiddens, hidden, tree_hidden, b))
                        continue
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings, hidden, tree_hidden = self.decoder(parent, left, prev, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask, hidden, tree_hidden)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        left, parent, prev = [], [], []
                        current_node_stack = self.copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = self.copy_list(b.embedding_stack)
                        current_nodes = copy.deepcopy(nodes)
                        current_nodes_hidden = self.copy_list(nodes_hiddens)
                        current_out = copy.deepcopy(b.out)
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(out_token)
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            current_nodes[0].append(out_token)
                            current_nodes_hidden[0].append(hidden[0])
                            generate_input = torch.LongTensor([out_token])
                            left_child, right_child, node_label = self.node_generater(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_nodes[0].pop()
                                current_nodes[0].pop()
                                current_nodes_hidden[0].pop()
                                current_nodes_hidden[0].pop()
                                current_num = sub_stree.embedding
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                            current_nodes[0].append(out_token)
                            current_nodes_hidden[0].append(hidden[0])
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        parent_flag = True
                        if len(current_nodes[0]) == 0:
                            left.append(self.root)
                            prev.append(self.root)
                            parent.append(hidden[0])
                        elif current_nodes[0][-1] < self.num_start:
                            left.append(self.root)
                            prev.append(self.node_generater.embeddings(torch.LongTensor([current_nodes[0][-1]])))
                            parent.append(current_nodes_hidden[0][-1])
                        else:
                            left.append(current_nums_embeddings[0, current_nodes[0][-1] - self.num_start].unsqueeze(0))
                            prev.append(current_nums_embeddings[0, current_nodes[0][-1] - self.num_start].unsqueeze(0))
                            for i in range(len(current_nodes[0]) - 1, -1, -1):
                                if current_nodes[0][i] < self.num_start:
                                    parent.append(current_nodes_hidden[0][i])
                                    parent_flag = False
                                    break
                            if parent_flag:
                                parent.append(hidden[0])
                        current_beams.append((parent, left, prev, current_nodes, current_nodes_hidden, hidden, tree_hidden, TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit)))
                beams = sorted(current_beams, key=lambda x: x[7].score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for _, _, _, _, _, _, _, b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['token_logits'] = token_logits
            all_layer_outputs['outputs'] = outputs
            all_layer_outputs['target'] = target
        return token_logits, outputs, all_layer_outputs

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [(1) for _ in range(hidden_size)]
        temp_0 = [(0) for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                if i == -1:
                    indices.append(0)
                    masked_index.append(temp_1)
                    continue
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [(0) for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        all_outputs = encoder_outputs.contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def copy_list(self, l):
        r = []
        if len(l) == 0:
            return r
        for i in l:
            if type(i) is list:
                r.append(self.copy_list(i))
            else:
                r.append(i)
        return r

    def convert_idx2symbol(self, output, num_list, num_stack):
        """batch_size=1"""
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if 'NUM' in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class BinaryTree(AbstractTree):
    """binary tree
    """

    def __init__(self, root_node=None):
        super().__init__()
        self.root = root_node

    def equ2tree(self, equ_list, out_idx2symbol, op_list, input_var, emb):
        stack = []
        for idx in equ_list:
            if idx == out_idx2symbol.index(SpecialTokens.PAD_TOKEN):
                break
            if idx == out_idx2symbol.index(SpecialTokens.EOS_TOKEN):
                break
            if out_idx2symbol[idx] in op_list:
                node = Node(idx, isleaf=False)
                node.set_right_node(stack.pop())
                node.set_left_node(stack.pop())
                stack.append(node)
            else:
                node = Node(idx, isleaf=True)
                position = (input_var == idx).nonzero()
                node.node_embeding = emb[position]
                stack.append(node)
        self.root = stack.pop()

    def equ2tree_(self, equ_list):
        stack = []
        for symbol in equ_list:
            if symbol in [SpecialTokens.EOS_TOKEN, SpecialTokens.PAD_TOKEN]:
                break
            if symbol in ['+', '-', '*', '/', '^', '=', SpecialTokens.BRG_TOKEN, SpecialTokens.OPT_TOKEN]:
                node = Node(symbol, isleaf=False)
                node.set_right_node(stack.pop())
                node.set_left_node(stack.pop())
                stack.append(node)
            else:
                node = Node(symbol, isleaf=True)
                stack.append(node)
        if len(stack) > 1:
            raise IndexError
        self.root = stack.pop()

    def tree2equ(self, node):
        equation = []
        if node.is_leaf:
            equation.append(node.node_value)
            return equation
        right_equ = self.tree2equ(node.right_node)
        left_equ = self.tree2equ(node.left_node)
        equation = left_equ + right_equ + [node.node_value]
        return equation


class RecursiveNN(nn.Module):

    def __init__(self, emb_size, op_size, op_list):
        super().__init__()
        self.emb_size = emb_size
        self.op_size = op_size
        self.W = nn.Linear(emb_size * 2, emb_size, bias=True)
        self.generate_linear = nn.Linear(emb_size, op_size, bias=True)
        self.classes = op_list

    def forward(self, expression_tree, num_embedding, look_up, out_idx2symbol):
        device = num_embedding.device
        self.out_idx2symbol = out_idx2symbol
        self.leaf_emb(expression_tree, num_embedding, look_up)
        self.nodeProbList = []
        self.labelList = []
        _ = self.traverse(expression_tree)
        if self.nodeProbList != []:
            nodeProb = torch.cat(self.nodeProbList, dim=0)
            label = torch.tensor(self.labelList)
        else:
            nodeProb = self.nodeProbList
            label = self.labelList
        return nodeProb, label

    def test(self, expression_tree, num_embedding, look_up, out_idx2symbol):
        device = num_embedding.device
        self.out_idx2symbol = out_idx2symbol
        self.leaf_emb(expression_tree, num_embedding, look_up)
        self.nodeProbList = []
        self.labelList = []
        _ = self.test_traverse(expression_tree)
        if self.nodeProbList != []:
            nodeProb = torch.cat(self.nodeProbList, dim=0)
            label = torch.tensor(self.labelList)
        else:
            nodeProb = self.nodeProbList
            label = self.labelList
        return nodeProb, label, expression_tree

    def leaf_emb(self, node, num_embed, look_up):
        if node.is_leaf:
            symbol = node.node_value
            if symbol not in look_up:
                node.embedding = num_embed[0]
            else:
                node.embedding = num_embed[look_up.index(symbol)]
        else:
            self.leaf_emb(node.left_node, num_embed, look_up)
            self.leaf_emb(node.right_node, num_embed, look_up)

    def traverse(self, node):
        if node.is_leaf:
            currentNode = node.embedding.unsqueeze(0)
        else:
            left_vector = self.traverse(node.left_node)
            right_vector = self.traverse(node.right_node)
            combined_v = torch.cat((left_vector, right_vector), 1)
            currentNode, op_prob = self.RecurCell(combined_v)
            node.embedding = currentNode.squeeze(0)
            self.nodeProbList.append(op_prob)
            self.labelList.append(self.classes.index(node.node_value))
        return currentNode

    def test_traverse(self, node):
        if node.is_leaf:
            currentNode = node.embedding.unsqueeze(0)
        else:
            left_vector = self.test_traverse(node.left_node)
            right_vector = self.test_traverse(node.right_node)
            combined_v = torch.cat((left_vector, right_vector), 1)
            currentNode, op_prob = self.RecurCell(combined_v)
            node.embedding = currentNode.squeeze(0)
            op_idx = torch.topk(op_prob, 1, 1)[1]
            self.nodeProbList.append(op_prob)
            node.node_value = self.classes[op_idx]
            self.labelList.append(self.classes.index(node.node_value))
        return currentNode

    def RecurCell(self, combine_emb):
        node_embedding = torch.tanh(self.W(combine_emb))
        op = self.generate_linear(node_embedding)
        return node_embedding, op


class TRNN(nn.Module):
    """
    Reference:
        Wang et al. "Template-Based Math Word Problem Solvers with Recursive Neural Networks" in AAAI 2019.
    """

    def __init__(self, config, dataset):
        super(TRNN, self).__init__()
        self.device = config['device']
        self.seq2seq_embedding_size = config['seq2seq_embedding_size']
        self.seq2seq_encode_hidden_size = config['seq2seq_encode_hidden_size']
        self.seq2seq_decode_hidden_size = config['seq2seq_decode_hidden_size']
        self.num_layers = config['seq2seq_num_layers']
        self.teacher_force_ratio = config['teacher_force_ratio']
        self.seq2seq_dropout_ratio = config['seq2seq_dropout_ratio']
        self.ans_embedding_size = config['ans_embedding_size']
        self.ans_hidden_size = config['ans_hidden_size']
        self.ans_dropout_ratio = config['ans_dropout_ratio']
        self.ans_num_layers = config['ans_num_layers']
        self.encoder_rnn_cell_type = config['encoder_rnn_cell_type']
        self.decoder_rnn_cell_type = config['decoder_rnn_cell_type']
        self.max_gen_len = config['max_output_len']
        self.bidirectional = config['bidirectional']
        self.attention = True
        self.share_vocab = config['share_vocab']
        self.embedding = config['embedding']
        self.mask_list = NumMask.number
        self.in_idx2word = dataset.in_idx2word
        self.out_idx2symbol = dataset.out_idx2symbol
        self.temp_idx2symbol = dataset.temp_idx2symbol
        self.vocab_size = len(dataset.in_idx2word)
        self.symbol_size = len(dataset.out_idx2symbol)
        self.temp_symbol_size = len(dataset.temp_idx2symbol)
        self.operator_nums = len(dataset.operator_list)
        self.operator_list = dataset.operator_list
        self.generate_list = [SpecialTokens.UNK_TOKEN] + dataset.generate_list
        self.generate_idx = [self.in_idx2word.index(num) for num in self.generate_list]
        if self.share_vocab:
            self.sos_token_idx = dataset.in_word2idx[SpecialTokens.SOS_TOKEN]
        else:
            self.sos_token_idx = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        try:
            self.out_sos_token = dataset.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = dataset.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        try:
            self.temp_sos_token = dataset.temp_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.temp_sos_token = None
        try:
            self.temp_eos_token = dataset.temp_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.temp_eos_token = None
        try:
            self.temp_pad_token = dataset.temp_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.temp_pad_token = None
        if config['embedding'] == 'roberta':
            self.seq2seq_in_embedder = RobertaEmbedder(self.vocab_size, config['pretrained_model_path'])
            self.seq2seq_in_embedder.token_resize(self.vocab_size)
        elif config['embedding'] == 'bert':
            self.seq2seq_in_embedder = BertEmbedder(self.vocab_size, config['pretrained_model_path'])
            self.seq2seq_in_embedder.token_resize(self.vocab_size)
        else:
            self.seq2seq_in_embedder = BasicEmbedder(self.vocab_size, self.seq2seq_embedding_size, self.seq2seq_dropout_ratio)
        if self.share_vocab:
            self.seq2seq_out_embedder = self.seq2seq_in_embedder
        else:
            self.seq2seq_out_embedder = BasicEmbedder(self.temp_symbol_size, self.seq2seq_embedding_size, self.seq2seq_dropout_ratio)
        self.seq2seq_encoder = BasicRNNEncoder(self.seq2seq_embedding_size, self.seq2seq_encode_hidden_size, self.num_layers, self.encoder_rnn_cell_type, self.seq2seq_dropout_ratio, self.bidirectional)
        self.seq2seq_decoder = AttentionalRNNDecoder(self.seq2seq_embedding_size, self.seq2seq_decode_hidden_size, self.seq2seq_encode_hidden_size, self.num_layers, self.decoder_rnn_cell_type, self.seq2seq_dropout_ratio)
        self.seq2seq_gen_linear = nn.Linear(self.seq2seq_encode_hidden_size, self.temp_symbol_size)
        if config['embedding'] == 'roberta':
            self.answer_in_embedder = RobertaEmbedder(self.vocab_size, config['pretrained_model_path'])
            self.answer_in_embedder.token_resize(self.vocab_size)
        elif config['embedding'] == 'bert':
            self.answer_in_embedder = BertEmbedder(self.vocab_size, config['pretrained_model_path'])
            self.answer_in_embedder.token_resize(self.vocab_size)
        else:
            self.answer_in_embedder = BasicEmbedder(self.vocab_size, self.ans_embedding_size, self.ans_dropout_ratio)
        self.answer_encoder = SelfAttentionRNNEncoder(self.ans_embedding_size, self.ans_hidden_size, self.ans_embedding_size, self.num_layers, self.encoder_rnn_cell_type, self.ans_dropout_ratio, self.bidirectional)
        self.answer_rnn = RecursiveNN(self.ans_embedding_size, self.operator_nums, self.operator_list)
        weight = torch.ones(self.temp_symbol_size)
        pad = dataset.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        self.seq2seq_loss = NLLLoss(weight, pad)
        weight2 = torch.ones(self.operator_nums)
        self.ans_module_loss = NLLLoss(weight2, size_average=True)
        self.wrong = 0

    def forward(self, seq, seq_length, seq_mask, num_pos, template_target=None, equation_target=None, output_all_layers=False):
        seq2seq_token_logits, seq2seq_outputs, seq2seq_layer_outputs = self.seq2seq_forward(seq, seq_length, template_target, output_all_layers)
        if equation_target:
            template = None
        else:
            template = self.convert_temp_idx2symbol(seq2seq_outputs)
        ans_token_logits, ans_outputs, ans_module_layer_outputs = self.ans_module_forward(seq, seq_length, seq_mask, template, num_pos, equation_target, output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(seq2seq_layer_outputs)
            model_all_outputs.update(ans_module_layer_outputs)
        return (seq2seq_token_logits, ans_token_logits), (seq2seq_outputs, ans_outputs), model_all_outputs

    def calculate_loss(self, batch_data: dict) ->Tuple[float, float]:
        """Finish forward-propagating, calculating loss and back-propagation.

        :param batch_data: one batch data.
        :return: seq2seq module loss, answer module loss.
        """
        seq2seq_loss = self.seq2seq_calculate_loss(batch_data)
        answer_loss = self.ans_module_calculate_loss(batch_data)
        return seq2seq_loss, answer_loss

    def model_test(self, batch_data: dict) ->tuple:
        """Model test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.
        batch_data should include keywords 'question', 'ques len', 'equation', 'ques mask',
        'num pos', 'num list', 'template'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        seq_mask = torch.BoolTensor(batch_data['ques mask'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        template_target = self.convert_temp_idx2symbol(torch.tensor(batch_data['template']))
        _, output_template, _ = self.seq2seq_forward(seq, seq_length)
        template = self.convert_temp_idx2symbol(output_template)
        _, _, ans_module_layers = self.ans_module_forward(seq, seq_length, seq_mask, template, num_pos, output_all_layers=True)
        equations = ans_module_layers['ans_model_equation_outputs']
        _, _, ans_module_layers = self.ans_module_forward(seq, seq_length, seq_mask, template_target, num_pos, output_all_layers=True)
        ans_module_test = ans_module_layers['ans_model_equation_outputs']
        equations = self.mask2num(equations, num_list)
        ans_module_test = self.mask2num(ans_module_test, num_list)
        targets = self.convert_idx2symbol(target, num_list)
        temp_t = template_target
        return equations, targets, template, temp_t, ans_module_test, targets

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        ques_mask = torch.BoolTensor(batch_data['ques mask'])
        num_pos = batch_data['num pos']
        token_logits, symbol_outputs, model_all_outputs = self.forward(seq, seq_length, ques_mask, num_pos, output_all_layers=output_all_layers)
        return token_logits, symbol_outputs, model_all_outputs

    def seq2seq_calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation of seq2seq module.

        :param batch_data: one batch data.
        :return: loss value of seq2seq module.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['template'])
        token_logits, _, _ = self.seq2seq_forward(seq, seq_length, target)
        if self.share_vocab:
            target = self.convert_in_idx_2_temp_idx(target)
        outputs = torch.nn.functional.log_softmax(token_logits, dim=-1)
        self.seq2seq_loss.reset()
        self.seq2seq_loss.eval_batch(outputs.view(-1, outputs.size(-1)), target.view(-1))
        self.seq2seq_loss.backward()
        return self.seq2seq_loss.get_loss()

    def ans_module_calculate_loss(self, batch_data):
        """Finish forward-propagating, calculating loss and back-propagation of answer module.

        :param batch_data: one batch data.
        :return: loss value of answer module.
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        seq_mask = torch.BoolTensor(batch_data['ques mask'])
        num_pos = batch_data['num pos']
        equ_source = copy.deepcopy(batch_data['equ_source'])
        for idx, equ in enumerate(equ_source):
            equ_source[idx] = equ.split(' ')
        template = equ_source
        token_logits, _, ans_module_layers = self.ans_module_forward(seq, seq_length, seq_mask, template, num_pos, equation_target=template, output_all_layers=True)
        target = ans_module_layers['ans_module_target']
        self.ans_module_loss.reset()
        for b_i in range(len(target)):
            if not isinstance(token_logits[b_i], list):
                output = torch.nn.functional.log_softmax(token_logits[b_i], dim=1)
                self.ans_module_loss.eval_batch(output, target[b_i].view(-1))
        self.ans_module_loss.backward()
        return self.ans_module_loss.get_loss()

    def seq2seq_generate_t(self, encoder_outputs, encoder_hidden, decoder_inputs):
        with_t = random.random()
        if with_t < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden)
            token_logits = self.seq2seq_gen_linear(decoder_outputs)
            token_logits = token_logits.view(-1, token_logits.size(-1))
            token_logits = torch.nn.functional.log_softmax(token_logits, dim=1)
        else:
            seq_len = decoder_inputs.size(1)
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            token_logits = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden)
                step_output = decoder_output.squeeze(1)
                token_logit = self.seq2seq_gen_linear(step_output)
                predict = torch.nn.functional.log_softmax(token_logit, dim=1)
                output = predict.topk(1, dim=1)[1]
                token_logits.append(predict)
                if self.share_vocab:
                    output = self.convert_temp_idx_2_in_idx(output)
                    decoder_input = self.seq2seq_out_embedder(output)
                else:
                    decoder_input = self.seq2seq_out_embedder(output)
            token_logits = torch.stack(token_logits, dim=1)
            token_logits = token_logits.view(-1, token_logits.size(-1))
        return token_logits

    def seq2seq_generate_without_t(self, encoder_outputs, encoder_hidden, decoder_input):
        all_outputs = []
        decoder_hidden = encoder_hidden
        for idx in range(self.max_gen_len):
            if self.attention:
                decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden)
            step_output = decoder_output.squeeze(1)
            token_logits = self.seq2seq_gen_linear(step_output)
            predict = torch.nn.functional.log_softmax(token_logits, dim=1)
            output = predict.topk(1, dim=1)[1]
            all_outputs.append(output)
            if self.share_vocab:
                output = self.convert_temp_idx_2_in_idx(output)
                decoder_input = self.seq2seq_out_embedder(output)
            else:
                decoder_input = self.seq2seq_out_embedder(output)
        all_outputs = torch.cat(all_outputs, dim=1)
        return all_outputs

    def seq2seq_forward(self, seq, seq_length, target=None, output_all_layers=False):
        batch_size = seq.size(0)
        device = seq.device
        seq_emb = self.seq2seq_in_embedder(seq)
        encoder_outputs, encoder_hidden, encoder_layer_outputs = self.seq2seq_encoder_forward(seq_emb, seq_length, output_all_layers)
        decoder_inputs = self.init_seq2seq_decoder_inputs(target, device, batch_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.seq2seq_decoder_forward(encoder_outputs, encoder_hidden, decoder_inputs, target, output_all_layers)
        seq2seq_all_outputs = {}
        if output_all_layers:
            seq2seq_all_outputs['seq2seq_inputs_embedding'] = seq_emb
            seq2seq_all_outputs.update(encoder_layer_outputs)
            seq2seq_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, seq2seq_all_outputs

    def ans_module_forward(self, seq, seq_length, seq_mask, template, num_pos, equation_target=None, output_all_layers=False):
        if self.embedding == 'roberta':
            seq_emb = self.answer_in_embedder(seq, seq_mask)
        else:
            seq_emb = self.answer_in_embedder(seq)
        encoder_output, encoder_hidden = self.answer_encoder(seq_emb, seq_length)
        batch_size = encoder_output.size(0)
        generate_num = torch.tensor(self.generate_idx)
        if self.embedding == 'roberta':
            generate_emb = self.answer_in_embedder(generate_num, None)
        else:
            generate_emb = self.answer_in_embedder(generate_num)
        batch_prob = []
        batch_target = []
        outputs = []
        equations = []
        input_template = equation_target if equation_target else template
        if equation_target is not None:
            for b_i in range(batch_size):
                try:
                    tree_i = self.template2tree(input_template[b_i])
                except IndexError:
                    outputs.append([])
                    continue
                look_up = self.generate_list + NumMask.number[:len(num_pos[b_i])]
                num_encoding = seq_emb[b_i, num_pos[b_i]] + encoder_output[b_i, num_pos[b_i]]
                num_embedding = torch.cat([generate_emb, num_encoding], dim=0)
                assert len(look_up) == len(num_embedding)
                prob, target = self.answer_rnn(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
                batch_prob.append(prob)
                batch_target.append(target)
                if not isinstance(prob, list):
                    output = torch.topk(prob, 1)[1]
                    outputs.append(output)
                else:
                    outputs.append([])
        else:
            for b_i in range(batch_size):
                try:
                    tree_i = self.template2tree(input_template[b_i])
                except IndexError:
                    outputs.append([])
                    continue
                look_up = self.generate_list + NumMask.number[:len(num_pos[b_i])]
                num_encoding = seq_emb[b_i, num_pos[b_i]] + encoder_output[b_i, num_pos[b_i]]
                num_embedding = torch.cat([generate_emb, num_encoding], dim=0)
                assert len(look_up) == len(num_embedding)
                prob, output, node_pred = self.answer_rnn.test(tree_i.root, num_embedding, look_up, self.out_idx2symbol)
                batch_prob.append(prob)
                tree_i.root = node_pred
                outputs.append(output)
                equation = self.tree2equation(tree_i)
                equations.append(equation)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['ans_module_token_logits'] = batch_prob
            all_layer_outputs['ans_module_target'] = batch_target
            all_layer_outputs['ans_model_outputs'] = outputs
            all_layer_outputs['ans_model_equation_outputs'] = equations
        return batch_prob, outputs, all_layer_outputs

    def seq2seq_encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_outputs, encoder_hidden = self.seq2seq_encoder(seq_emb, seq_length)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, self.seq2seq_encode_hidden_size:] + encoder_outputs[:, :, :self.seq2seq_encode_hidden_size]
            if self.encoder_rnn_cell_type == 'lstm':
                encoder_hidden = encoder_hidden[0][::2].contiguous(), encoder_hidden[1][::2].contiguous()
            else:
                encoder_hidden = encoder_hidden[::2].contiguous()
        if self.encoder_rnn_cell_type == self.decoder_rnn_cell_type:
            pass
        elif self.encoder_rnn_cell_type == 'gru' and self.decoder_rnn_cell_type == 'lstm':
            encoder_hidden = encoder_hidden, encoder_hidden
        elif self.encoder_rnn_cell_type == 'rnn' and self.decoder_rnn_cell_type == 'lstm':
            encoder_hidden = encoder_hidden, encoder_hidden
        elif self.encoder_rnn_cell_type == 'lstm' and (self.decoder_rnn_cell_type == 'gru' or self.decoder_rnn_cell_type == 'rnn'):
            encoder_hidden = encoder_hidden[0]
        else:
            pass
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['seq2seq_encoder_outputs'] = encoder_outputs
            all_layer_outputs['seq2seq_encoder_hidden'] = encoder_hidden
        return encoder_outputs, encoder_hidden, all_layer_outputs

    def seq2seq_decoder_forward(self, encoder_outputs, encoder_hidden, decoder_inputs, target=None, output_all_layers=False):
        if target is not None and random.random() < self.teacher_force_ratio:
            if self.attention:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden, encoder_outputs)
            else:
                decoder_outputs, decoder_states = self.seq2seq_decoder(decoder_inputs, encoder_hidden)
            token_logits = self.seq2seq_gen_linear(decoder_outputs)
            outputs = token_logits.topk(1, dim=-1)[1]
        else:
            seq_len = decoder_inputs.size(1) if target is not None else self.max_gen_len
            decoder_hidden = encoder_hidden
            decoder_input = decoder_inputs[:, 0, :].unsqueeze(1)
            decoder_outputs = []
            token_logits = []
            outputs = []
            for idx in range(seq_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden, encoder_outputs)
                else:
                    decoder_output, decoder_hidden = self.seq2seq_decoder(decoder_input, decoder_hidden)
                step_output = decoder_output.squeeze(1)
                token_logit = self.seq2seq_gen_linear(step_output)
                output = token_logit.topk(1, dim=-1)[1]
                decoder_outputs.append(step_output)
                token_logits.append(token_logit)
                outputs.append(output)
                if self.share_vocab:
                    output = self.convert_temp_idx_2_in_idx(output)
                    decoder_input = self.seq2seq_out_embedder(output)
                else:
                    decoder_input = self.seq2seq_out_embedder(output)
            decoder_outputs = torch.stack(decoder_outputs, dim=1)
            token_logits = torch.stack(token_logits, dim=1)
            outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['seq2seq_decoder_outputs'] = decoder_outputs
            all_layer_outputs['seq2seq_token_logits'] = token_logits
            all_layer_outputs['seq2seq_outputs'] = outputs
        return token_logits, outputs, all_layer_outputs

    def template2tree(self, template):
        tree = BinaryTree()
        tree.equ2tree_(template)
        return tree

    def tree2equation(self, tree):
        equation = tree.tree2equ(tree.root)
        return equation

    def init_seq2seq_decoder_inputs(self, target, device, batch_size):
        pad_var = torch.LongTensor([self.sos_token_idx] * batch_size).view(batch_size, 1)
        if target is not None:
            decoder_inputs = torch.cat((pad_var, target), dim=1)[:, :-1]
        else:
            decoder_inputs = pad_var
        decoder_inputs = self.seq2seq_out_embedder(decoder_inputs)
        return decoder_inputs

    def convert_temp_idx_2_in_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.in_word2idx[self.temp_idx2symbol[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_in_idx_2_temp_idx(self, output):
        device = output.device
        batch_size = output.size(0)
        seq_len = output.size(1)
        decoded_output = []
        for b_i in range(batch_size):
            output_i = []
            for s_i in range(seq_len):
                output_i.append(self.temp_symbol2idx[self.in_idx2word[output[b_i, s_i]]])
            decoded_output.append(output_i)
        decoded_output = torch.tensor(decoded_output).view(batch_size, -1)
        return decoded_output

    def convert_temp_idx2symbol(self, output):
        batch_size = output.size(0)
        seq_len = output.size(1)
        symbol_list = []
        for b_i in range(batch_size):
            symbols = []
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.temp_sos_token, self.temp_eos_token, self.temp_pad_token]:
                    break
                symbol = self.temp_idx2symbol[idx]
                symbols.append(symbol)
            symbol_list.append(symbols)
        return symbol_list

    def convert_idx2symbol(self, output, num_list):
        batch_size = output.size(0)
        seq_len = output.size(1)
        output_list = []
        for b_i in range(batch_size):
            res = []
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                idx = output[b_i][s_i]
                if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                    break
                symbol = self.out_idx2symbol[idx]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list

    def symbol2idx(self, symbols):
        """symbol to idx
        equation symbol to equation idx
        """
        outputs = []
        for symbol in symbols:
            if symbol not in self.out_idx2symbol:
                idx = self.out_idx2symbol.index(SpecialTokens.UNK_TOKEN)
            else:
                idx = self.out_idx2symbol.index(symbol)
            outputs.append(idx)
        return outputs

    def mask2num(self, output, num_list):
        batch_size = len(output)
        output_list = []
        for b_i in range(batch_size):
            res = []
            seq_len = len(output[b_i])
            num_len = len(num_list[b_i])
            for s_i in range(seq_len):
                symbol = output[b_i][s_i]
                if 'NUM' in symbol:
                    num_idx = self.mask_list.index(symbol)
                    if num_idx >= num_len:
                        res.append(symbol)
                    else:
                        res.append(num_list[b_i][num_idx])
                else:
                    res.append(symbol)
            output_list.append(res)
        return output_list


def cosine_sim(logits, logits_1):
    device = logits.device
    return torch.ones(logits.size(0)) + torch.cosine_similarity(logits, logits_1, dim=1)


def cosine_loss(logits, logits_1, length):
    loss_total = []
    for predict, label in zip(logits.split(1, dim=1), logits_1.split(1, dim=1)):
        predict = predict.squeeze()
        label = label.squeeze()
        loss_t = cosine_sim(predict, label)
        loss_total.append(loss_t)
    loss_total = torch.stack(loss_total, dim=0).transpose(1, 0)
    loss_total = loss_total.sum() / length.float().sum()
    return loss_total


def soft_cross_entropy_loss(predict_score, label_score):
    log_softmax = torch.nn.LogSoftmax(dim=1)
    softmax = torch.nn.Softmax(dim=1)
    predict_prob_log = log_softmax(predict_score).float()
    label_prob = softmax(label_score).float()
    loss_elem = -label_prob * predict_prob_log
    loss = loss_elem.sum(dim=1)
    return loss


def soft_target_loss(logits, soft_target, length):
    loss_total = []
    for predict, label in zip(logits.split(1, dim=1), soft_target.split(1, dim=1)):
        predict = predict.squeeze()
        label = label.squeeze()
        loss_t = soft_cross_entropy_loss(predict, label)
        loss_total.append(loss_t)
    loss_total = torch.stack(loss_total, dim=0).transpose(1, 0)
    loss_total = loss_total.sum() / length.float().sum()
    return loss_total


class TSN(nn.Module):
    """
    Reference:
        Zhang et al. "Teacher-Student Networks with Multiple Decoders for Solving Math Word Problem" in IJCAI 2020.
    """

    def __init__(self, config, dataset):
        super(TSN, self).__init__()
        self.hidden_size = config['hidden_size']
        self.bidirectional = config['bidirectional']
        self.device = config['device']
        self.USE_CUDA = True if self.device == torch.device('cuda') else False
        self.beam_size = config['beam_size']
        self.max_out_len = config['max_output_len']
        self.embedding_size = config['embedding_size']
        self.dropout_ratio = config['dropout_ratio']
        self.num_layers = config['num_layers']
        self.rnn_cell_type = config['rnn_cell_type']
        self.alpha = 0.15
        self.max_encoder_mask_len = config['max_encoder_mask_len']
        if self.max_encoder_mask_len == None:
            self.max_encoder_mask_len = 128
        self.vocab_size = len(dataset.in_idx2word)
        self.out_symbol2idx = dataset.out_symbol2idx
        self.out_idx2symbol = dataset.out_idx2symbol
        generate_list = dataset.generate_list
        self.generate_nums = [self.out_symbol2idx[symbol] for symbol in generate_list]
        self.mask_list = NumMask.number
        self.num_start = dataset.num_start
        self.operator_nums = dataset.operator_nums
        self.generate_size = len(generate_list)
        self.unk_token = self.out_symbol2idx[SpecialTokens.UNK_TOKEN]
        try:
            self.out_sos_token = self.out_symbol2idx[SpecialTokens.SOS_TOKEN]
        except:
            self.out_sos_token = None
        try:
            self.out_eos_token = self.out_symbol2idx[SpecialTokens.EOS_TOKEN]
        except:
            self.out_eos_token = None
        try:
            self.out_pad_token = self.out_symbol2idx[SpecialTokens.PAD_TOKEN]
        except:
            self.out_pad_token = None
        self.t_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.t_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio, batch_first=False)
        self.t_decoder = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.t_node_generater = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.t_merge = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.s_embedder = BasicEmbedder(self.vocab_size, self.embedding_size, self.dropout_ratio)
        self.s_encoder = BasicRNNEncoder(self.embedding_size, self.hidden_size, self.num_layers, self.rnn_cell_type, self.dropout_ratio, batch_first=False)
        self.s_decoder_1 = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.s_node_generater_1 = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.s_merge_1 = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.s_decoder_2 = Prediction(self.hidden_size, self.operator_nums, self.generate_size, self.dropout_ratio)
        self.s_node_generater_2 = GenerateNode(self.hidden_size, self.operator_nums, self.embedding_size, self.dropout_ratio)
        self.s_merge_2 = Merge(self.hidden_size, self.embedding_size, self.dropout_ratio)
        self.loss = MaskedCrossEntropyLoss()
        self.soft_target = {}

    def forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False):
        """

        :param seq:
        :param seq_length:
        :param nums_stack:
        :param num_size:
        :param num_pos:
        :param target:
        :param output_all_layers:
        :return:
        """
        t_token_logits, t_symbol_outputs, t_net_all_outputs = self.teacher_net_forward(seq, seq_length, nums_stack, num_size, num_pos, target=target, output_all_layers=output_all_layers)
        s_token_logits, s_symbol_outputs, s_net_all_outputs = self.student_net_forward(seq, seq_length, nums_stack, num_size, num_pos, target=target, output_all_layers=output_all_layers)
        model_all_outputs = {}
        if output_all_layers:
            model_all_outputs.update(t_net_all_outputs)
            model_all_outputs.update(s_net_all_outputs)
            model_all_outputs['soft_target'] = t_token_logits.clone().detach()
        return (t_token_logits, s_token_logits[0], s_token_logits[1]), (t_symbol_outputs, s_symbol_outputs[0], s_symbol_outputs[1]), model_all_outputs

    def teacher_net_forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False) ->Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:[batch_size, output_length, output_size], symbol_outputs:[batch_size,output_length], model_all_outputs.
        :rtype: tuple(torch.Tensor, torch.Tensor, dict)
        """
        seq_mask = torch.eq(seq, self.in_pad_token)
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        batch_size = len(seq_length)
        seq_emb = self.t_embedder(seq)
        problem_output, encoder_outputs, encoder_layer_outputs = self.teacher_net_encoder_forward(seq_emb, seq_length, output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, self.hidden_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.teacher_net_decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target, output_all_layers)
        teacher_net_all_outputs = {}
        if output_all_layers:
            teacher_net_all_outputs['teacher_inputs_embedding'] = seq_emb
            teacher_net_all_outputs.update(encoder_layer_outputs)
            teacher_net_all_outputs['teacher_number_representation'] = all_nums_encoder_outputs
            teacher_net_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, teacher_net_all_outputs

    def student_net_forward(self, seq, seq_length, nums_stack, num_size, num_pos, target=None, output_all_layers=False) ->Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """

        :param torch.Tensor seq: input sequence, shape: [batch_size, seq_length].
        :param torch.Tensor seq_length: the length of sequence, shape: [batch_size].
        :param list nums_stack: different positions of the same number, length:[batch_size]
        :param list num_size: number of numbers of input sequence, length:[batch_size].
        :param list num_pos: number positions of input sequence, length:[batch_size].
        :param torch.Tensor | None target: target, shape: [batch_size, target_length], default None.
        :param bool output_all_layers: return output of all layers if output_all_layers is True, default False.
        :return : token_logits:(token_logits_1,token_logits_2), symbol_outputs:(symbol_outputs_1,symbol_outputs_2), model_all_outputs.
        :rtype: tuple(tuple(torch.Tensor), tuple(torch.Tensor), dict)
        """
        seq_mask = torch.eq(seq, self.in_pad_token)
        num_mask = []
        max_num_size = max(num_size) + len(self.generate_nums)
        for i in num_size:
            d = i + len(self.generate_nums)
            num_mask.append([0] * d + [1] * (max_num_size - d))
        num_mask = torch.BoolTensor(num_mask)
        batch_size = len(seq_length)
        seq_emb = self.t_embedder(seq)
        problem_output, encoder_outputs, encoder_layer_outputs = self.student_net_encoder_forward(seq_emb, seq_length, output_all_layers)
        copy_num_len = [len(_) for _ in num_pos]
        max_num_size = max(copy_num_len)
        all_nums_encoder_outputs = self.get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, self.hidden_size)
        token_logits, symbol_outputs, decoder_layer_outputs = self.student_net_decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target, output_all_layers)
        student_net_all_outputs = {}
        if output_all_layers:
            student_net_all_outputs['student_inputs_embedding'] = seq_emb
            student_net_all_outputs.update(encoder_layer_outputs)
            student_net_all_outputs['student_number_representation'] = all_nums_encoder_outputs
            student_net_all_outputs.update(decoder_layer_outputs)
        return token_logits, symbol_outputs, student_net_all_outputs

    def teacher_calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation of teacher net.

        :param batch_data: one batch data.
        :return: loss value

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.LongTensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        token_logits, _, t_net_layer_outputs = self.teacher_net_forward(seq, seq_length, nums_stack, num_size, num_pos, target, output_all_layers=True)
        target = t_net_layer_outputs['teacher_target']
        loss = masked_cross_entropy(token_logits, target, target_length)
        loss.backward()
        return loss.item()

    def student_calculate_loss(self, batch_data: dict) ->float:
        """Finish forward-propagating, calculating loss and back-propagation of student net.

        :param batch_data: one batch data.
        :return: loss value.

        batch_data should include keywords 'question', 'ques len', 'equation', 'equ len',
        'num stack', 'num size', 'num pos', 'id'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.LongTensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        batch_id = batch_data['id']
        soft_target = self.get_soft_target(batch_id)
        soft_target = torch.cat(soft_target, dim=0)
        token_logits, _, s_net_layer_outputs = self.student_net_forward(seq, seq_length, nums_stack, num_size, num_pos, target, output_all_layers=True)
        token_logits_1, token_logits_2 = token_logits
        target1 = s_net_layer_outputs['student_1_target']
        target2 = s_net_layer_outputs['student_2_target']
        loss1 = masked_cross_entropy(token_logits_1, target1, target_length)
        loss2 = soft_target_loss(token_logits_1, soft_target, target_length)
        loss3 = masked_cross_entropy(token_logits_2, target2, target_length)
        loss4 = soft_target_loss(token_logits_2, soft_target, target_length)
        cos_loss = cosine_loss(token_logits_1, token_logits_2, target_length)
        loss = 0.85 * loss1 + 0.15 * loss2 + 0.85 * loss3 + 0.15 * loss4 + 0.1 * cos_loss
        loss.backward()
        return loss.item()

    def teacher_test(self, batch_data: dict) ->tuple:
        """Teacher net test.

        :param batch_data: one batch data.
        :return: predicted equation, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        _, outputs, _ = self.forward(seq, seq_length, nums_stack, num_size, num_pos)
        all_output = self.convert_idx2symbol(outputs, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output, targets

    def student_test(self, batch_data: dict) ->Tuple[list, float, list, float, list]:
        """Student net test.

        :param batch_data: one batch data.
        :return: predicted equation1, score1, predicted equation2, score2, target equation.

        batch_data should include keywords 'question', 'ques len', 'equation',
        'num stack', 'num pos', 'num list'
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_pos = batch_data['num pos']
        num_list = batch_data['num list']
        num_size = batch_data['num size']
        _, outputs, s_net_layer_outputs = self.student_net_forward(seq, seq_length, nums_stack, num_size, num_pos, output_all_layers=True)
        outputs_1, outputs_2 = outputs
        score1 = s_net_layer_outputs['student_1_score']
        score2 = s_net_layer_outputs['student_2_score']
        all_output1 = self.convert_idx2symbol(outputs_1, num_list[0], copy_list(nums_stack[0]))
        all_output2 = self.convert_idx2symbol(outputs_2, num_list[0], copy_list(nums_stack[0]))
        targets = self.convert_idx2symbol(target[0], num_list[0], copy_list(nums_stack[0]))
        return all_output1, score1, all_output2, score2, targets

    def model_test(self, batch_data):
        return

    def predict(self, batch_data: dict, output_all_layers=False):
        """
        predict samples without target.

        :param dict batch_data: one batch data.
        :param bool output_all_layers: return all layer outputs of model.
        :return: token_logits, symbol_outputs, all_layer_outputs
        """
        raise NotImplementedError

    def teacher_net_encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_inputs = seq_emb.transpose(0, 1)
        pade_outputs, hidden_states = self.t_encoder(encoder_inputs, seq_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['teacher_encoder_outputs'] = encoder_outputs
            all_layer_outputs['teacher_encoder_hidden'] = hidden_states
            all_layer_outputs['teacher_inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def teacher_net_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = problem_output.size(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        if target is not None:
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.t_decoder(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.t_node_generater(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.t_merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.t_decoder(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.t_node_generater(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.t_merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['teacher_token_logits'] = token_logits
            all_layer_outputs['teacher_outputs'] = outputs
            all_layer_outputs['teacher_target'] = target
        return token_logits, outputs, all_layer_outputs

    def student_net_encoder_forward(self, seq_emb, seq_length, output_all_layers=False):
        encoder_inputs = seq_emb.transpose(0, 1)
        pade_outputs, hidden_states = self.s_encoder(encoder_inputs, seq_length)
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        encoder_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['student_encoder_outputs'] = encoder_outputs
            all_layer_outputs['student_encoder_hidden'] = hidden_states
            all_layer_outputs['student_inputs_representation'] = problem_output
        return problem_output, encoder_outputs, all_layer_outputs

    def student_net_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        s_1_token_logits, s_1_outputs, s_1_all_layer_outputs = self.student_net_1_decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=target, output_all_layers=output_all_layers)
        s_2_token_logits, s_2_outputs, s_2_all_layer_outputs = self.student_net_2_decoder_forward(encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=target, output_all_layers=output_all_layers)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs.update(s_1_all_layer_outputs)
            all_layer_outputs.update(s_2_all_layer_outputs)
        return (s_1_token_logits, s_2_token_logits), (s_1_outputs, s_2_outputs), all_layer_outputs

    def student_net_1_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = problem_output.size(0)
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        score = None
        if target is not None:
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.s_node_generater_1(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.s_node_generater_1(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
            score = beams[0].score
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['student_1_token_logits'] = token_logits
            all_layer_outputs['student_1_outputs'] = outputs
            all_layer_outputs['student_1_target'] = target
            all_layer_outputs['student_1_score'] = score
        return token_logits, outputs, all_layer_outputs

    def student_net_2_decoder_forward(self, encoder_outputs, problem_output, all_nums_encoder_outputs, nums_stack, seq_mask, num_mask, target=None, output_all_layers=False):
        batch_size = encoder_outputs.size(1)
        seq_size = encoder_outputs.size(0)
        encoder_outputs_mask = self.encoder_mask[:batch_size, :seq_size, :].transpose(1, 0).float()
        encoder_outputs = encoder_outputs * encoder_outputs_mask.float()
        node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
        padding_hidden = torch.FloatTensor([(0.0) for _ in range(self.hidden_size)]).unsqueeze(0)
        embeddings_stacks = [[] for _ in range(batch_size)]
        left_childs = [None for _ in range(batch_size)]
        token_logits = []
        outputs = []
        score = None
        if target is not None:
            max_target_length = target.size(0)
            for t in range(max_target_length):
                num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                token_logit = torch.cat((op_score, num_score), 1)
                output = torch.topk(token_logit, 1, dim=-1)[1]
                token_logits.append(token_logit)
                outputs.append(output)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), token_logit, nums_stack, self.num_start, self.unk_token)
                target[t] = target_t
                if self.USE_CUDA:
                    generate_input = generate_input
                left_child, right_child, node_label = self.s_node_generater_1(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < self.num_start:
                        node_stack.append(TreeNode(r))
                        node_stack.append(TreeNode(l, left_flag=True))
                        o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - self.num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
                        o.append(TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
        else:
            beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [], [])]
            max_gen_len = self.max_out_len
            for t in range(max_gen_len):
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs
                    num_score, op_score, current_embeddings, current_context, current_nums_embeddings = self.s_decoder_1(b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    token_logit = torch.cat((op_score, num_score), 1)
                    out_score = nn.functional.log_softmax(token_logit, dim=1)
                    topv, topi = out_score.topk(self.beam_size)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy_list(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy_list(b.embedding_stack)
                        current_out = [tl for tl in b.out]
                        current_token_logit = [tl for tl in b.token_logit]
                        current_token_logit.append(token_logit)
                        out_token = int(ti)
                        current_out.append(torch.squeeze(ti, dim=1))
                        node = current_node_stack[0].pop()
                        if out_token < self.num_start:
                            generate_input = torch.LongTensor([out_token])
                            if self.USE_CUDA:
                                generate_input = generate_input
                            left_child, right_child, node_label = self.s_node_generater_1(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(TreeNode(right_child))
                            current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - self.num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.s_merge_1(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks, current_left_childs, current_out, current_token_logit))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beam_size]
                flag = True
                for b in beams:
                    if len(b.node_stack[0]) != 0:
                        flag = False
                if flag:
                    break
            token_logits = beams[0].token_logit
            outputs = beams[0].out
            score = beams[0].score
        token_logits = torch.stack(token_logits, dim=1)
        outputs = torch.stack(outputs, dim=1)
        all_layer_outputs = {}
        if output_all_layers:
            all_layer_outputs['student_2_token_logits'] = token_logits
            all_layer_outputs['student_2_outputs'] = outputs
            all_layer_outputs['student_2_target'] = target
            all_layer_outputs['student_2_score'] = score
        return token_logits, outputs, all_layer_outputs

    def build_graph(self, seq_length, num_list, num_pos, group_nums):
        max_len = seq_length.max()
        batch_size = len(seq_length)
        batch_graph = []
        for b_i in range(batch_size):
            x = torch.zeros((max_len, max_len))
            for idx in range(seq_length[b_i]):
                x[idx, idx] = 1
            quantity_cell_graph = torch.clone(x)
            graph_greater = torch.clone(x)
            graph_lower = torch.clone(x)
            graph_quanbet = torch.clone(x)
            graph_attbet = torch.clone(x)
            for idx, n_pos in enumerate(num_pos[b_i]):
                for pos in group_nums[b_i][idx]:
                    quantity_cell_graph[n_pos, pos] = 1
                    quantity_cell_graph[pos, n_pos] = 1
                    graph_quanbet[n_pos, pos] = 1
                    graph_quanbet[pos, n_pos] = 1
                    graph_attbet[n_pos, pos] = 1
                    graph_attbet[pos, n_pos] = 1
            for idx_i in range(len(num_pos[b_i])):
                for idx_j in range(len(num_pos[b_i])):
                    num_i = str2float(num_list[b_i][idx_i])
                    num_j = str2float(num_list[b_i][idx_j])
                    if num_i > num_j:
                        graph_greater[num_pos[b_i][idx_i]][num_pos[b_i][idx_j]] = 1
                        graph_lower[num_pos[b_i][idx_j]][num_pos[b_i][idx_i]] = 1
                    else:
                        graph_greater[num_pos[b_i][idx_j]][num_pos[b_i][idx_i]] = 1
                        graph_lower[num_pos[b_i][idx_i]][num_pos[b_i][idx_j]] = 1
            group_num_ = itertools.chain.from_iterable(group_nums[b_i])
            combn = itertools.permutations(group_num_, 2)
            for idx in combn:
                graph_quanbet[idx] = 1
                graph_quanbet[idx] = 1
                graph_attbet[idx] = 1
                graph_attbet[idx] = 1
            quantity_cell_graph = quantity_cell_graph
            graph_greater = graph_greater
            graph_lower = graph_lower
            graph_quanbet = graph_quanbet
            graph_attbet = graph_attbet
            graph = torch.stack([quantity_cell_graph, graph_greater, graph_lower, graph_quanbet, graph_attbet], dim=0)
            batch_graph.append(graph)
        batch_graph = torch.stack(batch_graph)
        return batch_graph

    def init_encoder_mask(self, batch_size):
        encoder_mask = torch.FloatTensor(batch_size, self.max_encoder_mask_len, self.hidden_size).uniform_() < 0.99
        self.encoder_mask = encoder_mask.float()

    @torch.no_grad()
    def init_soft_target(self, batch_data):
        """Build soft target
        
        Args:
            batch_data (dict): one batch data.
        
        """
        seq = torch.tensor(batch_data['question'])
        seq_length = torch.tensor(batch_data['ques len']).long()
        target = torch.tensor(batch_data['equation'])
        target_length = torch.tensor(batch_data['equ len'])
        nums_stack = copy.deepcopy(batch_data['num stack'])
        num_size = batch_data['num size']
        num_pos = batch_data['num pos']
        ques_id = batch_data['id']
        all_node_outputs, _, t_net_layer_outputs = self.teacher_net_forward(seq, seq_length, nums_stack, num_size, num_pos, target)
        all_node_outputs = all_node_outputs.cpu()
        for id_, soft_target in zip(ques_id, all_node_outputs.split(1)):
            self.soft_target[id_] = soft_target

    def get_soft_target(self, batch_id):
        soft_tsrget = []
        for id_ in batch_id:
            soft_tsrget.append(self.soft_target[id_])
        return soft_tsrget

    def get_all_number_encoder_outputs(self, encoder_outputs, num_pos, batch_size, num_size, hidden_size):
        indices = list()
        sen_len = encoder_outputs.size(0)
        masked_index = []
        temp_1 = [(1) for _ in range(hidden_size)]
        temp_0 = [(0) for _ in range(hidden_size)]
        for b in range(batch_size):
            for i in num_pos[b]:
                indices.append(i + b * sen_len)
                masked_index.append(temp_0)
            indices += [(0) for _ in range(len(num_pos[b]), num_size)]
            masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
        indices = torch.LongTensor(indices)
        masked_index = torch.BoolTensor(masked_index)
        masked_index = masked_index.view(batch_size, num_size, hidden_size)
        if self.USE_CUDA:
            indices = indices
            masked_index = masked_index
        all_outputs = encoder_outputs.transpose(0, 1).contiguous()
        all_embedding = all_outputs.view(-1, encoder_outputs.size(2))
        all_num = all_embedding.index_select(0, indices)
        all_num = all_num.view(batch_size, num_size, hidden_size)
        return all_num.masked_fill_(masked_index, 0.0)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                max_score = -float('1e12')
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return torch.LongTensor(target), torch.LongTensor(target_input)

    def convert_idx2symbol(self, output, num_list, num_stack):
        """batch_size=1"""
        seq_len = len(output)
        num_len = len(num_list)
        output_list = []
        res = []
        for s_i in range(seq_len):
            idx = output[s_i]
            if idx in [self.out_sos_token, self.out_eos_token, self.out_pad_token]:
                break
            symbol = self.out_idx2symbol[idx]
            if 'NUM' in symbol:
                num_idx = self.mask_list.index(symbol)
                if num_idx >= num_len:
                    res = []
                    break
                res.append(num_list[num_idx])
            elif symbol == SpecialTokens.UNK_TOKEN:
                try:
                    pos_list = num_stack.pop()
                    c = num_list[pos_list[0]]
                    res.append(c)
                except:
                    return None
            else:
                res.append(symbol)
        output_list.append(res)
        return output_list


class AbstractModel(nn.Module):

    def __init__(self):
        super(AbstractModel, self).__init__()

    def calculate_loss(self, batch_data: dict):
        raise NotImplementedError

    def model_test(self, batch_data: dict):
        raise NotImplementedError

    def predict(self, batch_data: dict, output_all_layers: bool=False):
        raise NotImplementedError

    @classmethod
    def load_from_pretrained(cls, pretrained_dir):
        raise NotImplementedError

    def save_model(self, trained_dir):
        raise NotImplementedError

    def __str__(self):
        info = super().__str__()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        parameters = '\ntotal parameters : {} \ntrainable parameters : {}'.format(total, trainable)
        return info + parameters


class MultiHeadAttention(nn.Module):
    """Multi-head Attention is proposed in the following paper:
            Attention Is All You Need.
    """

    def __init__(self, embedding_size, num_heads, dropout_ratio=0.0):
        super(MultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        assert self.head_size * num_heads == self.embedding_size, 'embedding size must be divisible by num_heads'
        self.scaling = self.head_size ** -0.5
        self.linear_query = nn.Linear(embedding_size, embedding_size)
        self.linear_key = nn.Linear(embedding_size, embedding_size)
        self.linear_value = nn.Linear(embedding_size, embedding_size)
        nn.init.normal_(self.linear_query.weight, mean=0, std=0.02)
        nn.init.normal_(self.linear_key.weight, mean=0, std=0.02)
        nn.init.normal_(self.linear_value.weight, mean=0, std=0.02)
        self.linear_out = nn.Linear(embedding_size, embedding_size)
        nn.init.normal_(self.linear_out.weight, mean=0, std=0.02)
        self.weight_dropout = nn.Dropout(dropout_ratio)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """
        Multi-head attention

        Args:
            query (torch.Tensor): shape [batch_size, tgt_len, embedding_size].
            key (torch.Tensor): shape [batch_size, src_len, embedding_size].
            value (torch.Tensor): shape [batch_size, src_len, embedding_size].
            key_padding_mask (torch.Tensor): shape [batch_size, src_len].
            attn_mask (torch.BoolTensor): shape [batch_size, tgt_len, src_len].

        Return:
            tuple(torch.Tensor, torch.Tensor):
                attn_repre, shape [batch_size, tgt_len, embedding_size].
                attn_weights, shape [batch_size, tgt_len, src_len].
        """
        device = query.device
        batch_size, tgt_len, embedding_size = query.size()
        src_len = key.size(1)
        assert key.size() == value.size()
        q = self.linear_query(query) * self.scaling
        k = self.linear_key(key)
        v = self.linear_value(value)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        k = k.view(batch_size, src_len, self.num_heads, self.head_size).permute(0, 2, 3, 1)
        v = v.view(batch_size, src_len, self.num_heads, self.head_size).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k)
        assert list(attn_weights.size()) == [batch_size, self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_weights.masked_fill_(attn_mask.unsqueeze(0).unsqueeze(1), float('-inf'))
        if key_padding_mask is not None:
            attn_weights.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = self.weight_dropout(F.softmax(attn_weights, dim=-1))
        attn_repre = torch.matmul(attn_weights, v)
        assert list(attn_repre.size()) == [batch_size, self.num_heads, tgt_len, self.head_size]
        attn_repre = attn_repre.transpose(1, 2).contiguous().view(batch_size, tgt_len, embedding_size)
        attn_repre = self.linear_out(attn_repre)
        attn_weights, _ = attn_weights.max(dim=1)
        return attn_repre, attn_weights


class SelfAttention(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        batch_size = inputs.size(1)
        max_len = inputs.size(0)
        repeat_dims1 = [1, 1, max_len, 1]
        repeat_dims2 = [1, max_len, 1, 1]
        sen1 = inputs.transpose(0, 1).unsqueeze(2)
        sen2 = inputs.transpose(0, 1).unsqueeze(1)
        sen1 = sen1.repeat(repeat_dims1)
        sen2 = sen2.repeat(repeat_dims2)
        energy_in = torch.cat((sen1, sen2), 3)
        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature).squeeze(3)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)
        return attn_energies


class TreeDecoder(nn.Module):
    """
    Seq2tree decoder with Problem aware dynamic encoding
    """

    def __init__(self, hidden_size, op_nums, generate_size, dropout=0.5):
        super(TreeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.generate_size = generate_size
        self.op_nums = op_nums
        self.dropout = nn.Dropout(dropout)
        self.embedding_weight = nn.Parameter(torch.randn(1, generate_size, hidden_size))
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)
        self.ops = nn.Linear(hidden_size * 2, op_nums)
        self.attn = TreeAttention(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, nums_mask):
        current_embeddings = []
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)
        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)
        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node)
        current_attn = self.attn(current_embeddings, encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs)
        batch_size = current_embeddings.size(0)
        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)
        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, nums_mask)
        op = self.ops(leaf_input)
        return num_score, op, current_node, current_context, embedding_weight


class SARTreeDecoder(nn.Module):
    """
    Seq2tree decoder with Semantically-Aligned Regularization
    """

    def __init__(self, hidden_size, op_nums, generate_size, dropout=0.5):
        super(SARTreeDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.generate_size = generate_size
        self.op_nums = op_nums
        self.dropout = nn.Dropout(dropout)
        self.embedding_weight = nn.Parameter(torch.randn(1, generate_size, hidden_size))
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)
        self.ops = nn.Linear(hidden_size * 2, op_nums)
        self.attn = TreeAttention(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)
        self.saligned_attn = TreeAttention(hidden_size, hidden_size)
        self.encoder_linear1 = nn.Linear(hidden_size, hidden_size)
        self.encoder_linear2 = nn.Linear(hidden_size, hidden_size)
        self.decoder_linear1 = nn.Linear(hidden_size, hidden_size)
        self.decoder_linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, nums_mask):
        """
        Args:
            node_stacks (list): node stacks.
            left_childs (list): representation of left childs.
            encoder_outputs (torch.Tensor): output from encoder, shape [sequence_length, batch_size, hidden_size].
            num_pades (torch.Tensor): number representation, shape [batch_size, number_size, hidden_size].
            padding_hidden (torch.Tensor): padding hidden, shape [1,hidden_size].
            seq_mask (torch.BoolTensor): sequence mask, shape [batch_size, sequence_length].
            mask_nums (torch.BoolTensor): number mask, shape [batch_size, number_size]

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
                num_score, number score, shape [batch_size, number_size].
                op, operator score, shape [batch_size, operator_size].
                current_node, current node representation, shape [batch_size, 1, hidden_size].
                current_context, current context representation, shape [batch_size, 1, hidden_size].
                embedding_weight, embedding weight, shape [batch_size, number_size, hidden_size].
        """
        current_embeddings = []
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)
        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                node_emb = g * t
                current_node_temp.append(node_emb)
        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node)
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))
        batch_size = current_embeddings.size(0)
        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)
        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, nums_mask)
        op = self.ops(leaf_input)
        return num_score, op, current_node, current_context, embedding_weight

    def Semantically_Aligned_Regularization(self, subtree_emb, s_aligned_vector):
        """
        Args:
            subtree_emb (torch.Tensor):
            s_aligned_vector (torch.Tensor):

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                s_aligned_a
                s_aligned_d
        """
        s_aligned_a = self.encoder_linear2(torch.tanh(self.encoder_linear1(s_aligned_vector)))
        s_aligned_d = self.decoder_linear2(torch.tanh(self.decoder_linear1(subtree_emb)))
        return s_aligned_a, s_aligned_d


class PositionEmbedder_x(nn.Module):

    def __init__(self, embedding_size, max_len=1024):
        super(PositionEmbedder_x, self).__init__()
        pe = torch.zeros(max_len, embedding_size)
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, input_embedding):
        """
        Args:
            input_embedding (torch.Tensor): shape [batch_size, sequence_length, embedding_size].
        """
        seq_len = input_embedding.size(1)
        outputs = input_embedding + self.pe.squeeze()[:seq_len]
        return outputs


class DisPositionalEncoding(nn.Module):

    def __init__(self, embedding_size, max_len):
        super(DisPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_size)
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.position_encoding = nn.Embedding(max_len, embedding_size)
        self.position_encoding.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, dis_graph, category_num):
        dis_graph_expend = dis_graph.unsqueeze(1)
        ZeroPad = nn.ZeroPad2d(padding=(0, category_num, 0, category_num))
        dis_graph_expend = ZeroPad(dis_graph_expend)
        input_pos = dis_graph_expend.squeeze(1).long()
        return self.position_encoding(input_pos)


class TransformerEncoder(nn.Module):
    """
    The stacked Transformer encoder layers.
    """

    def __init__(self, embedding_size, ffn_size, num_encoder_layers, num_heads, attn_dropout_ratio=0.0, attn_weight_dropout_ratio=0.0, ffn_dropout_ratio=0.0):
        super(TransformerEncoder, self).__init__()
        self.transformer_layers = nn.ModuleList()
        for _ in range(num_encoder_layers):
            self.transformer_layers.append(TransformerLayer(embedding_size, ffn_size, num_heads, attn_dropout_ratio, attn_weight_dropout_ratio, ffn_dropout_ratio))

    def forward(self, x, kv=None, self_padding_mask=None, output_all_encoded_layers=False):
        """ Implement the encoding process step by step.

        Args:
            x (torch.Tensor): target sequence embedding, shape: [batch_size, sequence_length, embedding_size].
            kv (torch.Tensor): the cached history latent vector, shape: [batch_size, sequence_length, embedding_size], default: None.
            self_padding_mask (torch.Tensor): padding mask of target sequence, shape: [batch_size, sequence_length], default: None.
            output_all_encoded_layers (Bool): whether to output all the encoder layers, default: ``False``.

        Returns:
            torch.Tensor: output features, shape: [batch_size, sequence_length, ffn_size].
        """
        all_encoded_layers = []
        for idx, layer in enumerate(self.transformer_layers):
            x, _, _ = layer(x, kv, self_padding_mask)
            all_encoded_layers.append(x)
        if output_all_encoded_layers:
            return all_encoded_layers
        return all_encoded_layers[-1]


class GenVar(nn.Module):
    """ Module to generate variable embedding.

    Args:
        dim_encoder_state (int): Dimension of the last cell state of encoder
            RNN (output of Encoder module).
        dim_context (int): Dimension of RNN in GenVar module.
        dim_attn_hidden (int): Dimension of hidden layer in attention.
        dim_mlp_hiddens (int): Dimension of hidden layers in the MLP
            that transform encoder state to query of attention.
        dropout_rate (int): Dropout rate for attention and MLP.
    """

    def __init__(self, dim_encoder_state, dim_context, dim_attn_hidden=256, dropout_rate=0.5):
        super(GenVar, self).__init__()
        self.attention = Attention(dim_context, dim_encoder_state, dim_attn_hidden, dropout_rate)

    def forward(self, encoder_state, context, context_lens):
        """ Generate embedding for an unknown variable.

        Args:
            encoder_state (torch.FloatTensor): Last cell state of the encoder (output of Encoder module).
            context (torch.FloatTensor): Encoded context, with size [batch_size, text_len, dim_hidden].

        Return:
            torch.FloatTensor: Embedding of an unknown variable, with size [batch_size, dim_context]
        """
        attn = self.attention(context, encoder_state.squeeze(0), context_lens)
        return attn


class NodeGenerater(nn.Module):

    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(NodeGenerater, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_left = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_right = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_left_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_right_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)
        l_child = torch.tanh(self.generate_left(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_left_g(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_right(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_right_g(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class SubTreeMerger(nn.Module):

    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(SubTreeMerger, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)
        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AveragePooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DQN,
     lambda: ([], {'input_size': 4, 'embedding_size': 4, 'hidden_size': 4, 'output_size': 4, 'dropout_ratio': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Dec_LSTM,
     lambda: ([], {'embedding_size': 4, 'hidden_size': 4, 'dropout_ratio': 0.5}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (EPTPositionalEncoding,
     lambda: ([], {'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GCN,
     lambda: ([], {'in_feat_dim': 4, 'nhid': 4, 'out_feat_dim': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GateNN,
     lambda: ([], {'hidden_size': 4, 'input1_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GraphConvolution,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Merge,
     lambda: ([], {'hidden_size': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (MultiHeadAttention,
     lambda: ([], {'embedding_size': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (Num_Graph_Module,
     lambda: ([], {'node_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (Parse_Graph_Module,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PositionEmbedder,
     lambda: ([], {'embedding_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (PositionEmbedder_x,
     lambda: ([], {'embedding_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'pos_size': 4, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RelevantScore,
     lambda: ([], {'dim_value': 4, 'dim_query': 4, 'hidden1': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ScoreModel,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SelfAttention,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (SelfAttentionMask,
     lambda: ([], {}),
     lambda: ([0], {}),
     False),
    (Squeeze,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SubTreeMerger,
     lambda: ([], {'hidden_size': 4, 'embedding_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (SublayerConnection,
     lambda: ([], {'size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
]

class Test_LYH_YF_MWPToolkit(_paritybench_base):
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

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

