import sys
_module = sys.modules[__name__]
del sys
master = _module
common = _module
constants = _module
evaluate = _module
evaluators = _module
bert_evaluator = _module
bow_evaluator = _module
classification_evaluator = _module
evaluator = _module
relevance_transfer_evaluator = _module
train = _module
trainers = _module
bert_trainer = _module
bow_trainer = _module
classification_trainer = _module
relevance_transfer_trainer = _module
trainer = _module
datasets = _module
aapd = _module
ag_news = _module
bert_processors = _module
aapd_processor = _module
abstract_processor = _module
agnews_processor = _module
imdb_processor = _module
reuters_processor = _module
robust45_processor = _module
sogou_processor = _module
sst_processor = _module
yelp2014_processor = _module
bow_processors = _module
abstract_processor = _module
dbpedia = _module
download_datasets = _module
imdb = _module
imdb_torchtext = _module
ohsumed = _module
process_datasets = _module
r52 = _module
r8 = _module
reuters = _module
robust04 = _module
robust05 = _module
robust45 = _module
sogou_news = _module
sst = _module
trec6 = _module
twenty_news = _module
yahoo_answers = _module
yelp2014 = _module
yelp_review_polarity = _module
models = _module
args = _module
bert = _module
char_cnn = _module
model = _module
fasttext = _module
model = _module
han = _module
model = _module
sent_level_rnn = _module
word_level_rnn = _module
hbert = _module
model = _module
sentence_encoder = _module
kim_cnn = _module
model = _module
lr = _module
model = _module
reg_lstm = _module
embed_regularize = _module
locked_dropout = _module
model = _module
weight_drop = _module
xml_cnn = _module
model = _module
setup = _module
tasks = _module
relevance_transfer = _module
rerank = _module
resample = _module
utils = _module
optimization = _module
preprocessing = _module

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


import warnings


import numpy as np


import torch


import torch.nn.functional as F


from sklearn import metrics


from torch.utils.data import DataLoader


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data import RandomSampler


import time


from torchtext.vocab import Vectors


import functools


import re


from torch import tensor


from torch.utils.data import Dataset


import random


import logging


from copy import deepcopy


import torch.nn as nn


import torch.onnx


from torch import nn


from sklearn.feature_extraction.text import TfidfVectorizer


from torch.nn import Parameter


from functools import wraps


from collections import defaultdict


import torch.utils.data


import math


class CharCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.is_cuda_enabled = config.cuda
        num_conv_filters = config.num_conv_filters
        output_channel = config.output_channel
        num_affine_neurons = config.num_affine_neurons
        target_class = config.target_class
        input_channel = 68
        self.conv1 = nn.Conv1d(input_channel, num_conv_filters, kernel_size=7)
        self.conv2 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=7)
        self.conv3 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv4 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv5 = nn.Conv1d(num_conv_filters, num_conv_filters, kernel_size=3)
        self.conv6 = nn.Conv1d(num_conv_filters, output_channel, kernel_size=3)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(output_channel, num_affine_neurons)
        self.fc2 = nn.Linear(num_affine_neurons, num_affine_neurons)
        self.fc3 = nn.Linear(num_affine_neurons, target_class)

    def forward(self, x, **kwargs):
        if torch.cuda.is_available() and self.is_cuda_enabled:
            x = x.transpose(1, 2).type(torch.FloatTensor)
        else:
            x = x.transpose(1, 2).type(torch.FloatTensor)
        x = F.max_pool1d(F.relu(self.conv1(x)), 3)
        x = F.max_pool1d(F.relu(self.conv2(x)), 3)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class FastText(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            None
            exit()
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(words_dim, target_class)

    def forward(self, x, **kwargs):
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze(1)
        logit = self.fc1(x)
        return logit


class SentLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        sentence_num_hidden = config.sentence_num_hidden
        word_num_hidden = config.word_num_hidden
        target_class = config.target_class
        self.sentence_context_weights = nn.Parameter(torch.rand(2 * sentence_num_hidden, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_gru = nn.GRU(2 * word_num_hidden, sentence_num_hidden, bidirectional=True)
        self.sentence_linear = nn.Linear(2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        self.fc = nn.Linear(2 * sentence_num_hidden, target_class)
        self.soft_sent = nn.Softmax()

    def forward(self, x):
        sentence_h, _ = self.sentence_gru(x)
        x = torch.tanh(self.sentence_linear(sentence_h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1, 0))
        x = torch.mul(sentence_h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        x = self.fc(x.squeeze(0))
        return x


class WordLevelRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        word_num_hidden = config.word_num_hidden
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        if self.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif self.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif self.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            None
            exit()
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()

    def forward(self, x):
        if self.mode == 'rand':
            x = self.embed(x)
        elif self.mode == 'static':
            x = self.static_embed(x)
        elif self.mode == 'non-static':
            x = self.non_static_embed(x)
        else:
            None
            exit()
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        return x


class HAN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.mode = config.mode
        self.word_attention_rnn = WordLevelRNN(config)
        self.sentence_attention_rnn = SentLevelRNN(config)

    def forward(self, x, **kwargs):
        x = x.permute(1, 2, 0)
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        return self.sentence_attention_rnn(word_attentions)


class HierarchicalBert(nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        input_channels = 1
        ks = 3
        self.sentence_encoder = BertSentenceEncoder.from_pretrained(args.pretrained_model_path, num_labels=args.num_labels)
        self.conv1 = nn.Conv2d(input_channels, args.output_channel, (3, self.sentence_encoder.config.hidden_size), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channels, args.output_channel, (4, self.sentence_encoder.config.hidden_size), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channels, args.output_channel, (5, self.sentence_encoder.config.hidden_size), padding=(4, 0))
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(ks * args.output_channel, args.num_labels)

    def forward(self, input_ids, segment_ids=None, input_mask=None):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """
        input_ids = input_ids.permute(1, 0, 2)
        segment_ids = segment_ids.permute(1, 0, 2)
        input_mask = input_mask.permute(1, 0, 2)
        x_encoded = []
        for i0 in range(len(input_ids)):
            x_encoded.append(self.sentence_encoder(input_ids[i0], input_mask[i0], segment_ids[i0]))
        x = torch.stack(x_encoded)
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(1)
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        if self.args.dynamic_pool:
            x = [self.dynamic_pool(i).squeeze(2) for i in x]
            x = torch.cat(x, 1)
            x = x.view(-1, self.filter_widths * self.output_channel * self.dynamic_pool_length)
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
            x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc1(x)
        return logits, x


class KimCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        output_channel = config.output_channel
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        ks = 3
        input_channel = 1
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        elif config.mode == 'multichannel':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
            input_channel = 2
        else:
            None
            exit()
        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, words_dim), padding=(4, 0))
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(ks * output_channel, target_class)

    def forward(self, x, **kwargs):
        if self.mode == 'rand':
            word_input = self.embed(x)
            x = word_input.unsqueeze(1)
        elif self.mode == 'static':
            static_input = self.static_embed(x)
            x = static_input.unsqueeze(1)
        elif self.mode == 'non-static':
            non_static_input = self.non_static_embed(x)
            x = non_static_input.unsqueeze(1)
        elif self.mode == 'multichannel':
            non_static_input = self.non_static_embed(x)
            static_input = self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1)
        else:
            None
            exit()
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


class LogisticRegression(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.vocab_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = torch.squeeze(x)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


class LockedDropout(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class WeightDrop(torch.nn.Module):

    def __init__(self, module, weights, dropout=0, variational=False):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def null_function(*args, **kwargs):
        return

    def _setup(self):
        if issubclass(type(self.module), torch.nn.RNNBase):
            self.module.flatten_parameters = self.null_function
        for name_w in self.weights:
            None
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            if self.variational:
                mask = torch.autograd.Variable(torch.ones(raw_w.size(0), 1))
                if raw_w.is_cuda:
                    mask = mask
                mask = torch.nn.functional.dropout(mask, p=self.dropout, training=True)
                w = mask.expand_as(raw_w) * raw_w
            else:
                w = torch.nn.functional.dropout(raw_w, p=self.dropout, training=self.training)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1
    X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx, embed.max_norm, embed.norm_type, embed.scale_grad_by_freq, embed.sparse)
    return X


class RegLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        target_class = config.target_class
        self.is_bidirectional = config.bidirectional
        self.has_bottleneck_layer = config.bottleneck_layer
        self.mode = config.mode
        self.tar = config.tar
        self.ar = config.ar
        self.beta_ema = config.beta_ema
        self.wdrop = config.wdrop
        self.embed_droprate = config.embed_droprate
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(config.words_num, config.words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        else:
            None
            exit()
        self.lstm = nn.LSTM(config.words_dim, config.hidden_dim, dropout=config.dropout, num_layers=config.num_layers, bidirectional=self.is_bidirectional, batch_first=True)
        if self.wdrop:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=self.wdrop)
        self.dropout = nn.Dropout(config.dropout)
        if self.has_bottleneck_layer:
            if self.is_bidirectional:
                self.fc1 = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
                self.fc2 = nn.Linear(config.hidden_dim, target_class)
            else:
                self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
                self.fc2 = nn.Linear(config.hidden_dim // 2, target_class)
        elif self.is_bidirectional:
            self.fc1 = nn.Linear(2 * config.hidden_dim, target_class)
        else:
            self.fc1 = nn.Linear(config.hidden_dim, target_class)
        if self.beta_ema > 0:
            self.avg_param = deepcopy(list(p.data for p in self.parameters()))
            if torch.cuda.is_available():
                self.avg_param = [a for a in self.avg_param]
            self.steps_ema = 0.0

    def forward(self, x, lengths=None):
        if self.mode == 'rand':
            x = embedded_dropout(self.embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.embed(x)
        elif self.mode == 'static':
            x = embedded_dropout(self.static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.static_embed(x)
        elif self.mode == 'non-static':
            x = embedded_dropout(self.non_static_embed, x, dropout=self.embed_droprate if self.training else 0) if self.embed_droprate else self.non_static_embed(x)
        else:
            None
            exit()
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        rnn_outs, _ = self.lstm(x)
        rnn_outs_temp = rnn_outs
        if lengths is not None:
            rnn_outs, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=True)
            rnn_outs_temp, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs_temp, batch_first=True)
        x = F.relu(torch.transpose(rnn_outs_temp, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        if self.has_bottleneck_layer:
            x = F.relu(self.fc1(x))
            if self.tar or self.ar:
                return self.fc2(x), rnn_outs.permute(1, 0, 2)
            return self.fc2(x)
        else:
            if self.tar or self.ar:
                return self.fc1(x), rnn_outs.permute(1, 0, 2)
            return self.fc1(x)

    def update_ema(self):
        self.steps_ema += 1
        for p, avg_p in zip(self.parameters(), self.avg_param):
            avg_p.mul_(self.beta_ema).add_((1 - self.beta_ema) * p.data)

    def load_ema_params(self):
        for p, avg_p in zip(self.parameters(), self.avg_param):
            p.data.copy_(avg_p / (1 - self.beta_ema ** self.steps_ema))

    def load_params(self, params):
        for p, avg_p in zip(self.parameters(), params):
            p.data.copy_(avg_p)

    def get_params(self):
        params = deepcopy(list(p.data for p in self.parameters()))
        return params


class XmlCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        dataset = config.dataset
        self.output_channel = config.output_channel
        target_class = config.target_class
        words_num = config.words_num
        words_dim = config.words_dim
        self.mode = config.mode
        self.num_bottleneck_hidden = config.num_bottleneck_hidden
        self.dynamic_pool_length = config.dynamic_pool_length
        self.ks = 3
        input_channel = 1
        if config.mode == 'rand':
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif config.mode == 'static':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
        elif config.mode == 'non-static':
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
        elif config.mode == 'multichannel':
            self.static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=True)
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.TEXT_FIELD.vocab.vectors, freeze=False)
            input_channel = 2
        else:
            None
            exit()
        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (2, words_dim), padding=(1, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (4, words_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (8, words_dim), padding=(7, 0))
        self.dropout = nn.Dropout(config.dropout)
        self.bottleneck = nn.Linear(self.ks * self.output_channel * self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.fc1 = nn.Linear(self.num_bottleneck_hidden, target_class)
        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length)

    def forward(self, x, **kwargs):
        if self.mode == 'rand':
            word_input = self.embed(x)
            x = word_input.unsqueeze(1)
        elif self.mode == 'static':
            static_input = self.static_embed(x)
            x = static_input.unsqueeze(1)
        elif self.mode == 'non-static':
            non_static_input = self.non_static_embed(x)
            x = non_static_input.unsqueeze(1)
        elif self.mode == 'multichannel':
            non_static_input = self.non_static_embed(x)
            static_input = self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1)
        else:
            None
            exit()
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [self.pool(i).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = F.relu(self.bottleneck(x.view(-1, self.ks * self.output_channel * self.dynamic_pool_length)))
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit

