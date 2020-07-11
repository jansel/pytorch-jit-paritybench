import sys
_module = sys.modules[__name__]
del sys
download = _module
feedly_data = _module
conf = _module
bert = _module
runner = _module
cbow = _module
mlp_parameter_tuning = _module
news = _module
surnames = _module
training = _module
dataset = _module
lm_tuner_runner = _module
model = _module
dataset = _module
model = _module
utils = _module
setup = _module
test_vocabulary = _module
plugins = _module
test_config = _module
test_trainer = _module
trainer_utils = _module
test_config_loader = _module
test_experiment_runner = _module
transfer_nlp = _module
common = _module
tokenizers = _module
utils = _module
embeddings = _module
embeddings = _module
utils = _module
loaders = _module
loaders = _module
vectorizers = _module
vocabulary = _module
config = _module
helpers = _module
metrics = _module
predictors = _module
regularizers = _module
reporters = _module
trainer_abc = _module
trainers = _module
experiment_runner = _module

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


import logging


from typing import Dict


from typing import List


from typing import Any


import numpy as np


import pandas as pd


import torch


import string


from collections import Counter


from typing import Tuple


import random


import torch.nn as nn


import torch.optim as optim


import copy


from typing import Union


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import inspect


from itertools import zip_longest


import re


from abc import abstractmethod


from collections import defaultdict


class DatasetSplits:

    def __init__(self, train_set: Dataset, train_batch_size: int, val_set: Dataset, val_batch_size: int, test_set: Dataset=None, test_batch_size: int=None):
        self.train_set: Dataset = train_set
        self.train_batch_size: int = train_batch_size
        self.val_set: Dataset = val_set
        self.val_batch_size: int = val_batch_size
        self.test_set: Dataset = test_set
        self.test_batch_size: int = test_batch_size

    def train_data_loader(self):
        return DataLoader(self.train_set, self.train_batch_size, shuffle=True)

    def val_data_loader(self):
        return DataLoader(self.val_set, self.val_batch_size, shuffle=False)

    def test_data_loader(self):
        return DataLoader(self.test_set, self.test_batch_size, shuffle=False)


TQDM = True


REGISTRY = {}


def register_plugin(registrable: Any, alias: str=None):
    """
    Register a class, a function or a method to REGISTRY
    Args:
        registrable:
        alias:
    Returns:
    """
    alias = alias or registrable.__name__
    if alias in REGISTRY:
        raise ValueError(f'{alias} is already registered to registrable {REGISTRY[alias]}. Please select another name')
    REGISTRY[alias] = registrable
    return registrable


logger = logging.getLogger(__name__)


class CBOWClassifier(torch.nn.Module):

    def __init__(self, data: DatasetSplits, embedding_size: int, glove_path: str=None, padding_idx: int=0):
        super(CBOWClassifier, self).__init__()
        self.num_embeddings = len(data.vectorizer.data_vocab)
        self.embedding_size = embedding_size
        self.padding_idx = padding_idx
        if glove_path:
            logger.info('Using pre-trained word embeddings...')
            self.embeddings = Embedding(glove_filepath=glove_path, data=data).embeddings
            self.embeddings = torch.from_numpy(self.embeddings).float()
            glove_size = len(self.embeddings[0])
            self.embedding: torch.nn.Embedding = torch.nn.Embedding(embedding_dim=glove_size, num_embeddings=self.num_embeddings, padding_idx=self.padding_idx, _weight=self.embeddings)
        else:
            logger.info('Not using pre-trained word embeddings...')
            self.embedding: torch.nn.Embedding = torch.nn.Embedding(embedding_dim=self.embedding_size, num_embeddings=self.num_embeddings, padding_idx=self.padding_idx)
        self.fc1 = torch.nn.Linear(in_features=embedding_size, out_features=self.num_embeddings)
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x_in: torch.Tensor) ->torch.Tensor:
        """

        :param x_in: input data tensor. x_in.shape should be (batch, input_dim)
        :param apply_softmax: flag for the softmax activation
                should be false if used with the Cross Entropy losses
        :return: the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        x_embedded_sum = self.dropout(self.embedding(x_in).sum(dim=1))
        y_out = self.fc1(x_embedded_sum)
        return y_out


class NewsClassifier(torch.nn.Module):

    def __init__(self, data: DatasetSplits, embedding_size: int, num_channels: int, hidden_dim: int, dropout_p: float, padding_idx: int=0, glove_path: str=None):
        super(NewsClassifier, self).__init__()
        self.num_embeddings = len(data.vectorizer.data_vocab)
        self.num_classes = len(data.vectorizer.target_vocab)
        self.num_channels: int = num_channels
        self.embedding_size: int = embedding_size
        self.hidden_dim: int = hidden_dim
        self.padding_idx: int = padding_idx
        if glove_path:
            logger.info('Using pre-trained word embeddings...')
            self.embeddings = Embedding(glove_filepath=glove_path, data=data).embeddings
            self.embeddings = torch.from_numpy(self.embeddings).float()
            glove_size = len(self.embeddings[0])
            self.emb: torch.nn.Embedding = torch.nn.Embedding(embedding_dim=glove_size, num_embeddings=self.num_embeddings, padding_idx=self.padding_idx, _weight=self.embeddings)
        else:
            logger.info('Not using pre-trained word embeddings...')
            self.emb: torch.nn.Embedding = torch.nn.Embedding(embedding_dim=self.embedding_size, num_embeddings=self.num_embeddings, padding_idx=self.padding_idx)
        self.convnet = torch.nn.Sequential(torch.nn.Conv1d(in_channels=self.embedding_size, out_channels=self.num_channels, kernel_size=3), torch.nn.ELU(), torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=2), torch.nn.ELU(), torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=1), torch.nn.ELU(), torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3), torch.nn.ELU())
        self._dropout_p: float = dropout_p
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.fc1: torch.nn.Linear = torch.nn.Linear(self.num_channels, self.hidden_dim)
        self.fc2: torch.nn.Linear = torch.nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool=False) ->torch.Tensor:
        """

        :param x_in: input data tensor
        :param apply_softmax: flag for the softmax activation
                should be false if used with the Cross Entropy losses
        :return: the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        x_embedded = self.emb(x_in).permute(0, 2, 1)
        features = self.convnet(x_embedded)
        remaining_size = features.size(dim=2)
        features = torch.nn.functional.avg_pool1d(features, remaining_size).squeeze(dim=2)
        features = self.dropout(features)
        intermediate_vector = torch.nn.functional.relu(self.dropout(self.fc1(features)))
        prediction_vector = self.fc2(intermediate_vector)
        if apply_softmax:
            prediction_vector = torch.nn.functional.softmax(prediction_vector, dim=1)
        return prediction_vector


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, data: DatasetSplits, hidden_dim: int):
        super(MultiLayerPerceptron, self).__init__()
        self.input_dim = len(data.vectorizer.data_vocab)
        self.hidden_dim = hidden_dim
        self.output_dim = len(data.vectorizer.target_vocab)
        self.fc = torch.nn.Linear(in_features=self.input_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=self.output_dim)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool=False) ->torch.Tensor:
        """
        Linear -> ReLu -> Linear (+ softmax if probabilities needed)
        :param x_in: size (batch, input_dim)
        :param apply_softmax: False if used with the cross entropy loss, True if probability wanted
        :return:
        """
        intermediate = torch.nn.functional.relu(self.fc(x_in))
        output = self.fc2(intermediate)
        if self.output_dim == 1:
            output = output.squeeze()
        if apply_softmax:
            output = torch.nn.functional.softmax(output, dim=1)
        return output


class SurnameClassifierCNN(torch.nn.Module):

    def __init__(self, data: DatasetSplits, num_channels: int):
        super(SurnameClassifierCNN, self).__init__()
        self.initial_num_channels = len(data.vectorizer.data_vocab)
        self.num_classes = len(data.vectorizer.target_vocab)
        self.num_channels: int = num_channels
        self.convnet = torch.nn.Sequential(torch.nn.Conv1d(in_channels=self.initial_num_channels, out_channels=self.num_channels, kernel_size=3), torch.nn.ELU(), torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=2), torch.nn.ELU(), torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3, stride=2), torch.nn.ELU(), torch.nn.Conv1d(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=3), torch.nn.ELU())
        self.fc = torch.nn.Linear(self.num_channels, self.num_classes)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool=False) ->torch.Tensor:
        """
        Conv -> ELU -> ELU -> Conv -> ELU -> Linear
        :param x_in: size (batch, initial_num_channels, max_sequence)
        :param apply_softmax: False if used with the cross entropy loss, True if probability wanted
        :return:
        """
        features = self.convnet(x_in).squeeze(dim=2)
        prediction_vector = self.fc(features)
        if apply_softmax:
            prediction_vector = torch.nn.functional.softmax(prediction_vector, dim=1)
        return prediction_vector


class ElmanRNN(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, batch_first: bool=False):
        super(ElmanRNN, self).__init__()
        self.rnn_cell: torch.nn.RNNCell = torch.nn.RNNCell(input_size, hidden_size)
        self.batch_first: bool = batch_first
        self.hidden_size: int = hidden_size

    def _initial_hidden(self, batch_size: int) ->torch.tensor:
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x_in: torch.Tensor, initial_hidden: torch.Tensor=None) ->torch.Tensor:
        """

        :param x_in: an input data tensor.
                If self.batch_first: x_in.shape = (batch, seq_size, feat_size)
                Else: x_in.shape = (seq_size, batch, feat_size)
        :param initial_hidden: the initial hidden state for the RNN
        :return: The outputs of the RNN at each time step.
                If self.batch_first: hiddens.shape = (batch, seq_size, hidden_size)
                Else: hiddens.shape = (seq_size, batch, hidden_size)
        """
        if self.batch_first:
            batch_size, seq_size, feat_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feat_size = x_in.size()
        hiddens = []
        if initial_hidden is None:
            initial_hidden = self._initial_hidden(batch_size)
            initial_hidden = initial_hidden
        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)
        hiddens = torch.stack(hiddens)
        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)
        return hiddens


def column_gather(y_out: torch.FloatTensor, x_lengths: torch.LongTensor) ->torch.FloatTensor:
    x_lengths = x_lengths.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(x_lengths):
        out.append(y_out[batch_index, column_index])
    return torch.stack(out)


class SurnameClassifierRNN(torch.nn.Module):

    def __init__(self, data: DatasetSplits, embedding_size: int, rnn_hidden_size: int, batch_first: bool=True, padding_idx: int=0):
        super(SurnameClassifierRNN, self).__init__()
        self.num_embeddings = len(data.vectorizer.data_vocab)
        self.num_classes = len(data.vectorizer.target_vocab)
        self.emb: torch.nn.Embedding = torch.nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=embedding_size, padding_idx=padding_idx)
        self.rnn: ElmanRNN = ElmanRNN(input_size=embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)
        self.fc1: torch.nn.Linear = torch.nn.Linear(in_features=rnn_hidden_size, out_features=rnn_hidden_size)
        self.fc2: torch.nn.Linear = torch.nn.Linear(in_features=rnn_hidden_size, out_features=self.num_classes)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x_in: torch.Tensor, x_lengths: torch.Tensor=None, apply_softmax: bool=False) ->torch.Tensor:
        """

        :param x_in: an input data tensor.
                 x_in.shape should be (batch, input_dim)
        :param x_lengths: the lengths of each sequence in the batch.
                 They are used to find the final vector of each sequence
        :param apply_softmax: a flag for the softmax activation
                 should be false if used with the Cross Entropy losses
        :return: the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        x_embedded = self.emb(x_in)
        y_out = self.rnn(x_embedded)
        if x_lengths is not None:
            y_out = column_gather(y_out, x_lengths)
        else:
            y_out = y_out[:, (-1), :]
        y_out = torch.nn.functional.relu(self.fc1(self.dropout(y_out)))
        y_out = self.fc2(self.dropout(y_out))
        if apply_softmax:
            y_out = torch.nn.functional.softmax(y_out, dim=1)
        return y_out


class SurnameConditionedGenerationModel(torch.nn.Module):

    def __init__(self, data: DatasetSplits, char_embedding_size: int, rnn_hidden_size: int, batch_first: bool=True, padding_idx: int=0, dropout_p: float=0.5, conditioned: bool=False):
        super(SurnameConditionedGenerationModel, self).__init__()
        self.char_vocab_size = len(data.vectorizer.data_vocab)
        self.num_nationalities = len(data.vectorizer.target_vocab)
        self.char_emb: torch.nn.Embedding = torch.nn.Embedding(num_embeddings=self.char_vocab_size, embedding_dim=char_embedding_size, padding_idx=padding_idx)
        self.nation_emb: torch.nn.Embedding = None
        self.conditioned = conditioned
        if self.conditioned:
            self.nation_emb = torch.nn.Embedding(num_embeddings=self.num_nationalities, embedding_dim=rnn_hidden_size)
        self.rnn: torch.nn.GRU = torch.nn.GRU(input_size=char_embedding_size, hidden_size=rnn_hidden_size, batch_first=batch_first)
        self.fc: torch.nn.Linear = torch.nn.Linear(in_features=rnn_hidden_size, out_features=self.char_vocab_size)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, x_in: torch.Tensor, nationality_index: int=0, apply_softmax: bool=False) ->torch.Tensor:
        """

        :param x_in: input data tensor, x_in.shape should be (batch, max_seq_size)
        :param nationality_index: The index of the nationality for each data point
                Used to initialize the hidden state of the RNN
        :param apply_softmax: flag for the softmax activation
                should be false if used with the Cross Entropy losses
        :return: the resulting tensor. tensor.shape should be (batch, char_vocab_size)
        """
        x_embedded = self.char_emb(x_in)
        if self.conditioned:
            nationality_embedded = self.nation_emb(nationality_index).unsqueeze(0)
            y_out, _ = self.rnn(x_embedded, nationality_embedded)
        else:
            y_out, _ = self.rnn(x_embedded)
        batch_size, seq_size, feat_size = y_out.shape
        y_out = y_out.contiguous().view(batch_size * seq_size, feat_size)
        y_out = self.fc(self.dropout(y_out))
        if apply_softmax:
            y_out = torch.nn.functional.softmax(y_out, dim=1)
        new_feat_size = y_out.shape[-1]
        y_out = y_out.view(batch_size, seq_size, new_feat_size)
        return y_out


class Transformer(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_embeddings: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool):
        super().__init__()
        self.causal: bool = causal
        self.tokens_embeddings: torch.nn.Embedding = torch.nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings: torch.nn.Embedding = torch.nn.Embedding(num_max_positions, embed_dim)
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)
        self.attentions, self.feed_forwards = torch.nn.ModuleList(), torch.nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = torch.nn.ModuleList(), torch.nn.ModuleList()
        for _ in range(num_layers):
            self.attentions.append(torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(torch.nn.Sequential(torch.nn.Linear(embed_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(torch.nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(torch.nn.LayerNorm(embed_dim, eps=1e-12))
        self.attn_mask = None
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    def forward(self, x):
        """ x has shape [batch, seq length]"""
        padding_mask = x == self.tokenizer.vocab['[PAD]']
        x = x.transpose(0, 1).contiguous()
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)
        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions, self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h
            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
        return h


class TransformerWithLMHead(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool, initializer_range: float):
        """ Transformer with a language modeling head on top (tied weights) """
        super().__init__()
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        num_embeddings = len(tokenizer.vocab)
        self.initializer_range = initializer_range
        self.transformer = Transformer(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal=causal)
        self.lm_head = torch.nn.Linear(embed_dim, num_embeddings, bias=False)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

    def init_weights(self, module):
        """ initialize weights - nn.MultiheadAttention is already initalized by PyTorch (xavier) """
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        """ x has shape [batch, seq length]"""
        hidden_states = self.transformer(x)
        logits = self.lm_head(hidden_states)
        return logits


class TransformerWithClfHead(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool, initializer_range: float, num_classes: int):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.initializer_range = initializer_range
        self.transformer = Transformer(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal=causal)
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        clf_tokens_mask = x.transpose(0, 1).contiguous() == self.tokenizer.vocab['[CLS]']
        hidden_states = self.transformer(x)
        msk = clf_tokens_mask.unsqueeze(-1).float()
        clf_tokens_states = (hidden_states * msk).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)
        return clf_logits


class TransformerWithAdapters(Transformer):

    def __init__(self, adapters_dim, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal):
        """ Transformer with adapters (small bottleneck layers) """
        super().__init__(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal)
        self.adapters_1 = torch.nn.ModuleList()
        self.adapters_2 = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.adapters_1.append(torch.nn.Sequential(torch.nn.Linear(embed_dim, adapters_dim), torch.nn.ReLU(), torch.nn.Linear(adapters_dim, embed_dim)))
            self.adapters_2.append(torch.nn.Sequential(torch.nn.Linear(embed_dim, adapters_dim), torch.nn.ReLU(), torch.nn.Linear(adapters_dim, embed_dim)))

    def forward(self, x):
        """ x has shape [batch, seq length]"""
        padding_mask = x == self.tokenizer.vocab['[PAD]']
        x = x.transpose(0, 1).contiguous()
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)
        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)
        for layer_norm_1, attention, adapter_1, layer_norm_2, feed_forward, adapter_2 in zip(self.layer_norms_1, self.attentions, self.adapters_1, self.layer_norms_2, self.feed_forwards, self.adapters_2):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            x = adapter_1(x) + x
            h = x + h
            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            x = adapter_2(x) + x
            h = x + h
        return h


class TransformerWithClfHeadAndAdapters(torch.nn.Module):

    def __init__(self, adapters_dim: int, embed_dim: int, hidden_dim: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool, initializer_range: float, num_classes: int):
        """ Transformer with a classification head and adapters. """
        super().__init__()
        self.initializer_range: float = initializer_range
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.num_layers = num_layers
        self.transformer: TransformerWithAdapters = TransformerWithAdapters(adapters_dim, embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal=causal)
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        clf_tokens_mask = x.transpose(0, 1).contiguous() == self.tokenizer.vocab['[CLS]']
        hidden_states = self.transformer(x)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)
        return clf_logits


class TransformerWithClfHeadAndLMHead(torch.nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int, num_max_positions: int, num_heads: int, num_layers: int, dropout: float, causal: bool, initializer_range: float, num_classes: int):
        super().__init__()
        self.initializer_range: float = initializer_range
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        num_embeddings = len(self.tokenizer.vocab)
        self.num_layers = num_layers
        self.transformer = Transformer(embed_dim, hidden_dim, num_embeddings, num_max_positions, num_heads, num_layers, dropout, causal=causal)
        self.lm_head = torch.nn.Linear(embed_dim, num_embeddings, bias=False)
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.transformer.tokens_embeddings.weight

    def init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        """ x and clf_tokens_mask have shape [seq length, batch] padding_mask has shape [batch, seq length] """
        clf_tokens_mask = x.transpose(0, 1).contiguous() == self.tokenizer.vocab['[CLS]']
        hidden_states = self.transformer(x)
        lm_logits = self.lm_head(hidden_states)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1).float()).sum(dim=0)
        clf_logits = self.classification_head(clf_tokens_states)
        return lm_logits, clf_logits


class TestModel(torch.nn.Module):

    def __init__(self, data: DatasetSplits, hidden_dim: int):
        super(TestModel, self).__init__()
        self.input_dim = len(data.vectorizer.data_vocab)
        self.output_dim = len(data.vectorizer.target_vocab)
        self.hidden_dim = hidden_dim
        self.fc = torch.nn.Linear(in_features=self.input_dim, out_features=hidden_dim)
        self.fc2 = torch.nn.Linear(in_features=hidden_dim, out_features=self.output_dim)

    def forward(self, x_in: torch.Tensor, apply_softmax: bool=False) ->torch.Tensor:
        intermediate = torch.nn.functional.relu(self.fc(x_in))
        output = self.fc2(intermediate)
        if self.output_dim == 1:
            output = output.squeeze()
        if apply_softmax:
            output = torch.nn.functional.softmax(output, dim=1)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ElmanRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
]

class Test_feedly_transfer_nlp(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

