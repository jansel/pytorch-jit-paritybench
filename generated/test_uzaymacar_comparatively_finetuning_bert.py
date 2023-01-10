import sys
_module = sys.modules[__name__]
del sys
baseline_main = _module
main = _module
baseline_models = _module
finetuned_models = _module
data_utils = _module
model_utils = _module
visualization_utils = _module
visualize_attention = _module

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


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import DataLoader


import random


import numpy as np


from torch.autograd import Variable


from torch.utils.data import Dataset


import re


class SimpleRNN(nn.Module):
    """
    Simple model that utilizes BERT tokenizer, custom embedding, a recurrent neural network choice
    of LSTM, dropout, and finally a dense layer for classification.

    @param (str) pretrained_model_name_for_tokenizer: name of the pretrained BERT model for
           tokenizing input sequences
    @param (int) max_vocabulary_size: upper limit for number of tokens in the embedding layer
    @param (int) max_tokenization_length: number of tokens to pad / truncate input sequences to
    @param (int) embedding_dim: dimension size of each token representation for the embedding layer
    @param (int) num_classes: number of classes to distinct between for classification; specify
           2 for binary classification (default: 1)
    @param (int) num_recurrent_layers: number of LSTM layers to utilize (default: 1)
    @param (bool) use_bidirectional: whether to use a bidirectional LSTM or not (default: False)
    @param (int) hidden_size: number of recurrent units in each LSTM cell (default: 128)
    @param (float) dropout_rate: possibility of each neuron to be discarded (default: 0.10)
    @param (bool) use_gpu: whether to utilize GPU (CUDA) or not (default: False)
    """

    def __init__(self, pretrained_model_name_for_tokenizer, max_vocabulary_size, max_tokenization_length, embedding_dim, num_classes=1, num_recurrent_layers=1, use_bidirectional=False, hidden_size=128, dropout_rate=0.1, use_gpu=False):
        super(SimpleRNN, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.use_bidirectional = use_bidirectional
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_for_tokenizer)
        self.tokenizer.max_len = max_tokenization_length
        self.embedding = nn.Embedding(num_embeddings=max_vocabulary_size, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_recurrent_layers, bidirectional=use_bidirectional, batch_first=True)
        self.clf = nn.Linear(in_features=hidden_size * 2 if use_bidirectional else hidden_size, out_features=num_classes)

    def get_tokenizer(self):
        """Function to easily access the BERT tokenizer"""
        return self.tokenizer

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        """Function implementing a forward pass of the model"""
        embedded_output = self.embedding(input_ids)
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
            c0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
        else:
            h0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
            c0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
        lstm_output = self.lstm(embedded_output, (h0, c0))
        sequence_output, _ = lstm_output
        last_timesteps = []
        for i in range(len(attention_mask)):
            last_timesteps.append(attention_mask[i].tolist().index(0) if 0 in attention_mask[i].tolist() else self.tokenizer.max_len - 1)
        if self.use_gpu:
            last_timesteps = torch.tensor(data=last_timesteps)
        else:
            last_timesteps = torch.tensor(data=last_timesteps)
        relative_hidden_size = self.hidden_size * 2 if self.use_bidirectional else self.hidden_size
        last_timesteps = last_timesteps.repeat(1, relative_hidden_size)
        last_timesteps = last_timesteps.view(-1, 1, relative_hidden_size)
        pooled_sequence_output = sequence_output.gather(dim=1, index=last_timesteps).squeeze()
        pooled_sequence_output = self.dropout(pooled_sequence_output)
        logits = self.clf(pooled_sequence_output)
        return logits


class SimpleRNNWithBERTEmbeddings(nn.Module):
    """
    Simple model that utilizes BERT tokenizer, pretrained BERT embedding, a recurrent neural network
    choice of LSTM, dropout, and finally a dense layer for classification.

    @param (str) pretrained_model_name_for_embeddings: name of the pretrained BERT model for
           both tokenizing input sequences and extracting vector representations for each token
    @param (int) max_tokenization_length: number of tokens to pad / truncate input sequences to
    @param (int) num_classes: number of classes to distinct between for classification; specify
           2 for binary classification (default: 1)
    @param (int) num_recurrent_layers: number of LSTM layers to utilize (default: 1)
    @param (bool) use_bidirectional: whether to use a bidirectional LSTM or not (default: False)
    @param (int) hidden_size: number of recurrent units in each LSTM cell (default: 128)
    @param (float) dropout_rate: possibility of each neuron to be discarded (default: 0.10)
    @param (bool) use_gpu: whether to utilize GPU (CUDA) or not (default: False)
    """

    def __init__(self, pretrained_model_name_for_embeddings, max_tokenization_length, num_classes=1, num_recurrent_layers=1, use_bidirectional=False, hidden_size=128, dropout_rate=0.1, use_gpu=False):
        super(SimpleRNNWithBERTEmbeddings, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.use_bidirectional = use_bidirectional
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_for_embeddings)
        self.tokenizer.max_len = max_tokenization_length
        self.config = BertConfig.from_pretrained(pretrained_model_name_for_embeddings)
        all_states_dict = BertModel.from_pretrained(pretrained_model_name_for_embeddings, config=self.config).state_dict()
        self.config.max_position_embeddings = max_tokenization_length
        self.config.num_hidden_layers = 0
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(pretrained_model_name_for_embeddings, config=self.config)
        current_states_dict = self.bert.state_dict()
        for param in current_states_dict.keys():
            if 'embedding' in param:
                current_states_dict[param] = all_states_dict[param]
        self.bert.load_state_dict(current_states_dict)
        logging.info('Loaded %d learnable parameters from pretrained BERT model with %d layer(s)' % (len(list(self.bert.parameters())), 0))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(input_size=self.config.hidden_size, hidden_size=hidden_size, num_layers=num_recurrent_layers, bidirectional=use_bidirectional, batch_first=True)
        self.clf = nn.Linear(in_features=hidden_size * 2 if self.use_bidirectional else hidden_size, out_features=num_classes)

    def get_tokenizer(self):
        """Function to easily access the BERT tokenizer"""
        return self.tokenizer

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        """Function implementing a forward pass of the model"""
        embedded_output = self.bert.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
            c0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
        else:
            h0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
            c0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
        lstm_output = self.lstm(embedded_output, (h0, c0))
        sequence_output, _ = lstm_output
        last_timesteps = []
        for i in range(len(attention_mask)):
            last_timesteps.append(attention_mask[i].tolist().index(0) if 0 in attention_mask[i].tolist() else self.tokenizer.max_len - 1)
        if self.use_gpu:
            last_timesteps = torch.tensor(data=last_timesteps)
        else:
            last_timesteps = torch.tensor(data=last_timesteps)
        relative_hidden_size = self.hidden_size * 2 if self.use_bidirectional else self.hidden_size
        last_timesteps = last_timesteps.repeat(1, relative_hidden_size)
        last_timesteps = last_timesteps.view(-1, 1, relative_hidden_size)
        pooled_sequence_output = sequence_output.gather(dim=1, index=last_timesteps).squeeze()
        pooled_sequence_output = self.dropout(pooled_sequence_output)
        logits = self.clf(pooled_sequence_output)
        return logits


def get_features(input_ids, tokenizer, device):
    """
    Function to get BERT-related features, and helps to build the total input representation.

    @param (Tensor) input_ids: the encoded integer indexes of a batch, with shape: (B, P)
    @param (pytorch_transformers.BertTokenizer) tokenizer: tokenizer with pre-figured mappings
    @param (torch.device) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    @return (Tensor, Tensor) token_type_ids, attention_mask: features describe token type with
            a 0 for the first sentence and a 1 for the pair sentence; enable attention on a
            particular token with a 1 or disable it with a 0
    """
    token_type_ids, attention_mask = [], []
    for input_ids_example in input_ids:
        input_ids_example = input_ids_example.squeeze().tolist()
        if input_ids.shape[0] == 1:
            input_ids_example = input_ids.squeeze().tolist()
        padding_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
        padding_length = input_ids_example.count(padding_token_id)
        text_length = len(input_ids_example) - padding_length
        token_type_ids_example = [0] * len(input_ids_example)
        attention_mask_example = [1] * text_length + [0] * padding_length
        assert len(token_type_ids_example) == len(input_ids_example)
        assert len(attention_mask_example) == len(input_ids_example)
        token_type_ids.append(token_type_ids_example)
        attention_mask.append(attention_mask_example)
    token_type_ids = torch.tensor(data=token_type_ids, device=device)
    attention_mask = torch.tensor(data=attention_mask, device=device)
    return token_type_ids, attention_mask


def clean_text(text):
    """Function to clean text using RegEx operations, removal of stopwords, and lemmatization."""
    text = re.sub('[^\\w\\s]', '', text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(' ')]
    text = [lemmatizer.lemmatize(token, 'v') for token in text]
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = text.lstrip().rstrip()
    text = re.sub(' +', ' ', text)
    return text


def tokenize_and_encode(text, tokenizer, apply_cleaning=False, max_tokenization_length=512, truncation_method='head-only', split_head_density=0.5):
    """
    Function to tokenize & encode a given text.

    @param (str) text: a sequence of words to be tokenized in raw string format
    @param (pytorch_transformers.BertTokenizer) tokenizer: tokenizer with pre-figured mappings
    @param (bool) apply_cleaning: whether or not to perform common cleaning operations on texts;
           note that enabling only makes sense if language of the task is English (default: False)
    @param (int) max_tokenization_length: maximum number of positional embeddings, or the sequence
           length of an example that will be fed to BERT model (default: 512)
    @param (str) truncation_method: method that will be applied in case the text exceeds
           @max_tokenization_length; currently implemented methods include 'head-only', 'tail-only',
           and 'head+tail' (default: 'head-only')
    @param (float) split_head_density: weight on head when splitting between head and tail, only
           applicable if @truncation_method='head+tail' (default: 0.5)
    @return (list) input_ids: the encoded integer indexes of the given text; note that
            get_data_iterators() function converts this to a Tensor under the hood
    """
    if apply_cleaning:
        text = clean_text(text=text)
    tokenized_text = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    text_tokenization_length = max_tokenization_length - 2
    if len(input_ids) > text_tokenization_length:
        if truncation_method == 'head-only':
            input_ids = input_ids[:text_tokenization_length]
        elif truncation_method == 'tail-only':
            input_ids = input_ids[-text_tokenization_length:]
        elif truncation_method == 'head+tail':
            head_tokenization_length = int(text_tokenization_length * split_head_density)
            tail_tokenization_length = text_tokenization_length - head_tokenization_length
            input_head_ids = input_ids[:head_tokenization_length]
            input_tail_ids = input_ids[-tail_tokenization_length:]
            input_ids = input_head_ids + input_tail_ids
    cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    input_ids = [cls_id] + input_ids + [sep_id]
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    if len(input_ids) < max_tokenization_length:
        padding_length = max_tokenization_length - len(input_ids)
        input_ids = input_ids + [pad_id] * padding_length
    return input_ids


class FineTunedBert(nn.Module):
    """
    Finetuning model that utilizes BERT tokenizer, pretrained BERT embedding, pretrained BERT
    encoders, an optional recurrent neural network  choice of LSTM, dropout, and finally a dense
    layer for classification.

    @param (str) pretrained_model_name: name of the pretrained BERT model for tokenizing input
           sequences, extracting vector representations for each token, [...]
    @param (int) num_pretrained_bert_layers: number of BERT Encoder layers to be utilized
    @param (int) max_tokenization_length: maximum number of positional embeddings, or the sequence
           length of an example that will be fed to BERT model (default: 512)
    @param (int) num_classes: number of classes to distinct between for classification; specify
           2 for binary classification (default: 1)
    @param (bool) top_down: whether to assign parameters (weights and biases) in order or
           backwards (default: True)
    @param (int) num_recurrent_layers: number of LSTM layers to utilize (default: 1)
    @param (bool) use_bidirectional: whether to use a bidirectional LSTM or not (default: False)
    @param (int) hidden_size: number of recurrent units in each LSTM cell (default: 128)
    @param (bool) reinitialize_pooler_parameters: whether to use the pretrained pooler parameters
           or initialize weights as ones and biases zeros and train for scratch (default: False)
    @param (float) dropout_rate: possibility of each neuron to be discarded (default: 0.10)
    @param (bool) aggregate_on_cls_token: whether to pool on only the hidden states of the [CLS]
           token for classification or on the hidden states of all (512) tokens (default: True)
    @param (bool) concatenate_hidden_states: whether to concatenate all the available hidden states
           outputted by the embedding and encoder layers (K+1) or only use the latest hidden state
           (default: False)
    @param (bool) use_gpu: whether to utilize GPU (CUDA) or not (default: False)
    """

    def __init__(self, pretrained_model_name, num_pretrained_bert_layers, max_tokenization_length, num_classes=1, top_down=True, num_recurrent_layers=1, use_bidirectional=False, hidden_size=128, reinitialize_pooler_parameters=False, dropout_rate=0.1, aggregate_on_cls_token=True, concatenate_hidden_states=False, use_gpu=False):
        super(FineTunedBert, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.use_bidirectional = use_bidirectional
        self.hidden_size = hidden_size
        self.aggregate_on_cls_token = aggregate_on_cls_token
        self.concatenate_hidden_states = concatenate_hidden_states
        self.use_gpu = use_gpu
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.tokenizer.max_len = max_tokenization_length
        self.config = BertConfig.from_pretrained(pretrained_model_name)
        all_states_dict = BertModel.from_pretrained(pretrained_model_name, config=self.config).state_dict()
        self.config.max_position_embeddings = max_tokenization_length
        self.config.num_hidden_layers = num_pretrained_bert_layers
        self.config.output_hidden_states = True
        self.config.output_attentions = True
        self.bert = BertModel.from_pretrained(pretrained_model_name, config=self.config)
        current_states_dict = self.bert.state_dict()
        if top_down:
            for param in current_states_dict.keys():
                if 'pooler' not in param or not reinitialize_pooler_parameters:
                    current_states_dict[param] = all_states_dict[param]
                elif 'weight' in param:
                    current_states_dict[param] = torch.ones(self.config.hidden_size, self.config.hidden_size)
                elif 'bias' in param:
                    current_states_dict[param] = torch.zeros(self.config.hidden_size)
        else:
            align = 5 + (12 - num_pretrained_bert_layers) * 16
            for index, param in enumerate(current_states_dict.keys()):
                if index < 5 and 'embeddings' in param:
                    current_states_dict[param] = all_states_dict[param]
                elif index >= 5 and 'pooler' not in param:
                    current_states_dict[param] = list(all_states_dict.values())[align:][index - 5]
                elif not reinitialize_pooler_parameters:
                    current_states_dict[param] = all_states_dict[param]
                elif 'weight' in param:
                    current_states_dict[param] = torch.ones(self.config.hidden_size, self.config.hidden_size)
                elif 'bias' in param:
                    current_states_dict[param] = torch.zeros(self.config.hidden_size)
        del all_states_dict
        self.bert.load_state_dict(current_states_dict)
        logging.info('Loaded %d learnable parameters from pretrained BERT model with %d layer(s)' % (len(list(self.bert.parameters())), num_pretrained_bert_layers))
        input_hidden_dimension = None
        if concatenate_hidden_states:
            input_hidden_dimension = (num_pretrained_bert_layers + 1) * self.config.hidden_size
        else:
            input_hidden_dimension = self.config.hidden_size
        self.flatten_sequence_length = lambda t: t.view(-1, self.config.max_position_embeddings * input_hidden_dimension)
        self.dropout = nn.Dropout(p=dropout_rate)
        if self.num_recurrent_layers > 0:
            self.lstm = nn.LSTM(input_size=input_hidden_dimension, hidden_size=hidden_size, num_layers=num_recurrent_layers, bidirectional=use_bidirectional, batch_first=True)
            self.clf = nn.Linear(in_features=hidden_size * 2 if use_bidirectional else hidden_size, out_features=num_classes)
        elif aggregate_on_cls_token:
            self.clf = nn.Linear(in_features=input_hidden_dimension, out_features=num_classes)
        else:
            self.clf = nn.Linear(in_features=max_tokenization_length * input_hidden_dimension, out_features=num_classes)

    def get_tokenizer(self):
        """Function to easily access the BERT tokenizer"""
        return self.tokenizer

    def get_bert_attention(self, raw_sentence, device):
        """Function for getting the multi-head self-attention output from pretrained BERT"""
        x = tokenize_and_encode(text=raw_sentence, tokenizer=self.get_tokenizer(), max_tokenization_length=self.config.max_position_embeddings, truncation_method='head-only')
        x = torch.tensor(data=x, device=device)
        x = x.unsqueeze(dim=1).view(1, -1)
        token_type_ids, attention_mask = get_features(input_ids=x, tokenizer=self.get_tokenizer(), device=device)
        bert_outputs = self.bert(input_ids=x, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=None, head_mask=None)
        attention_outputs = bert_outputs[3]
        return attention_outputs

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        """Function implementing a forward pass of the model"""
        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask)
        sequence_output = bert_outputs[0]
        pooled_output = bert_outputs[1]
        hidden_outputs = bert_outputs[2]
        attention_outputs = bert_outputs[3]
        if self.concatenate_hidden_states:
            sequence_output = torch.cat(hidden_outputs, dim=-1)
        if self.num_recurrent_layers > 0:
            if self.use_gpu:
                h0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
                c0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
            else:
                h0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
                c0 = Variable(torch.zeros(self.num_recurrent_layers * 2 if self.use_bidirectional else self.num_recurrent_layers, input_ids.shape[0], self.hidden_size))
            lstm_output = self.lstm(sequence_output, (h0, c0))
            sequence_output, _ = lstm_output
            last_timesteps = []
            for i in range(len(attention_mask)):
                last_timesteps.append(attention_mask[i].tolist().index(0) if 0 in attention_mask[i].tolist() else self.tokenizer.max_len - 1)
            if self.use_gpu:
                last_timesteps = torch.tensor(data=last_timesteps)
            else:
                last_timesteps = torch.tensor(data=last_timesteps)
            relative_hidden_size = self.hidden_size * 2 if self.use_bidirectional else self.hidden_size
            last_timesteps = last_timesteps.repeat(1, relative_hidden_size)
            last_timesteps = last_timesteps.view(-1, 1, relative_hidden_size)
            pooled_sequence_output = sequence_output.gather(dim=1, index=last_timesteps).squeeze()
            pooled_sequence_output = self.dropout(pooled_sequence_output)
            logits = self.clf(pooled_sequence_output)
        else:
            if not self.aggregate_on_cls_token:
                pooled_output = self.flatten_sequence_length(sequence_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.clf(pooled_output)
        return logits

