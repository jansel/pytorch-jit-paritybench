import sys
_module = sys.modules[__name__]
del sys
convert_tf_checkpoint_to_pytorch = _module
modeling = _module
tokenization = _module
output_entity = _module
dbengine = _module
sqlova = _module
model = _module
nl2sql = _module
wikisql_models = _module
utils = _module
utils_wikisql = _module
wikisql_formatter = _module
train = _module
annotate = _module
evaluate = _module
common = _module
query = _module
table = _module

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


import re


import tensorflow as tf


import torch


import numpy as np


import copy


import math


import torch.nn as nn


from torch.nn import CrossEntropyLoss


import torch.utils.data


from matplotlib.pylab import *


from copy import deepcopy


import torch.nn.functional as F


import random as rd


import torchvision.datasets as dsets


import random as python_random


class BERTLayerNorm(nn.Module):

    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BERTEmbeddings(nn.Module):

    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):

    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        """ From this,

        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):

    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):

    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BERTIntermediate(nn.Module):

    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):

    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):

    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):

    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):

    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    def print_status(self):
        """
        Wonseok add this.
        """
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None
        None

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for key, value in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, 'r') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + '\n'


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output


class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class BertForQuestionAnswering(nn.Module):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class BertNoAnswer(nn.Module):

    def __init__(self, hidden_size, context_length=317):
        super(BertNoAnswer, self).__init__()
        self.context_length = context_length
        self.W_no = nn.Linear(hidden_size, 1)
        self.no_answer = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 2))

    def forward(self, sequence_output, start_logit, end_logit, mask=None):
        if mask is None:
            nbatch, length, _ = sequence_output.size()
            mask = torch.ones(nbatch, length)
        mask = mask.float()
        mask = mask.unsqueeze(-1)[:, 1:self.context_length + 1]
        mask = (1.0 - mask) * -10000.0
        sequence_output = sequence_output[:, 1:self.context_length + 1]
        start_logit = start_logit[:, 1:self.context_length + 1] + mask
        end_logit = end_logit[:, 1:self.context_length + 1] + mask
        pa_1 = nn.functional.softmax(start_logit.transpose(1, 2), -1)
        v1 = torch.bmm(pa_1, sequence_output).squeeze(1)
        pa_2 = nn.functional.softmax(end_logit.transpose(1, 2), -1)
        v2 = torch.bmm(pa_2, sequence_output).squeeze(1)
        pa_3 = self.W_no(sequence_output) + mask
        pa_3 = nn.functional.softmax(pa_3.transpose(1, 2), -1)
        v3 = torch.bmm(pa_3, sequence_output).squeeze(1)
        bias = self.no_answer(torch.cat([v1, v2, v3], -1))
        return bias


class BertForSQuAD2(nn.Module):

    def __init__(self, config, context_length=317):
        super(BertForSQuAD2, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.na_head = BertNoAnswer(config.hidden_size, context_length)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None, labels=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        span_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        na_logits = self.na_head(sequence_output, start_logits, end_logits, attention_mask)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = (na_logits + logits) / 2
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            unanswerable_loss = loss_fct(logits, labels)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss + unanswerable_loss
            return total_loss
        else:
            probs = nn.functional.softmax(logits, -1)
            _, probs = probs.split(1, dim=-1)
            return start_logits, end_logits, probs


class BertForWikiSQL(nn.Module):

    def __init__(self, config, context_length=317):
        super(BertForWikiSQL, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.na_head = BertNoAnswer(config.hidden_size, context_length)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None, labels=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        span_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        na_logits = self.na_head(sequence_output, start_logits, end_logits, attention_mask)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = (na_logits + logits) / 2
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            unanswerable_loss = loss_fct(logits, labels)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss + unanswerable_loss
            return total_loss
        else:
            probs = nn.functional.softmax(logits, -1)
            _, probs = probs.split(1, dim=-1)
            return start_logits, end_logits, probs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_perm_inv(perm):
    perm_inv = zeros(len(perm), dtype=int32)
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i
    return perm_inv


def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, last_only=False):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape
    l = array(l)
    perm_idx = argsort(-l)
    perm_idx_inv = generate_perm_inv(perm_idx)
    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :], l[perm_idx], batch_first=True)
    if hc0 is not None:
        hc0 = hc0[0][:, perm_idx], hc0[1][:, perm_idx]
    packed_wemb_l = packed_wemb_l.float()
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)
    hout, cout = hc_out
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)
    if last_only:
        wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]
        wenc.unsqueeze_(1)
    wenc = wenc[perm_idx_inv]
    if return_hidden:
        hout = hout[:, perm_idx_inv]
        cout = cout[:, perm_idx_inv]
        return wenc, hout, cout
    else:
        return wenc


def encode_hpu(lstm, wemb_hpu, l_hpu, l_hs):
    wenc_hpu, hout, cout = encode(lstm, wemb_hpu, l_hpu, return_hidden=True, hc0=None, last_only=True)
    wenc_hpu = wenc_hpu.squeeze(1)
    bS_hpu, mL_hpu, eS = wemb_hpu.shape
    hS = wenc_hpu.size(-1)
    wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
    wenc_hs = wenc_hs
    st = 0
    for i, l_hs1 in enumerate(l_hs):
        wenc_hs[i, :l_hs1] = wenc_hpu[st:st + l_hs1]
        st += l_hs1
    return wenc_hs


class SAP(nn.Module):

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_agg_ops=-1, old=False):
        super(SAP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.question_knowledge_dim = 5
        self.header_knowledge_dim = 3
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.W_att = nn.Linear(hS + self.question_knowledge_dim, hS + self.header_knowledge_dim)
        self.sa_out = nn.Sequential(nn.Linear(hS + self.question_knowledge_dim, hS), nn.Tanh(), nn.Linear(hS, n_agg_ops))
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)
        if old:
            self.W_c = nn.Linear(hS, hS)
            self.W_hs = nn.Linear(hS, hS)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=False, knowledge=None, knowledge_header=None):
        mL_n = max(l_n)
        bS = len(l_hs)
        wenc_n = encode(self.enc_n, wemb_n, l_n, return_hidden=False, hc0=None, last_only=False)
        knowledge = [(k + (mL_n - len(k)) * [0]) for k in knowledge]
        knowledge = torch.tensor(knowledge).unsqueeze(-1)
        feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1, index=knowledge, value=1)
        wenc_n = torch.cat([wenc_n, feature], -1)
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)
        knowledge_header = [(k + (max(l_hs) - len(k)) * [0]) for k in knowledge_header]
        knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1, index=knowledge_header, value=1)
        wenc_hs = torch.cat([wenc_hs, feature2], -1)
        bS = len(l_hs)
        mL_n = max(l_n)
        wenc_hs_ob = wenc_hs[list(range(bS)), pr_sc]
        att = torch.bmm(self.W_att(wenc_n), wenc_hs_ob.unsqueeze(2)).squeeze(2)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, l_n1:] = -10000000000
        p = self.softmax_dim1(att)
        if show_p_sa:
            if p.shape[0] != 1:
                raise Exception('Batch size should be 1.')
            fig = figure(2001)
            subplot(7, 2, 3)
            cla()
            plot(p[0].data.numpy(), '--rs', ms=7)
            title('sa: nlu_weight')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
        c_n = torch.mul(wenc_n, p.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_sa = self.sa_out(c_n)
        return s_sa


class SCP(nn.Module):

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(SCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.question_knowledge_dim = 5
        self.header_knowledge_dim = 3
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.W_att = nn.Linear(hS + self.question_knowledge_dim, hS + self.header_knowledge_dim)
        self.W_c = nn.Linear(hS + self.question_knowledge_dim, hS)
        self.W_hs = nn.Linear(hS + self.header_knowledge_dim, hS)
        self.sc_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=False, knowledge=None, knowledge_header=None):
        mL_n = max(l_n)
        bS = len(l_hs)
        wenc_n = encode(self.enc_n, wemb_n, l_n, return_hidden=False, hc0=None, last_only=False)
        knowledge = [(k + (mL_n - len(k)) * [0]) for k in knowledge]
        knowledge = torch.tensor(knowledge).unsqueeze(-1)
        feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1, index=knowledge, value=1)
        wenc_n = torch.cat([wenc_n, feature], -1)
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)
        knowledge_header = [(k + (max(l_hs) - len(k)) * [0]) for k in knowledge_header]
        knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1, index=knowledge_header, value=1)
        wenc_hs = torch.cat([wenc_hs, feature2], -1)
        bS = len(l_hs)
        mL_n = max(l_n)
        att_h = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_h[b, :, l_n1:] = -10000000000
        p_n = self.softmax_dim2(att_h)
        if show_p_sc:
            if p_n.shape[0] != 1:
                raise Exception('Batch size should be 1.')
            fig = figure(2001, figsize=(12, 3.5))
            subplot2grid((7, 2), (3, 0), rowspan=2)
            cla()
            _color = 'rgbkcm'
            _symbol = '.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p_n[0][i_h][:].data.numpy() - i_h, '--' + _symbol[color_idx] + _color[color_idx], ms=7)
            title('sc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
        c_n = torch.mul(p_n.unsqueeze(3), wenc_n.unsqueeze(1)).sum(dim=2)
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)
        s_sc = self.sc_out(vec).squeeze(2)
        mL_hs = max(l_hs)
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                s_sc[b, l_hs1:] = -10000000000
        return s_sc


class WCP(nn.Module):

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(WCP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.question_knowledge_dim = 5
        self.header_knowledge_dim = 3
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.W_att = nn.Linear(hS + self.question_knowledge_dim, hS + self.header_knowledge_dim)
        self.W_c = nn.Linear(hS + self.question_knowledge_dim, hS)
        self.W_hs = nn.Linear(hS + self.header_knowledge_dim, hS)
        self.W_out = nn.Sequential(nn.Tanh(), nn.Linear(2 * hS, 1))
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc, penalty=True, predict_select_column=None, knowledge=None, knowledge_header=None):
        mL_n = max(l_n)
        bS = len(l_hs)
        wenc_n = encode(self.enc_n, wemb_n, l_n, return_hidden=False, hc0=None, last_only=False)
        knowledge = [(k + (mL_n - len(k)) * [0]) for k in knowledge]
        knowledge = torch.tensor(knowledge).unsqueeze(-1)
        feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1, index=knowledge, value=1)
        wenc_n = torch.cat([wenc_n, feature], -1)
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)
        knowledge_header = [(k + (max(l_hs) - len(k)) * [0]) for k in knowledge_header]
        knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1, index=knowledge_header, value=1)
        wenc_hs = torch.cat([wenc_hs, feature2], -1)
        att = torch.bmm(wenc_hs, self.W_att(wenc_n).transpose(1, 2))
        mL_n = max(l_n)
        for b_n, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b_n, :, l_n1:] = -10000000000
        p = self.softmax_dim2(att)
        if show_p_wc:
            if p.shape[0] != 1:
                raise Exception('Batch size should be 1.')
            fig = figure(2001)
            subplot2grid((7, 2), (3, 1), rowspan=2)
            cla()
            _color = 'rgbkcm'
            _symbol = '.......'
            for i_h in range(l_hs[0]):
                color_idx = i_h % len(_color)
                plot(p[0][i_h][:].data.numpy() - i_h, '--' + _symbol[color_idx] + _color[color_idx], ms=7)
            title('wc: p_n for each h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
        wenc_n = wenc_n.unsqueeze(1)
        p = p.unsqueeze(3)
        c_n = torch.mul(wenc_n, p).sum(2)
        y = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs)], dim=2)
        score = self.W_out(y).squeeze(2)
        if penalty:
            for b, l_hs1 in enumerate(l_hs):
                score[b, l_hs1:] = -10000000000.0
        return score


class WNP(nn.Module):

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3):
        super(WNP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.mL_w = 4
        self.question_knowledge_dim = 5
        self.header_knowledge_dim = 3
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.W_att_h = nn.Linear(hS + self.header_knowledge_dim, 1)
        self.W_hidden = nn.Linear(hS + self.header_knowledge_dim, lS * hS)
        self.W_cell = nn.Linear(hS + self.header_knowledge_dim, lS * hS)
        self.W_att_n = nn.Linear(hS + self.question_knowledge_dim, 1)
        self.wn_out = nn.Sequential(nn.Linear(hS + self.question_knowledge_dim, hS), nn.Tanh(), nn.Linear(hS, self.mL_w + 1))
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=False, knowledge=None, knowledge_header=None):
        mL_n = max(l_n)
        bS = len(l_hs)
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)
        knowledge_header = [(k + (max(l_hs) - len(k)) * [0]) for k in knowledge_header]
        knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1, index=knowledge_header, value=1)
        wenc_hs = torch.cat([wenc_hs, feature2], -1)
        bS = len(l_hs)
        mL_n = max(l_n)
        mL_hs = max(l_hs)
        att_h = self.W_att_h(wenc_hs).squeeze(2)
        for b, l_hs1 in enumerate(l_hs):
            if l_hs1 < mL_hs:
                att_h[b, l_hs1:] = -10000000000
        p_h = self.softmax_dim1(att_h)
        if show_p_wn:
            if p_h.shape[0] != 1:
                raise Exception('Batch size should be 1.')
            fig = figure(2001)
            subplot(7, 2, 5)
            cla()
            plot(p_h[0].data.numpy(), '--rs', ms=7)
            title('wn: header_weight')
            grid(True)
            fig.canvas.draw()
            show()
        c_hs = torch.mul(wenc_hs, p_h.unsqueeze(2)).sum(1)
        hidden = self.W_hidden(c_hs)
        hidden = hidden.view(bS, self.lS * 2, int(self.hS / 2))
        hidden = hidden.transpose(0, 1).contiguous()
        cell = self.W_cell(c_hs)
        cell = cell.view(bS, self.lS * 2, int(self.hS / 2))
        cell = cell.transpose(0, 1).contiguous()
        wenc_n = encode(self.enc_n, wemb_n, l_n, return_hidden=False, hc0=(hidden, cell), last_only=False)
        knowledge = [(k + (mL_n - len(k)) * [0]) for k in knowledge]
        knowledge = torch.tensor(knowledge).unsqueeze(-1)
        feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1, index=knowledge, value=1)
        wenc_n = torch.cat([wenc_n, feature], -1)
        att_n = self.W_att_n(wenc_n).squeeze(2)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att_n[b, l_n1:] = -10000000000
        p_n = self.softmax_dim1(att_n)
        if show_p_wn:
            if p_n.shape[0] != 1:
                raise Exception('Batch size should be 1.')
            fig = figure(2001)
            subplot(7, 2, 6)
            cla()
            plot(p_n[0].data.numpy(), '--rs', ms=7)
            title('wn: nlu_weight')
            grid(True)
            fig.canvas.draw()
            show()
        c_n = torch.mul(wenc_n, p_n.unsqueeze(2).expand_as(wenc_n)).sum(dim=1)
        s_wn = self.wn_out(c_n)
        return s_wn


class WOP(nn.Module):

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=3):
        super(WOP, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.question_knowledge_dim = 0
        self.header_knowledge_dim = 0
        self.mL_w = 4
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.W_att = nn.Linear(hS + self.question_knowledge_dim, hS + self.header_knowledge_dim)
        self.W_c = nn.Linear(hS + self.question_knowledge_dim, hS)
        self.W_hs = nn.Linear(hS + self.header_knowledge_dim, hS)
        self.wo_out = nn.Sequential(nn.Linear(2 * hS, hS), nn.Tanh(), nn.Linear(hS, n_cond_ops))
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wenc_n=None, show_p_wo=False, knowledge=None, knowledge_header=None):
        mL_n = max(l_n)
        bS = len(l_hs)
        if not wenc_n:
            wenc_n = encode(self.enc_n, wemb_n, l_n, return_hidden=False, hc0=None, last_only=False)
            if self.question_knowledge_dim != 0:
                knowledge = [(k + (mL_n - len(k)) * [0]) for k in knowledge]
                knowledge = torch.tensor(knowledge).unsqueeze(-1)
                feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1, index=knowledge, value=1)
                wenc_n = torch.cat([wenc_n, feature], -1)
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)
        if self.header_knowledge_dim != 0:
            knowledge_header = [(k + (max(l_hs) - len(k)) * [0]) for k in knowledge_header]
            knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
            feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1, index=knowledge_header, value=1)
            wenc_hs = torch.cat([wenc_hs, feature2], -1)
        bS = len(l_hs)
        wenc_hs_ob = []
        for b in range(bS):
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]]
            wenc_hs_ob1 = torch.stack(real + pad)
            wenc_hs_ob.append(wenc_hs_ob1)
        wenc_hs_ob = torch.stack(wenc_hs_ob)
        wenc_hs_ob = wenc_hs_ob
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1), wenc_hs_ob.unsqueeze(3)).squeeze(3)
        mL_n = max(l_n)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000
        p = self.softmax_dim2(att)
        if show_p_wo:
            if p.shape[0] != 1:
                raise Exception('Batch size should be 1.')
            fig = figure(2001)
            subplot2grid((7, 2), (5, 0), rowspan=2)
            cla()
            _color = 'rgbkcm'
            _symbol = '.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--' + _symbol[color_idx] + _color[color_idx], ms=7)
            title('wo: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob)], dim=2)
        s_wo = self.wo_out(vec)
        return s_wo


class WVP_se(nn.Module):
    """
    Discriminative model
    Get start and end.
    Here, classifier for [ [투수], [팀1], [팀2], [연도], ...]
    Input:      Encoded nlu & selected column.
    Algorithm: Encoded nlu & selected column. -> classifier -> mask scores -> ...
    """

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_cond_ops=4, old=False):
        super(WVP_se, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops
        self.question_knowledge_dim = 5
        self.header_knowledge_dim = 3
        self.mL_w = 4
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.W_att = nn.Linear(hS + self.question_knowledge_dim, hS + self.header_knowledge_dim)
        self.W_c = nn.Linear(hS + self.question_knowledge_dim, hS)
        self.W_hs = nn.Linear(hS + self.header_knowledge_dim, hS)
        self.W_op = nn.Linear(n_cond_ops, hS)
        if old:
            self.wv_out = nn.Sequential(nn.Linear(4 * hS, 2))
        else:
            self.wv_out = nn.Sequential(nn.Linear(4 * hS + self.question_knowledge_dim, hS), nn.Tanh(), nn.Linear(hS, 2))
        self.softmax_dim1 = nn.Softmax(dim=1)
        self.softmax_dim2 = nn.Softmax(dim=2)

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn, wc, wo, wenc_n=None, show_p_wv=False, knowledge=None, knowledge_header=None):
        mL_n = max(l_n)
        bS = len(l_hs)
        if not wenc_n:
            wenc_n, hout, cout = encode(self.enc_n, wemb_n, l_n, return_hidden=True, hc0=None, last_only=False)
            knowledge = [(k + (mL_n - len(k)) * [0]) for k in knowledge]
            knowledge = torch.tensor(knowledge).unsqueeze(-1)
            feature = torch.zeros(bS, mL_n, self.question_knowledge_dim).scatter_(dim=-1, index=knowledge, value=1)
            wenc_n = torch.cat([wenc_n, feature], -1)
        wenc_hs = encode_hpu(self.enc_h, wemb_hpu, l_hpu, l_hs)
        knowledge_header = [(k + (max(l_hs) - len(k)) * [0]) for k in knowledge_header]
        knowledge_header = torch.tensor(knowledge_header).unsqueeze(-1)
        feature2 = torch.zeros(bS, max(l_hs), self.header_knowledge_dim).scatter_(dim=-1, index=knowledge_header, value=1)
        wenc_hs = torch.cat([wenc_hs, feature2], -1)
        wenc_hs_ob = []
        for b in range(bS):
            real = [wenc_hs[b, col] for col in wc[b]]
            pad = (self.mL_w - wn[b]) * [wenc_hs[b, 0]]
            wenc_hs_ob1 = torch.stack(real + pad)
            wenc_hs_ob.append(wenc_hs_ob1)
        wenc_hs_ob = torch.stack(wenc_hs_ob)
        wenc_hs_ob = wenc_hs_ob
        att = torch.matmul(self.W_att(wenc_n).unsqueeze(1), wenc_hs_ob.unsqueeze(3)).squeeze(3)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                att[b, :, l_n1:] = -10000000000
        p = self.softmax_dim2(att)
        if show_p_wv:
            if p.shape[0] != 1:
                raise Exception('Batch size should be 1.')
            fig = figure(2001)
            subplot2grid((7, 2), (5, 1), rowspan=2)
            cla()
            _color = 'rgbkcm'
            _symbol = '.......'
            for i_wn in range(self.mL_w):
                color_idx = i_wn % len(_color)
                plot(p[0][i_wn][:].data.numpy() - i_wn, '--' + _symbol[color_idx] + _color[color_idx], ms=7)
            title('wv: p_n for selected h')
            grid(True)
            fig.tight_layout()
            fig.canvas.draw()
            show()
        c_n = torch.mul(wenc_n.unsqueeze(1), p.unsqueeze(3)).sum(dim=2)
        wenc_op = []
        for b in range(bS):
            wenc_op1 = torch.zeros(self.mL_w, self.n_cond_ops)
            wo1 = wo[b]
            idx_scatter = []
            l_wo1 = len(wo1)
            for i_wo11 in range(self.mL_w):
                if i_wo11 < l_wo1:
                    wo11 = wo1[i_wo11]
                    idx_scatter.append([int(wo11)])
                else:
                    idx_scatter.append([0])
            wenc_op1 = wenc_op1.scatter(1, torch.tensor(idx_scatter), 1)
            wenc_op.append(wenc_op1)
        wenc_op = torch.stack(wenc_op)
        wenc_op = wenc_op
        vec = torch.cat([self.W_c(c_n), self.W_hs(wenc_hs_ob), self.W_op(wenc_op)], dim=2)
        vec1e = vec.unsqueeze(2).expand(-1, -1, mL_n, -1)
        wenc_ne = wenc_n.unsqueeze(1).expand(-1, 4, -1, -1)
        vec2 = torch.cat([vec1e, wenc_ne], dim=3)
        s_wv = self.wv_out(vec2)
        for b, l_n1 in enumerate(l_n):
            if l_n1 < mL_n:
                s_wv[b, :, l_n1:, :] = -10000000000
        return s_wv


def check_sc_sa_pairs(tb, pr_sc, pr_sa):
    """
    Check whether pr_sc, pr_sa are allowed pairs or not.
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']

    """
    bS = len(pr_sc)
    check = [False] * bS
    for b, pr_sc1 in enumerate(pr_sc):
        pr_sa1 = pr_sa[b]
        hd_types1 = tb[b]['types']
        hd_types11 = hd_types1[pr_sc1]
        if hd_types11 == 'text':
            if pr_sa1 == 0 or pr_sa1 == 3:
                check[b] = True
            else:
                check[b] = False
        elif hd_types11 == 'real':
            check[b] = True
        else:
            raise Exception('New TYPE!!')
    return check


def convert_pr_wvi_to_string(pr_wvi, nlu_t, nlu_wp_t, wp_to_wh_index, nlu):
    """
    - Convert to the string in whilte-space-separated tokens
    - Add-hoc addition.
    """
    pr_wv_str_wp = []
    pr_wv_str = []
    for b, pr_wvi1 in enumerate(pr_wvi):
        pr_wv_str_wp1 = []
        pr_wv_str1 = []
        wp_to_wh_index1 = wp_to_wh_index[b]
        nlu_wp_t1 = nlu_wp_t[b]
        nlu_t1 = nlu_t[b]
        for i_wn, pr_wvi11 in enumerate(pr_wvi1):
            st_idx, ed_idx = pr_wvi11
            pr_wv_str_wp11 = nlu_wp_t1[st_idx:ed_idx + 1]
            pr_wv_str_wp1.append(pr_wv_str_wp11)
            st_wh_idx = wp_to_wh_index1[st_idx]
            ed_wh_idx = wp_to_wh_index1[ed_idx]
            pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx + 1]
            pr_wv_str1.append(pr_wv_str11)
        pr_wv_str_wp.append(pr_wv_str_wp1)
        pr_wv_str.append(pr_wv_str1)
    return pr_wv_str, pr_wv_str_wp


def merge_wv_t1_eng(where_str_tokens, NLq):
    """
    Almost copied of SQLNet.
    The main purpose is pad blank line while combining tokens.
    """
    nlq = NLq.lower()
    where_str_tokens = [tok.lower() for tok in where_str_tokens]
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$'
    special = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '``': '"', "''": '"'}
    ret = ''
    double_quote_appear = 0
    for raw_w_token in where_str_tokens:
        if not raw_w_token:
            continue
        w_token = special.get(raw_w_token, raw_w_token)
        if w_token == '"':
            double_quote_appear = 1 - double_quote_appear
        if len(ret) == 0:
            pass
        elif len(ret) > 0 and ret + ' ' + w_token in nlq:
            ret = ret + ' '
        elif len(ret) > 0 and ret + w_token in nlq:
            pass
        elif w_token == '"':
            if double_quote_appear:
                ret = ret + ' '
        elif w_token[0] not in alphabet:
            pass
        elif ret[-1] not in ['(', '/', '–', '#', '$', '&'] and (ret[-1] != '"' or not double_quote_appear):
            ret = ret + ' '
        ret = ret + w_token
    return ret.strip()


def pred_sa(s_sa):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    pr_sa = []
    for s_sa1 in s_sa:
        pr_sa.append(s_sa1.argmax().item())
    return pr_sa


def pred_sc(s_sc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    pr_sc = []
    for s_sc1 in s_sc:
        pr_sc.append(s_sc1.argmax().item())
    return pr_sc


def pred_sc_beam(s_sc, beam_size):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    pr_sc_beam = []
    for s_sc1 in s_sc:
        val, idxes = s_sc1.topk(k=beam_size)
        pr_sc_beam.append(idxes.tolist())
    return pr_sc_beam


def pred_wc(wn, s_wc):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    ! Returned index is sorted!
    """
    pr_wc = []
    for b, wn1 in enumerate(wn):
        s_wc1 = s_wc[b]
        pr_wc1 = argsort(-s_wc1.data.cpu().numpy())[:wn1]
        pr_wc1.sort()
        pr_wc.append(list(pr_wc1))
    return pr_wc


def pred_wn(s_wn):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    pr_wn = []
    for s_wn1 in s_wn:
        pr_wn.append(s_wn1.argmax().item())
    return pr_wn


def pred_wo(wn, s_wo):
    """
    return: [ pr_wc1_i, pr_wc2_i, ...]
    """
    pr_wo_a = s_wo.argmax(dim=2)
    pr_wo = []
    for b, pr_wo_a1 in enumerate(pr_wo_a):
        wn1 = wn[b]
        pr_wo.append(list(pr_wo_a1.data.cpu().numpy()[:wn1]))
    return pr_wo


def pred_wvi_se_beam(max_wn, s_wv, beam_size):
    """
    s_wv: [B, 4, mL, 2]
    - predict best st-idx & ed-idx


    output:
    pr_wvi_beam = [B, max_wn, n_pairs, 2]. 2 means [st, ed].
    prob_wvi_beam = [B, max_wn, n_pairs]
    """
    bS = s_wv.shape[0]
    s_wv_st, s_wv_ed = s_wv.split(1, dim=3)
    s_wv_st = s_wv_st.squeeze(3)
    s_wv_ed = s_wv_ed.squeeze(3)
    prob_wv_st = F.softmax(s_wv_st, dim=-1).detach().numpy()
    prob_wv_ed = F.softmax(s_wv_ed, dim=-1).detach().numpy()
    k_logit = int(ceil(sqrt(beam_size)))
    n_pairs = k_logit ** 2
    assert n_pairs >= beam_size
    values_st, idxs_st = s_wv_st.topk(k_logit)
    values_ed, idxs_ed = s_wv_ed.topk(k_logit)
    pr_wvi_beam = []
    prob_wvi_beam = zeros([bS, max_wn, n_pairs])
    for b in range(bS):
        pr_wvi_beam1 = []
        idxs_st1 = idxs_st[b]
        idxs_ed1 = idxs_ed[b]
        for i_wn in range(max_wn):
            idxs_st11 = idxs_st1[i_wn]
            idxs_ed11 = idxs_ed1[i_wn]
            pr_wvi_beam11 = []
            pair_idx = -1
            for i_k in range(k_logit):
                for j_k in range(k_logit):
                    pair_idx += 1
                    st = idxs_st11[i_k].item()
                    ed = idxs_ed11[j_k].item()
                    pr_wvi_beam11.append([st, ed])
                    p1 = prob_wv_st[b, i_wn, st]
                    p2 = prob_wv_ed[b, i_wn, ed]
                    prob_wvi_beam[b, i_wn, pair_idx] = p1 * p2
            pr_wvi_beam1.append(pr_wvi_beam11)
        pr_wvi_beam.append(pr_wvi_beam1)
    return pr_wvi_beam, prob_wvi_beam


def remap_sc_idx(idxs, pr_sc_beam):
    for b, idxs1 in enumerate(idxs):
        for i_beam, idxs11 in enumerate(idxs1):
            sc_beam_idx = idxs[b][i_beam][0]
            sc_idx = pr_sc_beam[b][sc_beam_idx]
            idxs[b][i_beam][0] = sc_idx
    return idxs


def topk_multi_dim(tensor, n_topk=1, batch_exist=True):
    if batch_exist:
        idxs = []
        for b, tensor1 in enumerate(tensor):
            idxs1 = []
            tensor1_1d = tensor1.reshape(-1)
            values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
            idxs_list = unravel_index(idxs_1d.cpu().numpy(), tensor1.shape)
            for i_beam in range(n_topk):
                idxs11 = []
                for idxs_list1 in idxs_list:
                    idxs11.append(idxs_list1[i_beam])
                idxs1.append(idxs11)
            idxs.append(idxs1)
    else:
        tensor1 = tensor
        idxs1 = []
        tensor1_1d = tensor1.reshape(-1)
        values_1d, idxs_1d = tensor1_1d.topk(k=n_topk)
        idxs_list = unravel_index(idxs_1d.numpy(), tensor1.shape)
        for i_beam in range(n_topk):
            idxs11 = []
            for idxs_list1 in idxs_list:
                idxs11.append(idxs_list1[i_beam])
            idxs1.append(idxs11)
        idxs = idxs1
    return idxs


class Seq2SQL_v1(nn.Module):

    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, old=False):
        super(Seq2SQL_v1, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr
        self.max_wn = 4
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.scp = SCP(iS, hS, lS, dr)
        self.sap = SAP(iS, hS, lS, dr, n_agg_ops, old=old)
        self.wnp = WNP(iS, hS, lS, dr)
        self.wcp = WCP(iS, hS, lS, dr)
        self.wop = WOP(iS, hS, lS, dr, n_cond_ops)
        self.wvp = WVP_se(iS, hS, lS, dr, n_cond_ops, old=old)

    def forward(self, wemb_n, l_n, wemb_h, l_hpu, l_hs, g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None, show_p_sc=False, show_p_sa=False, show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False, knowledge=None, knowledge_header=None):
        s_sc = self.scp(wemb_n, l_n, wemb_h, l_hpu, l_hs, show_p_sc=show_p_sc, knowledge=knowledge, knowledge_header=knowledge_header)
        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(s_sc)
        s_sa = self.sap(wemb_n, l_n, wemb_h, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa, knowledge=knowledge, knowledge_header=knowledge_header)
        if g_sa:
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(s_sa)
        s_wn = self.wnp(wemb_n, l_n, wemb_h, l_hpu, l_hs, show_p_wn=show_p_wn, knowledge=knowledge, knowledge_header=knowledge_header)
        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)
        s_wc = self.wcp(wemb_n, l_n, wemb_h, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True, predict_select_column=pr_sc, knowledge=knowledge, knowledge_header=knowledge_header)
        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)
        s_wo = self.wop(wemb_n, l_n, wemb_h, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, show_p_wo=show_p_wo, knowledge=knowledge, knowledge_header=knowledge_header)
        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)
        s_wv = self.wvp(wemb_n, l_n, wemb_h, l_hpu, l_hs, wn=pr_wn, wc=pr_wc, wo=pr_wo, show_p_wv=show_p_wv, knowledge=knowledge, knowledge_header=knowledge_header)
        return s_sc, s_sa, s_wn, s_wc, s_wo, s_wv

    def beam_forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, engine, tb, nlu_t, nlu_wp_t, wp_to_wh_index, nlu, beam_size=4, show_p_sc=False, show_p_sa=False, show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False, knowledge=None, knowledge_header=None):
        """
        Execution-guided beam decoding.
        """
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc, knowledge=knowledge, knowledge_header=knowledge_header)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops])
        prob_sca = torch.zeros_like(prob_sc_sa)
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)
        for i_beam in range(beam_size):
            pr_sc = list(array(pr_sc_beam)[:, i_beam])
            s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa, knowledge=knowledge, knowledge_header=knowledge_header)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa
            prob_sc_selected = prob_sc[range(bS), pr_sc]
            prob_sca[:, i_beam, :] = (prob_sa.t() * prob_sc_selected).t()
        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        idxs = remap_sc_idx(idxs, pr_sc_beam)
        idxs_arr = array(idxs)
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS), beam_idx_sca, 0]
            pr_sa = idxs_arr[range(bS), beam_idx_sca, 1]
            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)
            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1:
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True
            if sum(beam_meet_the_final) == bS:
                break
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn, knowledge=knowledge, knowledge_header=knowledge_header)
        prob_wn = F.softmax(s_wn, dim=-1).detach().numpy()
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True, knowledge=knowledge, knowledge_header=knowledge_header)
        prob_wc = F.sigmoid(s_wc).detach().numpy()
        pr_wn_max = [self.max_wn] * bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc)
        prob_wc_max = zeros([bS, self.max_wn])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b, :] = prob_wc[b, pr_wc_max1]
        s_wo_max = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, show_p_wo=show_p_wo, knowledge=knowledge, knowledge_header=knowledge_header)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().numpy()
        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        for i_op in range(self.n_cond_ops - 1):
            pr_wo_temp = [[i_op] * self.max_wn] * bS
            s_wv = self.wvp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, wo=pr_wo_temp, show_p_wv=show_p_wv, knowledge=knowledge, knowledge_header=knowledge_header)
            prob_wv = F.softmax(s_wv, dim=-2).detach().numpy()
            pr_wvi_beam, prob_wvi_beam = pred_wvi_se_beam(self.max_wn, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)
            prob_wvi_beam_op_list.append(prob_wvi_beam)
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.max_wn, self.n_cond_ops - 1, n_wv_beam_pairs])
        for b in range(bS):
            for i_wn in range(self.max_wn):
                for i_op in range(self.n_cond_ops - 1):
                    for i_wv_beam in range(n_wv_beam_pairs):
                        p_wc = prob_wc_max[b, i_wn]
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]
                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv
        conds_max = []
        prob_conds_max = []
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [wp_to_wh_index[b]], [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]
                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2]]
                pr_ans = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], [conds11])
                if bool(pr_ans):
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)
        pr_sql_i = []
        prob_wn_w = []
        pr_wn_based_on_prob = []
        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len(conds_max[b])
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn + 1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)
            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_sql_i.append(pr_sql_i1)
        return prob_sca, prob_w, prob_wn_w, pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_sql_i


class SAP_agg(nn.Module):

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, n_agg_ops=-1, old=False):
        super(SAP_agg, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.sa_out = nn.Sequential(nn.Linear(hS, hS), nn.Tanh(), nn.Linear(hS, n_agg_ops))

    def forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=False):
        wenc_n = encode(self.enc_n, wemb_n, l_n, return_hidden=False, hc0=None, last_only=False)
        s_sa = self.sa_out(wenc_n).sum(dim=1)
        return s_sa


class Seq2SQL_v1_agg(nn.Module):

    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, old=False):
        super(Seq2SQL_v1_agg, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr
        self.max_wn = 4
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.sap = SAP_agg(iS, hS, lS, dr, n_agg_ops, old=old)

    def forward(self, wemb_n, l_n, wemb_h, l_hpu, l_hs, g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None, show_p_sc=False, show_p_sa=False, show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        s_sa = self.sap(wemb_n, l_n, wemb_h, l_hpu, l_hs, None, show_p_sa=show_p_sa)
        return None, s_sa, None, None, None, None

    def beam_forward(self, wemb_n, l_n, wemb_hpu, l_hpu, l_hs, engine, tb, nlu_t, nlu_wp_t, wp_to_wh_index, nlu, beam_size=4, show_p_sc=False, show_p_sa=False, show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        """
        Execution-guided beam decoding.
        """
        s_sc = self.scp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_sc=show_p_sc)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops])
        prob_sca = torch.zeros_like(prob_sc_sa)
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)
        for i_beam in range(beam_size):
            pr_sc = list(array(pr_sc_beam)[:, i_beam])
            s_sa = self.sap(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, pr_sc, show_p_sa=show_p_sa)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa
            prob_sc_selected = prob_sc[range(bS), pr_sc]
            prob_sca[:, i_beam, :] = (prob_sa.t() * prob_sc_selected).t()
        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        idxs = remap_sc_idx(idxs, pr_sc_beam)
        idxs_arr = array(idxs)
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS), beam_idx_sca, 0]
            pr_sa = idxs_arr[range(bS), beam_idx_sca, 1]
            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)
            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1:
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True
            if sum(beam_meet_the_final) == bS:
                break
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)
        s_wn = self.wnp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wn=show_p_wn)
        prob_wn = F.softmax(s_wn, dim=-1).detach().numpy()
        s_wc = self.wcp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, show_p_wc=show_p_wc, penalty=True)
        prob_wc = F.sigmoid(s_wc).detach().numpy()
        pr_wn_max = [self.max_wn] * bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc)
        prob_wc_max = zeros([bS, self.max_wn])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b, :] = prob_wc[b, pr_wc_max1]
        s_wo_max = self.wop(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, show_p_wo=show_p_wo)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().numpy()
        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        for i_op in range(self.n_cond_ops - 1):
            pr_wo_temp = [[i_op] * self.max_wn] * bS
            s_wv = self.wvp(wemb_n, l_n, wemb_hpu, l_hpu, l_hs, wn=pr_wn_max, wc=pr_wc_max, wo=pr_wo_temp, show_p_wv=show_p_wv)
            prob_wv = F.softmax(s_wv, dim=-2).detach().numpy()
            pr_wvi_beam, prob_wvi_beam = pred_wvi_se_beam(self.max_wn, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)
            prob_wvi_beam_op_list.append(prob_wvi_beam)
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.max_wn, self.n_cond_ops - 1, n_wv_beam_pairs])
        for b in range(bS):
            for i_wn in range(self.max_wn):
                for i_op in range(self.n_cond_ops - 1):
                    for i_wv_beam in range(n_wv_beam_pairs):
                        p_wc = prob_wc_max[b, i_wn]
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]
                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv
        conds_max = []
        prob_conds_max = []
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [wp_to_wh_index[b]], [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]
                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2]]
                pr_ans = engine.execute(tb[b]['id'], pr_sc[b], pr_sa[b], [conds11])
                if bool(pr_ans):
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)
        pr_sql_i = []
        prob_wn_w = []
        pr_wn_based_on_prob = []
        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len(conds_max[b])
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn + 1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)
            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_sql_i.append(pr_sql_i1)
        return prob_sca, prob_w, prob_wn_w, pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_sql_i


def find_where_pnt_belong(pnt, vg):
    idx_sub = -1
    for i, st_ed in enumerate(vg):
        st, ed = st_ed
        if pnt < ed and pnt >= st:
            idx_sub = i
    return idx_sub


def gen_pnt_i_from_pnt(pnt, i_sql_vocab1, i_nlu1, i_hds1):
    vg_list = [i_sql_vocab1, [i_nlu1], i_hds1]
    i_vg = -1
    i_vg_sub = -1
    for i, vg in enumerate(vg_list):
        idx_sub = find_where_pnt_belong(pnt, vg)
        if idx_sub > -1:
            i_vg = i
            i_vg_sub = idx_sub
            break
    return i_vg, i_vg_sub


def gen_i_vg_from_pnt_idxs(pnt_idxs, i_sql_vocab, i_nlu, i_hds):
    i_vg_list = []
    i_vg_sub_list = []
    for b, pnt_idxs1 in enumerate(pnt_idxs):
        sql_q1_list = []
        i_vg_list1 = []
        i_vg_sub_list1 = []
        for t, pnt in enumerate(pnt_idxs1):
            i_vg, i_vg_sub = gen_pnt_i_from_pnt(pnt, i_sql_vocab[b], i_nlu[b], i_hds[b])
            i_vg_list1.append(i_vg)
            i_vg_sub_list1.append(i_vg_sub)
        i_vg_list.append(i_vg_list1)
        i_vg_sub_list.append(i_vg_sub_list1)
    return i_vg_list, i_vg_sub_list


def gen_sql_q_from_i_vg(tokens, nlu, nlu_t, hds, tt_to_t_idx, pnt_start_tok, pnt_end_tok, pnt_idxs, i_vg_list, i_vg_sub_list):
    """
    (
        "none", "max", "min", "count", "sum", "average",
        "select", "where", "and",
        "equal", "greater than", "less than",
        "start", "end"
    ),
    """
    sql_q = []
    sql_i = []
    for b, nlu_t1 in enumerate(nlu_t):
        sql_q1_list = []
        sql_i1 = {}
        tt_to_t_idx1 = tt_to_t_idx[b]
        nlu_st_observed = False
        agg_observed = False
        wc_obs = False
        wo_obs = False
        conds = []
        for t, i_vg in enumerate(i_vg_list[b]):
            i_vg_sub = i_vg_sub_list[b][t]
            pnt = pnt_idxs[b][t]
            if i_vg == 0:
                if pnt == pnt_start_tok or pnt == pnt_end_tok:
                    pass
                else:
                    tok = tokens[b][pnt]
                    if tok in ['none', 'max', 'min', 'count', 'sum', 'average']:
                        agg_observed = True
                        if tok == 'none':
                            pass
                        sql_i1['agg'] = ['none', 'max', 'min', 'count', 'sum', 'average'].index(tok)
                    else:
                        if tok in ['greater', 'less', 'equal']:
                            if tok == 'greater':
                                tok = '>'
                            elif tok == 'less':
                                tok = '<'
                            elif tok == 'equal':
                                tok = '='
                            if wc_obs:
                                conds1.append(['=', '>', '<'].index(tok))
                                wo_obs = True
                        sql_q1_list.append(tok)
            elif i_vg == 1:
                if not nlu_st_observed:
                    idx_nlu_st = pnt
                    nlu_st_observed = True
                else:
                    idx_nlu_ed = pnt
                    st_wh_idx = tt_to_t_idx1[idx_nlu_st - pnt_end_tok - 2]
                    ed_wh_idx = tt_to_t_idx1[idx_nlu_ed - pnt_end_tok - 2]
                    pr_wv_str11 = nlu_t1[st_wh_idx:ed_wh_idx + 1]
                    merged_wv11 = merge_wv_t1_eng(pr_wv_str11, nlu[b])
                    sql_q1_list.append(merged_wv11)
                    nlu_st_observed = False
                    if wc_obs and wo_obs:
                        conds1.append(merged_wv11)
                        conds.append(conds1)
                        wc_obs = False
                        wo_obs = False
            elif i_vg == 2:
                tok = hds[b][i_vg_sub]
                if agg_observed:
                    sql_q1_list.append(f'({tok})')
                    sql_i1['sel'] = i_vg_sub
                    agg_observed = False
                else:
                    wc_obs = True
                    conds1 = [i_vg_sub]
                    sql_q1_list.append(tok)
        sql_i1['conds'] = conds
        sql_i.append(sql_i1)
        sql_q1 = ' '.join(sql_q1_list)
        sql_q.append(sql_q1)
    return sql_q, sql_i


class Decoder_s2s(nn.Module):

    def __init__(self, iS=300, hS=100, lS=2, dr=0.3, max_seq_length=222, n_cond_ops=3):
        super(Decoder_s2s, self).__init__()
        self.iS = iS
        self.hS = hS
        self.lS = lS
        self.dr = dr
        self.mL = max_seq_length
        self.Tmax = 200
        self.enc_h = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.enc_n = nn.LSTM(input_size=iS, hidden_size=int(hS / 2), num_layers=lS, batch_first=True, dropout=dr, bidirectional=True)
        self.decode_pn = nn.LSTM(input_size=max_seq_length, hidden_size=hS, num_layers=lS, batch_first=True, dropout=dr)
        self.W_s2s = nn.Linear(iS, hS)
        self.W_pnt = nn.Linear(hS, hS)
        self.wv_out = nn.Sequential(nn.Tanh(), nn.Linear(hS, 1))

    def forward(self, wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs=None):
        bS, mL_input, iS = wenc_s2s.shape
        ipnt = wenc_s2s.new_zeros(bS, 1, mL_input)
        ipnt[:, 0, pnt_start_tok] = 1
        cpnt = ipnt
        wenc_s2s = wenc_s2s.unsqueeze(1)
        h_0 = torch.zeros([self.lS, bS, self.hS])
        c_0 = torch.zeros([self.lS, bS, self.hS])
        for i_layer in range(self.lS):
            h_st = 2 * i_layer * self.hS
            h_ed = h_st + self.hS
            c_st = (2 * i_layer + 1) * self.hS
            c_ed = c_st + self.hS
            h_0[i_layer] = cls_vec[:, h_st:h_ed]
            c_0[i_layer] = cls_vec[:, c_st:c_ed]
        if g_pnt_idxs:
            pnt_n = torch.zeros(bS, self.Tmax, mL_input)
            for b, g_pnt_idxs1 in enumerate(g_pnt_idxs):
                for t, g_pnt_idx in enumerate(g_pnt_idxs1):
                    pnt_n[b, t, g_pnt_idx] = 1
            dec_pn, _ = self.decode_pn(pnt_n, (h_0, c_0))
            dec_pn = dec_pn.contiguous()
            dec_pn = dec_pn.unsqueeze(2)
            s_wv = self.wv_out(self.W_s2s(wenc_s2s) + self.W_pnt(dec_pn)).squeeze(3)
            for b, l_input1 in enumerate(l_input):
                if l_input1 < mL_input:
                    s_wv[b, :, l_input1:] = -10000000000
        else:
            t = 0
            s_wv_list = []
            cpnt_h = h_0, c_0
            while t < self.Tmax:
                dec_pn, cpnt_h = self.decode_pn(cpnt, cpnt_h)
                dec_pn = dec_pn.unsqueeze(2)
                s_wv1 = self.wv_out(self.W_s2s(wenc_s2s) + self.W_pnt(dec_pn)).squeeze(3)
                for b, l_input1 in enumerate(l_input):
                    if l_input1 < mL_input:
                        s_wv1[b, :, l_input1:] = -10000000000
                s_wv_list.append(s_wv1)
                _val, pnt_n = s_wv1.view(bS, -1).max(dim=1)
                cpnt = torch.zeros(bS, mL_input)
                cpnt = cpnt.scatter_(1, pnt_n.unsqueeze(1), 1)
                cpnt = cpnt.unsqueeze(1)
                t += 1
            s_wv = torch.stack(s_wv_list, 1)
            s_wv = s_wv.squeeze(2)
        return s_wv

    def EG_forward(self, wenc_s2s, l_input, cls_vec, pnt_start_tok, pnt_end_tok, i_sql_vocab, i_nlu, i_hds, tokens, nlu, nlu_t, hds, tt_to_t_idx, tb, engine, beam_size, beam_only=True):
        bS, mL_input, iS = wenc_s2s.shape
        wenc_s2s = wenc_s2s.unsqueeze(1)
        h_0 = torch.zeros([self.lS, bS, self.hS])
        c_0 = torch.zeros([self.lS, bS, self.hS])
        for i_layer in range(self.lS):
            h_st = 2 * i_layer * self.hS
            h_ed = h_st + self.hS
            c_st = (2 * i_layer + 1) * self.hS
            c_ed = c_st + self.hS
            h_0[i_layer] = cls_vec[:, h_st:h_ed]
            c_0[i_layer] = cls_vec[:, c_st:c_ed]
        pnt_list_beam = []
        cpnt_beam = []
        cpnt_h_beam = []
        for i_beam in range(beam_size):
            pnt_list_beam1 = []
            for b in range(bS):
                pnt_list_beam1.append([[pnt_start_tok], 0])
            pnt_list_beam.append(pnt_list_beam1)
            ipnt = wenc_s2s.new_zeros(bS, 1, mL_input)
            ipnt[:, 0, pnt_start_tok] = 1
            cpnt_beam.append(ipnt)
            cpnt_h_beam.append((h_0, c_0))
        t = 0
        while t < self.Tmax:
            candidates = [[] for b in range(bS)]
            for i_beam, cpnt in enumerate(cpnt_beam):
                cpnt_h = cpnt_h_beam[i_beam]
                pnt_list_beam1 = pnt_list_beam[i_beam]
                dec_pn, cpnt_h = self.decode_pn(cpnt, cpnt_h)
                cpnt_h_beam[i_beam] = cpnt_h
                dec_pn = dec_pn.unsqueeze(2)
                s_wv1 = self.wv_out(self.W_s2s(wenc_s2s) + self.W_pnt(dec_pn)).squeeze(3)
                for b, l_input1 in enumerate(l_input):
                    if l_input1 < mL_input:
                        s_wv1[b, :, l_input1:] = -10000000000
                prob, idxs = F.softmax(s_wv1.view(bS, -1), dim=1).topk(dim=1, k=max(l_input))
                log_prob = torch.log(prob)
                for b, log_prob1 in enumerate(log_prob):
                    pnt_list11, score = pnt_list_beam1[b]
                    for i_can, log_prob11 in enumerate(log_prob1):
                        previous_pnt = pnt_list11[-1]
                        if previous_pnt == pnt_end_tok:
                            new_seq = pnt_list11
                            new_score = score
                        else:
                            new_seq = pnt_list11 + [idxs[b][i_can].item()]
                            new_score = score + log_prob11.item()
                        _candidate = [new_seq, new_score]
                        candidates[b].append(_candidate)
            for b, candidates1 in enumerate(candidates):
                new_pnt_list_batch1 = sorted(candidates1, key=lambda list1: list1[-1], reverse=True)
                cnt = 0
                selected_candidates1 = []
                for new_pnt_list_batch11 in new_pnt_list_batch1:
                    if new_pnt_list_batch11 not in selected_candidates1:
                        if beam_only:
                            selected_candidates1.append(new_pnt_list_batch11)
                            pnt_list_beam[cnt][b] = new_pnt_list_batch11
                            cnt += 1
                        else:
                            executable = False
                            testable = False
                            pr_i_vg_list, pr_i_vg_sub_list = gen_i_vg_from_pnt_idxs([new_pnt_list_batch11[0]], [i_sql_vocab[b]], [i_nlu[b]], [i_hds[b]])
                            pr_sql_q_s2s, pr_sql_i = gen_sql_q_from_i_vg([tokens[b]], [nlu[b]], [nlu_t[b]], [hds[b]], [tt_to_t_idx[b]], pnt_start_tok, pnt_end_tok, [new_pnt_list_batch11[0]], pr_i_vg_list, pr_i_vg_sub_list)
                            try:
                                idx_agg = pr_sql_i[0]['agg']
                                idx_sel = pr_sql_i[0]['sel']
                                testable = True
                            except:
                                testable = False
                                pass
                            if testable:
                                try:
                                    conds = pr_sql_i[0]['conds']
                                except:
                                    conds = []
                                try:
                                    pr_ans1 = engine.execute(tb[b]['id'], idx_sel, idx_agg, conds)
                                    executable = bool(pr_ans1)
                                except:
                                    executable = False
                            if testable:
                                if executable:
                                    add_candidate = True
                                else:
                                    add_candidate = False
                            else:
                                add_candidate = True
                            if add_candidate:
                                selected_candidates1.append(new_pnt_list_batch11)
                                pnt_list_beam[cnt][b] = new_pnt_list_batch11
                                cnt += 1
                    if cnt == beam_size:
                        break
                if cnt < beam_size:
                    for i_junk in range(cnt, beam_size):
                        pnt_list_beam[i_junk][b] = [[pnt_end_tok], -9999999]
            for i_beam in range(beam_size):
                cpnt = torch.zeros(bS, mL_input)
                idx_batch = [seq_score[0][-1] for seq_score in pnt_list_beam[i_beam]]
                pnt_n = torch.tensor(idx_batch)
                cpnt = cpnt.scatter_(1, pnt_n.unsqueeze(1), 1)
                cpnt = cpnt.unsqueeze(1)
                cpnt_beam[i_beam] = cpnt
            t += 1
        pr_pnt_idxs = []
        p_list = []
        for b in range(bS):
            pnt_list_beam_best = pnt_list_beam[0]
            pr_pnt_idxs.append(pnt_list_beam_best[b][0])
            p_list.append(pnt_list_beam_best[b][1])
        return pr_pnt_idxs, p_list, pnt_list_beam


class FT_s2s_1(nn.Module):
    """ Decoder-Layer """

    def __init__(self, iS, hS, lS, dr, max_seq_length, n_cond_ops, n_agg_ops, old=False):
        super(FT_s2s_1, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_where_num = 4
        self.decoder_s2s = Decoder_s2s(iS, hS, lS, dr, max_seq_length)

    def forward(self, wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs=None):
        score = self.decoder_s2s(wenc_s2s, l_input, cls_vec, pnt_start_tok, g_pnt_idxs)
        return score

    def EG_forward(self, wenc_s2s, l_input, cls_vec, pnt_start_tok, pnt_end_tok, i_sql_vocab, i_nlu, i_hds, tokens, nlu, nlu_t, hds, tt_to_t_idx, tb, engine, beam_size=4, beam_only=True):
        """ EG-guided beam-search """
        score = self.decoder_s2s.EG_forward(wenc_s2s, l_input, cls_vec, pnt_start_tok, pnt_end_tok, i_sql_vocab, i_nlu, i_hds, tokens, nlu, nlu_t, hds, tt_to_t_idx, tb, engine, beam_size, beam_only)
        return score


def cal_prob_sa(s_sa, pr_sa):
    ps = F.softmax(s_sa, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_sa1 = pr_sa[b]
        p1 = ps1[pr_sa1]
        p.append(p1.item())
    return p


def cal_prob_sc(s_sc, pr_sc):
    ps = F.softmax(s_sc, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_sc1 = pr_sc[b]
        p1 = ps1[pr_sc1]
        p.append(p1.item())
    return p


def cal_prob_select(p_sc, p_sa):
    p_select = []
    for b, p_sc1 in enumerate(p_sc):
        p1 = 1.0
        p1 *= p_sc1
        p1 *= p_sa[b]
        p_select.append(p1)
    return p_select


def cal_prob_tot(p_select, p_where):
    p_tot = []
    for b, p_select1 in enumerate(p_select):
        p_where1 = p_where[b]
        p_tot.append(p_select1 * p_where1)
    return p_tot


def cal_prob_where(p_wn, p_wc, p_wo, p_wvi):
    p_where = []
    for b, p_wn1 in enumerate(p_wn):
        p1 = 1.0
        p1 *= p_wn1
        p_wc1 = p_wc[b]
        for i_wn, p_wc11 in enumerate(p_wc1):
            p_wo11 = p_wo[b][i_wn]
            p_wv11_st, p_wv11_ed = p_wvi[b][i_wn]
            p1 *= p_wc11
            p1 *= p_wo11
            p1 *= p_wv11_st
            p1 *= p_wv11_ed
        p_where.append(p1)
    return p_where


def cal_prob_wn(s_wn, pr_wn):
    ps = F.softmax(s_wn, dim=1)
    p = []
    for b, ps1 in enumerate(ps):
        pr_wn1 = pr_wn[b]
        p1 = ps1[pr_wn1]
        p.append(p1.item())
    return p


class FT_Scalar_1(nn.Module):
    """ Shallow-Layer """

    def __init__(self, iS, hS, lS, dr, n_cond_ops, n_agg_ops, old=False):
        super(FT_Scalar_1, self).__init__()
        self.iS = iS
        self.hS = hS
        self.ls = lS
        self.dr = dr
        self.n_cond_ops = n_cond_ops
        self.n_agg_ops = n_agg_ops
        self.n_where_num = 4

    def scp(self, wemb_h, l_hs):
        bS, max_header_len, _ = wemb_h.shape
        s_sc = torch.zeros(bS, max_header_len)
        s_sc[:, :] = wemb_h[:, :, 0]
        for b, l_hs1 in enumerate(l_hs):
            s_sc[b, l_hs1:] = -9999999999.0
        return s_sc

    def sap(self, wemb_h, pr_sc, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape
        s_sa = torch.zeros([bS, self.n_agg_ops])
        for b, pr_sc1 in enumerate(pr_sc):
            s_sa[b, :] = wemb_h[b, pr_sc1, idx_st:idx_ed]
        return s_sa

    def wnp(self, cls_vec):
        bS = cls_vec.shape[0]
        s_wn = torch.zeros(bS, self.n_where_num + 1)
        s_wn[:, :] = cls_vec[:, 0:self.n_where_num + 1]
        return s_wn

    def wcp(self, wemb_h, l_hs, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape
        s_wc = torch.zeros(bS, max_header_len, 1)
        s_wc[:, :, :] = wemb_h[:, :, idx_st:idx_ed]
        s_wc = s_wc.squeeze(2)
        for b, l_hs1 in enumerate(l_hs):
            s_wc[b, l_hs1:] = -99999999999.0
        return s_wc

    def wop(self, wemb_h, pr_wc, idx_st, idx_ed):
        bS, max_header_len, _ = wemb_h.shape
        s_wo = torch.zeros([bS, self.n_where_num, self.n_cond_ops])
        for b, pr_wc1 in enumerate(pr_wc):
            if len(pr_wc1) > 0:
                s_wo[b, 0:len(pr_wc1), :] = wemb_h[b, pr_wc1, idx_st:idx_ed]
            else:
                pass
        return s_wo

    def wvp(self, wemb_n, l_n, pr_wc):
        bS, _, _ = wemb_n.shape
        s_wv = torch.zeros([bS, self.n_where_num, max(l_n), 2])
        for b, pr_wc1 in enumerate(pr_wc):
            if len(pr_wc1) > 0:
                s_wv[b, 0:len(pr_wc1), :, 0] = wemb_n[b, :, pr_wc1].transpose(0, 1)
                s_wv[b, 0:len(pr_wc1), :, 1] = wemb_n[b, :, [(pr_wc11 + 100) for pr_wc11 in pr_wc1]].transpose(0, 1)
            else:
                pass
        for b, l_n1 in enumerate(l_n):
            if l_n1 < max(l_n):
                s_wv[b, :, l_n1:, :] = -100000000000.0
        return s_wv

    def forward(self, wemb_n, l_n, wemb_h, l_hs, cls_vec, g_sc=None, g_sa=None, g_wn=None, g_wc=None, g_wo=None, g_wvi=None, show_p_sc=False, show_p_sa=False, show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        s_sc = self.scp(wemb_h, l_hs)
        if g_sc:
            pr_sc = g_sc
        else:
            pr_sc = pred_sc(s_sc)
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops
        s_sa = self.sap(wemb_h, pr_sc, idx_st, idx_ed)
        if g_sa:
            pr_sa = g_sa
        else:
            pr_sa = pred_sa(s_sa)
        s_wn = self.wnp(cls_vec)
        if g_wn:
            pr_wn = g_wn
        else:
            pr_wn = pred_wn(s_wn)
        idx_st = idx_ed + 1
        idx_ed = idx_st + 1
        s_wc = self.wcp(wemb_h, l_hs, idx_st, idx_ed)
        if g_wc:
            pr_wc = g_wc
        else:
            pr_wc = pred_wc(pr_wn, s_wc)
        idx_st = idx_ed + 1
        idx_ed = idx_st + self.n_cond_ops
        s_wo = self.wop(wemb_h, pr_wc, idx_st, idx_ed)
        if g_wo:
            pr_wo = g_wo
        else:
            pr_wo = pred_wo(pr_wn, s_wo)
        s_wv = self.wvp(wemb_n, l_n, pr_wc)
        return s_sc, s_sa, s_wn, s_wc, s_wo, s_wv

    def forward_EG(self, wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu, beam_size=4):
        """
        Execution-guided beam decoding.
        Essentially identical with that of NL2SQL Layer.
        """
        prob_sca, pr_sc_best, pr_sa_best, p_sc_best, p_sa_best, p_select = self.EG_decoding_select(wemb_h, l_hs, tb, beam_size=beam_size)
        prob_w, prob_wn_w, pr_wn_based_on_prob, pr_sql_i, pr_wvi_best, p_where, p_wn_best, p_wc_best, p_wo_best, p_wvi_best = self.EG_decoding_where(wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb, nlu_t, nlu_tt, tt_to_t_idx, nlu, pr_sc_best, pr_sa_best, beam_size=4)
        p_tot = cal_prob_tot(p_select, p_where)
        return pr_sc_best, pr_sa_best, pr_wn_based_on_prob, pr_wvi_best, pr_sql_i, p_tot, p_select, p_where, p_sc_best, p_sa_best, p_wn_best, p_wc_best, p_wo_best, p_wvi_best

    def EG_decoding_select(self, wemb_h, l_hs, tb, beam_size=4, show_p_sc=False, show_p_sa=False):
        s_sc = self.scp(wemb_h, l_hs)
        prob_sc = F.softmax(s_sc, dim=-1)
        bS, mcL = s_sc.shape
        prob_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops])
        score_sc_sa = torch.zeros([bS, beam_size, self.n_agg_ops])
        prob_sca = torch.zeros_like(prob_sc_sa)
        pr_sc_beam = pred_sc_beam(s_sc, beam_size)
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops
        for i_beam in range(beam_size):
            pr_sc = list(array(pr_sc_beam)[:, i_beam])
            s_sa = self.sap(wemb_h, pr_sc, idx_st, idx_ed)
            prob_sa = F.softmax(s_sa, dim=-1)
            prob_sc_sa[:, i_beam, :] = prob_sa
            score_sc_sa[:, i_beam, :] = s_sa
            prob_sc_selected = prob_sc[range(bS), pr_sc]
            prob_sca[:, i_beam, :] = (prob_sa.t() * prob_sc_selected).t()
        idxs = topk_multi_dim(torch.tensor(prob_sca), n_topk=beam_size, batch_exist=True)
        idxs = remap_sc_idx(idxs, pr_sc_beam)
        idxs_arr = array(idxs)
        beam_idx_sca = [0] * bS
        beam_meet_the_final = [False] * bS
        while True:
            pr_sc = idxs_arr[range(bS), beam_idx_sca, 0]
            pr_sa = idxs_arr[range(bS), beam_idx_sca, 1]
            check = check_sc_sa_pairs(tb, pr_sc, pr_sa)
            if sum(check) == bS:
                break
            else:
                for b, check1 in enumerate(check):
                    if not check1:
                        beam_idx_sca[b] += 1
                        if beam_idx_sca[b] >= beam_size:
                            beam_meet_the_final[b] = True
                            beam_idx_sca[b] -= 1
                    else:
                        beam_meet_the_final[b] = True
            if sum(beam_meet_the_final) == bS:
                break
        pr_sc_best = list(pr_sc)
        pr_sa_best = list(pr_sa)
        p_sc_best = cal_prob_sc(s_sc, pr_sc_best)
        p_sa_best = cal_prob_sa(score_sc_sa[range(bS), beam_idx_sca, :].squeeze(1), pr_sa_best)
        p_select = cal_prob_select(p_sc_best, p_sa_best)
        return prob_sca, pr_sc_best, pr_sa_best, p_sc_best, p_sa_best, p_select

    def EG_decoding_where(self, wemb_n, l_n, wemb_h, l_hs, cls_vec, engine, tb, nlu_t, nlu_wp_t, tt_to_t_idx, nlu, pr_sc_best, pr_sa_best, beam_size=4, show_p_wn=False, show_p_wc=False, show_p_wo=False, show_p_wv=False):
        bS, max_header_len, _ = wemb_h.shape
        idx_st = 1
        idx_ed = 1 + self.n_agg_ops
        s_wn = self.wnp(cls_vec)
        prob_wn = F.softmax(s_wn, dim=-1).detach().numpy()
        idx_st = idx_ed + 1
        idx_ed = idx_st + 1
        s_wc = self.wcp(wemb_h, l_hs, idx_st, idx_ed)
        prob_wc = torch.sigmoid(s_wc).detach().numpy()
        pr_wn_max = [self.n_where_num] * bS
        pr_wc_max = pred_wc(pr_wn_max, s_wc)
        prob_wc_max = zeros([bS, self.n_where_num])
        for b, pr_wc_max1 in enumerate(pr_wc_max):
            prob_wc_max[b, :] = prob_wc[b, pr_wc_max1]
        idx_st = idx_ed + 1
        idx_ed = idx_st + self.n_cond_ops
        s_wo_max = self.wop(wemb_h, pr_wc_max, idx_st, idx_ed)
        prob_wo_max = F.softmax(s_wo_max, dim=-1).detach().numpy()
        pr_wvi_beam_op_list = []
        prob_wvi_beam_op_list = []
        prob_wvi_beam_st_op_list = []
        prob_wvi_beam_ed_op_list = []
        for i_op in range(self.n_cond_ops - 1):
            pr_wo_temp = [[i_op] * self.n_where_num] * bS
            s_wv = self.wvp(wemb_n, l_n, pr_wc_max)
            prob_wv = F.softmax(s_wv, dim=-2).detach().numpy()
            pr_wvi_beam, prob_wvi_beam, prob_wvi_beam_st, prob_wvi_beam_ed = pred_wvi_se_beam(self.n_where_num, s_wv, beam_size)
            pr_wvi_beam_op_list.append(pr_wvi_beam)
            prob_wvi_beam_op_list.append(prob_wvi_beam)
            prob_wvi_beam_st_op_list.append(prob_wvi_beam_st)
            prob_wvi_beam_ed_op_list.append(prob_wvi_beam_ed)
        n_wv_beam_pairs = prob_wvi_beam.shape[2]
        prob_w = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wc_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wo_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wvi_st_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        prob_wvi_ed_dupl = zeros([bS, self.n_where_num, self.n_cond_ops - 1, n_wv_beam_pairs])
        for b in range(bS):
            for i_wn in range(self.n_where_num):
                for i_op in range(self.n_cond_ops - 1):
                    p_wc = prob_wc_max[b, i_wn]
                    for i_wv_beam in range(n_wv_beam_pairs):
                        p_wo = prob_wo_max[b, i_wn, i_op]
                        p_wv = prob_wvi_beam_op_list[i_op][b, i_wn, i_wv_beam]
                        prob_w[b, i_wn, i_op, i_wv_beam] = p_wc * p_wo * p_wv
                        prob_wc_dupl[b, i_wn, i_op, i_wv_beam] = p_wc
                        prob_wo_dupl[b, i_wn, i_op, i_wv_beam] = p_wo
                        p_wv_st = prob_wvi_beam_st_op_list[i_op][b, i_wn, i_wv_beam]
                        p_wv_ed = prob_wvi_beam_ed_op_list[i_op][b, i_wn, i_wv_beam]
                        prob_wvi_st_dupl[b, i_wn, i_op, i_wv_beam] = p_wv_st
                        prob_wvi_ed_dupl[b, i_wn, i_op, i_wv_beam] = p_wv_ed
        conds_max = []
        prob_conds_max = []
        idxs = topk_multi_dim(torch.tensor(prob_w), n_topk=beam_size, batch_exist=True)
        pr_wvi_max = []
        p_wc_max = []
        p_wo_max = []
        p_wvi_max = []
        for b, idxs1 in enumerate(idxs):
            conds_max1 = []
            prob_conds_max1 = []
            pr_wvi1_max = []
            p_wc1_max = []
            p_wo1_max = []
            p_wvi1_max = []
            for i_wn, idxs11 in enumerate(idxs1):
                i_wc = pr_wc_max[b][idxs11[0]]
                i_op = idxs11[1]
                wvi = pr_wvi_beam_op_list[i_op][b][idxs11[0]][idxs11[2]]
                temp_pr_wv_str, _ = convert_pr_wvi_to_string([[wvi]], [nlu_t[b]], [nlu_wp_t[b]], [tt_to_t_idx[b]], [nlu[b]])
                merged_wv11 = merge_wv_t1_eng(temp_pr_wv_str[0][0], nlu[b])
                conds11 = [i_wc, i_op, merged_wv11]
                prob_conds11 = prob_w[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wc11_max = prob_wc_dupl[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wo11_max = prob_wo_dupl[b, idxs11[0], idxs11[1], idxs11[2]]
                p_wvi11_max = [prob_wvi_st_dupl[b, idxs11[0], idxs11[1], idxs11[2]], prob_wvi_ed_dupl[b, idxs11[0], idxs11[1], idxs11[2]]]
                pr_ans = engine.execute(tb[b]['id'], pr_sc_best[b], pr_sa_best[b], [conds11])
                if bool(pr_ans):
                    conds_max1.append(conds11)
                    prob_conds_max1.append(prob_conds11)
                    pr_wvi1_max.append(wvi)
                    p_wc1_max.append(p_wc11_max)
                    p_wo1_max.append(p_wo11_max)
                    p_wvi1_max.append(p_wvi11_max)
            conds_max.append(conds_max1)
            prob_conds_max.append(prob_conds_max1)
            pr_wvi_max.append(pr_wvi1_max)
            p_wc_max.append(p_wc1_max)
            p_wo_max.append(p_wo1_max)
            p_wvi_max.append(p_wvi1_max)
        pr_sql_i = []
        prob_wn_w = []
        pr_wn_based_on_prob = []
        pr_wvi_best = []
        p_wc = []
        p_wo = []
        p_wvi = []
        for b, prob_wn1 in enumerate(prob_wn):
            max_executable_wn1 = len(conds_max[b])
            prob_wn_w1 = []
            prob_wn_w1.append(prob_wn1[0])
            for i_wn in range(max_executable_wn1):
                prob_wn_w11 = prob_wn1[i_wn + 1] * prob_conds_max[b][i_wn]
                prob_wn_w1.append(prob_wn_w11)
            pr_wn_based_on_prob.append(argmax(prob_wn_w1))
            prob_wn_w.append(prob_wn_w1)
            pr_sql_i1 = {'agg': pr_sa_best[b], 'sel': pr_sc_best[b], 'conds': conds_max[b][:pr_wn_based_on_prob[b]]}
            pr_wvi_best1 = pr_wvi_max[b][:pr_wn_based_on_prob[b]]
            pr_sql_i.append(pr_sql_i1)
            pr_wvi_best.append(pr_wvi_best1)
            p_wc.append(p_wc_max[b][:pr_wn_based_on_prob[b]])
            p_wo.append(p_wo_max[b][:pr_wn_based_on_prob[b]])
            p_wvi.append(p_wvi_max[b][:pr_wn_based_on_prob[b]])
        p_wn = cal_prob_wn(s_wn, pr_wn_based_on_prob)
        p_where = cal_prob_where(p_wn, p_wc, p_wo, p_wvi)
        return prob_w, prob_wn_w, pr_wn_based_on_prob, pr_sql_i, pr_wvi_best, p_where, p_wn, p_wc, p_wo, p_wvi


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BERTAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BERTIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BERTLayer,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5, hidden_dropout_prob=0.5, intermediate_size=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BERTLayerNorm,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BERTOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BERTPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BERTSelfAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (BERTSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_guotong1988_NL2SQL_RULE(_paritybench_base):
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

