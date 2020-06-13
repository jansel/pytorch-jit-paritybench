import sys
_module = sys.modules[__name__]
del sys
claf = _module
__version__ = _module
config = _module
args = _module
namespace = _module
pattern = _module
registry = _module
utils = _module
data = _module
collate = _module
data_handler = _module
dataset = _module
base = _module
bert = _module
multi_task = _module
regression = _module
seq_cls = _module
squad = _module
tok_cls = _module
wikisql = _module
dto = _module
batch = _module
bert_feature = _module
helper = _module
reader = _module
conll2003 = _module
glue = _module
cola = _module
mnli = _module
mrpc = _module
qnli = _module
qqp = _module
rte = _module
sst = _module
stsb = _module
wnli = _module
decorator = _module
arguments = _module
register = _module
factory = _module
data_loader = _module
data_reader = _module
model = _module
optimizer = _module
tokens = _module
learn = _module
experiment = _module
mode = _module
optimization = _module
exponential_moving_avarage = _module
learning_rate_scheduler = _module
tensorboard = _module
trainer = _module
utils = _module
machine = _module
components = _module
retrieval = _module
tfidf = _module
ensemble_topk = _module
knowlege_base = _module
docs = _module
module = _module
nlu = _module
open_qa = _module
metric = _module
classification = _module
korquad_v1_official = _module
squad_v1_official = _module
squad_v2_official = _module
wikisql_lib = _module
dbengine = _module
query = _module
wikisql_official = _module
base = _module
cls_utils = _module
bert = _module
category = _module
mixin = _module
reading_comprehension = _module
bert = _module
bidaf = _module
bidaf_no_answer = _module
docqa = _module
docqa_no_answer = _module
drqa = _module
mixin = _module
qanet = _module
roberta = _module
bert = _module
roberta = _module
semantic_parsing = _module
sqlnet = _module
sequence_classification = _module
bert = _module
roberta = _module
structured_self_attention = _module
token_classification = _module
bert = _module
modules = _module
activation = _module
attention = _module
bi_attention = _module
co_attention = _module
docqa_attention = _module
multi_head_attention = _module
seq_attention = _module
conv = _module
depthwise_separable_conv = _module
pointwise_conv = _module
encoder = _module
lstm_cell_with_projection = _module
positional = _module
functional = _module
initializer = _module
layer = _module
highway = _module
normalization = _module
positionwise = _module
residual = _module
scalar_mix = _module
nsml = _module
cove = _module
elmo = _module
embedding = _module
base = _module
bert_embedding = _module
char_embedding = _module
cove_embedding = _module
elmo_embedding = _module
frequent_word_embedding = _module
sparse_feature = _module
word_embedding = _module
hangul = _module
indexer = _module
bert_indexer = _module
char_indexer = _module
elmo_indexer = _module
exact_match_indexer = _module
linguistic_indexer = _module
word_indexer = _module
linguistic = _module
text_handler = _module
token_embedder = _module
base = _module
basic_embedder = _module
reading_comprehension_embedder = _module
token_maker = _module
tokenizer = _module
bpe = _module
char = _module
pass_text = _module
sent = _module
subword = _module
word = _module
vocabulary = _module
conf = _module
eval = _module
predict = _module
convert_checkpoint_to_bert_model = _module
convert_embedding_to_vocab_txt = _module
make_squad_synthetic_data = _module
plot = _module
setup = _module
tests = _module
test_batch = _module
test_docs = _module
test_functional = _module
test_vocabulary = _module
test_config = _module
test_machine = _module
test_multi_task = _module
test_reading_comprehension = _module
test_semantic_parsing = _module
test_sequence_classification = _module
test_token_classification = _module
test_tokenizers = _module
train = _module

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


import torch


import logging


import random


from torch.nn.utils import clip_grad_norm_


from collections import OrderedDict


import re


from torch.nn import DataParallel


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import math


import itertools


from typing import Callable


from typing import List


from typing import Tuple


from typing import Union


from typing import Optional


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn import ParameterList


from torch.nn import Parameter


from torch import nn


from typing import Dict


from typing import Any


import warnings


import numpy


from torch.nn.modules import Dropout


class ModelBase(nn.Module):
    """
    Model Base Class

    Args:
        token_embedder: (claf.tokens.token_embedder.base) TokenEmbedder
    """

    def __init__(self):
        super(ModelBase, self).__init__()

    def forward(self, inputs):
        raise NotImplementedError

    def make_metrics(self, predictions):
        raise NotImplementedError

    def make_predictions(self, features):
        """
        for Metrics
        """
        raise NotImplementedError

    def predict(self, features):
        """
        Inference
        """
        raise NotImplementedError

    def print_examples(self, params):
        """
        Print evaluation examples
        """
        raise NotImplementedError

    def write_predictions(self, predictions, file_path=None, is_dict=True):
        data_type = 'train' if self.training else 'valid'
        pred_dir = Path(self._log_dir) / 'predictions'
        pred_dir.mkdir(exist_ok=True)
        if file_path is None:
            file_path = (
                f'predictions-{data_type}-{self._train_counter.get_display()}.json'
                )
        pred_path = pred_dir / file_path
        with open(pred_path, 'w') as out_file:
            if is_dict:
                out_file.write(json.dumps(predictions, indent=4))
            else:
                out_file.write(predictions)

    def is_ready(self):
        properties = [self._config, self._log_dir, self._train_counter,
            self._vocabs]
        return all([(p is not None) for p in properties])

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config):
        self._config = config

    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        self._log_dir = log_dir

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    @property
    def train_counter(self):
        return self._train_counter

    @train_counter.setter
    def train_counter(self, train_counter):
        self._train_counter = train_counter

    @property
    def vocabs(self):
        return self._vocabs

    @vocabs.setter
    def vocabs(self, vocabs):
        self._vocabs = vocabs


class SelfAttention(nn.Module):
    """
        Same bi-attention mechanism, only now between the passage and itself.
    """

    def __init__(self, rnn_dim, linear_dim, dropout=0.2, weight_init=True):
        super(SelfAttention, self).__init__()
        self.self_attention = attention.DocQAAttention(rnn_dim, linear_dim,
            self_attn=True, weight_init=weight_init)
        self.self_attn_Linear = nn.Linear(rnn_dim * 6, linear_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fn = F.relu
        if weight_init:
            initializer.weight(self.self_attn_Linear)

    def forward(self, context, context_mask):
        context_self_attnded = self.self_attention(context, context_mask,
            context, context_mask)
        return self.activation_fn(self.self_attn_Linear(context_self_attnded))


class SelfAttention(nn.Module):
    """
        Same bi-attention mechanism, only now between the passage and itself.
    """

    def __init__(self, rnn_dim, linear_dim, dropout=0.2, weight_init=True):
        super(SelfAttention, self).__init__()
        self.self_attention = attention.DocQAAttention(rnn_dim, linear_dim,
            self_attn=True, weight_init=weight_init)
        self.self_attn_Linear = nn.Linear(rnn_dim * 6, linear_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation_fn = F.relu
        if weight_init:
            initializer.weight(self.self_attn_Linear)

    def forward(self, context, context_mask):
        context_self_attnded = self.self_attention(context, context_mask,
            context, context_mask)
        context_self_attnded = self.activation_fn(self.self_attn_Linear(
            context_self_attnded))
        return context_self_attnded


class NoAnswer(nn.Module):
    """
        No-Answer Option

        * Args:
            embed_dim: the number of passage embedding dimension
            bias_hidden_dim: bias use two layer mlp, the number of hidden_size
    """

    def __init__(self, embed_dim, bias_hidden_dim):
        super(NoAnswer, self).__init__()
        self.self_attn = nn.Linear(embed_dim, 1)
        self.bias_mlp = nn.Sequential(nn.Linear(embed_dim * 3,
            bias_hidden_dim), nn.ReLU(), nn.Linear(bias_hidden_dim, 1))

    def forward(self, context_embed, span_start_logits, span_end_logits):
        p_1_h = F.softmax(span_start_logits, -1).unsqueeze(1)
        p_2_h = F.softmax(span_end_logits, -1).unsqueeze(1)
        p_3_h = self.self_attn(context_embed).transpose(1, 2)
        v_1 = torch.matmul(p_1_h, context_embed)
        v_2 = torch.matmul(p_2_h, context_embed)
        v_3 = torch.matmul(p_3_h, context_embed)
        return self.bias_mlp(torch.cat([v_1, v_2, v_3], -1)).squeeze(-1)


class EncoderBlock(nn.Module):
    """
        Encoder Block

        []: residual
        position_encoding -> [convolution-layer] x # -> [self-attention-layer] -> [feed-forward-layer]

        - convolution-layer: depthwise separable convolutions
        - self-attention-layer: multi-head attention
        - feed-forward-layer: pointwise convolution

        * Args:
            model_dim: the number of model dimension
            num_heads: the number of head in multi-head attention
            kernel_size: convolution kernel size
            num_conv_block: the number of convolution block
            dropout: the dropout probability
            layer_dropout: the layer dropout probability
                (cf. Deep Networks with Stochastic Depth(https://arxiv.org/abs/1603.09382) )
    """

    def __init__(self, model_dim=128, num_head=8, kernel_size=5,
        num_conv_block=4, dropout=0.1, layer_dropout=0.9):
        super(EncoderBlock, self).__init__()
        self.position_encoding = encoder.PositionalEncoding(model_dim)
        self.dropout = nn.Dropout(dropout)
        self.num_conv_block = num_conv_block
        self.conv_blocks = nn.ModuleList([conv.DepSepConv(model_dim,
            model_dim, kernel_size) for _ in range(num_conv_block)])
        self.self_attention = attention.MultiHeadAttention(num_head=
            num_head, model_dim=model_dim, dropout=dropout)
        self.feedforward_layer = layer.PositionwiseFeedForward(model_dim, 
            model_dim * 4, dropout=dropout)
        if layer_dropout < 1.0:
            L = num_conv_block + 2 - 1
            layer_dropout_prob = round(1 - 1 / L * (1 - layer_dropout), 3)
            self.residuals = nn.ModuleList(layer.ResidualConnection(
                model_dim, layer_dropout=layer_dropout_prob, layernorm=True
                ) for l in range(num_conv_block + 2))
        else:
            self.residuals = nn.ModuleList(layer.ResidualConnection(
                model_dim, layernorm=True) for l in range(num_conv_block + 2))

    def forward(self, x, mask=None):
        x = self.position_encoding(x)
        for i, conv_block in enumerate(self.conv_blocks):
            x = self.residuals[i](x, sub_layer_fn=conv_block)
            x = self.dropout(x)
        self_attention = lambda x: self.self_attention(q=x, k=x, v=x, mask=mask
            )
        x = self.residuals[self.num_conv_block](x, sub_layer_fn=self_attention)
        x = self.dropout(x)
        x = self.residuals[self.num_conv_block + 1](x, sub_layer_fn=self.
            feedforward_layer)
        x = self.dropout(x)
        return x


class AggPredictor(nn.Module):

    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout, agg_count
        ):
        super(AggPredictor, self).__init__()
        self.question_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.seq_attn = attention.LinearSeqAttn(model_dim)
        self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh(),
            nn.Linear(model_dim, agg_count))

    def forward(self, question_embed, question_mask):
        encoded_question, _ = self.question_rnn(question_embed)
        attn_matrix = self.seq_attn(encoded_question, question_mask)
        attn_question = f.weighted_sum(attn_matrix, encoded_question)
        logits = self.mlp(attn_question)
        return logits


class SelPredictor(nn.Module):

    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout,
        column_attention=None):
        super(SelPredictor, self).__init__()
        self.column_attention = column_attention
        self.question_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        if column_attention:
            self.linear_attn = nn.Linear(model_dim, model_dim)
        else:
            self.seq_attn = attention.LinearSeqAttn(model_dim)
        self.column_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.linear_question = nn.Linear(model_dim, model_dim)
        self.linear_column = nn.Linear(model_dim, model_dim)
        self.mlp = nn.Sequential(nn.Tanh(), nn.Linear(model_dim, 1))

    def forward(self, question_embed, question_mask, column_embed,
        column_name_mask, column_mask):
        B, C_L, N_L, embed_D = list(column_embed.size())
        encoded_column = utils.encode_column(column_embed, column_name_mask,
            self.column_rnn)
        encoded_question, _ = self.question_rnn(question_embed)
        if self.column_attention:
            attn_matrix = torch.bmm(encoded_column, self.linear_attn(
                encoded_question).transpose(1, 2))
            attn_matrix = f.add_masked_value(attn_matrix, question_mask.
                unsqueeze(1), value=-10000000.0)
            attn_matrix = F.softmax(attn_matrix, dim=-1)
            attn_question = (encoded_question.unsqueeze(1) * attn_matrix.
                unsqueeze(3)).sum(2)
        else:
            attn_matrix = self.seq_attn(encoded_question, question_mask)
            attn_question = f.weighted_sum(attn_matrix, encoded_question)
            attn_question = attn_question.unsqueeze(1)
        logits = self.mlp(self.linear_question(attn_question) + self.
            linear_column(encoded_column)).squeeze()
        logits = f.add_masked_value(logits, column_mask, value=-10000000.0)
        return logits


class CondsPredictor(nn.Module):

    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout,
        conds_op_count, column_maxlen, token_maxlen, column_attention=None):
        super(CondsPredictor, self).__init__()
        self.num_predictor = CondsNumPredictor(embed_dim, model_dim,
            rnn_num_layer, dropout, column_maxlen)
        self.column_predictor = CondsColPredictor(embed_dim, model_dim,
            rnn_num_layer, dropout, column_attention=column_attention)
        self.op_predictor = CondsOpPredictor(embed_dim, model_dim,
            rnn_num_layer, dropout, conds_op_count, column_maxlen,
            column_attention=column_attention)
        self.value_pointer = CondsValuePointer(embed_dim, model_dim,
            rnn_num_layer, dropout, column_maxlen, token_maxlen)

    def forward(self, question_embed, question_mask, column_embed,
        column_name_mask, column_mask, col_idx, conds_val_pos):
        num_logits = self.num_predictor(question_embed, question_mask,
            column_embed, column_name_mask, column_mask)
        column_logits = self.column_predictor(question_embed, question_mask,
            column_embed, column_name_mask, column_mask)
        if col_idx is None:
            col_idx = []
            preds_num = torch.argmax(num_logits, dim=-1)
            for i in range(column_logits.size(0)):
                _, pred_conds_column_idx = torch.topk(column_logits[i],
                    preds_num[i])
                col_idx.append(pred_conds_column_idx.tolist())
        op_logits = self.op_predictor(question_embed, question_mask,
            column_embed, column_name_mask, col_idx)
        value_logits = self.value_pointer(question_embed, question_mask,
            column_embed, column_name_mask, col_idx, conds_val_pos)
        return num_logits, column_logits, op_logits, value_logits


class CondsNumPredictor(nn.Module):

    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout,
        column_maxlen):
        super(CondsNumPredictor, self).__init__()
        self.model_dim = model_dim
        self.column_maxlen = column_maxlen
        self.column_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.column_seq_attn = attention.LinearSeqAttn(model_dim)
        self.column_to_hidden_state = nn.Linear(model_dim, 2 * model_dim)
        self.column_to_cell_state = nn.Linear(model_dim, 2 * model_dim)
        self.question_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.question_seq_attn = attention.LinearSeqAttn(model_dim)
        self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh(),
            nn.Linear(model_dim, column_maxlen + 1))

    def forward(self, question_embed, question_mask, column_embed,
        column_name_mask, column_mask):
        B, C_L, N_L, embed_D = list(column_embed.size())
        encoded_column = utils.encode_column(column_embed, column_name_mask,
            self.column_rnn)
        attn_column = self.column_seq_attn(encoded_column, column_mask)
        out_column = f.weighted_sum(attn_column, encoded_column)
        question_rnn_hidden_state = self.column_to_hidden_state(out_column
            ).view(B, self.column_maxlen, self.model_dim // 2).transpose(0, 1
            ).contiguous()
        question_rnn_cell_state = self.column_to_cell_state(out_column).view(B,
            self.column_maxlen, self.model_dim // 2).transpose(0, 1
            ).contiguous()
        encoded_question, _ = self.question_rnn(question_embed, (
            question_rnn_hidden_state, question_rnn_cell_state))
        attn_question = self.question_seq_attn(encoded_question, question_mask)
        out_question = f.weighted_sum(attn_question, encoded_question)
        return self.mlp(out_question)


class CondsColPredictor(nn.Module):

    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout,
        column_attention=None):
        super(CondsColPredictor, self).__init__()
        self.column_attention = column_attention
        self.question_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        if column_attention:
            self.linear_attn = nn.Linear(model_dim, model_dim)
        else:
            self.seq_attn = attention.LinearSeqAttn(model_dim)
        self.column_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.linear_question = nn.Linear(model_dim, model_dim)
        self.linear_column = nn.Linear(model_dim, model_dim)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(model_dim, 1))

    def forward(self, question_embed, question_mask, column_embed,
        column_name_mask, column_mask):
        B, C_L, N_L, embed_D = list(column_embed.size())
        encoded_column = utils.encode_column(column_embed, column_name_mask,
            self.column_rnn)
        encoded_question, _ = self.question_rnn(question_embed)
        if self.column_attention:
            attn_matrix = torch.bmm(encoded_column, self.linear_attn(
                encoded_question).transpose(1, 2))
            attn_matrix = f.add_masked_value(attn_matrix, question_mask.
                unsqueeze(1), value=-10000000.0)
            attn_matrix = F.softmax(attn_matrix, dim=-1)
            attn_question = (encoded_question.unsqueeze(1) * attn_matrix.
                unsqueeze(3)).sum(2)
        else:
            attn_matrix = self.seq_attn(encoded_question, question_mask)
            attn_question = f.weighted_sum(attn_matrix, encoded_question)
            attn_question = attn_question.unsqueeze(1)
        logits = self.mlp(self.linear_question(attn_question) + self.
            linear_column(encoded_column)).squeeze()
        logits = f.add_masked_value(logits, column_mask, value=-10000000.0)
        return logits


class CondsOpPredictor(nn.Module):

    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout,
        op_count, column_maxlen, column_attention=None):
        super(CondsOpPredictor, self).__init__()
        self.column_attention = column_attention
        self.column_maxlen = column_maxlen
        self.question_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        if column_attention:
            self.linear_attn = nn.Linear(model_dim, model_dim)
        else:
            self.seq_attn = attention.LinearSeqAttn(model_dim)
        self.column_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.linear_question = nn.Linear(model_dim, model_dim)
        self.linear_column = nn.Linear(model_dim, model_dim)
        self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim), nn.Tanh(),
            nn.Linear(model_dim, op_count))

    def forward(self, question_embed, question_mask, column_embed,
        column_name_mask, col_idx):
        B, C_L, N_L, embed_D = list(column_embed.size())
        encoded_column = utils.encode_column(column_embed, column_name_mask,
            self.column_rnn)
        encoded_used_column = utils.filter_used_column(encoded_column,
            col_idx, padding_count=self.column_maxlen)
        encoded_question, _ = self.question_rnn(question_embed)
        if self.column_attention:
            attn_matrix = torch.matmul(self.linear_attn(encoded_question).
                unsqueeze(1), encoded_used_column.unsqueeze(3)).squeeze()
            attn_matrix = f.add_masked_value(attn_matrix, question_mask.
                unsqueeze(1), value=-10000000.0)
            attn_matrix = F.softmax(attn_matrix, dim=-1)
            attn_question = (encoded_question.unsqueeze(1) * attn_matrix.
                unsqueeze(3)).sum(2)
        else:
            attn_matrix = self.seq_attn(encoded_question, question_mask)
            attn_question = f.weighted_sum(attn_matrix, encoded_question)
            attn_question = attn_question.unsqueeze(1)
        return self.mlp(self.linear_question(attn_question) + self.
            linear_column(encoded_used_column)).squeeze()


class CondsValuePointer(nn.Module):

    def __init__(self, embed_dim, model_dim, rnn_num_layer, dropout,
        column_maxlen, token_maxlen):
        super(CondsValuePointer, self).__init__()
        self.model_dim = model_dim
        self.column_maxlen = column_maxlen
        self.token_maxlen = token_maxlen
        self.question_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.seq_attn = attention.LinearSeqAttn(model_dim)
        self.column_rnn = nn.LSTM(input_size=embed_dim, hidden_size=
            model_dim // 2, num_layers=rnn_num_layer, batch_first=True,
            dropout=dropout, bidirectional=True)
        self.decoder = nn.LSTM(input_size=self.token_maxlen, hidden_size=
            model_dim, num_layers=rnn_num_layer, batch_first=True, dropout=
            dropout)
        self.linear_column = nn.Linear(model_dim, model_dim)
        self.linear_conds = nn.Linear(model_dim, model_dim)
        self.linear_question = nn.Linear(model_dim, model_dim)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(model_dim, 1))

    def forward(self, question_embed, question_mask, column_embed,
        column_name_mask, col_idx, conds_val_pos):
        B, C_L, N_L, embed_D = list(column_embed.size())
        question_embed, question_mask = self.concat_start_and_end_zero_padding(
            question_embed, question_mask)
        encoded_column = utils.encode_column(column_embed, column_name_mask,
            self.column_rnn)
        encoded_used_column = utils.filter_used_column(encoded_column,
            col_idx, padding_count=self.column_maxlen)
        encoded_question, _ = self.question_rnn(question_embed)
        encoded_used_column = encoded_used_column.unsqueeze(2).unsqueeze(2)
        encoded_question = encoded_question.unsqueeze(1).unsqueeze(1)
        if conds_val_pos is None:
            MAX_DECODER_STEP = 50
            decoder_input = torch.zeros(4 * B, 1, self.token_maxlen)
            decoder_input[:, (0), (0)] = 2
            if torch.cuda.is_available():
                decoder_input = decoder_input
            decoder_hidden = None
            logits = []
            for _ in range(MAX_DECODER_STEP):
                step_logit, decoder_hidden = self.decode_then_output(
                    encoded_used_column, encoded_question, question_mask,
                    decoder_input, decoder_hidden=decoder_hidden)
                step_logit = step_logit.unsqueeze(1)
                logits.append(step_logit)
                _, decoder_idxs = step_logit.view(B * self.column_maxlen, -1
                    ).max(1)
                decoder_input = torch.zeros(B * self.column_maxlen, self.
                    token_maxlen).scatter_(1, decoder_idxs.cpu().unsqueeze(
                    1), 1)
                if torch.cuda.is_available():
                    decoder_input = decoder_input
            logits = torch.stack(logits, 2)
        else:
            decoder_input, _ = utils.convert_position_to_decoder_input(
                conds_val_pos, token_maxlen=self.token_maxlen)
            logits, _ = self.decode_then_output(encoded_used_column,
                encoded_question, question_mask, decoder_input)
        return logits

    def concat_start_and_end_zero_padding(self, question_embed, mask):
        B, Q_L, embed_D = list(question_embed.size())
        zero_padding = torch.zeros(B, 1, embed_D)
        mask_with_start_end = torch.zeros(B, Q_L + 2)
        if torch.cuda.is_available():
            zero_padding = zero_padding
            mask_with_start_end = mask_with_start_end
        question_embed_with_start_end = torch.cat([zero_padding,
            question_embed, zero_padding], dim=1)
        mask_with_start_end[:, (0)] = 1
        mask_with_start_end[:, 1:Q_L + 1] = mask
        question_lengths = torch.sum(mask, dim=-1).byte()
        for i in range(B):
            mask_with_start_end[i, question_lengths[i].item() + 1] = 1
        return question_embed_with_start_end, mask_with_start_end

    def decode_then_output(self, encoded_used_column, encoded_question,
        question_mask, decoder_input, decoder_hidden=None):
        B = encoded_used_column.size(0)
        decoder_output, decoder_hidden = self.decoder(decoder_input.view(B *
            self.column_maxlen, -1, self.token_maxlen), decoder_hidden)
        decoder_output = decoder_output.contiguous().view(B, self.
            column_maxlen, -1, self.model_dim)
        decoder_output = decoder_output.unsqueeze(3)
        logits = self.mlp(self.linear_column(encoded_used_column) + self.
            linear_conds(decoder_output) + self.linear_question(
            encoded_question)).squeeze()
        logits = f.add_masked_value(logits, question_mask.unsqueeze(1).
            unsqueeze(1), value=-10000000.0)
        return logits, decoder_hidden


class BiAttention(nn.Module):
    """
    Attention Flow Layer
        in BiDAF (https://arxiv.org/pdf/1611.01603.pdf)

    The Similarity matrix
    Context-to-query Attention (C2Q)
    Query-to-context Attention (Q2C)

    * Args:
        model_dim: The number of module dimension
    """

    def __init__(self, model_dim):
        super(BiAttention, self).__init__()
        self.model_dim = model_dim
        self.W = nn.Linear(6 * model_dim, 1, bias=False)

    def forward(self, context, context_mask, query, query_mask):
        c, c_mask, q, q_mask = context, context_mask, query, query_mask
        S = self._make_similiarity_matrix(c, q)
        masked_S = f.add_masked_value(S, query_mask.unsqueeze(1), value=-
            10000000.0)
        c2q = self._context2query(S, q, q_mask)
        q2c = self._query2context(masked_S.max(dim=-1)[0], c, c_mask)
        G = torch.cat((c, c2q, c * c2q, c * q2c), dim=-1)
        return G

    def _make_similiarity_matrix(self, c, q):
        B, C_L, Q_L = c.size(0), c.size(1), q.size(1)
        matrix_shape = B, C_L, Q_L, self.model_dim * 2
        c_aug = c.unsqueeze(2).expand(matrix_shape)
        q_aug = q.unsqueeze(1).expand(matrix_shape)
        c_q = torch.mul(c_aug, q_aug)
        concated_vector = torch.cat((c_aug, q_aug, c_q), dim=3)
        return self.W(concated_vector).view(c.size(0), C_L, Q_L)

    def _context2query(self, S, q, q_mask):
        attention = f.last_dim_masked_softmax(S, q_mask)
        c2q = f.weighted_sum(attention=attention, matrix=q)
        return c2q

    def _query2context(self, S, c, c_mask):
        attention = f.masked_softmax(S, c_mask)
        q2c = f.weighted_sum(attention=attention, matrix=c)
        return q2c.unsqueeze(1).expand(c.size())


class CoAttention(nn.Module):
    """
    CoAttention encoder
        in Dynamic Coattention Networks For Question Answering (https://arxiv.org/abs/1611.01604)

    check the Figure 2 in paper

    * Args:
        embed_dim: the number of input embedding dimension
    """

    def __init__(self, embed_dim):
        super(CoAttention, self).__init__()
        self.W_0 = nn.Linear(embed_dim * 3, 1, bias=False)

    def forward(self, context_embed, question_embed, context_mask=None,
        question_mask=None):
        C, Q = context_embed, question_embed
        B, C_L, Q_L, D = C.size(0), C.size(1), Q.size(1), Q.size(2)
        similarity_matrix_shape = torch.zeros(B, C_L, Q_L, D)
        C_ = C.unsqueeze(2).expand_as(similarity_matrix_shape)
        Q_ = Q.unsqueeze(1).expand_as(similarity_matrix_shape)
        C_Q = torch.mul(C_, Q_)
        S = self.W_0(torch.cat([C_, Q_, C_Q], 3)).squeeze(3)
        S_question = S
        if question_mask is not None:
            S_question = f.add_masked_value(S_question, question_mask.
                unsqueeze(1), value=-10000000.0)
        S_q = F.softmax(S_question, 2)
        S_context = S.transpose(1, 2)
        if context_mask is not None:
            S_context = f.add_masked_value(S_context, context_mask.
                unsqueeze(1), value=-10000000.0)
        S_c = F.softmax(S_context, 2)
        A = torch.bmm(S_q, Q)
        B = torch.bmm(S_q, S_c).bmm(C)
        out = torch.cat([C, A, C * A, C * B], dim=-1)
        return out


class DocQAAttention(nn.Module):
    """
        Bi-Attention Layer + (Self-Attention)
            in DocumentQA (https://arxiv.org/abs/1710.10723)

        * Args:
            rnn_dim: the number of GRU cell hidden size
            linear_dim: the number of linear hidden size

        * Kwargs:
            self_attn: (bool) self-attention
            weight_init: (bool) weight initialization

    """

    def __init__(self, rnn_dim, linear_dim, self_attn=False, weight_init=True):
        super(DocQAAttention, self).__init__()
        self.self_attn = self_attn
        self.input_w = nn.Linear(2 * rnn_dim, 1, bias=False)
        self.key_w = nn.Linear(2 * rnn_dim, 1, bias=False)
        self.dot_w = nn.Parameter(torch.randn(1, 1, rnn_dim * 2))
        torch.nn.init.xavier_uniform_(self.dot_w)
        self.bias = nn.Parameter(torch.FloatTensor([[1]]))
        self.diag_mask = nn.Parameter(torch.eye(5000))
        if weight_init:
            initializer.weight(self.input_w)
            initializer.weight(self.key_w)

    def forward(self, x, x_mask, key, key_mask):
        S = self._trilinear(x, key)
        if self.self_attn:
            seq_length = x.size(1)
            diag_mask = self.diag_mask.narrow(0, 0, seq_length).narrow(1, 0,
                seq_length)
            joint_mask = 1 - self._compute_attention_mask(x_mask, key_mask)
            mask = torch.clamp(diag_mask + joint_mask, 0, 1)
            masked_S = S + mask * -10000000.0
            x2key = self._x2key(masked_S, key, key_mask)
            return torch.cat((x, x2key, x * x2key), dim=-1)
        else:
            joint_mask = 1 - self._compute_attention_mask(x_mask, key_mask)
            masked_S = S + joint_mask * -10000000.0
            x2key = self._x2key(masked_S, key, key_mask)
            masked_S = f.add_masked_value(S, key_mask.unsqueeze(1), value=-
                10000000.0)
            key2x = self._key2x(masked_S.max(dim=-1)[0], x, x_mask)
            return torch.cat((x, x2key, x * x2key, x * key2x), dim=-1)

    def _compute_attention_mask(self, x_mask, key_mask):
        x_mask = x_mask.unsqueeze(2)
        key_mask = key_mask.unsqueeze(1)
        joint_mask = torch.mul(x_mask, key_mask)
        return joint_mask

    def _trilinear(self, x, key):
        B, X_L, K_L = x.size(0), x.size(1), key.size(1)
        matrix_shape = B, X_L, K_L
        x_logits = self.input_w(x).expand(matrix_shape)
        key_logits = self.key_w(key).transpose(1, 2).expand(matrix_shape)
        x_dots = torch.mul(x, self.dot_w)
        x_key = torch.matmul(x_dots, key.transpose(1, 2))
        return x_logits + key_logits + x_key

    def _x2key(self, S, key, key_mask):
        if self.self_attn:
            bias = torch.exp(self.bias)
            S = torch.exp(S)
            attention = S / (S.sum(dim=-1, keepdim=True).expand(S.size()) +
                bias.expand(S.size()))
        else:
            attention = F.softmax(S, dim=-1)
        x2key = f.weighted_sum(attention=attention, matrix=key)
        return x2key

    def _key2x(self, S, x, x_mask):
        attention = f.masked_softmax(S, x_mask)
        key2x = f.weighted_sum(attention=attention, matrix=x)
        return key2x.unsqueeze(1).expand(x.size())


class MultiHeadAttention(nn.Module):
    """
    Transformer's Multi-Head Attention
        in "Attention is All You Need" (https://arxiv.org/abs/1706.03762)

    * Kwargs:
        num_head: the number of Head
        model_dim: the number of model dimension
        linear_key_dim: the number of linear key dimemsion
        linear_value_dim: the number of linear value dimension
    """

    def __init__(self, num_head=8, model_dim=100, dropout=0.1,
        linear_key_dim=None, linear_value_dim=None):
        super(MultiHeadAttention, self).__init__()
        if linear_key_dim is None:
            linear_key_dim = model_dim
        if linear_value_dim is None:
            linear_value_dim = model_dim
        assert linear_key_dim % num_head == 0
        assert linear_value_dim % num_head == 0
        self.model_dim = model_dim
        self.num_head = num_head
        self.projection = nn.ModuleList([nn.Linear(model_dim,
            linear_key_dim, bias=False), nn.Linear(model_dim,
            linear_key_dim, bias=False), nn.Linear(model_dim,
            linear_value_dim, bias=False)])
        self.out_linear = nn.Linear(linear_value_dim, model_dim)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x

    def forward(self, q, k, v, mask=None):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs, mask=mask)
        output = self._concat_heads(outputs)
        return self.out_linear(output)

    def _linear_projection(self, query, key, value):
        q = self.projection[0](query)
        k = self.projection[1](key)
        v = self.projection[2](value)
        return q, k, v

    def _split_heads(self, query, key, value):
        B = query.size(0)
        qs, ks, vs = [x.view(B, -1, self.num_head, x.size(-1) // self.
            num_head).transpose(1, 2) for x in [query, key, value]]
        return qs, ks, vs

    def _scaled_dot_product(self, query, key, value, mask=None):
        K_D = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(K_D)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = f.add_masked_value(scores, mask, value=-10000000.0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, value)

    def _concat_heads(self, outputs):
        B = outputs.size(0)
        num_head, dim = outputs.size()[-2:]
        return outputs.transpose(1, 2).contiguous().view(B, -1, self.
            num_head * dim)


class SeqAttnMatch(nn.Module):
    """
    Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, embed_dim, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(embed_dim, embed_dim)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores = scores.masked_fill(y_mask == 0, -1e+30)
        alpha_flat = F.softmax(scores.view(-1, y.size(1)), -1)
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))
        matched_seq = alpha.bmm(y)
        return matched_seq


class LinearSeqAttn(nn.Module):
    """
    Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask == 0, -1e+30)
        alpha = F.softmax(scores, dim=-1)
        return alpha


class BilinearSeqAttn(nn.Module):
    """
    A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask == 0, -1e+30)
        if self.normalize:
            if self.training:
                alpha = F.log_softmax(xWy, dim=-1)
            else:
                alpha = F.softmax(xWy, dim=-1)
        else:
            alpha = xWy.exp()
        return alpha


class DepSepConv(nn.Module):
    """
    Depthwise Separable Convolutions
        in Xception: Deep Learning with Depthwise Separable Convolutions (https://arxiv.org/abs/1610.02357)

    depthwise -> pointwise (1x1 conv)

    * Args:
        input_size: the number of input tensor's dimension
        num_filters: the number of convolution filter
        kernel_size: the number of convolution kernel size
    """

    def __init__(self, input_size=None, num_filters=None, kernel_size=None):
        super(DepSepConv, self).__init__()
        self.depthwise = nn.Conv1d(in_channels=input_size, out_channels=
            input_size, kernel_size=kernel_size, groups=input_size, padding
            =kernel_size // 2)
        nn.init.kaiming_normal_(self.depthwise.weight)
        self.pointwise = PointwiseConv(input_size=input_size, num_filters=
            num_filters)
        self.activation_fn = F.relu

    def forward(self, x):
        x = self.depthwise(x.transpose(1, 2))
        x = self.pointwise(x.transpose(1, 2))
        x = self.activation_fn(x)
        return x


class PointwiseConv(nn.Module):
    """
    Pointwise Convolution (1x1 Conv)

    Convolution 1 Dimension (Faster version)
    (cf. https://github.com/huggingface/pytorch-openai-transformer-lm/blob/        eafc28abdfadfa0732f03a0fc65805c5bfb2ffe7/model_pytorch.py#L45)

    * Args:
        input_size: the number of input tensor's dimension
        num_filters: the number of convolution filter
    """

    def __init__(self, input_size, num_filters):
        super(PointwiseConv, self).__init__()
        self.kernel_size = 1
        self.num_filters = num_filters
        weight = torch.empty(input_size, num_filters)
        nn.init.normal_(weight, std=0.02)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.num_filters,)
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)),
            self.weight)
        x = x.view(*size_out)
        return x


def block_orthogonal(tensor: torch.Tensor, split_sizes: List[int], gain:
    float=1.0) ->None:
    """
    An initializer which allows initializing model parameters in "blocks". This is helpful
    in the case of recurrent models which use multiple gates applied to linear projections,
    which can be computed efficiently if they are concatenated together. However, they are
    separate parameters which should be initialized independently.
    Parameters
    ----------
    tensor : ``torch.Tensor``, required.
        A tensor to initialize.
    split_sizes : List[int], required.
        A list of length ``tensor.ndim()`` specifying the size of the
        blocks along that particular dimension. E.g. ``[10, 20]`` would
        result in the tensor being split into chunks of size 10 along the
        first dimension and 20 along the second.
    gain : float, optional (default = 1.0)
        The gain (scaling) applied to the orthogonal initialization.
    """
    data = tensor.data
    sizes = list(tensor.size())
    if any([(a % b != 0) for a, b in zip(sizes, split_sizes)]):
        raise ValueError(
            'tensor dimensions must be divisible by their respective split_sizes. Found size: {} and split_sizes: {}'
            .format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split)) for max_size, split in zip(
        sizes, split_sizes)]
    for block_start_indices in itertools.product(*indexes):
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        block_slice = tuple([slice(start_index, start_index + step) for 
            start_index, step in index_and_step_tuples])
        data[block_slice] = torch.nn.init.orthogonal_(tensor[block_slice].
            contiguous(), gain=gain)


def get_dropout_mask(dropout_probability: float, tensor_for_masking: torch.
    Tensor):
    """
    Computes and returns an element-wise dropout mask for a given tensor, where
    each element in the mask is dropped out with probability dropout_probability.
    Note that the mask is NOT applied to the tensor - the tensor is passed to retain
    the correct CUDA tensor type for the mask.
    Parameters
    ----------
    dropout_probability : float, required.
        Probability of dropping a dimension of the input.
    tensor_for_masking : torch.Tensor, required.
    Returns
    -------
    A torch.FloatTensor consisting of the binary mask scaled by 1/ (1 - dropout_probability).
    This scaling ensures expected values and variances of the output of applying this mask
     and the original tensor are the same.
    """
    binary_mask = tensor_for_masking.new_tensor(torch.rand(
        tensor_for_masking.size()) > dropout_probability)
    dropout_mask = binary_mask.float().div(1.0 - dropout_probability)
    return dropout_mask


class LstmCellWithProjection(torch.nn.Module):
    """
    An LSTM with Recurrent Dropout and a projected and clipped hidden state and
    memory. Note: this implementation is slower than the native Pytorch LSTM because
    it cannot make use of CUDNN optimizations for stacked RNNs due to and
    variational dropout and the custom nature of the cell state.
    Parameters
    ----------
    input_size : ``int``, required.
        The dimension of the inputs to the LSTM.
    hidden_size : ``int``, required.
        The dimension of the outputs of the LSTM.
    cell_size : ``int``, required.
        The dimension of the memory cell used for the LSTM.
    go_forward: ``bool``, optional (default = True)
        The direction in which the LSTM is applied to the sequence.
        Forwards by default, or backwards if False.
    recurrent_dropout_probability: ``float``, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ . Implementation wise, this simply
        applies a fixed dropout mask per sequence to the recurrent connection of the
        LSTM.
    state_projection_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the memory cell.
    Returns
    -------
    output_accumulator : ``torch.FloatTensor``
        The outputs of the LSTM for each timestep. A tensor of shape
        (batch_size, max_timesteps, hidden_size) where for a given batch
        element, all outputs past the sequence length for that batch are
        zero tensors.
    final_state: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
        The final (state, memory) states of the LSTM, with shape
        (1, batch_size, hidden_size) and  (1, batch_size, cell_size)
        respectively. The first dimension is 1 in order to match the Pytorch
        API for returning stacked LSTM states.
    """

    def __init__(self, input_size: int, hidden_size: int, cell_size: int,
        go_forward: bool=True, recurrent_dropout_probability: float=0.0,
        memory_cell_clip_value: Optional[float]=None,
        state_projection_clip_value: Optional[float]=None) ->None:
        super(LstmCellWithProjection, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_size = cell_size
        self.go_forward = go_forward
        self.state_projection_clip_value = state_projection_clip_value
        self.memory_cell_clip_value = memory_cell_clip_value
        self.recurrent_dropout_probability = recurrent_dropout_probability
        self.input_linearity = torch.nn.Linear(input_size, 4 * cell_size,
            bias=False)
        self.state_linearity = torch.nn.Linear(hidden_size, 4 * cell_size,
            bias=True)
        self.state_projection = torch.nn.Linear(cell_size, hidden_size,
            bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        block_orthogonal(self.input_linearity.weight.data, [self.cell_size,
            self.input_size])
        block_orthogonal(self.state_linearity.weight.data, [self.cell_size,
            self.hidden_size])
        self.state_linearity.bias.data.fill_(0.0)
        self.state_linearity.bias.data[self.cell_size:2 * self.cell_size
            ].fill_(1.0)

    def forward(self, inputs: torch.FloatTensor, batch_lengths: List[int],
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]]=None):
        """
        Parameters
        ----------
        inputs : ``torch.FloatTensor``, required.
            A tensor of shape (batch_size, num_timesteps, input_size)
            to apply the LSTM over.
        batch_lengths : ``List[int]``, required.
            A list of length batch_size containing the lengths of the sequences in batch.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        Returns
        -------
        output_accumulator : ``torch.FloatTensor``
            The outputs of the LSTM for each timestep. A tensor of shape
            (batch_size, max_timesteps, hidden_size) where for a given batch
            element, all outputs past the sequence length for that batch are
            zero tensors.
        final_state : ``Tuple[``torch.FloatTensor, torch.FloatTensor]``
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM. The ``state`` has shape (1, batch_size, hidden_size) and the
            ``memory`` has shape (1, batch_size, cell_size).
        """
        batch_size = inputs.size()[0]
        total_timesteps = inputs.size()[1]
        output_accumulator = inputs.new_zeros(batch_size, total_timesteps,
            self.hidden_size)
        if initial_state is None:
            full_batch_previous_memory = inputs.new_zeros(batch_size, self.
                cell_size)
            full_batch_previous_state = inputs.new_zeros(batch_size, self.
                hidden_size)
        else:
            full_batch_previous_state = initial_state[0].squeeze(0)
            full_batch_previous_memory = initial_state[1].squeeze(0)
        current_length_index = batch_size - 1 if self.go_forward else 0
        if self.recurrent_dropout_probability > 0.0 and self.training:
            dropout_mask = get_dropout_mask(self.
                recurrent_dropout_probability, full_batch_previous_state)
        else:
            dropout_mask = None
        for timestep in range(total_timesteps):
            index = (timestep if self.go_forward else total_timesteps -
                timestep - 1)
            if self.go_forward:
                while batch_lengths[current_length_index] <= index:
                    current_length_index -= 1
            else:
                while current_length_index < len(batch_lengths
                    ) - 1 and batch_lengths[current_length_index + 1] > index:
                    current_length_index += 1
            previous_memory = full_batch_previous_memory[0:
                current_length_index + 1].clone()
            previous_state = full_batch_previous_state[0:
                current_length_index + 1].clone()
            timestep_input = inputs[0:current_length_index + 1, (index)]
            projected_input = self.input_linearity(timestep_input)
            projected_state = self.state_linearity(previous_state)
            input_gate = torch.sigmoid(projected_input[:, 0 * self.
                cell_size:1 * self.cell_size] + projected_state[:, 0 * self
                .cell_size:1 * self.cell_size])
            forget_gate = torch.sigmoid(projected_input[:, 1 * self.
                cell_size:2 * self.cell_size] + projected_state[:, 1 * self
                .cell_size:2 * self.cell_size])
            memory_init = torch.tanh(projected_input[:, 2 * self.cell_size:
                3 * self.cell_size] + projected_state[:, 2 * self.cell_size
                :3 * self.cell_size])
            output_gate = torch.sigmoid(projected_input[:, 3 * self.
                cell_size:4 * self.cell_size] + projected_state[:, 3 * self
                .cell_size:4 * self.cell_size])
            memory = input_gate * memory_init + forget_gate * previous_memory
            if self.memory_cell_clip_value:
                memory = torch.clamp(memory, -self.memory_cell_clip_value,
                    self.memory_cell_clip_value)
            pre_projection_timestep_output = output_gate * torch.tanh(memory)
            timestep_output = self.state_projection(
                pre_projection_timestep_output)
            if self.state_projection_clip_value:
                timestep_output = torch.clamp(timestep_output, -self.
                    state_projection_clip_value, self.
                    state_projection_clip_value)
            if dropout_mask is not None:
                timestep_output = timestep_output * dropout_mask[0:
                    current_length_index + 1]
            full_batch_previous_memory = full_batch_previous_memory.clone()
            full_batch_previous_state = full_batch_previous_state.clone()
            full_batch_previous_memory[0:current_length_index + 1] = memory
            full_batch_previous_state[0:current_length_index + 1
                ] = timestep_output
            output_accumulator[0:current_length_index + 1, (index)
                ] = timestep_output
        final_state = full_batch_previous_state.unsqueeze(0
            ), full_batch_previous_memory.unsqueeze(0)
        return output_accumulator, final_state


RnnStateStorage = Tuple[torch.Tensor, ...]


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Sort a batch first tensor by some specified lengths.
    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.
    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """
    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths,
        torch.Tensor):
        raise ValueError(
            'Both the tensor and sequence lengths must be torch.Tensors.')
    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0,
        descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    index_range = sequence_lengths.new_tensor(torch.arange(0, len(
        sequence_lengths)))
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return (sorted_tensor, sorted_sequence_lengths, restoration_indices,
        permutation_index)


RnnState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class _EncoderBase(torch.nn.Module):
    """
    This abstract class serves as a base for the 3 ``Encoder`` abstractions in AllenNLP.
    - :class:`~allennlp.modules.seq2seq_encoders.Seq2SeqEncoders`
    - :class:`~allennlp.modules.seq2vec_encoders.Seq2VecEncoders`
    Additionally, this class provides functionality for sorting sequences by length
    so they can be consumed by Pytorch RNN classes, which require their inputs to be
    sorted by length. Finally, it also provides optional statefulness to all of it's
    subclasses by allowing the caching and retrieving of the hidden states of RNNs.
    """

    def __init__(self, stateful: bool=False) ->None:
        super(_EncoderBase, self).__init__()
        self.stateful = stateful
        self._states: Optional[RnnStateStorage] = None

    def sort_and_run_forward(self, module: Callable[[PackedSequence,
        Optional[RnnState]], Tuple[Union[PackedSequence, torch.Tensor],
        RnnState]], inputs: torch.Tensor, mask: torch.Tensor, hidden_state:
        Optional[RnnState]=None):
        """
        This function exists because Pytorch RNNs require that their inputs be sorted
        before being passed as input. As all of our Seq2xxxEncoders use this functionality,
        it is provided in a base class. This method can be called on any module which
        takes as input a ``PackedSequence`` and some ``hidden_state``, which can either be a
        tuple of tensors or a tensor.
        As all of our Seq2xxxEncoders have different return types, we return `sorted`
        outputs from the module, which is called directly. Additionally, we return the
        indices into the batch dimension required to restore the tensor to it's correct,
        unsorted order and the number of valid batch elements (i.e the number of elements
        in the batch which are not completely masked). This un-sorting and re-padding
        of the module outputs is left to the subclasses because their outputs have different
        types and handling them smoothly here is difficult.
        Parameters
        ----------
        module : ``Callable[[PackedSequence, Optional[RnnState]],
                            Tuple[Union[PackedSequence, torch.Tensor], RnnState]]``, required.
            A function to run on the inputs. In most cases, this is a ``torch.nn.Module``.
        inputs : ``torch.Tensor``, required.
            A tensor of shape ``(batch_size, sequence_length, embedding_size)`` representing
            the inputs to the Encoder.
        mask : ``torch.Tensor``, required.
            A tensor of shape ``(batch_size, sequence_length)``, representing masked and
            non-masked elements of the sequence for each element in the batch.
        hidden_state : ``Optional[RnnState]``, (default = None).
            A single tensor of shape (num_layers, batch_size, hidden_size) representing the
            state of an RNN with or a tuple of
            tensors of shapes (num_layers, batch_size, hidden_size) and
            (num_layers, batch_size, memory_size), representing the hidden state and memory
            state of an LSTM-like RNN.
        Returns
        -------
        module_output : ``Union[torch.Tensor, PackedSequence]``.
            A Tensor or PackedSequence representing the output of the Pytorch Module.
            The batch size dimension will be equal to ``num_valid``, as sequences of zero
            length are clipped off before the module is called, as Pytorch cannot handle
            zero length sequences.
        final_states : ``Optional[RnnState]``
            A Tensor representing the hidden state of the Pytorch Module. This can either
            be a single tensor of shape (num_layers, num_valid, hidden_size), for instance in
            the case of a GRU, or a tuple of tensors, such as those required for an LSTM.
        restoration_indices : ``torch.LongTensor``
            A tensor of shape ``(batch_size,)``, describing the re-indexing required to transform
            the outputs back to their original batch order.
        """
        batch_size = mask.size(0)
        num_valid = torch.sum(mask[:, (0)]).int().item()
        sequence_lengths = mask.long().sum(-1)
        (sorted_inputs, sorted_sequence_lengths, restoration_indices,
            sorting_indices) = sort_batch_by_length(inputs, sequence_lengths)
        packed_sequence_input = pack_padded_sequence(sorted_inputs[:
            num_valid, :, :], sorted_sequence_lengths[:num_valid].data.
            tolist(), batch_first=True)
        if not self.stateful:
            if hidden_state is None:
                initial_states = hidden_state
            elif isinstance(hidden_state, tuple):
                initial_states = [state.index_select(1, sorting_indices)[:,
                    :num_valid, :].contiguous() for state in hidden_state]
            else:
                initial_states = hidden_state.index_select(1, sorting_indices)[
                    :, :num_valid, :].contiguous()
        else:
            initial_states = self._get_initial_states(batch_size, num_valid,
                sorting_indices)
        module_output, final_states = module(packed_sequence_input,
            initial_states)
        return module_output, final_states, restoration_indices

    def _get_initial_states(self, batch_size: int, num_valid: int,
        sorting_indices: torch.LongTensor) ->Optional[RnnState]:
        """
        Returns an initial state for use in an RNN. Additionally, this method handles
        the batch size changing across calls by mutating the state to append initial states
        for new elements in the batch. Finally, it also handles sorting the states
        with respect to the sequence lengths of elements in the batch and removing rows
        which are completely padded. Importantly, this `mutates` the state if the
        current batch size is larger than when it was previously called.
        Parameters
        ----------
        batch_size : ``int``, required.
            The batch size can change size across calls to stateful RNNs, so we need
            to know if we need to expand or shrink the states before returning them.
            Expanded states will be set to zero.
        num_valid : ``int``, required.
            The batch may contain completely padded sequences which get removed before
            the sequence is passed through the encoder. We also need to clip these off
            of the state too.
        sorting_indices ``torch.LongTensor``, required.
            Pytorch RNNs take sequences sorted by length. When we return the states to be
            used for a given call to ``module.forward``, we need the states to match up to
            the sorted sequences, so before returning them, we sort the states using the
            same indices used to sort the sequences.
        Returns
        -------
        This method has a complex return type because it has to deal with the first time it
        is called, when it has no state, and the fact that types of RNN have heterogeneous
        states.
        If it is the first time the module has been called, it returns ``None``, regardless
        of the type of the ``Module``.
        Otherwise, for LSTMs, it returns a tuple of ``torch.Tensors`` with shape
        ``(num_layers, num_valid, state_size)`` and ``(num_layers, num_valid, memory_size)``
        respectively, or for GRUs, it returns a single ``torch.Tensor`` of shape
        ``(num_layers, num_valid, state_size)``.
        """
        if self._states is None:
            return None
        if batch_size > self._states[0].size(1):
            num_states_to_concat = batch_size - self._states[0].size(1)
            resized_states = []
            for state in self._states:
                zeros = state.new_zeros(state.size(0), num_states_to_concat,
                    state.size(2))
                resized_states.append(torch.cat([state, zeros], 1))
            self._states = tuple(resized_states)
            correctly_shaped_states = self._states
        elif batch_size < self._states[0].size(1):
            correctly_shaped_states = tuple(state[:, :batch_size, :] for
                state in self._states)
        else:
            correctly_shaped_states = self._states
        if len(self._states) == 1:
            correctly_shaped_state = correctly_shaped_states[0]
            sorted_state = correctly_shaped_state.index_select(1,
                sorting_indices)
            return sorted_state[:, :num_valid, :]
        else:
            sorted_states = [state.index_select(1, sorting_indices) for
                state in correctly_shaped_states]
            return tuple(state[:, :num_valid, :] for state in sorted_states)

    def _update_states(self, final_states: RnnStateStorage,
        restoration_indices: torch.LongTensor) ->None:
        """
        After the RNN has run forward, the states need to be updated.
        This method just sets the state to the updated new state, performing
        several pieces of book-keeping along the way - namely, unsorting the
        states and ensuring that the states of completely padded sequences are
        not updated. Finally, it also detaches the state variable from the
        computational graph, such that the graph can be garbage collected after
        each batch iteration.
        Parameters
        ----------
        final_states : ``RnnStateStorage``, required.
            The hidden states returned as output from the RNN.
        restoration_indices : ``torch.LongTensor``, required.
            The indices that invert the sorting used in ``sort_and_run_forward``
            to order the states with respect to the lengths of the sequences in
            the batch.
        """
        new_unsorted_states = [state.index_select(1, restoration_indices) for
            state in final_states]
        if self._states is None:
            self._states = tuple(state.data for state in new_unsorted_states)
        else:
            current_state_batch_size = self._states[0].size(1)
            new_state_batch_size = final_states[0].size(1)
            used_new_rows_mask = [(state[(0), :, :].sum(-1) != 0.0).float()
                .view(1, new_state_batch_size, 1) for state in
                new_unsorted_states]
            new_states = []
            if current_state_batch_size > new_state_batch_size:
                for old_state, new_state, used_mask in zip(self._states,
                    new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state[:, :new_state_batch_size, :
                        ] * (1 - used_mask)
                    old_state[:, :new_state_batch_size, :
                        ] = new_state + masked_old_state
                    new_states.append(old_state.detach())
            else:
                new_states = []
                for old_state, new_state, used_mask in zip(self._states,
                    new_unsorted_states, used_new_rows_mask):
                    masked_old_state = old_state * (1 - used_mask)
                    new_state += masked_old_state
                    new_states.append(new_state.detach())
            self._states = tuple(new_states)

    def reset_states(self):
        self._states = None


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
        in "Attention is All You Need" (https://arxiv.org/abs/1706.03762)

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).

    (cf. https://github.com/tensorflow/tensor2tensor/blob/42c3f377f441e5a0f431127d63e71414ead291c4/        tensor2tensor/layers/common_attention.py#L388)

    * Args:
        embed_dim: the number of embedding dimension

    * Kwargs:
        max_len: the number of maximum sequence length
    """

    def __init__(self, embed_dim, max_length=2000):
        super(PositionalEncoding, self).__init__()
        signal_sinusoid = self._get_timing_signal(max_length, embed_dim)
        self.register_buffer('position_encoding', signal_sinusoid)

    def _get_timing_signal(self, length, channels, min_timescale=1.0,
        max_timescale=10000.0):
        position = np.arange(length)
        num_timescales = channels // 2
        log_timescale_increment = math.log(float(max_timescale) / float(
            min_timescale) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * np.exp(np.arange(num_timescales).
            astype(np.float) * -log_timescale_increment)
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(
            inv_timescales, 0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)],
            axis=1)
        signal = np.pad(signal, [[0, 0], [0, channels % 2]], 'constant',
            constant_values=[0.0, 0.0])
        signal = signal.reshape([1, length, channels])
        return torch.from_numpy(signal).type(torch.FloatTensor)

    def forward(self, x):
        x = x + self.position_encoding[:, :x.size(1)]
        return x


def get_activation_fn(name):
    """ PyTorch built-in activation functions """
    activation_functions = {'linear': lambda : lambda x: x, 'relu': nn.ReLU,
        'relu6': nn.ReLU6, 'elu': nn.ELU, 'prelu': nn.PReLU, 'leaky_relu':
        nn.LeakyReLU, 'threshold': nn.Threshold, 'hardtanh': nn.Hardtanh,
        'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'log_sigmoid': nn.
        LogSigmoid, 'softplus': nn.Softplus, 'softshrink': nn.Softshrink,
        'softsign': nn.Softsign, 'tanhshrink': nn.Tanhshrink}
    if name not in activation_functions:
        raise ValueError(
            f"""'{name}' is not included in activation_functions. use below one. 
 {activation_functions.keys()}"""
            )
    return activation_functions[name]


class Highway(nn.Module):
    """
    Highway Networks (https://arxiv.org/abs/1505.00387)
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/highway.py

    * Args:
        input_size: The number of expected features in the input `x`
        num_layers: The number of Highway layers.
        activation: Activation Function (ReLU is default)
    """

    def __init__(self, input_size, num_layers=2, activation='relu'):
        super(Highway, self).__init__()
        self.activation_fn = activation
        if type(activation) == str:
            self.activation_fn = get_activation_fn(activation)()
        self._layers = torch.nn.ModuleList([nn.Linear(input_size, 
            input_size * 2) for _ in range(num_layers)])
        for layer in self._layers:
            layer.bias[input_size:].data.fill_(1)

    def forward(self, x):
        current_input = x
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation_fn(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class LayerNorm(nn.Module):
    """
    Layer Normalization
    (https://arxiv.org/abs/1607.06450)
    """

    def __init__(self, normalized_shape, eps=1e-05):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):
    """
    Pointwise Feed-Forward Layer

    * Args:
        input_size: the number of input size
        hidden_size: the number of hidden size

    * Kwargs:
        dropout: the probability of dropout
    """

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.pointwise_conv1 = PointwiseConv(input_size=input_size,
            num_filters=hidden_size)
        self.pointwise_conv2 = PointwiseConv(input_size=hidden_size,
            num_filters=input_size)
        self.activation_fn = F.relu
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.activation_fn(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x


class ResidualConnection(nn.Module):
    """
    ResidualConnection
        in Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)

    => f(x) + x

    * Args:
        dim: the number of dimension

    * Kwargs:
        layer_dropout: layer dropout probability (stochastic depth)
        dropout: dropout probability
    """

    def __init__(self, dim, layer_dropout=None, layernorm=False):
        super(ResidualConnection, self).__init__()
        self.survival = None
        if layer_dropout < 1:
            self.survival = torch.FloatTensor([layer_dropout])
        if layernorm:
            self.norm = LayerNorm(dim)
        else:
            self.norm = lambda x: x

    def forward(self, x, sub_layer_fn):
        if self.training and self.survival is not None:
            survival_prob = torch.bernoulli(self.survival).item()
            if survival_prob == 1:
                return x + sub_layer_fn(self.norm(x))
            else:
                return x
        else:
            return x + sub_layer_fn(self.norm(x))


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.
    In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
    before weighting.
    """

    def __init__(self, mixture_size: int, do_layer_norm: bool=False,
        initial_scalar_parameters: List[float]=None, trainable: bool=True
        ) ->None:
        super(ScalarMix, self).__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm
        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ValueError(
                'Length of initial_scalar_parameters {} differs from mixture_size {}'
                .format(initial_scalar_parameters, mixture_size))
        self.scalar_parameters = ParameterList([Parameter(torch.FloatTensor
            ([initial_scalar_parameters[i]]), requires_grad=trainable) for
            i in range(mixture_size)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=
            trainable)

    def forward(self, tensors: List[torch.Tensor], mask: torch.Tensor=None
        ) ->torch.Tensor:
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.
        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.
        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise ValueError(
                '{} tensors were passed, but the module was initialized to mix {} tensors.'
                .format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2
                ) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + 1e-12)
        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for
            parameter in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)
        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)
        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                    broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)


logger = logging.getLogger(__name__)


def remove_sentence_boundaries(tensor: torch.Tensor, mask: torch.Tensor
    ) ->Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps, dim)``
    this returns a tensor of shape ``(batch_size, timesteps - 2, dim)`` after removing
    the beginning and end sentence markers.  The sentences are assumed to be padded on the right,
    with the beginning of each sentence assumed to occur at index 0 (i.e., ``mask[:, 0]`` is assumed
    to be 1).
    Returns both the new tensor and updated mask.
    This function is the inverse of ``add_sentence_boundary_token_ids``.
    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)``
    Returns
    -------
    tensor_without_boundary_tokens : ``torch.Tensor``
        The tensor after removing the boundary tokens of shape ``(batch_size, timesteps - 2, dim)``
    new_mask : ``torch.Tensor``
        The new mask for the tensor of shape ``(batch_size, timesteps - 2)``.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.long)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[(i), :j - 2, :] = tensor[(i), 1:
                j - 1, :]
            new_mask[(i), :j - 2] = 1
    return tensor_without_boundary_tokens, new_mask


class Elmo(torch.nn.Module):
    """
    Compute ELMo representations using a pre-trained bidirectional language model.
    See "Deep contextualized word representations", Peters et al. for details.
    This module takes character id input and computes ``num_output_representations`` different layers
    of ELMo representations.  Typically ``num_output_representations`` is 1 or 2.  For example, in
    the case of the SRL model in the above paper, ``num_output_representations=1`` where ELMo was included at
    the input token representation layer.  In the case of the SQuAD model, ``num_output_representations=2``
    as ELMo was also included at the GRU output layer.
    In the implementation below, we learn separate scalar weights for each output layer,
    but only run the biLM once on each input sequence for efficiency.
    Parameters
    ----------
    options_file : ``str``, required.
        ELMo JSON options file
    weight_file : ``str``, required.
        ELMo hdf5 weight file
    num_output_representations: ``int``, required.
        The number of ELMo representation layers to output.
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    do_layer_norm : ``bool``, optional, (default=False).
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional, (default = 0.5).
        The dropout to be applied to the ELMo representations.
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, Elmo expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    module : ``torch.nn.Module``, optional, (default = None).
        If provided, then use this module instead of the pre-trained ELMo biLM.
        If using this option, then pass ``None`` for both ``options_file``
        and ``weight_file``.  The module must provide a public attribute
        ``num_layers`` with the number of internal layers and its ``forward``
        method must return a ``dict`` with ``activations`` and ``mask`` keys
        (see `_ElmoBilm`` for an example).  Note that ``requires_grad`` is also
        ignored with this option.
    """

    def __init__(self, options_file: str, weight_file: str,
        num_output_representations: int, requires_grad: bool=False,
        do_layer_norm: bool=False, dropout: float=0.5, vocab_to_cache: List
        [str]=None, module: torch.nn.Module=None) ->None:
        super(Elmo, self).__init__()
        logging.info('Initializing ELMo')
        if module is not None:
            if options_file is not None or weight_file is not None:
                raise ValueError(
                    "Don't provide options_file or weight_file with module")
            self._elmo_lstm = module
        else:
            self._elmo_lstm = _ElmoBiLm(options_file, weight_file,
                requires_grad=requires_grad, vocab_to_cache=vocab_to_cache)
        self._has_cached_vocab = vocab_to_cache is not None
        self._dropout = Dropout(p=dropout)
        self._scalar_mixes: Any = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(self._elmo_lstm.num_layers,
                do_layer_norm=do_layer_norm)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def get_output_dim(self):
        return self._elmo_lstm.get_output_dim()

    def forward(self, inputs: torch.Tensor, word_inputs: torch.Tensor=None
        ) ->Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
        Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.
        Returns
        -------
        Dict with keys:
        ``'elmo_representations'``: ``List[torch.Tensor]``
            A ``num_output_representations`` list of ELMo representations for the input sequence.
            Each representation is shape ``(batch_size, timesteps, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps)`` long tensor with sequence mask.
        """
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs
        if word_inputs is not None:
            original_word_size = word_inputs.size()
            if self._has_cached_vocab and len(original_word_size) > 2:
                reshaped_word_inputs = word_inputs.view(-1,
                    original_word_size[-1])
                logger.warning(
                    'Word inputs were passed to ELMo but it does not have a cached vocab.'
                    )
                reshaped_word_inputs = None
            else:
                reshaped_word_inputs = word_inputs
        else:
            reshaped_word_inputs = word_inputs
        bilm_output = self._elmo_lstm(reshaped_inputs, reshaped_word_inputs)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation_with_bos_eos = scalar_mix(layer_activations,
                mask_with_bos_eos)
            representation_without_bos_eos, mask_without_bos_eos = (
                remove_sentence_boundaries(representation_with_bos_eos,
                mask_with_bos_eos))
            representations.append(self._dropout(
                representation_without_bos_eos))
        if word_inputs is not None and len(original_word_size) > 2:
            mask = mask_without_bos_eos.view(original_word_size)
            elmo_representations = [representation.view(original_word_size +
                (-1,)) for representation in representations]
        elif len(original_shape) > 3:
            mask = mask_without_bos_eos.view(original_shape[:-1])
            elmo_representations = [representation.view(original_shape[:-1] +
                (-1,)) for representation in representations]
        else:
            mask = mask_without_bos_eos
            elmo_representations = representations
        return {'elmo_representations': elmo_representations, 'mask': mask}

    @classmethod
    def from_params(cls, params) ->'Elmo':
        params.add_file_to_archive('options_file')
        params.add_file_to_archive('weight_file')
        options_file = params.pop('options_file')
        weight_file = params.pop('weight_file')
        requires_grad = params.pop('requires_grad', False)
        num_output_representations = params.pop('num_output_representations')
        do_layer_norm = params.pop_bool('do_layer_norm', False)
        dropout = params.pop_float('dropout', 0.5)
        params.assert_empty(cls.__name__)
        return cls(options_file=options_file, weight_file=weight_file,
            num_output_representations=num_output_representations,
            requires_grad=requires_grad, do_layer_norm=do_layer_norm,
            dropout=dropout)


class ElmoLstm(_EncoderBase):
    """
    A stacked, bidirectional LSTM which uses
    :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`'s
    with highway layers between the inputs to layers.
    The inputs to the forward and backward directions are independent - forward and backward
    states are not concatenated between layers.
    Additionally, this LSTM maintains its `own` state, which is updated every time
    ``forward`` is called. It is dynamically resized for different batch sizes and is
    designed for use with non-continuous inputs (i.e inputs which aren't formatted as a stream,
    such as text used for a language modelling task, which is how stateful RNNs are typically used).
    This is non-standard, but can be thought of as having an "end of sentence" state, which is
    carried across different sentences.
    Parameters
    ----------
    input_size : ``int``, required
        The dimension of the inputs to the LSTM.
    hidden_size : ``int``, required
        The dimension of the outputs of the LSTM.
    cell_size : ``int``, required.
        The dimension of the memory cell of the
        :class:`~allennlp.modules.lstm_cell_with_projection.LstmCellWithProjection`.
    num_layers : ``int``, required
        The number of bidirectional LSTMs to use.
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    recurrent_dropout_probability: ``float``, optional (default = 0.0)
        The dropout probability to be used in a dropout scheme as stated in
        `A Theoretically Grounded Application of Dropout in Recurrent Neural Networks
        <https://arxiv.org/abs/1512.05287>`_ .
    state_projection_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the hidden_state after projecting it.
    memory_cell_clip_value: ``float``, optional, (default = None)
        The magnitude with which to clip the memory cell.
    """

    def __init__(self, input_size: int, hidden_size: int, cell_size: int,
        num_layers: int, requires_grad: bool=False,
        recurrent_dropout_probability: float=0.0, memory_cell_clip_value:
        Optional[float]=None, state_projection_clip_value: Optional[float]=None
        ) ->None:
        super(ElmoLstm, self).__init__(stateful=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_size = cell_size
        self.requires_grad = requires_grad
        forward_layers = []
        backward_layers = []
        lstm_input_size = input_size
        go_forward = True
        for layer_index in range(num_layers):
            forward_layer = LstmCellWithProjection(lstm_input_size,
                hidden_size, cell_size, go_forward,
                recurrent_dropout_probability, memory_cell_clip_value,
                state_projection_clip_value)
            backward_layer = LstmCellWithProjection(lstm_input_size,
                hidden_size, cell_size, not go_forward,
                recurrent_dropout_probability, memory_cell_clip_value,
                state_projection_clip_value)
            lstm_input_size = hidden_size
            self.add_module('forward_layer_{}'.format(layer_index),
                forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index),
                backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self, inputs: torch.Tensor, mask: torch.LongTensor
        ) ->torch.Tensor:
        """
        Parameters
        ----------
        inputs : ``torch.Tensor``, required.
            A Tensor of shape ``(batch_size, sequence_length, hidden_size)``.
        mask : ``torch.LongTensor``, required.
            A binary mask of shape ``(batch_size, sequence_length)`` representing the
            non-padded elements in each sequence in the batch.
        Returns
        -------
        A ``torch.Tensor`` of shape (num_layers, batch_size, sequence_length, hidden_size),
        where the num_layers dimension represents the LSTM output from that layer.
        """
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = (self.
            sort_and_run_forward(self._lstm_forward, inputs, mask))
        num_layers, num_valid, returned_timesteps, encoder_dim = (
            stacked_sequence_output.size())
        if num_valid < batch_size:
            zeros = stacked_sequence_output.new_zeros(num_layers, 
                batch_size - num_valid, returned_timesteps, encoder_dim)
            stacked_sequence_output = torch.cat([stacked_sequence_output,
                zeros], 1)
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.new_zeros(num_layers, batch_size - num_valid,
                    state_dim)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.new_zeros(num_layers,
                batch_size, sequence_length_difference,
                stacked_sequence_output[0].size(-1))
            stacked_sequence_output = torch.cat([stacked_sequence_output,
                zeros], 2)
        self._update_states(final_states, restoration_indices)
        return stacked_sequence_output.index_select(1, restoration_indices)

    def _lstm_forward(self, inputs: PackedSequence, initial_state: Optional
        [Tuple[torch.Tensor, torch.Tensor]]=None) ->Tuple[torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        inputs : ``PackedSequence``, required.
            A batch first ``PackedSequence`` to run the stacked LSTM over.
        initial_state : ``Tuple[torch.Tensor, torch.Tensor]``, optional, (default = None)
            A tuple (state, memory) representing the initial hidden state and memory
            of the LSTM, with shape (num_layers, batch_size, 2 * hidden_size) and
            (num_layers, batch_size, 2 * cell_size) respectively.
        Returns
        -------
        output_sequence : ``torch.FloatTensor``
            The encoded sequence of shape (num_layers, batch_size, sequence_length, hidden_size)
        final_states: ``Tuple[torch.FloatTensor, torch.FloatTensor]``
            The per-layer final (state, memory) states of the LSTM, with shape
            (num_layers, batch_size, 2 * hidden_size) and  (num_layers, batch_size, 2 * cell_size)
            respectively. The last dimension is duplicated because it contains the state/memory
            for both the forward and backward layers.
        """
        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor, torch.Tensor]]
                ] = [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise ValueError(
                'Initial states were passed to forward() but the number of initial states does not match the number of layers.'
                )
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0),
                initial_state[1].split(1, 0)))
        inputs, batch_lengths = pad_packed_sequence(inputs, batch_first=True)
        forward_output_sequence = inputs
        backward_output_sequence = inputs
        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(
                layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(
                layer_index))
            forward_cache = forward_output_sequence
            backward_cache = backward_output_sequence
            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(
                    self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(
                    self.cell_size, 2)
                forward_state = forward_hidden_state, forward_memory_state
                backward_state = backward_hidden_state, backward_memory_state
            else:
                forward_state = None
                backward_state = None
            forward_output_sequence, forward_state = forward_layer(
                forward_output_sequence, batch_lengths, forward_state)
            backward_output_sequence, backward_state = backward_layer(
                backward_output_sequence, batch_lengths, backward_state)
            if layer_index != 0:
                forward_output_sequence += forward_cache
                backward_output_sequence += backward_cache
            sequence_outputs.append(torch.cat([forward_output_sequence,
                backward_output_sequence], -1))
            final_states.append((torch.cat([forward_state[0],
                backward_state[0]], -1), torch.cat([forward_state[1],
                backward_state[1]], -1)))
        stacked_sequence_outputs: torch.FloatTensor = torch.stack(
            sequence_outputs)
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor, torch.FloatTensor] = (torch
            .cat(final_hidden_states, 0), torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple

    def load_weights(self, weight_file: str) ->None:
        """
        Load the pre-trained weights from the file.
        """
        requires_grad = self.requires_grad
        with h5py.File(weight_file, 'r') as fin:
            for i_layer, lstms in enumerate(zip(self.forward_layers, self.
                backward_layers)):
                for j_direction, lstm in enumerate(lstms):
                    cell_size = lstm.cell_size
                    dataset = fin['RNN_%s' % j_direction]['RNN']['MultiRNNCell'
                        ]['Cell%s' % i_layer]['LSTMCell']
                    tf_weights = numpy.transpose(dataset['W_0'][...])
                    torch_weights = tf_weights.copy()
                    input_size = lstm.input_size
                    input_weights = torch_weights[:, :input_size]
                    recurrent_weights = torch_weights[:, input_size:]
                    tf_input_weights = tf_weights[:, :input_size]
                    tf_recurrent_weights = tf_weights[:, input_size:]
                    for torch_w, tf_w in [[input_weights, tf_input_weights],
                        [recurrent_weights, tf_recurrent_weights]]:
                        torch_w[1 * cell_size:2 * cell_size, :] = tf_w[2 *
                            cell_size:3 * cell_size, :]
                        torch_w[2 * cell_size:3 * cell_size, :] = tf_w[1 *
                            cell_size:2 * cell_size, :]
                    lstm.input_linearity.weight.data.copy_(torch.
                        FloatTensor(input_weights))
                    lstm.state_linearity.weight.data.copy_(torch.
                        FloatTensor(recurrent_weights))
                    lstm.input_linearity.weight.requires_grad = requires_grad
                    lstm.state_linearity.weight.requires_grad = requires_grad
                    tf_bias = dataset['B'][...]
                    tf_bias[2 * cell_size:3 * cell_size] += 1
                    torch_bias = tf_bias.copy()
                    torch_bias[1 * cell_size:2 * cell_size] = tf_bias[2 *
                        cell_size:3 * cell_size]
                    torch_bias[2 * cell_size:3 * cell_size] = tf_bias[1 *
                        cell_size:2 * cell_size]
                    lstm.state_linearity.bias.data.copy_(torch.FloatTensor(
                        torch_bias))
                    lstm.state_linearity.bias.requires_grad = requires_grad
                    proj_weights = numpy.transpose(dataset['W_P_0'][...])
                    lstm.state_projection.weight.data.copy_(torch.
                        FloatTensor(proj_weights))
                    lstm.state_projection.weight.requires_grad = requires_grad


def add_sentence_boundary_token_ids(tensor: torch.Tensor, mask: torch.
    Tensor, sentence_begin_token: Any, sentence_end_token: Any) ->Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Add begin/end of sentence tokens to the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps)`` or
    ``(batch_size, timesteps, dim)`` this returns a tensor of shape
    ``(batch_size, timesteps + 2)`` or ``(batch_size, timesteps + 2, dim)`` respectively.
    Returns both the new tensor and updated mask.
    Parameters
    ----------
    tensor : ``torch.Tensor``
        A tensor of shape ``(batch_size, timesteps)`` or ``(batch_size, timesteps, dim)``
    mask : ``torch.Tensor``
         A tensor of shape ``(batch_size, timesteps)``
    sentence_begin_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the <S> id. For 3D input, a tensor with length dim.
    sentence_end_token: Any (anything that can be broadcast in torch for assignment)
        For 2D input, a scalar with the </S> id. For 3D input, a tensor with length dim.
    Returns
    -------
    tensor_with_boundary_tokens : ``torch.Tensor``
        The tensor with the appended and prepended boundary tokens. If the input was 2D,
        it has shape (batch_size, timesteps + 2) and if the input was 3D, it has shape
        (batch_size, timesteps + 2, dim).
    new_mask : ``torch.Tensor``
        The new mask for the tensor, taking into account the appended tokens
        marking the beginning and end of the sentence.
    """
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, (0)] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens != 0).long()
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[(i), (0), :] = sentence_begin_token
            tensor_with_boundary_tokens[(i), (j + 1), :] = sentence_end_token
        new_mask = ((tensor_with_boundary_tokens > 0).long().sum(dim=-1) > 0
            ).long()
    else:
        raise ValueError(
            'add_sentence_boundary_token_ids only accepts 2D and 3D input')
    return tensor_with_boundary_tokens, new_mask


class _ElmoBiLm(torch.nn.Module):
    """
    Run a pre-trained bidirectional language model, outputing the activations at each
    layer for weighting together into an ELMo representation (with
    ``allennlp.modules.seq2seq_encoders.Elmo``).  This is a lower level class, useful
    for advanced uses, but most users should use ``allennlp.modules.seq2seq_encoders.Elmo``
    directly.
    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, _ElmoBiLm expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    """

    def __init__(self, options_file: str, weight_file: str, requires_grad:
        bool=False, vocab_to_cache: List[str]=None) ->None:
        super(_ElmoBiLm, self).__init__()
        self._token_embedder = _ElmoCharacterEncoder(options_file,
            weight_file, requires_grad=requires_grad)
        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning(
                'You are fine tuning ELMo and caching char CNN word vectors. This behaviour is not guaranteed to be well defined, particularly. if not all of your inputs will occur in the vocabulary cache.'
                )
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None
        with open(options_file, 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ValueError(
                'We only support pretrained biLMs with residual connections')
        self._elmo_lstm = ElmoLstm(input_size=options['lstm'][
            'projection_dim'], hidden_size=options['lstm']['projection_dim'
            ], cell_size=options['lstm']['dim'], num_layers=options['lstm']
            ['n_layers'], memory_cell_clip_value=options['lstm'][
            'cell_clip'], state_projection_clip_value=options['lstm'][
            'proj_clip'], requires_grad=requires_grad)
        self._elmo_lstm.load_weights(weight_file)
        self.num_layers = options['lstm']['n_layers'] + 1

    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()

    def forward(self, inputs: torch.Tensor, word_inputs: torch.Tensor=None
        ) ->Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape ``(batch_size, timesteps)``,
            which represent word ids which have been pre-cached.
        Returns
        -------
        Dict with keys:
        ``'activations'``: ``List[torch.Tensor]``
            A list of activations at each layer of the network, each of shape
            ``(batch_size, timesteps + 2, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.
        Note that the output tensors all include additional special begin and end of sequence
        markers.
        """
        if self._word_embedding is not None and word_inputs is not None:
            try:
                mask_without_bos_eos = (word_inputs > 0).long()
                embedded_inputs = self._word_embedding(word_inputs)
                type_representation, mask = add_sentence_boundary_token_ids(
                    embedded_inputs, mask_without_bos_eos, self.
                    _bos_embedding, self._eos_embedding)
            except RuntimeError:
                token_embedding = self._token_embedder(inputs)
                mask = token_embedding['mask']
                type_representation = token_embedding['token_embedding']
        else:
            token_embedding = self._token_embedder(inputs)
            mask = token_embedding['mask']
            type_representation = token_embedding['token_embedding']
        lstm_outputs = self._elmo_lstm(type_representation, mask)
        output_tensors = [torch.cat([type_representation,
            type_representation], dim=-1) * mask.float().unsqueeze(-1)]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.
            size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))
        return {'activations': output_tensors, 'mask': mask}


def _make_bos_eos(character: int, padding_character: int,
    beginning_of_word_character: int, end_of_word_character: int,
    max_word_length: int):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class TokenEmbedding(torch.nn.Module):
    """
    Token Embedding

    It can be embedding matrix, language model (ELMo), neural machine translation model (CoVe) and features.

    * Args:
        vocab: Vocab (rqa.tokens.vocab)
    """

    def __init__(self, vocab):
        super(TokenEmbedding, self).__init__()
        self.vocab = vocab

    def forward(self, tokens):
        """ embedding look-up """
        raise NotImplementedError

    def get_output_dim(self):
        """ get embedding dimension """
        raise NotImplementedError

    def get_vocab_size(self):
        return len(self.vocab)


class TokenEmbedder(torch.nn.Module):
    """
    Token Embedder

    Take a tensor(indexed token) look up Embedding modules.

    * Args:
        token_makers: dictionary of TokenMaker (claf.token_makers.token)
    """

    def __init__(self, token_makers):
        super(TokenEmbedder, self).__init__()
        self.embed_dims = {}
        self.vocabs = {token_name: token_maker.vocab for token_name,
            token_maker in token_makers.items()}
        self.add_embedding_modules(token_makers)

    def add_embedding_modules(self, token_makers):
        """ add embedding module to TokenEmbedder """
        self.token_names = []
        for token_name, token_maker in token_makers.items():
            self.token_names.append(token_name)
            vocab = token_maker.vocab
            embedding = token_maker.embedding_fn(vocab)
            self.add_module(token_name, embedding)
            self.embed_dims[token_name] = embedding.get_output_dim()

    def get_embed_dim(self):
        raise NotImplementedError

    def forward(self, inputs, params={}):
        raise NotImplementedError


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_naver_claf(_paritybench_base):
    pass
    def test_000(self):
        self._check(BilinearSeqAttn(*[], **{'x_size': 4, 'y_size': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(CoAttention(*[], **{'embed_dim': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(EncoderBlock(*[], **{}), [torch.rand([4, 1, 128])], {})

    def test_003(self):
        self._check(Highway(*[], **{'input_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(LayerNorm(*[], **{'normalized_shape': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(NoAnswer(*[], **{'embed_dim': 4, 'bias_hidden_dim': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(PointwiseConv(*[], **{'input_size': 4, 'num_filters': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(PositionalEncoding(*[], **{'embed_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(PositionwiseFeedForward(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(ScalarMix(*[], **{'mixture_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(SeqAttnMatch(*[], **{'embed_dim': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {})

