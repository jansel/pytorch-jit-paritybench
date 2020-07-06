import sys
_module = sys.modules[__name__]
del sys
RUN_mnist = _module
datasets = _module
models = _module
LSTMATTHighway = _module
args = _module
LSTMATT = _module
TextCNN = _module
TextCNNHighway = _module
TextRCNN = _module
TextRCNNHighway = _module
TextRNN = _module
TextRNNHighway = _module
Decoder = _module
DecoderLayer = _module
Embeddings = _module
Encoder = _module
EncoderLayer = _module
LayerNorm = _module
MultiHeadAttention = _module
PositionalEncoding = _module
PositionwiseFeedForward = _module
ScaleDotProductAttention = _module
SublayerConnection = _module
Transformer = _module
utils = _module
TransformerText = _module
SST2_utils = _module
utils = _module
Conv = _module
Embedding = _module
Highway = _module
LSTM = _module
Linear = _module
run_Highway_SST = _module
run_SST = _module
train_eval = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import torch.nn as nn


import torchvision


import torchvision.transforms as transforms


import torch.nn.functional as F


import math


import copy


from copy import deepcopy


from torchtext import data


from torchtext import datasets


from torchtext import vocab


import time


from sklearn import metrics


import random


import numpy as np


import torch.optim as optim


class LogisticRegressionBinary(nn.Module):

    def __init__(self, config):
        super(LogicRegression, self).__init__()
        self.LR = nn.Linear(config.input_size, config.output_size)

    def forward(self, x):
        out = self.LR(x)
        out = torch.sigmoid(out)
        return out


class LogisticRegressionMulti(nn.Module):

    def __init__(self, config):
        super(LogisticRegressionMulti, self).__init__()
        self.config = config
        self.LR = nn.Linear(config.input_size, config.output_size)

    def forward(self, x):
        x = x.reshape(-1, self.config.input_size)
        return self.LR(x)


class LinearRegression(nn.Module):

    def __init__(self, config):
        super(LinearRegression, self).__init__()
        self.LR = nn.Linear(config.input_size, config.output_size)

    def forward(self, x):
        out = self.LR(x)
        return out


class FNN(nn.Module):

    def __init__(self, config):
        super(FNN, self).__init__()
        self.config = config
        self.input_layer = nn.Linear(config.input_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.hidden_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_layer = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, x):
        x = x.reshape(-1, self.config.input_size)
        out = self.input_layer(x)
        out = self.relu(out)
        out = self.hidden_layer(out)
        out = self.relu(out)
        out = self.output_layer(out)
        return out


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Highway(nn.Module):
    """
    Input shape=(batch_size,dim,dim)
    Output shape=(batch_size,dim,dim)
    """

    def __init__(self, layer_num, dim=600):
        super(Highway, self).__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim) for _ in range(self.layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x


class Embedding(nn.Module):
    """
    word and char embedding

    Input shape: word_emb=(batch_size,sentence_length,emb_size) char_emb=(batch_size,sentence_length,word_length,emb_size)
    Output shape: y= (batch_size,sentence_length,word_emb_size+char_emb_size)
    """

    def __init__(self, highway_layers, word_dim, char_dim):
        super(Embedding, self).__init__()
        self.highway = Highway(highway_layers, word_dim + char_dim)

    def forward(self, word_emb, char_emb):
        char_emb, _ = torch.max(char_emb, 2)
        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.highway(emb)
        return emb


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        """
        Args: 
            input_size: x 的特征维度
            hidden_size: 隐层的特征维度
            num_layers: LSTM 层数
        """
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.init_params()

    def init_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)
            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x, lengths):
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        packed_output, (hidden, cell) = self.rnn(packed_x)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        return hidden, output


class LSTMATTHighway(nn.Module):

    def __init__(self, word_dim, char_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, word_emb, char_emb, highway_layers):
        super(LSTMATTHighway, self).__init__()
        self.char_embedding = nn.Embedding.from_pretrained(char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)
        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)
        self.rnn = LSTM(word_dim + char_dim, hidden_size, num_layers, bidirectional, dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.W_w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, text_word, text_char):
        text_word, text_lengths = text_word
        word_emb = self.dropout(self.word_embedding(text_word))
        char_emb = self.dropout(self.char_embedding(text_char))
        char_emb = char_emb.permute(1, 0, 2, 3)
        text_emb = self.text_embedding(word_emb, char_emb)
        hidden, outputs = self.rnn(text_emb, text_lengths)
        outputs = outputs.permute(1, 0, 2)
        """ tanh attention 的实现 """
        score = torch.tanh(torch.matmul(outputs, self.W_w))
        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        scored_x = outputs * attention_weights
        feat = torch.sum(scored_x, dim=1)
        return self.fc(feat)


class LSTMATT(nn.Module):

    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pretrained_embeddings):
        super(LSTMATT, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = LSTM(embedding_dim, hidden_size, num_layers, bidirectional, dropout)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.W_w = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_w = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

    def forward(self, x):
        text, text_lengths = x
        embedded = self.dropout(self.embedding(text))
        hidden, outputs = self.rnn(embedded, text_lengths)
        outputs = outputs.permute(1, 0, 2)
        """ tanh attention 的实现 """
        score = torch.tanh(torch.matmul(outputs, self.W_w))
        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        scored_x = outputs * attention_weights
        feat = torch.sum(scored_x, dim=1)
        return self.fc(feat)


class Conv1d(nn.Module):

    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=fs) for fs in filter_sizes])
        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]


class Linear(nn.Module):

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return x


class TextCNN(nn.Module):

    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pretrained_embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)
        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        text, _ = x
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conved = self.convs(embedded)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class TextCNNHighway(nn.Module):

    def __init__(self, word_dim, char_dim, n_filters, filter_sizes, output_dim, dropout, word_emb, char_emb, highway_layers):
        super().__init__()
        self.char_embedding = nn.Embedding.from_pretrained(char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)
        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)
        self.convs = Conv1d(word_dim + char_dim, n_filters, filter_sizes)
        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_word, text_char):
        text_word, _ = text_word
        word_emb = self.word_embedding(text_word)
        char_emb = self.char_embedding(text_char)
        char_emb = char_emb.permute(1, 0, 2, 3)
        text_emb = self.text_embedding(word_emb, char_emb)
        text_emb = text_emb.permute(1, 2, 0)
        conved = self.convs(text_emb)
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


class TextRCNN(nn.Module):

    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pretrained_embeddings):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        self.W2 = Linear(2 * hidden_size + embedding_dim, hidden_size * 2)
        self.fc = Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        text, text_lengths = x
        embedded = self.dropout(self.embedding(text))
        outputs, _ = self.rnn(embedded)
        outputs = outputs.permute(1, 0, 2)
        embedded = embedded.permute(1, 0, 2)
        x = torch.cat((outputs, embedded), 2)
        y2 = torch.tanh(self.W2(x)).permute(0, 2, 1)
        y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
        return self.fc(y3)


class TextRCNNHighway(nn.Module):

    def __init__(self, word_dim, char_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, word_emb, char_emb, highway_layers):
        super(TextRCNNHighway, self).__init__()
        self.char_embedding = nn.Embedding.from_pretrained(char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)
        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)
        self.rnn = nn.LSTM(word_dim + char_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        self.W2 = Linear(2 * hidden_size + word_dim + char_dim, hidden_size * 2)
        self.fc = Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_word, text_char):
        text_word, text_lengths = text_word
        word_emb = self.dropout(self.word_embedding(text_word))
        char_emb = self.dropout(self.char_embedding(text_char))
        char_emb = char_emb.permute(1, 0, 2, 3)
        text_emb = self.text_embedding(word_emb, char_emb)
        outputs, _ = self.rnn(text_emb)
        outputs = outputs.permute(1, 0, 2)
        text_emb = text_emb.permute(1, 0, 2)
        x = torch.cat((outputs, text_emb), 2)
        y2 = torch.tanh(self.W2(x)).permute(0, 2, 1)
        y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
        return self.fc(y3)


class TextRNN(nn.Module):

    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, pretrained_embeddings):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.rnn = LSTM(embedding_dim, hidden_size, num_layers, bidirectional, dropout)
        self.fc = Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        text, text_lengths = x
        embedded = self.dropout(self.embedding(text))
        hidden, outputs = self.rnn(embedded, text_lengths)
        hidden = self.dropout(torch.cat((hidden[(-2), :, :], hidden[(-1), :, :]), dim=1))
        return self.fc(hidden)


class TextRNNHighway(nn.Module):

    def __init__(self, word_dim, char_dim, output_dim, hidden_size, num_layers, bidirectional, dropout, word_emb, char_emb, highway_layers):
        super(TextRNNHighway, self).__init__()
        self.char_embedding = nn.Embedding.from_pretrained(char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)
        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)
        self.rnn = LSTM(word_dim + char_dim, hidden_size, num_layers, bidirectional, dropout)
        self.fc = Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_word, text_char):
        text_word, text_lengths = text_word
        word_emb = self.dropout(self.word_embedding(text_word))
        char_emb = self.dropout(self.char_embedding(text_char))
        char_emb = char_emb.permute(1, 0, 2, 3)
        text_emb = self.text_embedding(word_emb, char_emb)
        hidden, outputs = self.rnn(text_emb, text_lengths)
        hidden = self.dropout(torch.cat((hidden[(-2), :, :], hidden[(-1), :, :]), dim=1))
        return self.fc(hidden)


class LayerNorm(nn.Module):
    """
    Layer Normalization 的实现
    """

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def clones(module, N):
    """
    clone N 个完全相同的 module
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    residual connection + layer norma lization
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        """
        return x + self.dropout(sublayer(self.norm(x)))


class DecoderLayer(nn.Module):
    """ Self-attention + encoder self-attention + feed forward """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Embeddings(nn.Module):

    def __init__(self, pretrained_embeddings, d_model, freeze=False):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Encoder(nn.Module):
    """ Transformer Encoder
    It includes N layer of EncoderLayer
    """

    def __init__(self, layer, N):
        """
        layer: EncoderLayer, 每层的网络
        N: Encoder 包含 N 层 EncoderLayer
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder Layer:
        self-attention + feed-forward layer
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        """

        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class ScaledDotProduction(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, dropout=0.1):
        super(ScaledDotProduction, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """ Q_len == K_len == V_len
        Args:
            Q: [batch_size, Q_len, dim]
            K: [batch_size, K_len, dim]
            V: [batch_size, V_len, dim]
            mask: 是否 mask， 只有 decoder 才需要
        Returns:
            output: [batch_size, Q_len, dim], attention value
            attn: [batch_size, Q_len, K_len]， scores 
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1000000000.0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = ScaledDotProduction(dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q: [batch_size, Q_len, d_model]
            K: [batch_size, K_len, d_model]
            V: [batch-size, V_len, d_model]
            mask: 
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = Q.size(0)
        Q, K, V = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (Q, K, V))]
        x, attn = self.attn(Q, K, V, mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -1 * (math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class TransformerText(nn.Module):
    """ 用 Transformer 来作为特征抽取的基本单元 """

    def __init__(self, head, n_layer, emd_dim, d_model, d_ff, output_dim, dropout, pretrained_embeddings):
        super(TransformerText, self).__init__()
        self.word_embedding = Embeddings(pretrained_embeddings, emd_dim)
        self.position_embedding = PositionalEncoding(emd_dim, dropout)
        self.trans_linear = nn.Linear(emd_dim, d_model)
        multi_attn = MultiHeadAttention(head, d_model)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, multi_attn, feed_forward, dropout), n_layer)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x:
            text: [sent len, batch size], 文本数据
            text_lens: [batch_size], 文本数据长度
        """
        text, _ = x
        text = text.permute(1, 0)
        embeddings = self.word_embedding(text)
        embeddings = self.position_embedding(embeddings)
        embeddings = self.trans_linear(embeddings)
        embeddings = self.encoder(embeddings)
        features = embeddings[:, (-1), :]
        return self.fc(features)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv1d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'filter_sizes': [4, 4]}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (Embedding,
     lambda: ([], {'highway_layers': 1, 'word_dim': 4, 'char_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FNN,
     lambda: ([], {'config': _mock_config(input_size=4, hidden_size=4, output_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Highway,
     lambda: ([], {'layer_num': 1}),
     lambda: ([torch.rand([600, 600])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LinearRegression,
     lambda: ([], {'config': _mock_config(input_size=4, output_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LogisticRegressionMulti,
     lambda: ([], {'config': _mock_config(input_size=4, output_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'h': 4, 'd_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_model': 4, 'd_ff': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ScaledDotProduction,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SublayerConnection,
     lambda: ([], {'size': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4]), _mock_layer()], {}),
     False),
]

class Test_songyingxin_TextClassification(_paritybench_base):
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

