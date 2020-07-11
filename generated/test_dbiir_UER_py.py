import sys
_module = sys.modules[__name__]
del sys
preprocess = _module
pretrain = _module
run_classifier = _module
run_mrc = _module
run_ner = _module
average_model = _module
build_vocab = _module
check_model = _module
cloze_test = _module
convert_bert_from_google_to_uer = _module
convert_bert_from_huggingface_to_uer = _module
convert_bert_from_uer_to_google = _module
convert_bert_from_uer_to_huggingface = _module
diff_vocab = _module
dynamic_vocab_adapter = _module
extract_embedding = _module
extract_feature = _module
generate = _module
multi_single_convert = _module
topn_words_dep = _module
topn_words_indep = _module
uer = _module
encoders = _module
attn_encoder = _module
bert_encoder = _module
birnn_encoder = _module
cnn_encoder = _module
gpt_encoder = _module
mixed_encoder = _module
rnn_encoder = _module
layers = _module
embeddings = _module
layer_norm = _module
multi_headed_attn = _module
position_ffn = _module
transformer = _module
model_builder = _module
model_loader = _module
model_saver = _module
models = _module
bert_model = _module
model = _module
subencoders = _module
avg_subencoder = _module
cnn_subencoder = _module
rnn_subencoder = _module
targets = _module
bert_target = _module
bilm_target = _module
cls_target = _module
lm_target = _module
mlm_target = _module
nsp_target = _module
s2s_target = _module
trainer = _module
utils = _module
act_fun = _module
config = _module
constants = _module
data = _module
misc = _module
optimizers = _module
seed = _module
subword = _module
tokenizer = _module
vocab = _module

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


import random


import collections


import torch.nn as nn


from collections import Counter


from collections import OrderedDict


import string


import re


from torch.nn import CrossEntropyLoss


import tensorflow as tf


import numpy as np


from tensorflow.python import pywrap_tensorflow


import torch.nn.functional as F


import math


import time


import torch.distributed as dist


import torch.multiprocessing as mp


from torch.nn.parallel import DistributedDataParallel


from torch.optim import Optimizer


from torch.optim.lr_scheduler import LambdaLR


class BertClassifier(nn.Module):

    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, src, label, mask):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        emb = self.embedding(src, mask)
        output = self.encoder(emb, mask)
        if self.pooling == 'mean':
            output = torch.mean(output, dim=1)
        elif self.pooling == 'max':
            output = torch.max(output, dim=1)[0]
        elif self.pooling == 'last':
            output = output[:, (-1), :]
        else:
            output = output[:, (0), :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


class BertQuestionAnswering(nn.Module):

    def __init__(self, args, bert_model):
        super(BertQuestionAnswering, self).__init__()
        self.embedding = bert_model.embedding
        self.encoder = bert_model.encoder
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, src, mask, start_positions, end_positions):
        """
        Args:
            src: [batch_size x seq_length]
            start_positions,end_positions: [batch_size]
            mask: [batch_size x seq_length]
        """
        emb = self.embedding(src, mask)
        output = self.encoder(emb, mask)
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        start_loss = self.criterion(self.softmax(start_logits), start_positions)
        end_loss = self.criterion(self.softmax(end_logits), end_positions)
        loss = (start_loss + end_loss) / 2
        return loss, start_logits, end_logits


class BertTagger(nn.Module):

    def __init__(self, args, model):
        super(BertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, label, mask):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]

        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        emb = self.embedding(src, mask)
        output = self.encoder(emb, mask)
        output = self.output_layer(output)
        output = output.contiguous().view(-1, self.labels_num)
        output = self.softmax(output)
        label = label.contiguous().view(-1, 1)
        label_mask = (label > 0).float()
        one_hot = torch.zeros(label_mask.size(0), self.labels_num).scatter_(1, label, 1.0)
        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-06
        loss = numerator / denominator
        predict = output.argmax(dim=-1)
        correct = torch.sum(label_mask * predict.eq(label).float())
        return loss, correct, predict, label


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ClozeModel(torch.nn.Module):

    def __init__(self, args, model):
        super(ClozeModel, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.eval()

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = gelu(self.target.mlm_linear_1(output))
        output = self.target.layer_norm(output)
        output = self.target.mlm_linear_2(output)
        prob = torch.nn.Softmax(dim=-1)(output)
        return prob


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class BertEmbedding(nn.Module):
    """
    BERT embedding consists of three parts:
    word embedding, position embedding, and segment embedding.
    """

    def __init__(self, args, vocab_size):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.max_length = 512
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.position_embedding = nn.Embedding(self.max_length, args.emb_size)
        self.segment_embedding = nn.Embedding(3, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, seg):
        word_emb = self.word_embedding(src)
        pos_emb = self.position_embedding(torch.arange(0, word_emb.size(1), device=word_emb.device, dtype=torch.long).unsqueeze(0).repeat(word_emb.size(0), 1))
        seg_emb = self.segment_embedding(seg)
        emb = word_emb + pos_emb + seg_emb
        emb = self.dropout(self.layer_norm(emb))
        return emb


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x.contiguous().view(batch_size, seq_length, heads_num, per_head_size).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_size)
        query, key, value = [l(x).view(batch_size, -1, heads_num, per_head_size).transpose(1, 2) for l, x in zip(self.linear_layers, (query, key, value))]
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output


class PositionwiseFeedForward(nn.Module):
    """ Feed Forward Layer """

    def __init__(self, hidden_size, feedforward_size):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, feedforward_size)
        self.linear_2 = nn.Linear(feedforward_size, hidden_size)

    def forward(self, x):
        inter = gelu(self.linear_1(x))
        output = self.linear_2(inter)
        return output


class TransformerLayer(nn.Module):
    """
    Transformer layer mainly consists of two parts:
    multi-headed self-attention and feed forward layer.
    """

    def __init__(self, args):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(args.hidden_size, args.heads_num, args.dropout)
        self.dropout_1 = nn.Dropout(args.dropout)
        self.layer_norm_1 = LayerNorm(args.hidden_size)
        self.feed_forward = PositionwiseFeedForward(args.hidden_size, args.feedforward_size)
        self.dropout_2 = nn.Dropout(args.dropout)
        self.layer_norm_2 = LayerNorm(args.hidden_size)

    def forward(self, hidden, mask):
        """
        Args:
            hidden: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        inter = self.dropout_1(self.self_attn(hidden, hidden, hidden, mask))
        inter = self.layer_norm_1(inter + hidden)
        output = self.dropout_2(self.feed_forward(inter))
        output = self.layer_norm_2(output + inter)
        return output


class BertEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([TransformerLayer(args) for _ in range(self.layers_num)])

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        seq_length = emb.size(1)
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        return hidden


class SequenceEncoder(torch.nn.Module):

    def __init__(self, args, vocab):
        super(SequenceEncoder, self).__init__()
        self.embedding = BertEmbedding(args, len(vocab))
        self.encoder = BertEncoder(args)

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        return output


class GenerateModel(torch.nn.Module):

    def __init__(self, args, model):
        super(GenerateModel, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.eval()

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = gelu(self.target.output_layer(output))
        return output


class AttnEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(AttnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.self_attn = MultiHeadedAttention(args.hidden_size, args.heads_num, args.dropout)
        self.self_attn = nn.ModuleList([MultiHeadedAttention(args.hidden_size, args.heads_num, args.dropout) for _ in range(self.layers_num)])

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            mask: [batch_size x 1 x seq_length x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        seq_length = emb.size(1)
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        hidden = emb
        for i in range(self.layers_num):
            hidden = self.self_attn[i](hidden, hidden, hidden, mask)
        return hidden


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class BilstmEncoder(nn.Module):

    def __init__(self, args):
        super(BilstmEncoder, self).__init__()
        assert args.hidden_size % 2 == 0
        self.hidden_size = args.hidden_size // 2
        self.layers_num = args.layers_num
        self.rnn_forward = nn.LSTM(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)
        self.rnn_backward = nn.LSTM(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        emb_forward = emb
        hidden_forward = self.init_hidden(emb_forward.size(0), emb_forward.device)
        output_forward, hidden_forward = self.rnn_forward(emb_forward, hidden_forward)
        output_forward = self.drop(output_forward)
        emb_backward = flip(emb, 1)
        hidden_backward = self.init_hidden(emb_backward.size(0), emb_backward.device)
        output_backward, hidden_backward = self.rnn_backward(emb_backward, hidden_backward)
        output_backward = self.drop(output_backward)
        output_backward = flip(output_backward, 1)
        return torch.cat([output_forward, output_backward], 2)

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device), torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class CnnEncoder(nn.Module):

    def __init__(self, args):
        super(CnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) for _ in range(args.layers_num)])

    def forward(self, emb, seg):
        batch_size, seq_len, _ = emb.size()
        padding = torch.zeros([batch_size, self.kernel_size - 1, self.emb_size])
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)
        hidden = self.conv_1(emb)
        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size - 1, 1])
        hidden = torch.cat([padding, hidden], dim=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:, :, self.kernel_size - 1:, :]
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return output


class GatedcnnEncoder(nn.Module):

    def __init__(self, args):
        super(GatedcnnEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.kernel_size = args.kernel_size
        self.block_size = args.block_size
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.gate_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) for _ in range(args.layers_num - 1)])
        self.gate = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) for _ in range(args.layers_num - 1)])

    def forward(self, emb, seg):
        batch_size, seq_len, _ = emb.size()
        res_input = torch.transpose(emb.unsqueeze(3), 1, 2)
        padding = torch.zeros([batch_size, self.kernel_size - 1, self.emb_size])
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)
        hidden = self.conv_1(emb)
        gate = self.gate_1(emb)
        hidden = hidden * torch.sigmoid(gate)
        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size - 1, 1])
        hidden = torch.cat([padding, hidden], dim=2)
        for i, (conv_i, gate_i) in enumerate(zip(self.conv, self.gate)):
            hidden, gate = conv_i(hidden), gate_i(hidden)
            hidden = hidden * torch.sigmoid(gate)
            if (i + 1) % self.block_size:
                hidden = hidden + res_input
                res_input = hidden
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:, :, self.kernel_size - 1:, :]
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return output


class GptEncoder(nn.Module):
    """
    BERT encoder exploits 12 or 24 transformer layers to extract features.
    """

    def __init__(self, args):
        super(GptEncoder, self).__init__()
        self.layers_num = args.layers_num
        self.transformer = nn.ModuleList([TransformerLayer(args) for _ in range(self.layers_num)])

    def forward(self, emb, seg):
        """
        Args:
            emb: [batch_size x seq_length x emb_size]
            seg: [batch_size x seq_length]

        Returns:
            hidden: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = emb.size()
        mask = torch.ones(seq_length, seq_length, device=emb.device)
        mask = torch.tril(mask)
        mask = (1.0 - mask) * -10000
        mask = mask.repeat(batch_size, 1, 1, 1)
        hidden = emb
        for i in range(self.layers_num):
            hidden = self.transformer[i](hidden, mask)
        return hidden


class RcnnEncoder(nn.Module):

    def __init__(self, args):
        super(RcnnEncoder, self).__init__()
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        self.layers_num = args.layers_num
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=args.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)
        self.drop = nn.Dropout(args.dropout)
        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) for _ in range(args.layers_num - 1)])

    def forward(self, emb, seg):
        batch_size, seq_len, _ = emb.size()
        hidden = self.init_hidden(batch_size, emb.device)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        padding = torch.zeros([batch_size, self.kernel_size - 1, self.emb_size])
        hidden = torch.cat([padding, output], dim=1).unsqueeze(1)
        hidden = self.conv_1(hidden)
        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size - 1, 1])
        hidden = torch.cat([padding, hidden], dim=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:, :, self.kernel_size - 1:, :]
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return output

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device), torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class CrnnEncoder(nn.Module):

    def __init__(self, args):
        super(CrnnEncoder, self).__init__()
        self.emb_size = args.emb_size
        self.hidden_size = args.hidden_size
        self.kernel_size = args.kernel_size
        self.layers_num = args.layers_num
        self.conv_1 = nn.Conv2d(1, args.hidden_size, (args.kernel_size, args.emb_size))
        self.conv = nn.ModuleList([nn.Conv2d(args.hidden_size, args.hidden_size, (args.kernel_size, 1)) for _ in range(args.layers_num - 1)])
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=args.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        batch_size, seq_len, _ = emb.size()
        padding = torch.zeros([batch_size, self.kernel_size - 1, self.emb_size])
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)
        hidden = self.conv_1(emb)
        padding = torch.zeros([batch_size, self.hidden_size, self.kernel_size - 1, 1])
        hidden = torch.cat([padding, hidden], dim=2)
        for i, conv_i in enumerate(self.conv):
            hidden = conv_i(hidden)
            hidden = torch.cat([padding, hidden], dim=2)
        hidden = hidden[:, :, self.kernel_size - 1:, :]
        output = hidden.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        hidden = self.init_hidden(batch_size, emb.device)
        output, hidden = self.rnn(output, hidden)
        output = self.drop(output)
        return output

    def init_hidden(self, batch_size, device):
        return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device), torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class LstmEncoder(nn.Module):

    def __init__(self, args):
        super(LstmEncoder, self).__init__()
        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0
            self.hidden_size = args.hidden_size // 2
        else:
            self.hidden_size = args.hidden_size
        self.layers_num = args.layers_num
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True, bidirectional=self.bidirectional)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return torch.zeros(self.layers_num * 2, batch_size, self.hidden_size, device=device), torch.zeros(self.layers_num * 2, batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device), torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class GruEncoder(nn.Module):

    def __init__(self, args):
        super(GruEncoder, self).__init__()
        self.bidirectional = args.bidirectional
        if self.bidirectional:
            assert args.hidden_size % 2 == 0
            self.hidden_size = args.hidden_size // 2
        else:
            self.hidden_size = args.hidden_size
        self.layers_num = args.layers_num
        self.rnn = nn.GRU(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=args.layers_num, dropout=args.dropout, batch_first=True, bidirectional=self.bidirectional)
        self.drop = nn.Dropout(args.dropout)

    def forward(self, emb, seg):
        hidden = self.init_hidden(emb.size(0), emb.device)
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        return output

    def init_hidden(self, batch_size, device):
        if self.bidirectional:
            return torch.zeros(self.layers_num * 2, batch_size, self.hidden_size, device=device)
        else:
            return torch.zeros(self.layers_num, batch_size, self.hidden_size, device=device)


class WordEmbedding(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(WordEmbedding, self).__init__()
        self.dropout = nn.Dropout(args.dropout)
        self.word_embedding = nn.Embedding(vocab_size, args.emb_size)
        self.layer_norm = LayerNorm(args.emb_size)

    def forward(self, src, _):
        emb = self.word_embedding(src)
        emb = self.dropout(self.layer_norm(emb))
        return emb


class BertModel(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """

    def __init__(self, args, embedding, encoder, target):
        super(BertModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

    def forward(self, src, tgt_mlm, tgt_nsp, seg):
        emb = self.embedding(src, seg)
        seq_length = emb.size(1)
        mask = (seg > 0).unsqueeze(1).repeat(1, seq_length, 1).unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        output = self.encoder(emb, mask)
        loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator = self.target(output, tgt_mlm, tgt_nsp)
        return loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator


UNK_ID = 100


def word2sub(word_ids, vocab, sub_vocab, subword_type):
    """
    word_ids: batch_size, seq_length
    """
    batch_size, seq_length = word_ids.size()
    device = word_ids.device
    word_ids = word_ids.contiguous().view(-1).tolist()
    words = [vocab.i2w[i] for i in word_ids]
    max_length = max([len(w) for w in words])
    sub_ids = torch.zeros((len(words), max_length), dtype=torch.long)
    for i in range(len(words)):
        for j, c in enumerate(words[i]):
            sub_ids[i, j] = sub_vocab.w2i.get(c, UNK_ID)
    return sub_ids


class Model(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm and nsp tasks
    """

    def __init__(self, args, embedding, encoder, target, subencoder=None):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target
        if subencoder is not None:
            self.vocab, self.sub_vocab = args.vocab, args.sub_vocab
            self.subword_type = args.subword_type
            self.subencoder = subencoder
        else:
            self.subencoder = None

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        if self.subencoder is not None:
            sub_ids = word2sub(src, self.vocab, self.sub_vocab, self.subword_type)
            emb = emb + self.subencoder(sub_ids).contiguous().view(*emb.size())
        output = self.encoder(emb, seg)
        loss_info = self.target(output, tgt)
        return loss_info


class AvgSubencoder(nn.Module):

    def __init__(self, args, vocab_size):
        super(AvgSubencoder, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, args.emb_size)

    def forward(self, ids):
        emb = self.embedding_layer(ids)
        output = emb.mean(1)
        return output


class CnnSubencoder(nn.Module):

    def __init__(self, args, vocab_size):
        super(CnnSubencoder, self).__init__()
        self.kernel_size = args.kernel_size
        self.emb_size = args.emb_size
        self.embedding_layer = nn.Embedding(vocab_size, args.emb_size)
        self.cnn = nn.Conv2d(1, args.emb_size, (args.kernel_size, args.emb_size))

    def forward(self, ids):
        emb = self.embedding_layer(ids)
        padding = torch.zeros([emb.size(0), self.kernel_size - 1, self.emb_size])
        emb = torch.cat([padding, emb], dim=1).unsqueeze(1)
        conv_output = F.relu(self.cnn(emb)).squeeze(3)
        conv_output = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)
        return conv_output


class LstmSubencoder(nn.Module):

    def __init__(self, args, vocab_size):
        super(LstmSubencoder, self).__init__()
        self.hidden_size = args.emb_size
        self.layers_num = args.sub_layers_num
        self.embedding_layer = nn.Embedding(vocab_size, args.emb_size)
        self.rnn = nn.LSTM(input_size=args.emb_size, hidden_size=self.hidden_size, num_layers=self.layers_num, dropout=args.dropout, batch_first=True)

    def forward(self, ids):
        batch_size, _ = ids.size()
        hidden = torch.zeros(self.layers_num, batch_size, self.hidden_size), torch.zeros(self.layers_num, batch_size, self.hidden_size)
        emb = self.embedding_layer(ids)
        output, hidden = self.rnn(emb, hidden)
        output = output.mean(1)
        return output


class BertTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM) 
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(BertTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = LayerNorm(args.hidden_size)
        self.mlm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)
        self.nsp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.nsp_linear_2 = nn.Linear(args.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def mlm(self, memory_bank, tgt_mlm):
        output_mlm = gelu(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[(tgt_mlm > 0), :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        one_hot = torch.zeros(output_mlm.size(0), self.vocab_size).scatter_(1, tgt_mlm.contiguous().view(-1, 1), 1.0)
        numerator = -torch.sum(output_mlm * one_hot, 1)
        denominator = torch.tensor(output_mlm.size(0) + 1e-06)
        loss_mlm = torch.sum(numerator) / denominator
        correct_mlm = torch.sum(output_mlm.argmax(dim=-1).eq(tgt_mlm).float())
        return loss_mlm, correct_mlm, denominator

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_nsp [batch_size]

        Returns:
            loss_mlm: Masked language model loss.
            loss_nsp: Next sentence prediction loss.
            correct_mlm: Number of words that are predicted correctly.
            correct_nsp: Number of sentences that are predicted correctly.
            denominator: Number of masked words.
        """
        assert type(tgt) == tuple
        tgt_mlm, tgt_nsp = tgt[0], tgt[1]
        loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)
        output_nsp = torch.tanh(self.nsp_linear_1(memory_bank[:, (0), :]))
        output_nsp = self.nsp_linear_2(output_nsp)
        loss_nsp = self.criterion(self.softmax(output_nsp), tgt_nsp)
        correct_nsp = self.softmax(output_nsp).argmax(dim=-1).eq(tgt_nsp).sum()
        return loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator


class BilmTarget(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(BilmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size // 2
        self.softmax = nn.LogSoftmax(dim=-1)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """
        assert type(tgt) == tuple
        tgt_forward, tgt_backward = tgt[0], tgt[1]
        output_forward = self.output_layer(memory_bank[:, :, :self.hidden_size])
        output_forward = output_forward.contiguous().view(-1, self.vocab_size)
        output_forward = self.softmax(output_forward)
        tgt_forward = tgt_forward.contiguous().view(-1, 1)
        label_mask_forward = (tgt_forward > 0).float()
        one_hot_forward = torch.zeros(label_mask_forward.size(0), self.vocab_size).scatter_(1, tgt_forward, 1.0)
        numerator_forward = -torch.sum(output_forward * one_hot_forward, 1)
        label_mask_forward = label_mask_forward.contiguous().view(-1)
        tgt_forward = tgt_forward.contiguous().view(-1)
        numerator_forward = torch.sum(label_mask_forward * numerator_forward)
        denominator_forward = torch.sum(label_mask_forward) + 1e-06
        loss_forward = numerator_forward / denominator_forward
        correct_forward = torch.sum(label_mask_forward * output_forward.argmax(dim=-1).eq(tgt_forward).float())
        output_backward = self.output_layer(memory_bank[:, :, self.hidden_size:])
        output_backward = output_backward.contiguous().view(-1, self.vocab_size)
        output_backward = self.softmax(output_backward)
        tgt_backward = tgt_backward.contiguous().view(-1, 1)
        label_mask_backward = (tgt_backward > 0).float()
        one_hot_backward = torch.zeros(label_mask_backward.size(0), self.vocab_size).scatter_(1, tgt_backward, 1.0)
        numerator_backward = -torch.sum(output_backward * one_hot_backward, 1)
        label_mask_backward = label_mask_backward.contiguous().view(-1)
        tgt_backward = tgt_backward.contiguous().view(-1)
        numerator_backward = torch.sum(label_mask_backward * numerator_backward)
        denominator_backward = torch.sum(label_mask_backward) + 1e-06
        loss_backward = numerator_backward / denominator_backward
        correct_backward = torch.sum(label_mask_backward * output_backward.argmax(dim=-1).eq(tgt_backward).float())
        return loss_forward, loss_backward, correct_forward, correct_backward, denominator_backward


class ClsTarget(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(ClsTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Classification loss.
            correct: Number of sentences that are predicted correctly.
        """
        output = torch.tanh(self.linear_1(memory_bank[:, (0), :]))
        logits = self.linear_2(output)
        loss = self.criterion(self.softmax(logits), tgt)
        correct = self.softmax(logits).argmax(dim=-1).eq(tgt).sum()
        return loss, correct


class LmTarget(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(LmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.softmax = nn.LogSoftmax(dim=-1)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of predicted words.
        """
        output = self.output_layer(memory_bank)
        output = output.contiguous().view(-1, self.vocab_size)
        output = self.softmax(output)
        tgt = tgt.contiguous().view(-1, 1)
        label_mask = (tgt > 0).float()
        one_hot = torch.zeros(label_mask.size(0), self.vocab_size).scatter_(1, tgt, 1.0)
        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        tgt = tgt.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-06
        loss = numerator / denominator
        correct = torch.sum(label_mask * output.argmax(dim=-1).eq(tgt).float())
        return loss, correct, denominator


class MlmTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM) 
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(MlmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.mlm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = LayerNorm(args.hidden_size)
        self.mlm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def mlm(self, memory_bank, tgt_mlm):
        output_mlm = gelu(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[(tgt_mlm > 0), :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        one_hot = torch.zeros(output_mlm.size(0), self.vocab_size).scatter_(1, tgt_mlm.contiguous().view(-1, 1), 1.0)
        numerator = -torch.sum(output_mlm * one_hot, 1)
        denominator = torch.tensor(output_mlm.size(0) + 1e-06)
        loss_mlm = torch.sum(numerator) / denominator
        if output_mlm.size(0) == 0:
            correct_mlm = torch.tensor(0.0)
        else:
            correct_mlm = torch.sum(output_mlm.argmax(dim=-1).eq(tgt_mlm).float())
        return loss_mlm, correct_mlm, denominator

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Masked language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """
        loss, correct, denominator = self.mlm(memory_bank, tgt)
        return loss, correct, denominator


class NspTarget(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(NspTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.linear = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size]

        Returns:
            loss: Next sentence prediction loss.
            correct: Number of sentences that are predicted correctly.
        """
        output = self.linear(memory_bank[:, (0), :])
        loss = self.criterion(self.softmax(output), tgt)
        correct = self.softmax(output).argmax(dim=-1).eq(tgt).sum()
        return loss, correct


class S2sTarget(nn.Module):
    """
    """

    def __init__(self, args, vocab_size):
        super(S2sTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.embedding_layer = nn.Embedding(vocab_size, args.emb_size)
        self.decoder = nn.LSTM(args.emb_size, args.hidden_size, 1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, memory_bank, tgt):
        emb = self.embedding_layer(tgt[:, :])
        output = []
        hidden_state = memory_bank[:, (-1), :].unsqueeze(0).contiguous(), memory_bank[:, (-1), :].unsqueeze(0).contiguous()
        for i, emb_i in enumerate(emb.split(1, dim=1)):
            output_i, hidden_state = self.decoder(emb_i, hidden_state)
            output.append(self.output_layer(output_i))
        output = torch.cat(output, dim=1)
        output = output.contiguous().view(-1, self.vocab_size)
        output = self.softmax(output)
        tgt = tgt.contiguous().view(-1, 1)
        label_mask = (tgt > 0).float()
        one_hot = torch.zeros(label_mask.size(0), self.vocab_size).scatter_(1, tgt, 1.0)
        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        tgt = tgt.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-06
        loss = numerator / denominator
        correct = torch.sum(label_mask * output.argmax(dim=-1).eq(tgt).float())
        return loss, correct, denominator


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttnEncoder,
     lambda: ([], {'args': _mock_config(layers_num=1, hidden_size=4, heads_num=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (AvgSubencoder,
     lambda: ([], {'args': _mock_config(emb_size=4), 'vocab_size': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (BertEmbedding,
     lambda: ([], {'args': _mock_config(dropout=0.5, emb_size=4), 'vocab_size': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (BertEncoder,
     lambda: ([], {'args': _mock_config(layers_num=1, hidden_size=4, heads_num=4, dropout=0.5, feedforward_size=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
    (BilstmEncoder,
     lambda: ([], {'args': _mock_config(hidden_size=4, layers_num=1, emb_size=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CnnEncoder,
     lambda: ([], {'args': _mock_config(layers_num=1, kernel_size=4, block_size=4, emb_size=4, hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (CnnSubencoder,
     lambda: ([], {'args': _mock_config(kernel_size=4, emb_size=4), 'vocab_size': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     True),
    (CrnnEncoder,
     lambda: ([], {'args': _mock_config(emb_size=4, hidden_size=4, kernel_size=4, layers_num=1, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GatedcnnEncoder,
     lambda: ([], {'args': _mock_config(layers_num=1, kernel_size=4, block_size=4, emb_size=4, hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GptEncoder,
     lambda: ([], {'args': _mock_config(layers_num=1, hidden_size=4, heads_num=4, dropout=0.5, feedforward_size=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LmTarget,
     lambda: ([], {'args': _mock_config(hidden_size=4), 'vocab_size': 4}),
     lambda: ([torch.rand([64, 4, 4, 4]), torch.zeros([1024], dtype=torch.int64)], {}),
     True),
    (LstmSubencoder,
     lambda: ([], {'args': _mock_config(emb_size=4, sub_layers_num=1, dropout=0.5), 'vocab_size': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     True),
    (MultiHeadedAttention,
     lambda: ([], {'hidden_size': 4, 'heads_num': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 64, 4]), torch.rand([4, 16, 4, 4]), torch.rand([4, 64, 4]), torch.rand([4, 4, 64, 64])], {}),
     False),
    (PositionwiseFeedForward,
     lambda: ([], {'hidden_size': 4, 'feedforward_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (RcnnEncoder,
     lambda: ([], {'args': _mock_config(emb_size=4, hidden_size=4, kernel_size=4, layers_num=1, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequenceEncoder,
     lambda: ([], {'args': _mock_config(dropout=0.5, emb_size=4, layers_num=1, hidden_size=4, heads_num=4, feedforward_size=4), 'vocab': [4, 4]}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
    (TransformerLayer,
     lambda: ([], {'args': _mock_config(hidden_size=4, heads_num=4, dropout=0.5, feedforward_size=4)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (WordEmbedding,
     lambda: ([], {'args': _mock_config(dropout=0.5, emb_size=4), 'vocab_size': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64), torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_dbiir_UER_py(_paritybench_base):
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

