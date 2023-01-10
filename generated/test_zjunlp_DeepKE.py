import sys
_module = sys.modules[__name__]
del sys
conf = _module
predict = _module
run = _module
get_doc_statistics = _module
process_ace05e = _module
process_ace05ep = _module
split_dataset = _module
split_dataset_dygie = _module
generate_data = _module
predict = _module
run = _module
predict = _module
run = _module
predict = _module
run = _module
prepare_weaksupervised_data = _module
run_bert = _module
run_lstmcrf = _module
predict = _module
run = _module
DA = _module
merge_dataset = _module
predict = _module
run = _module
predict = _module
run = _module
ds_label_data = _module
predict = _module
run = _module
InferBert = _module
LMModel = _module
predict = _module
preprocess = _module
utils = _module
setup = _module
deepke = _module
attribution_extraction = _module
standard = _module
BasicModule = _module
BiLSTM = _module
Capsule = _module
GCN = _module
LM = _module
PCNN = _module
Transformer = _module
models = _module
Attention = _module
CNN = _module
Capsule = _module
Embedding = _module
GCN = _module
RNN = _module
Transformer = _module
module = _module
tools = _module
dataset = _module
metrics = _module
serializer = _module
trainer = _module
vocab = _module
ioUtils = _module
nnUtils = _module
event_extraction = _module
degree = _module
data = _module
model = _module
template_generate_ace = _module
retrieve = _module
retrieve_utils = _module
ann2brat = _module
annotation = _module
phrase = _module
rule = _module
wsd = _module
name_entity_re = _module
few_shot = _module
model = _module
modeling_bart = _module
datasets = _module
mapping_type = _module
train = _module
util = _module
multimodal = _module
IFA_model = _module
clip = _module
configuration_clip = _module
feature_extraction_clip = _module
feature_extraction_utils = _module
file_utils = _module
image_utils = _module
modeling_clip = _module
processing_clip = _module
tokenization_clip = _module
modeling_IFA = _module
modules = _module
dataset = _module
train = _module
BiLSTM_CRF = _module
InferBert = _module
dataset = _module
relation_extraction = _module
document = _module
evaluation = _module
losses = _module
model = _module
module = _module
prepro = _module
utils = _module
base_data_module = _module
dialogue = _module
processor = _module
generate_k_shot = _module
get_label_word = _module
lit_models = _module
base = _module
transformer = _module
IFA_model = _module
feature_extraction_clip = _module
feature_extraction_utils = _module
file_utils = _module
image_utils = _module
modeling_clip = _module
processing_clip = _module
modeling_IFA = _module
dataset = _module
train = _module
BasicModule = _module
BiLSTM = _module
Capsule = _module
GCN = _module
LM = _module
PCNN = _module
Transformer = _module
Attention = _module
CNN = _module
Capsule = _module
Embedding = _module
GCN = _module
RNN = _module
Transformer = _module
dataset = _module
loss = _module
metrics = _module
trainer = _module
nnUtils = _module
transform_data = _module
modeling_bart = _module

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


import logging


import matplotlib.pyplot as plt


import torch.nn as nn


from torch import optim


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


import re


import random


from typing import List


from typing import Dict


from typing import Any


from typing import Tuple


import numpy as np


import time


import warnings


import torch.nn.functional as F


from torch import nn


from torch.utils.data import RandomSampler


from torch.utils.data import SequentialSampler


from torch.utils.data import TensorDataset


from torch.utils.data.distributed import DistributedSampler


from sklearn.metrics import f1_score


from sklearn.metrics import classification_report


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.optim.lr_scheduler import StepLR


from torch.optim import Adam


from logging import debug


from torch.utils.data.dataloader import DataLoader


from torchvision import transforms


from collections import OrderedDict


from typing import Union


import math


from torch.utils.data import Dataset


from abc import ABCMeta


from abc import abstractmethod


from sklearn.metrics import precision_recall_fscore_support


from collections import namedtuple


from torch.nn import functional as F


from functools import partial


from typing import Optional


from torch import Tensor


from torch.nn import CrossEntropyLoss


from itertools import chain


from torch.nn.utils.rnn import pad_sequence


import copy


from collections import UserDict


from typing import TYPE_CHECKING


from enum import Enum


import functools


import types


from functools import wraps


from types import ModuleType


from typing import BinaryIO


from typing import ContextManager


from uuid import uuid4


import torch.utils.checkpoint


from torch import device


from torch.autograd import Variable


class BasicModule(nn.Module):
    """
    封装nn.Module, 提供 save 和 load 方法
    """

    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path, device):
        """
        加载指定路径的模型
        """
        self.load_state_dict(torch.load(path, map_location=device))

    def save(self, epoch=0, cfg=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名
        """
        time_prefix = time.strftime('%Y-%m-%d_%H-%M-%S')
        prefix = os.path.join(cfg.cwd, 'checkpoints', time_prefix)
        os.makedirs(prefix, exist_ok=True)
        name = os.path.join(prefix, cfg.model_name + '_' + f'epoch{epoch}' + '.pth')
        torch.save(self.state_dict(), name)
        return name


class RNN(nn.Module):

    def __init__(self, config):
        """
        type_rnn: RNN, GRU, LSTM 可选
        """
        super(RNN, self).__init__()
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size // 2 if config.bidirectional else config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.last_layer_hn = config.last_layer_hn
        self.type_rnn = config.type_rnn
        rnn = eval(f'nn.{self.type_rnn}')
        self.rnn = rnn(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional, bias=True, batch_first=True)

    def forward(self, x, x_len):
        """
        Args: 
            torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H_in] 一般是经过embedding后的值
            x_len: torch.Tensor [L] 已经排好序的句长值
        Returns:
            output: torch.Tensor [B, L, H_out] 序列标注的使用结果
            hn:     torch.Tensor [B, N, H_out] / [B, H_out] 分类的结果，当 last_layer_hn 时只有最后一层结果
        """
        B, L, _ = x.size()
        H, N = self.hidden_size, self.num_layers
        x_len = x_len.cpu()
        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=True)
        output, hn = self.rnn(x)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=L)
        if self.type_rnn == 'LSTM':
            hn = hn[0]
        if self.bidirectional:
            hn = hn.view(N, 2, B, H).transpose(1, 2).contiguous().view(N, B, 2 * H).transpose(0, 1)
        else:
            hn = hn.transpose(0, 1)
        if self.last_layer_hn:
            hn = hn[:, -1, :]
        return output, hn


def seq_len_to_mask(seq_len: Union[List, np.ndarray, torch.Tensor], max_len=512, mask_pos_to_true=True):
    """
    将一个表示sequence length的一维数组转换为二维的mask，默认pad的位置为1。
    转变 1-d seq_len到2-d mask。

    Args :
        seq_len (list, np.ndarray, torch.LongTensor) : shape将是(B,)
        max_len (int): 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    Return: 
        mask (np.ndarray, torch.Tensor) : shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)
    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)
    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f'seq_len can only have one dimension, got {seq_len.dim()} != 1.')
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error('Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.')
    return mask


class LM(BasicModule):

    def __init__(self, cfg):
        super(LM, self).__init__()
        self.bert = BertModel.from_pretrained(cfg.lm_file, num_hidden_layers=cfg.num_hidden_layers)
        self.bilstm = RNN(cfg)
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_relations)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        word, lens = x['word'], x['lens']
        mask = seq_len_to_mask(lens, mask_pos_to_true=False)
        a = self.bert(word, attention_mask=mask)
        last_hidden_state = a[0]
        pooler_output = a[1]
        out, out_pool = self.bilstm(last_hidden_state, lens)
        out_pool = self.dropout(out_pool)
        output = self.fc(out_pool)
        return output


class Embedding(nn.Module):

    def __init__(self, config):
        """
        word embedding: 一般 0 为 padding
        pos embedding:  一般 0 为 padding
        dim_strategy: [cat, sum]  多个 embedding 是拼接还是相加
        """
        super(Embedding, self).__init__()
        self.vocab_size = config.vocab_size
        self.word_dim = config.word_dim
        self.pos_size = config.pos_size
        self.pos_dim = config.pos_dim if config.dim_strategy == 'cat' else config.word_dim
        self.dim_strategy = config.dim_strategy
        self.wordEmbed = nn.Embedding(self.vocab_size, self.word_dim, padding_idx=0)
        self.headPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)
        self.tailPosEmbed = nn.Embedding(self.pos_size, self.pos_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(self.word_dim)

    def forward(self, *x):
        word, head, tail = x
        word_embedding = self.wordEmbed(word)
        head_embedding = self.headPosEmbed(head)
        tail_embedding = self.tailPosEmbed(tail)
        if self.dim_strategy == 'cat':
            return torch.cat((word_embedding, head_embedding, tail_embedding), -1)
        elif self.dim_strategy == 'sum':
            return self.layer_norm(word_embedding + head_embedding + tail_embedding)
        else:
            raise Exception('dim_strategy must choose from [sum, cat]')


class BiLSTM(BasicModule):

    def __init__(self, cfg):
        super(BiLSTM, self).__init__()
        if cfg.dim_strategy == 'cat':
            cfg.input_size = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.input_size = cfg.word_dim
        self.embedding = Embedding(cfg)
        self.bilstm = RNN(cfg)
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_relations)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        inputs = self.embedding(word, head_pos, tail_pos)
        out, out_pool = self.bilstm(inputs, lens)
        output = self.fc(out_pool)
        return output


class Capsule(nn.Module):

    def __init__(self, cfg):
        super(Capsule, self).__init__()
        self.input_dim_capsule = cfg.input_dim_capsule
        self.dim_capsule = cfg.dim_capsule
        self.num_capsule = cfg.num_capsule
        self.batch_size = cfg.batch_size
        self.share_weights = cfg.share_weights
        self.num_iterations = cfg.num_iterations
        if self.share_weights:
            W = torch.zeros(1, self.input_dim_capsule, self.num_capsule * self.dim_capsule)
        else:
            W = torch.zeros(self.batch_size, self.input_dim_capsule, self.num_capsule * self.dim_capsule)
        W = nn.init.xavier_normal_(W)
        self.W = nn.Parameter(W)

    def forward(self, x):
        """
        x: [B, L, H]      # 从 CNN / RNN 得到的结果
            L 作为 input_num_capsules, H 作为 input_dim_capsule
        """
        B, I, _ = x.size()
        O, F = self.num_capsule, self.dim_capsule
        u = torch.matmul(x, self.W)
        u = u.view(B, I, O, F).transpose(1, 2)
        b = torch.zeros_like(u[:, :, :, 0])
        for i in range(self.num_iterations):
            c = torch.softmax(b, dim=1)
            v = torch.einsum('boi,boif->bof', [c, u])
            v = self.squash(v)
            b = torch.einsum('bof,boif->boi', [v, u])
        return v

    @staticmethod
    def squash(x: torch.Tensor):
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        mag = x_norm ** 2
        out = x / x_norm * mag / (1 + mag)
        return out


class GCN(nn.Module):

    def __init__(self, cfg):
        super(GCN, self).__init__()
        self.num_layers = cfg.num_layers
        self.input_size = cfg.input_size
        self.hidden_size = cfg.hidden_size
        self.dropout = cfg.dropout
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.weight_list = nn.ModuleList()
        for i in range(self.num_layers):
            self.weight_list.append(nn.Linear(self.hidden_size * (i + 1), self.hidden_size))
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, adj):
        L = adj.sum(2).unsqueeze(2) + 1
        outputs = self.fc1(x)
        cache_list = [outputs]
        output_list = []
        for l in range(self.num_layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW + self.weight_list[l](outputs)
            AxW = AxW / L
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.dropout(gAxW))
        gcn_outputs = output_list[self.num_layers - 1]
        gcn_outputs = gcn_outputs + self.fc1(x)
        out = self.fc(gcn_outputs)
        return out


class GELU(nn.Module):

    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class CNN(nn.Module):
    """
    nlp 里为了保证输出的句长 = 输入的句长，一般使用奇数 kernel_size，如 [3, 5, 7, 9]
    当然也可以不等长输出，keep_length 设为 False
    此时，padding = k // 2
    stride 一般为 1
    """

    def __init__(self, config):
        """
        in_channels      : 一般就是 word embedding 的维度，或者 hidden size 的维度
        out_channels     : int
        kernel_sizes     : list 为了保证输出长度=输入长度，必须为奇数: 3, 5, 7...
        activation       : [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]
        pooling_strategy : [max, avg, cls]
        dropout:         : float
        """
        super(CNN, self).__init__()
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.kernel_sizes = config.kernel_sizes
        self.activation = config.activation
        self.pooling_strategy = config.pooling_strategy
        self.dropout = config.dropout
        self.keep_length = config.keep_length
        for kernel_size in self.kernel_sizes:
            assert kernel_size % 2 == 1, 'kernel size has to be odd numbers.'
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=1, padding=k // 2 if self.keep_length else 0, dilation=1, groups=1, bias=False) for k in self.kernel_sizes])
        assert self.activation in ['relu', 'lrelu', 'prelu', 'selu', 'celu', 'gelu', 'sigmoid', 'tanh'], 'activation function must choose from [relu, lrelu, prelu, selu, celu, gelu, sigmoid, tanh]'
        self.activations = nn.ModuleDict([['relu', nn.ReLU()], ['lrelu', nn.LeakyReLU()], ['prelu', nn.PReLU()], ['selu', nn.SELU()], ['celu', nn.CELU()], ['gelu', GELU()], ['sigmoid', nn.Sigmoid()], ['tanh', nn.Tanh()]])
        assert self.pooling_strategy in ['max', 'avg', 'cls'], 'pooling strategy must choose from [max, avg, cls]'
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        """
            :param x: torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H] 一般是经过embedding后的值
            :param mask: [batch_size, max_len], 句长部分为0，padding部分为1。不影响卷积运算，max-pool一定不会pool到pad为0的位置
            :return:
        """
        x = torch.transpose(x, 1, 2)
        act_fn = self.activations[self.activation]
        x = [act_fn(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1)
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill_(mask, 1e-12)
        if self.pooling_strategy == 'max':
            xp = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        elif self.pooling_strategy == 'avg':
            x_len = mask.squeeze().eq(0).sum(-1).unsqueeze(-1).to(torch.float)
            xp = torch.sum(x, dim=-1) / x_len
        else:
            xp = x[:, :, 0]
        x = x.transpose(1, 2)
        x = self.dropout(x)
        xp = self.dropout(xp)
        return x, xp


class PCNN(BasicModule):

    def __init__(self, cfg):
        super(PCNN, self).__init__()
        self.use_pcnn = cfg.use_pcnn
        if cfg.dim_strategy == 'cat':
            cfg.in_channels = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.in_channels = cfg.word_dim
        self.embedding = Embedding(cfg)
        self.cnn = CNN(cfg)
        self.fc1 = nn.Linear(len(cfg.kernel_sizes) * cfg.out_channels, cfg.intermediate)
        self.fc2 = nn.Linear(cfg.intermediate, cfg.num_relations)
        self.dropout = nn.Dropout(cfg.dropout)
        if self.use_pcnn:
            self.fc_pcnn = nn.Linear(3 * len(cfg.kernel_sizes) * cfg.out_channels, len(cfg.kernel_sizes) * cfg.out_channels)
            self.pcnn_mask_embedding = nn.Embedding(4, 3)
            masks = torch.tensor([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
            self.pcnn_mask_embedding.weight.data.copy_(masks)
            self.pcnn_mask_embedding.weight.requires_grad = False

    def forward(self, x):
        word, lens, head_pos, tail_pos = x['word'], x['lens'], x['head_pos'], x['tail_pos']
        mask = seq_len_to_mask(lens)
        inputs = self.embedding(word, head_pos, tail_pos)
        out, out_pool = self.cnn(inputs, mask=mask)
        if self.use_pcnn:
            out = out.unsqueeze(-1)
            pcnn_mask = x['pcnn_mask']
            pcnn_mask = self.pcnn_mask_embedding(pcnn_mask).unsqueeze(-2)
            out = out + pcnn_mask
            out = out.max(dim=1)[0] - 100
            out_pool = out.view(out.size(0), -1)
            out_pool = F.leaky_relu(self.fc_pcnn(out_pool))
            out_pool = self.dropout(out_pool)
        output = self.fc1(out_pool)
        output = F.leaky_relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output


class DotAttention(nn.Module):

    def __init__(self, dropout=0.0):
        super(DotAttention, self).__init__()
        self.dropout = dropout

    def forward(self, Q, K, V, mask_out=None, head_mask=None):
        """
        一般输入信息 X 时，假设 K = V = X

        att_weight = softmax( score_func(q, k) )
        att = sum( att_weight * v )

        :param Q: [..., L, H]
        :param K: [..., S, H]
        :param V: [..., S, H]
        :param mask_out: [..., 1, S]
        :return:
        """
        H = Q.size(-1)
        scale = float(H) ** 0.5
        attention_weight = torch.matmul(Q, K.transpose(-1, -2)) / scale
        if mask_out is not None:
            while mask_out.dim() != Q.dim():
                mask_out = mask_out.unsqueeze(1)
            attention_weight.masked_fill_(mask_out, -100000000.0)
        attention_weight = F.softmax(attention_weight, dim=-1)
        attention_weight = F.dropout(attention_weight, self.dropout)
        if head_mask is not None:
            attention_weight = attention_weight * head_mask
        attention_out = torch.matmul(attention_weight, V)
        return attention_out, attention_weight


class MultiHeadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, output_attentions=True):
        """
        :param embed_dim: 输入的维度，必须能被 num_heads 整除
        :param num_heads: attention 的个数
        :param dropout: float。
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.output_attentions = output_attentions
        self.head_dim = int(embed_dim / num_heads)
        self.all_head_dim = self.head_dim * num_heads
        assert self.all_head_dim == embed_dim, logger.error(f'embed_dim{embed_dim} must be divisible by num_heads{num_heads}')
        self.q_in = nn.Linear(embed_dim, self.all_head_dim)
        self.k_in = nn.Linear(embed_dim, self.all_head_dim)
        self.v_in = nn.Linear(embed_dim, self.all_head_dim)
        self.attention = DotAttention(dropout=dropout)
        self.out = nn.Linear(self.all_head_dim, embed_dim)

    def forward(self, Q, K, V, key_padding_mask=None, attention_mask=None, head_mask=None):
        """
        :param Q: [B, L, Hs]
        :param K: [B, S, Hs]
        :param V: [B, S, Hs]
        :param key_padding_mask: [B, S]                为 1/True 的地方需要 mask
        :param attention_mask: [S] / [L, S] 指定位置 mask 掉， 为 1/True 的地方需要 mask
        :param head_mask: [N] 指定 head mask 掉，        为 1/True 的地方需要 mask
        """
        B, L, Hs = Q.shape
        S = V.size(1)
        N, H = self.num_heads, self.head_dim
        q = self.q_in(Q).view(B, L, N, H).transpose(1, 2)
        k = self.k_in(K).view(B, S, N, H).transpose(1, 2)
        v = self.v_in(V).view(B, S, N, H).transpose(1, 2)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.ne(0)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)
        if attention_mask is not None:
            attention_mask = attention_mask.ne(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
            else:
                raise ValueError(f'attention_mask dim must be 1 or 2, can not be {attention_mask.dim()}')
        if key_padding_mask is None:
            mask_out = attention_mask if attention_mask is not None else None
        else:
            mask_out = (key_padding_mask + attention_mask).ne(0) if attention_mask is not None else key_padding_mask
        if head_mask is not None:
            head_mask = head_mask.eq(0)
            head_mask = head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        attention_out, attention_weight = self.attention(q, k, v, mask_out=mask_out, head_mask=head_mask)
        attention_out = attention_out.transpose(1, 2).reshape(B, L, N * H)
        attention_out = self.out(attention_out)
        if self.output_attentions:
            return attention_out, attention_weight
        else:
            return attention_out,


class TransformerAttention(nn.Module):

    def __init__(self, config):
        super(TransformerAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.dropout = config.dropout
        self.output_attentions = config.output_attentions
        self.layer_norm_eps = config.layer_norm_eps
        self.multihead_attention = MultiHeadAttention(self.hidden_size, self.num_heads, self.dropout, self.output_attentions)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.layerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(self, x, key_padding_mask=None, attention_mask=None, head_mask=None):
        """
        :param x: [B, L, Hs]
        :param attention_mask: [B, L] padding后的句子后面补0了，补0的位置为True，前面部分为False
        :param head_mask: [L] [N,L]
        :return:
        """
        attention_outputs = self.multihead_attention(x, x, x, key_padding_mask, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        attention_output = self.dense(attention_output)
        attention_output = self.dropout(attention_output)
        attention_output = self.layerNorm(attention_output + x)
        outputs = (attention_output,) + attention_outputs[1:]
        return outputs


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {'gelu': gelu, 'relu': torch.nn.functional.relu, 'swish': swish, 'gelu_new': gelu_new}


class TransformerOutput(nn.Module):

    def __init__(self, config):
        super(TransformerOutput, self).__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout = config.dropout
        self.layer_norm_eps = config.layer_norm_eps
        self.zoom_in = nn.Linear(self.hidden_size, self.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]
        self.zoom_out = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout)
        self.layerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(self, input_tensor):
        hidden_states = self.zoom_in(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.zoom_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerLayer(nn.Module):

    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.attention = TransformerAttention(config)
        self.output = TransformerOutput(config)

    def forward(self, hidden_states, key_padding_mask=None, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, key_padding_mask, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        layer_output = self.output(attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class Transformer(nn.Module):

    def __init__(self, config):
        super(Transformer, self).__init__()
        self.num_hidden_layers = config.num_hidden_layers
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TransformerLayer(config) for _ in range(self.num_hidden_layers)])

    def forward(self, hidden_states, key_padding_mask=None, attention_mask=None, head_mask=None):
        """
        :param hidden_states: [B, L, Hs]
        :param key_padding_mask: [B, S]                   为 1/True 的地方需要 mask
        :param attn_mask: [S] / [L, S] 指定位置 mask 掉，   为 1/True 的地方需要 mask
        :param head_mask: [N] / [L, N] 指定 head mask 掉， 为 1/True 的地方需要 mask
        """
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.expand((self.num_hidden_layers,) + head_mask.shape)
        else:
            head_mask = [None] * self.num_hidden_layers
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, key_padding_mask, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        outputs = hidden_states,
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class GenerativeModel(nn.Module):

    def __init__(self, config, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        logger.info(f'Loading pre-trained model {config.model_name}')
        self.model_config = AutoConfig.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        self.model = AutoModelForPreTraining.from_pretrained(config.model_name, cache_dir=config.cache_dir, config=self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch):
        outputs = self.model(input_ids=batch.enc_idxs, attention_mask=batch.enc_attn, decoder_input_ids=batch.dec_idxs, decoder_attention_mask=batch.dec_attn, labels=batch.lbl_idxs, return_dict=True)
        loss = outputs['loss']
        return loss

    def predict(self, batch, num_beams=4, max_length=50):
        self.eval()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=batch.enc_idxs, attention_mask=batch.enc_attn, num_beams=num_beams, max_length=max_length)
        final_output = []
        for bid in range(len(batch.enc_idxs)):
            output_sentence = self.tokenizer.decode(outputs[bid], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_output.append(output_sentence)
        self.train()
        return final_output


class PromptBartEncoder(nn.Module):

    def __init__(self, encoder):
        super(PromptBartEncoder, self).__init__()
        self.bart_encoder = encoder

    def forward(self, src_tokens, attention_mask=None, past_key_values=None):
        encoder_dicts = self.bart_encoder(input_ids=src_tokens, attention_mask=attention_mask, past_key_values=past_key_values, return_dict=True, output_hidden_states=True)
        return encoder_dicts.last_hidden_state, encoder_dicts.hidden_states


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def _prepare_bart_decoder_inputs(pad_token_id, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    if decoder_padding_mask is not None and decoder_padding_mask.shape[1] > 1:
        decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]
    tmp = fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp
    return decoder_input_ids, decoder_padding_mask, causal_mask


class PromptBartDecoder(nn.Module):

    def __init__(self, decoder, pad_token_id, label_ids, use_prompt=False, prompt_len=10, learn_weights=False):
        super(PromptBartDecoder, self).__init__()
        self.bart_decoder = decoder
        self.pad_token_id = pad_token_id
        self.use_prompt = use_prompt
        self.prompt_len = prompt_len
        self.learn_weights = learn_weights
        self.label_ids = label_ids
        None
        if self.learn_weights:
            self.averge_weights = nn.ParameterList(parameters=None)
            for id in label_ids:
                if len(id) > 1:
                    self.averge_weights.append(nn.Parameter(torch.FloatTensor(len(id)).uniform_(1.0, 2.5)))
            None
            mapping = [0, 2]
            if self.pad_token_id == 0:
                mapping = [101, 102]
            for id in label_ids:
                mapping += id[:1]
            mapping = torch.LongTensor(mapping)
        else:
            mapping = torch.LongTensor([0, 2] + label_ids)
            if self.pad_token_id == 0:
                mapping = torch.LongTensor([101, 102] + label_ids)
            self.label_start_id = min(label_ids)
            self.label_end_id = max(label_ids) + 1
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.bart_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(0.3), nn.ReLU(), nn.Linear(hidden_size, hidden_size))
        self.dropout_layer = nn.Dropout(0.3)

    def forward(self, tgt_tokens, prompt_state):
        cumsum = tgt_tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[-1]).ne(cumsum[:, -1:])
        encoder_outputs = prompt_state.encoder_output
        attention_mask = prompt_state.encoder_mask
        first = prompt_state.first
        src_tokens = prompt_state.src_tokens
        past_key_values = prompt_state.past_key_values
        mapping_token_mask = tgt_tokens.lt(self.src_start_index)
        mapped_tokens = tgt_tokens.masked_fill(tgt_tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]
        src_tokens_index = tgt_tokens - self.src_start_index
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)
        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
        decoder_input_ids, _, causal_mask = _prepare_bart_decoder_inputs(self.pad_token_id, tokens, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=self.bart_decoder.embed_tokens.weight.dtype)
        if self.use_prompt:
            assert past_key_values is not None
            _, _, seqlen, _ = past_key_values[0]['self']['prev_value'].shape
            tgt_len = decoder_input_ids.size(1)
            temp_mask = torch.zeros(tgt_len, seqlen)
            causal_mask = torch.cat([temp_mask, causal_mask], dim=1)
        if self.training:
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)
            dict = self.bart_decoder(input_ids=tokens, encoder_hidden_states=encoder_outputs, encoder_padding_mask=attention_mask, decoder_padding_mask=decoder_pad_mask, decoder_causal_mask=causal_mask[:tokens.size(1), :self.prompt_len + tokens.size(1)], output_hidden_states=True, past_key_values=past_key_values, return_dict=True)
        else:
            past_key_values = prompt_state.past_key_values
            dict = self.bart_decoder(input_ids=tokens, encoder_hidden_states=encoder_outputs, encoder_padding_mask=attention_mask, decoder_padding_mask=None, decoder_causal_mask=None, past_key_values=past_key_values, use_cache=True, return_dict=True)
        hidden_state = dict.last_hidden_state
        hidden_state = self.dropout_layer(hidden_state)
        if not self.training:
            prompt_state.past_key_values = dict.past_key_values
        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index + src_tokens.size(-1)), fill_value=-1e+24)
        eos_scores = F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[2:3]))
        if self.learn_weights:
            tag_scores = None
            idx = 0
            for ids in self.label_ids:
                if len(ids) <= 1:
                    temp_score = F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[ids]))
                else:
                    weight = F.softmax(self.averge_weights[idx])
                    temp_score = F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[[ids[0]]])) * weight[0]
                    for i in range(1, len(ids)):
                        temp_score = temp_score + F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[[ids[i]]])) * weight[i]
                    idx += 1
                if tag_scores is None:
                    tag_scores = temp_score
                else:
                    tag_scores = torch.cat((tag_scores, temp_score), dim=2)
        else:
            tag_scores = F.linear(hidden_state, self.dropout_layer(self.bart_decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))
        src_outputs = encoder_outputs
        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)
        if first is not None:
            mask = first.eq(0)
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = attention_mask.eq(0)
        mask = mask.unsqueeze(1)
        input_embed = self.dropout_layer(self.bart_decoder.embed_tokens(src_tokens))
        src_outputs = (src_outputs + input_embed) / 2
        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        word_scores = word_scores.masked_fill(mask, -1e+32)
        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = word_scores
        return logits, prompt_state

    def decode(self, tokens, state):
        return self(tokens, state)[0][:, -1]


BART_INPUTS_DOCSTRING = """
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using :class:`~transformers.BartTokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Provide for translation and summarization training. By default, the model will create this tensor by
            shifting the :obj:`input_ids` to the right, following the paper.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.

            If you want to change padding behavior, you should read :func:`modeling_bart._prepare_decoder_inputs` and
            modify to your needs. See diagram 1 in `the paper <https://arxiv.org/abs/1910.13461>`__ for more
            information on the default strategy.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, `optional`: :obj:`hidden_states`, `optional`: :obj:`attentions`)
            :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`) is a
            sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last
            ``decoder_input_ids`` (those that don't have their past key value states given to this model) of shape
            :obj:`(batch_size, 1)` instead of all ``decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


BART_START_DOCSTRING = """

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, encoder_decoder_attention=False, cache_key=None, preseqlen=-1, use_prompt=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        assert cache_key in ['self', 'encoder_decoder', 'encoder']
        if self.encoder_decoder_attention:
            assert cache_key == 'encoder_decoder'
        self.cache_key = cache_key
        self.use_prompt = use_prompt
        self.preseqlen = preseqlen

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(self, idx, query, key: Optional[Tensor], key_padding_mask: Optional[Tensor]=None, layer_state: Optional[Dict[str, Optional[Tensor]]]=None, attn_mask: Optional[Tensor]=None, output_attentions=False) ->Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        no_extend = False
        if layer_state is not None:
            saved_state = layer_state.get(self.cache_key, {})
            use_prompt = self.use_prompt
            preseqlen = self.preseqlen
            if 'prev_key' in saved_state and static_kv and use_prompt:
                computed_len = saved_state['prev_key'].size(2)
                if computed_len > preseqlen:
                    key = None
                    no_extend = True
            elif 'prev_key' in saved_state and static_kv and not use_prompt:
                key = None
                no_extend = True
        else:
            saved_state = None
            layer_state = {}
        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)
        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)
        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, no_extend, bsz)
        layer_state[self.cache_key] = {'prev_key': k.view(bsz, self.num_heads, -1, self.head_dim), 'prev_value': v.view(bsz, self.num_heads, -1, self.head_dim), 'prev_key_padding_mask': key_padding_mask}
        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)
        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len)
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        if 'prev_key' in saved_state:
            _prev_key = saved_state['prev_key']
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if 'prev_value' in saved_state:
            _prev_value = saved_state['prev_value']
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get('prev_key_padding_mask', None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            elif key_padding_mask is not None:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
            else:
                new_key_padding_mask = None
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask


def LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)
        else:
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f'odd embedding_dim {embedding_dim} not supported')
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
        The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array([[(pos / np.power(10000, 2 * (j // 2) / dim)) for j in range(dim)] for pos in range(n_pos)])
        out[:, 0:dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, dim // 2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)
        else:
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)


_CONFIG_FOR_DOC = 'BartConfig'


_TOKENIZER_FOR_DOC = 'BartTokenizer'


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


FLAX_BASE_MODEL_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""


FLAX_CAUSAL_LM_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="np")
    >>> outputs = model(**inputs)
    >>> # retrieve logts for next token
    >>> next_token_logits = outputs.logits[:, -1]
    ```
"""


FLAX_MASKED_LM_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="jax")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""


FLAX_MULTIPLE_CHOICE_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."
    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="jax", padding=True)
    >>> outputs = model(**{{k: v[None, :] for k, v in encoding.items()}})
    >>> logits = outputs.logits
    ```
"""


FLAX_QUESTION_ANSWERING_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="jax")
    >>> outputs = model(**inputs)
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
"""


FLAX_SEQUENCE_CLASSIFICATION_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""


FLAX_TOKEN_CLASSIFICATION_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="jax")
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```
"""


FLAX_SAMPLE_DOCSTRINGS = {'SequenceClassification': FLAX_SEQUENCE_CLASSIFICATION_SAMPLE, 'QuestionAnswering': FLAX_QUESTION_ANSWERING_SAMPLE, 'TokenClassification': FLAX_TOKEN_CLASSIFICATION_SAMPLE, 'MultipleChoice': FLAX_MULTIPLE_CHOICE_SAMPLE, 'MaskedLM': FLAX_MASKED_LM_SAMPLE, 'BaseModel': FLAX_BASE_MODEL_SAMPLE, 'LMHead': FLAX_CAUSAL_LM_SAMPLE}


PT_BASE_MODEL_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import torch
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""


PT_CAUSAL_LM_SAMPLE = """
    Example:
    ```python
    >>> import torch
    >>> from transformers import {processor_class}, {model_class}
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> outputs = model(**inputs, labels=inputs["input_ids"])
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""


PT_MASKED_LM_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import torch
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="pt")
    >>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
    >>> outputs = model(**inputs, labels=labels)
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""


PT_MULTIPLE_CHOICE_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import torch
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."
    >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1
    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)
    >>> outputs = model(**{{k: v.unsqueeze(0) for k, v in encoding.items()}}, labels=labels)  # batch size is 1
    >>> # the linear classifier still needs to be trained
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""


PT_QUESTION_ANSWERING_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import torch
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="pt")
    >>> start_positions = torch.tensor([1])
    >>> end_positions = torch.tensor([3])
    >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    >>> loss = outputs.loss
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
"""


PT_SEQUENCE_CLASSIFICATION_SAMPLE = """
    Example of single-label classification:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import torch
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    >>> outputs = model(**inputs, labels=labels)
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
    Example of multi-label classification:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import torch
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", problem_type="multi_label_classification")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> labels = torch.tensor([[1, 1]], dtype=torch.float)  # need dtype=float for BCEWithLogitsLoss
    >>> outputs = model(**inputs, labels=labels)
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""


PT_SPEECH_BASE_MODEL_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import torch
    >>> from datasets import load_dataset
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate
    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    >>> list(last_hidden_states.shape)
    {expected_output}
    ```
"""


PT_SPEECH_CTC_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> from datasets import load_dataset
    >>> import torch
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate
    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)
    >>> # transcribe speech
    >>> transcription = processor.batch_decode(predicted_ids)
    >>> transcription[0]
    {expected_output}
    ```
    ```python
    >>> with processor.as_target_processor():
    ...     inputs["labels"] = processor(dataset[0]["text"], return_tensors="pt").input_ids
    >>> # compute loss
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""


PT_SPEECH_FRAME_CLASS_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> from datasets import load_dataset
    >>> import torch
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate
    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> probabilities = torch.sigmoid(logits[0])
    >>> # labels is a one-hot array of shape (num_frames, num_speakers)
    >>> labels = (probabilities > 0.5).long()
    >>> labels[0].tolist()
    {expected_output}
    ```
"""


PT_SPEECH_SEQ_CLASS_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> from datasets import load_dataset
    >>> import torch
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate
    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> with torch.no_grad():
    ...     logits = model(**inputs).logits
    >>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
    >>> predicted_label = model.config.id2label[predicted_class_ids]
    >>> predicted_label
    {expected_output}
    ```
    ```python
    >>> # compute loss - target_label is e.g. "down"
    >>> target_label = model.config.id2label[0]
    >>> inputs["labels"] = torch.tensor([model.config.label2id[target_label]])
    >>> loss = model(**inputs).loss
    >>> round(loss.item(), 2)
    {expected_loss}
    ```
"""


PT_SPEECH_XVECTOR_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> from datasets import load_dataset
    >>> import torch
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate
    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(
    ...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ... )
    >>> with torch.no_grad():
    ...     embeddings = model(**inputs).embeddings
    >>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    >>> # the resulting embeddings can be used for cosine similarity-based retrieval
    >>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    >>> similarity = cosine_sim(embeddings[0], embeddings[1])
    >>> threshold = 0.7  # the optimal threshold is dataset-dependent
    >>> if similarity < threshold:
    ...     print("Speakers are not the same!")
    >>> round(similarity.item(), 2)
    {expected_output}
    ```
"""


PT_TOKEN_CLASSIFICATION_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import torch
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    >>> labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
    >>> outputs = model(**inputs, labels=labels)
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""


PT_SAMPLE_DOCSTRINGS = {'SequenceClassification': PT_SEQUENCE_CLASSIFICATION_SAMPLE, 'QuestionAnswering': PT_QUESTION_ANSWERING_SAMPLE, 'TokenClassification': PT_TOKEN_CLASSIFICATION_SAMPLE, 'MultipleChoice': PT_MULTIPLE_CHOICE_SAMPLE, 'MaskedLM': PT_MASKED_LM_SAMPLE, 'LMHead': PT_CAUSAL_LM_SAMPLE, 'BaseModel': PT_BASE_MODEL_SAMPLE, 'SpeechBaseModel': PT_SPEECH_BASE_MODEL_SAMPLE, 'CTC': PT_SPEECH_CTC_SAMPLE, 'AudioClassification': PT_SPEECH_SEQ_CLASS_SAMPLE, 'AudioFrameClassification': PT_SPEECH_FRAME_CLASS_SAMPLE, 'AudioXVector': PT_SPEECH_XVECTOR_SAMPLE}


TF_BASE_MODEL_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import tensorflow as tf
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)
    >>> last_hidden_states = outputs.last_hidden_state
    ```
"""


TF_CAUSAL_LM_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import tensorflow as tf
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> outputs = model(inputs)
    >>> logits = outputs.logits
    ```
"""


TF_MASKED_LM_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import tensorflow as tf
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("The capital of France is {mask}.", return_tensors="tf")
    >>> inputs["labels"] = tokenizer("The capital of France is Paris.", return_tensors="tf")["input_ids"]
    >>> outputs = model(inputs)
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""


TF_MULTIPLE_CHOICE_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import tensorflow as tf
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
    >>> choice0 = "It is eaten with a fork and a knife."
    >>> choice1 = "It is eaten while held in the hand."
    >>> encoding = tokenizer([prompt, prompt], [choice0, choice1], return_tensors="tf", padding=True)
    >>> inputs = {{k: tf.expand_dims(v, 0) for k, v in encoding.items()}}
    >>> outputs = model(inputs)  # batch size is 1
    >>> # the linear classifier still needs to be trained
    >>> logits = outputs.logits
    ```
"""


TF_QUESTION_ANSWERING_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import tensorflow as tf
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> input_dict = tokenizer(question, text, return_tensors="tf")
    >>> outputs = model(input_dict)
    >>> start_logits = outputs.start_logits
    >>> end_logits = outputs.end_logits
    >>> all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
    >>> answer = " ".join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0] + 1])
    ```
"""


TF_SEQUENCE_CLASSIFICATION_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import tensorflow as tf
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> inputs["labels"] = tf.reshape(tf.constant(1), (-1, 1))  # Batch size 1
    >>> outputs = model(inputs)
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""


TF_TOKEN_CLASSIFICATION_SAMPLE = """
    Example:
    ```python
    >>> from transformers import {processor_class}, {model_class}
    >>> import tensorflow as tf
    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
    >>> input_ids = inputs["input_ids"]
    >>> inputs["labels"] = tf.reshape(
    ...     tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids))
    >>> )  # Batch size 1
    >>> outputs = model(inputs)
    >>> loss = outputs.loss
    >>> logits = outputs.logits
    ```
"""


TF_SAMPLE_DOCSTRINGS = {'SequenceClassification': TF_SEQUENCE_CLASSIFICATION_SAMPLE, 'QuestionAnswering': TF_QUESTION_ANSWERING_SAMPLE, 'TokenClassification': TF_TOKEN_CLASSIFICATION_SAMPLE, 'MultipleChoice': TF_MULTIPLE_CHOICE_SAMPLE, 'MaskedLM': TF_MASKED_LM_SAMPLE, 'LMHead': TF_CAUSAL_LM_SAMPLE, 'BaseModel': TF_BASE_MODEL_SAMPLE}


PT_RETURN_INTRODUCTION = """
    Returns:
        [`{full_output_type}`] or `tuple(torch.FloatTensor)`: A [`{full_output_type}`] or a tuple of
        `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
        elements depending on the configuration ([`{config_class}`]) and inputs.
"""


TF_RETURN_INTRODUCTION = """
    Returns:
        [`{full_output_type}`] or `tuple(tf.Tensor)`: A [`{full_output_type}`] or a tuple of `tf.Tensor` (if
        `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the
        configuration ([`{config_class}`]) and inputs.
"""


def _get_indent(t):
    """Returns the indentation in the first line of t"""
    search = re.search('^(\\s*)\\S', t)
    return '' if search is None else search.groups()[0]


def _convert_output_args_doc(output_args_doc):
    """Convert output_args_doc to display properly."""
    indent = _get_indent(output_args_doc)
    blocks = []
    current_block = ''
    for line in output_args_doc.split('\n'):
        if _get_indent(line) == indent:
            if len(current_block) > 0:
                blocks.append(current_block[:-1])
            current_block = f'{line}\n'
        else:
            current_block += f'{line[2:]}\n'
    blocks.append(current_block[:-1])
    for i in range(len(blocks)):
        blocks[i] = re.sub('^(\\s+)(\\S+)(\\s+)', '\\1- **\\2**\\3', blocks[i])
        blocks[i] = re.sub(':\\s*\\n\\s*(\\S)', ' -- \\1', blocks[i])
    return '\n'.join(blocks)


def _prepare_output_docstrings(output_type, config_class, min_indent=None):
    """
    Prepares the return part of the docstring using `output_type`.
    """
    output_docstring = output_type.__doc__
    lines = output_docstring.split('\n')
    i = 0
    while i < len(lines) and re.search('^\\s*(Args|Parameters):\\s*$', lines[i]) is None:
        i += 1
    if i < len(lines):
        params_docstring = '\n'.join(lines[i + 1:])
        params_docstring = _convert_output_args_doc(params_docstring)
    full_output_type = f'{output_type.__module__}.{output_type.__name__}'
    intro = TF_RETURN_INTRODUCTION if output_type.__name__.startswith('TF') else PT_RETURN_INTRODUCTION
    intro = intro.format(full_output_type=full_output_type, config_class=config_class)
    result = intro + params_docstring
    if min_indent is not None:
        lines = result.split('\n')
        i = 0
        while len(lines[i]) == 0:
            i += 1
        indent = len(_get_indent(lines[i]))
        if indent < min_indent:
            to_add = ' ' * (min_indent - indent)
            lines = [(f'{to_add}{line}' if len(line) > 0 else line) for line in lines]
            result = '\n'.join(lines)
    return result


def add_code_sample_docstrings(*docstr, processor_class=None, checkpoint=None, output_type=None, config_class=None, mask='[MASK]', model_cls=None, modality=None, expected_output='', expected_loss=''):

    def docstring_decorator(fn):
        model_class = fn.__qualname__.split('.')[0] if model_cls is None else model_cls
        if model_class[:2] == 'TF':
            sample_docstrings = TF_SAMPLE_DOCSTRINGS
        elif model_class[:4] == 'Flax':
            sample_docstrings = FLAX_SAMPLE_DOCSTRINGS
        else:
            sample_docstrings = PT_SAMPLE_DOCSTRINGS
        doc_kwargs = dict(model_class=model_class, processor_class=processor_class, checkpoint=checkpoint, mask=mask, expected_output=expected_output, expected_loss=expected_loss)
        if 'SequenceClassification' in model_class and modality == 'audio':
            code_sample = sample_docstrings['AudioClassification']
        elif 'SequenceClassification' in model_class:
            code_sample = sample_docstrings['SequenceClassification']
        elif 'QuestionAnswering' in model_class:
            code_sample = sample_docstrings['QuestionAnswering']
        elif 'TokenClassification' in model_class:
            code_sample = sample_docstrings['TokenClassification']
        elif 'MultipleChoice' in model_class:
            code_sample = sample_docstrings['MultipleChoice']
        elif 'MaskedLM' in model_class or model_class in ['FlaubertWithLMHeadModel', 'XLMWithLMHeadModel']:
            code_sample = sample_docstrings['MaskedLM']
        elif 'LMHead' in model_class or 'CausalLM' in model_class:
            code_sample = sample_docstrings['LMHead']
        elif 'CTC' in model_class:
            code_sample = sample_docstrings['CTC']
        elif 'AudioFrameClassification' in model_class:
            code_sample = sample_docstrings['AudioFrameClassification']
        elif 'XVector' in model_class and modality == 'audio':
            code_sample = sample_docstrings['AudioXVector']
        elif 'Model' in model_class and modality == 'audio':
            code_sample = sample_docstrings['SpeechBaseModel']
        elif 'Model' in model_class or 'Encoder' in model_class:
            code_sample = sample_docstrings['BaseModel']
        else:
            raise ValueError(f"Docstring can't be built for model {model_class}")
        func_doc = (fn.__doc__ or '') + ''.join(docstr)
        output_doc = '' if output_type is None else _prepare_output_docstrings(output_type, config_class)
        built_doc = code_sample.format(**doc_kwargs)
        fn.__doc__ = func_doc + output_doc + built_doc
        return fn
    return docstring_decorator


def add_start_docstrings(*docstr):

    def docstring_decorator(fn):
        fn.__doc__ = ''.join(docstr) + (fn.__doc__ if fn.__doc__ is not None else '')
        return fn
    return docstring_decorator


class PromptBartState(object):

    def __init__(self, encoder_output, encoder_mask, past_key_values, src_tokens, first, src_embed_outputs, preseqlen):
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self.past_key_values = past_key_values
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs
        self.preseqlen = preseqlen

    def _reorder_state(self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int=0):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f'Cannot reorder data of type:{type(state)}')
        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new

    def num_samples(self):
        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None


def seq_to_mask(seq_len, max_len):
    """[get attention mask with sequence length]

    Args:
        seq_len ([torch.tensor]): [shape: bsz, each sequence length in a batch]
    """
    max_len = int(max_len) if max_len else seq_len.max().long()
    cast_seq = torch.arange(max_len).expand(seq_len.size(0), -1)
    mask = cast_seq.lt(seq_len.unsqueeze(1))
    return mask


class PromptBartModel(nn.Module):

    def __init__(self, tokenizer, label_ids, args):
        super(PromptBartModel, self).__init__()
        self.use_prompt = args.use_prompt
        self.prompt_len = args.prompt_len
        self.prompt_dim = args.prompt_dim
        self.learn_weights = args.learn_weights
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        bart_name = args.bart_name
        self.bart_config = BartConfig.from_pretrained(bart_name)
        self.bart_config.use_prompt = args.use_prompt
        self.bart_config.preseqlen = args.prompt_len
        bart_config = self.bart_config
        bart_model = BartModel.from_pretrained(bart_name, config=bart_config)
        num_tokens, _ = bart_model.encoder.embed_tokens.weight.shape
        bart_model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens) + num_tokens)
        bart_model = avg_token_embeddings(tokenizer, bart_model, bart_name, num_tokens)
        self.prompt_encoder = PromptBartEncoder(bart_model.encoder)
        self.prompt_decoder = PromptBartDecoder(bart_model.decoder, tokenizer.pad_token_id, label_ids, self.use_prompt, self.prompt_len, self.learn_weights)
        self.prompt_inputs = torch.arange(self.prompt_len).long()
        self.encoder_prompt_embed = nn.Embedding(self.prompt_len, bart_config.d_model)
        self.encoder_mlp = nn.Sequential(nn.Linear(bart_config.d_model, self.prompt_dim), nn.Tanh(), nn.Linear(self.prompt_dim, bart_config.decoder_layers * 2 * bart_config.d_model))
        self.decoder_prompt_embed = nn.Embedding(self.prompt_len, bart_config.d_model)
        self.decoder_mlp = nn.Sequential(nn.Linear(bart_config.d_model, self.prompt_dim), nn.Tanh(), nn.Linear(self.prompt_dim, bart_config.decoder_layers * 2 * bart_config.d_model))
        self.prompt_cross_embed = nn.Embedding(self.prompt_len, bart_config.d_model)
        self.cross_mlp = nn.Sequential(nn.Linear(bart_config.d_model, self.prompt_dim), nn.Tanh(), nn.Linear(self.prompt_dim, bart_config.decoder_layers * 2 * bart_config.d_model))
        self.dropout = nn.Dropout(0.0)

    def forward(self, src_tokens, tgt_tokens, src_seq_len, first):
        prompt_state = self.generator(src_tokens, src_seq_len, first)
        decoder_outputs, prompt_state = self.prompt_decoder(tgt_tokens, prompt_state)
        return decoder_outputs

    def generator(self, src_tokens, src_seq_len, first):
        batch_size = src_tokens.size(0)
        past_key_values = self.get_prompt(batch_size) if self.use_prompt else None
        attention_mask = seq_to_mask(src_seq_len, max_len=src_tokens.size(1))
        encoder_outputs, hidden_states = self.prompt_encoder(src_tokens, attention_mask=attention_mask, past_key_values=past_key_values)
        prompt_state = PromptBartState(encoder_outputs, attention_mask, past_key_values, src_tokens, first, hidden_states[0], self.bart_config.preseqlen)
        return prompt_state

    def get_prompt(self, batch_size):
        input_tokens = self.prompt_inputs.unsqueeze(0).expand(batch_size, -1)
        encoder_embed = self.encoder_prompt_embed(input_tokens)
        past_key_values = self.encoder_mlp(encoder_embed)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.bart_config.decoder_layers * 2, self.bart_config.decoder_attention_heads, self.bart_config.d_model // self.bart_config.decoder_attention_heads)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        decoder_embed = self.decoder_prompt_embed(input_tokens)
        past_key_values2 = self.decoder_mlp(decoder_embed)
        past_key_values2 = past_key_values2.view(bsz, seqlen, self.bart_config.decoder_layers * 2, self.bart_config.decoder_attention_heads, self.bart_config.d_model // self.bart_config.decoder_attention_heads)
        past_key_values2 = self.dropout(past_key_values2)
        past_key_values2 = past_key_values2.permute([2, 0, 3, 1, 4]).split(2)
        cross_embed = self.prompt_cross_embed(input_tokens)
        past_key_values_enc = self.cross_mlp(cross_embed)
        past_key_values_enc = past_key_values_enc.view(bsz, seqlen, self.bart_config.decoder_layers * 2, self.bart_config.decoder_attention_heads, self.bart_config.d_model // self.bart_config.decoder_attention_heads)
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)
        result = []
        for i, key_val in enumerate(past_key_values):
            temp_dict = {'self': {'prev_key': key_val[0].contiguous(), 'prev_value': key_val[1].contiguous(), 'prev_key_padding_mask': torch.zeros(bsz, seqlen).bool()}}
            key_val2 = past_key_values2[i]
            temp_dict['encoder_decoder'] = {'prev_key': key_val2[0].contiguous(), 'prev_value': key_val2[1].contiguous(), 'prev_key_padding_mask': torch.zeros(bsz, seqlen).bool()}
            key_val_enc = past_key_values_enc[i]
            temp_dict['encoder'] = {'prev_key': key_val_enc[0].contiguous(), 'prev_value': key_val_enc[1].contiguous(), 'prev_key_padding_mask': torch.zeros(bsz, seqlen).bool()}
            result.append(temp_dict)
        return result


class BeamHypotheses(object):

    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.hyp = []
        self.worst_score = 1000000000.0

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.hyp)])
                del self.hyp[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / self.max_length ** self.length_penalty


def get_model_device(model):
    assert isinstance(model, nn.Module)
    parameters = list(model.parameters())
    if len(parameters) == 0:
        return None
    else:
        return parameters[0].device


def _beam_search_generate(decoder: PromptBartDecoder, tokens=None, state=None, max_length=20, max_len_a=0.0, num_beams=4, bos_token_id=None, eos_token_id=None, do_sample=True, repetition_penalty=1.0, length_penalty=None, pad_token_id=0, restricter=None) ->torch.LongTensor:
    assert do_sample is False
    device = get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError('You have to specify either `tokens` or `bos_token_id`.')
        batch_size = state.num_samples()
        if batch_size is None:
            raise RuntimeError('Cannot infer the number of samples from `state`.')
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long)
    batch_size = tokens.size(0)
    if state.num_samples:
        assert state.num_samples() == batch_size, 'The number of samples in `tokens` and `state` should match.'
    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id
    scores = decoder.decode(tokens=tokens, state=state)
    vocab_size = scores.size(1)
    assert vocab_size >= num_beams, 'num_beams should be smaller than the number of vocabulary size.'
    scores = F.log_softmax(scores, dim=-1)
    if restricter is not None:
        _next_scores, _next_tokens = restricter(state, tokens, scores, num_beams + 1)
    else:
        _next_scores, _next_tokens = torch.topk(scores, num_beams + 1, dim=1, largest=True, sorted=True)
    indices = torch.arange(batch_size, dtype=torch.long)
    indices = indices.repeat_interleave(num_beams)
    state.reorder_state(indices)
    tokens = tokens.index_select(dim=0, index=indices)
    if max_len_a != 0:
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float() * max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((batch_size * num_beams,), fill_value=max_length, dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long() * max_length
        else:
            max_lengths = tokens.new_full((batch_size * num_beams,), fill_value=max_length, dtype=torch.long)
    hypos = [BeamHypotheses(num_beams, real_max_length, length_penalty, early_stopping=False) for _ in range(batch_size)]
    not_eos_mask = _next_tokens.ne(_eos_token_id)
    keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)
    keep_mask = not_eos_mask.__and__(keep_mask)
    next_tokens = _next_tokens.masked_select(keep_mask).view(batch_size, num_beams)
    next_scores = _next_scores.masked_select(keep_mask).view(batch_size, num_beams)
    rows, cols = not_eos_mask.eq(0)[:, :num_beams].nonzero(as_tuple=True)
    if len(rows) > 0:
        for row, col in zip(rows.tolist(), cols.tolist()):
            _token = torch.cat([tokens[row * num_beams], _next_tokens[row, col:col + 1]], dim=0)
            hypos[row].add(_token.clone(), _next_scores[row, col].item())
    token_ids = torch.cat([tokens, next_tokens.view(-1, 1)], dim=-1)
    dones = [False] * batch_size
    beam_scores = next_scores.view(-1)
    cur_len = token_ids.size(1)
    batch_inds_with_numbeams_interval = (torch.arange(batch_size) * num_beams).view(-1, 1)
    while cur_len < real_max_length:
        scores = decoder.decode(token_ids, state)
        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)
        if _eos_token_id != -1:
            max_len_eos_mask = max_lengths.eq(cur_len + 1)
            eos_scores = scores[:, _eos_token_id]
            scores[:, _eos_token_id] = torch.where(max_len_eos_mask, eos_scores + 1e+32, eos_scores)
        scores = F.log_softmax(scores, dim=-1)
        _scores = scores + beam_scores[:, None]
        _scores = _scores.view(batch_size, -1)
        if restricter is not None:
            next_scores, ids = restricter(state, token_ids, _scores, 2 * num_beams)
        else:
            next_scores, ids = torch.topk(_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        from_which_beam = ids // vocab_size
        next_tokens = ids % vocab_size
        not_eos_mask = next_tokens.ne(_eos_token_id)
        keep_mask = not_eos_mask.cumsum(dim=1).le(num_beams)
        keep_mask = not_eos_mask.__and__(keep_mask)
        _next_tokens = next_tokens.masked_select(keep_mask).view(-1, 1)
        _from_which_beam = from_which_beam.masked_select(keep_mask).view(batch_size, num_beams)
        _next_scores = next_scores.masked_select(keep_mask).view(batch_size, num_beams)
        beam_scores = _next_scores.view(-1)
        flag = True
        if cur_len + 1 == real_max_length:
            eos_batch_idx = torch.arange(batch_size).repeat_interleave(repeats=num_beams, dim=0)
            eos_beam_ind = torch.arange(num_beams).repeat(batch_size)
            eos_beam_idx = from_which_beam[:, :num_beams].reshape(-1)
        else:
            effective_eos_mask = next_tokens[:, :num_beams].eq(_eos_token_id)
            if effective_eos_mask.sum().gt(0):
                eos_batch_idx, eos_beam_ind = effective_eos_mask.nonzero(as_tuple=True)
                eos_beam_idx = eos_batch_idx * num_beams * 2 + eos_beam_ind
                eos_beam_idx = from_which_beam.view(-1)[eos_beam_idx]
            else:
                flag = False
        if flag:
            _token_ids = torch.cat([token_ids, _next_tokens], dim=-1)
            for batch_idx, beam_ind, beam_idx in zip(eos_batch_idx.tolist(), eos_beam_ind.tolist(), eos_beam_idx.tolist()):
                if not dones[batch_idx]:
                    score = next_scores[batch_idx, beam_ind].item()
                    if _eos_token_id != -1:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx, :cur_len].clone(), score)
                    else:
                        hypos[batch_idx].add(_token_ids[batch_idx * num_beams + beam_idx].clone(), score)
        reorder_inds = (batch_inds_with_numbeams_interval + _from_which_beam).view(-1)
        state.reorder_state(reorder_inds)
        token_ids = torch.cat([token_ids.index_select(index=reorder_inds, dim=0), _next_tokens], dim=-1)
        for batch_idx in range(batch_size):
            dones[batch_idx] = dones[batch_idx] or hypos[batch_idx].is_done(next_scores[batch_idx, 0].item()) or max_lengths[batch_idx * num_beams] == cur_len + 1
        cur_len += 1
        if all(dones):
            break
    tgt_len = token_ids.new_zeros(batch_size)
    best = []
    for i, hypotheses in enumerate(hypos):
        best_hyp = max(hypotheses.hyp, key=lambda x: x[0])[1]
        if _eos_token_id != -1:
            best_hyp = torch.cat([best_hyp, best_hyp.new_ones(1) * _eos_token_id])
        tgt_len[i] = len(best_hyp)
        best.append(best_hyp)
    decoded = token_ids.new_zeros(batch_size, tgt_len.max().item()).fill_(pad_token_id)
    for i, hypo in enumerate(best):
        decoded[i, :tgt_len[i]] = hypo
    return decoded


def _no_beam_search_generate(decoder: PromptBartDecoder, state, tokens=None, max_length=20, max_len_a=0.0, bos_token_id=None, eos_token_id=None, repetition_penalty=1.0, length_penalty=1.0, pad_token_id=0, restricter=None):
    device = get_model_device(decoder)
    if tokens is None:
        if bos_token_id is None:
            raise RuntimeError('You have to specify either `tokens` or `bos_token_id`.')
        batch_size = state.num_samples()
        if batch_size is None:
            raise RuntimeError('Cannot infer the number of samples from `state`.')
        tokens = torch.full([batch_size, 1], fill_value=bos_token_id, dtype=torch.long)
    batch_size = tokens.size(0)
    if state.num_samples:
        assert state.num_samples() == batch_size, 'The number of samples in `tokens` and `state` should match.'
    if eos_token_id is None:
        _eos_token_id = -1
    else:
        _eos_token_id = eos_token_id
    scores = decoder.decode(tokens=tokens, state=state)
    if restricter is not None:
        _, next_tokens = restricter(state, tokens, scores, num_beams=1)
    else:
        next_tokens = scores.argmax(dim=-1, keepdim=True)
    token_ids = torch.cat([tokens, next_tokens], dim=1)
    cur_len = token_ids.size(1)
    dones = token_ids.new_zeros(batch_size).eq(1).__or__(next_tokens.squeeze(1).eq(eos_token_id))
    if max_len_a != 0:
        if state.encoder_mask is not None:
            max_lengths = (state.encoder_mask.sum(dim=1).float() * max_len_a).long() + max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)
        real_max_length = max_lengths.max().item()
    else:
        real_max_length = max_length
        if state.encoder_mask is not None:
            max_lengths = state.encoder_mask.new_ones(state.encoder_mask.size(0)).long() * max_length
        else:
            max_lengths = tokens.new_full((tokens.size(0),), fill_value=max_length, dtype=torch.long)
    while cur_len < real_max_length:
        scores = decoder.decode(tokens=token_ids, state=state)
        if repetition_penalty != 1.0:
            token_scores = scores.gather(dim=1, index=token_ids)
            lt_zero_mask = token_scores.lt(0).float()
            ge_zero_mask = lt_zero_mask.eq(0).float()
            token_scores = lt_zero_mask * repetition_penalty * token_scores + ge_zero_mask / repetition_penalty * token_scores
            scores.scatter_(dim=1, index=token_ids, src=token_scores)
        if eos_token_id is not None and length_penalty != 1.0:
            token_scores = scores / cur_len ** length_penalty
            eos_mask = scores.new_ones(scores.size(1))
            eos_mask[eos_token_id] = 0
            eos_mask = eos_mask.unsqueeze(0).eq(1)
            scores = scores.masked_scatter(eos_mask, token_scores)
        if restricter is not None:
            _, next_tokens = restricter(state, token_ids, scores, 1)
        else:
            next_tokens = scores.argmax(dim=-1, keepdim=True)
        next_tokens = next_tokens.squeeze(-1)
        if _eos_token_id != -1:
            next_tokens = next_tokens.masked_fill(max_lengths.eq(cur_len + 1), _eos_token_id)
        next_tokens = next_tokens.masked_fill(dones, pad_token_id)
        tokens = next_tokens.unsqueeze(1)
        token_ids = torch.cat([token_ids, tokens], dim=-1)
        end_mask = next_tokens.eq(_eos_token_id)
        dones = dones.__or__(end_mask)
        cur_len += 1
        if dones.min() == 1:
            break
    return token_ids


@torch.no_grad()
def greedy_generate(decoder, tokens=None, state=None, max_length=20, max_len_a=0.0, num_beams=1, bos_token_id=None, eos_token_id=None, pad_token_id=0, repetition_penalty=1, length_penalty=1.0, restricter=None):
    if num_beams == 1:
        token_ids = _no_beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a, bos_token_id=bos_token_id, eos_token_id=eos_token_id, repetition_penalty=repetition_penalty, length_penalty=length_penalty, pad_token_id=pad_token_id, restricter=restricter)
    else:
        token_ids = _beam_search_generate(decoder, tokens=tokens, state=state, max_length=max_length, max_len_a=max_len_a, num_beams=num_beams, bos_token_id=bos_token_id, eos_token_id=eos_token_id, do_sample=False, repetition_penalty=repetition_penalty, length_penalty=length_penalty, pad_token_id=pad_token_id, restricter=restricter)
    return token_ids


class PromptGeneratorModel(nn.Module):

    def __init__(self, prompt_model, max_length=20, max_len_a=0.0, num_beams=1, do_sample=False, bos_token_id=None, eos_token_id=None, repetition_penalty=1, length_penalty=1.0, pad_token_id=0, restricter=None):
        super(PromptGeneratorModel, self).__init__()
        self.prompt_model = prompt_model
        self.decoder = prompt_model.prompt_decoder
        self.generate_func = partial(greedy_generate, decoder=self.decoder, max_length=max_length, max_len_a=max_len_a, num_beams=num_beams, bos_token_id=bos_token_id, eos_token_id=eos_token_id, repetition_penalty=repetition_penalty, length_penalty=length_penalty, pad_token_id=pad_token_id, restricter=restricter)
        self.do_sample = do_sample
        self.max_length = max_length
        self.num_beams = num_beams
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.pad_token_id = pad_token_id
        self.restricter = restricter
        self.max_len_a = max_len_a

    def forward(self, src_tokens, tgt_tokens, src_seq_len=None, tgt_seq_len=None, first=None):
        """
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor tgt_tokens: bsz x max_len'
        :param torch.LongTensor src_seq_len: bsz
        :param torch.LongTensor tgt_seq_len: bsz
        :return:
        """
        return self.prompt_model(src_tokens, tgt_tokens, src_seq_len, first)

    def predict(self, src_tokens, src_seq_len=None, first=None):
        """
        :param torch.LongTensor src_tokens: bsz x max_len
        :param torch.LongTensor src_seq_len: bsz
        :return:
        """
        prompt_state = self.prompt_model.generator(src_tokens, src_seq_len, first)
        result = self.generate_func(tokens=None, state=prompt_state)
        return result


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def is_flax_available():
    return _flax_available


def is_tf_available():
    return _tf_available


def is_torch_available():
    return _torch_available


_torch_fx_available = _torch_onnx_dict_inputs_support_available = False


def is_torch_fx_available():
    return _torch_fx_available


def is_torch_fx_proxy(x):
    if is_torch_fx_available():
        import torch.fx
        return isinstance(x, torch.fx.Proxy)
    return False


def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray` or `np.ndarray`.
    """
    if is_torch_fx_proxy(x):
        return True
    if is_torch_available():
        import torch
        if isinstance(x, torch.Tensor):
            return True
    if is_tf_available():
        import tensorflow as tf
        if isinstance(x, tf.Tensor):
            return True
    if is_flax_available():
        if isinstance(x, (jnp.ndarray, Tracer)):
            return True
    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.
    <Tip warning={true}>
    You can't unpack a `ModelOutput` directly. Use the [`~file_utils.ModelOutput.to_tuple`] method to convert it to a
    tuple before.
    </Tip>
    """

    def __post_init__(self):
        class_fields = fields(self)
        if not len(class_fields):
            raise ValueError(f'{self.__class__.__name__} has no fields.')
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f'{self.__class__.__name__} should not have more than one required field.')
        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])
        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False
            if first_field_iterator:
                for element in iterator:
                    if not isinstance(element, (list, tuple)) or not len(element) == 2 or not isinstance(element[0], str):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f'You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.')

    def setdefault(self, *args, **kwargs):
        raise Exception(f'You cannot use ``setdefault`` on a {self.__class__.__name__} instance.')

    def pop(self, *args, **kwargs):
        raise Exception(f'You cannot use ``pop`` on a {self.__class__.__name__} instance.')

    def update(self, *args, **kwargs):
        raise Exception(f'You cannot use ``update`` on a {self.__class__.__name__} instance.')

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for k, v in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) ->Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert self.head_dim * self.num_heads == self.embed_dim, f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads}).'
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False, past_key_values: torch.Tensor=None, current_layer: int=None, output_qks=None) ->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        qks = (key_states, value_states) if output_qks else None
        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)
        proj_shape = bsz * self.num_heads, -1, self.head_dim
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(f'Attention weights should be of size {bsz * self.num_heads, tgt_len, src_len}, but is {attn_weights.size()}')
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(f'`attn_output` should be of size {bsz, self.num_heads, tgt_len, self.head_dim}, but is {attn_output.size()}')
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped, qks


class CLIPBaseModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    qks: Optional[Tuple[torch.FloatTensor]] = None


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


class CLIPMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = quick_gelu
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False, past_key_values: torch.Tensor=None, current_layer: int=None, output_qks=None):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, qks = self.self_attn(hidden_states=hidden_states, output_attentions=output_attentions, past_key_values=past_key_values, output_qks=output_qks, current_layer=current_layer)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = hidden_states,
        if output_attentions:
            outputs += attn_weights,
        if output_qks:
            outputs += qks,
        return outputs


class CLIPVisionEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)))
        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer('aux_position_ids', torch.arange(48).expand((1, -1)))
        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer('rcnn_position_ids', torch.arange(12).expand((1, -1)))

    def forward(self, pixel_values, aux_embeddings=None, rcnn_embeddings=None):
        batch_size = pixel_values.shape[0]
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = class_embeds
        if aux_embeddings is not None:
            aux_embeds = []
            for aux_embedding in aux_embeddings:
                aux_embed = self.patch_embedding(aux_embedding)
                aux_embed = aux_embed.flatten(2).transpose(1, 2).flatten(0, 1)
                aux_embeds.append(aux_embed)
            aux_embeds = torch.stack(aux_embeds)
            aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
            embeddings = torch.cat((embeddings, aux_embeds), dim=1)
        if rcnn_embeddings is not None:
            rcnn_embeds = []
            for rcnn_embedding in rcnn_embeddings:
                rcnn_embed = self.patch_embedding(rcnn_embedding)
                rcnn_embed = rcnn_embed.flatten(2).transpose(1, 2).flatten(0, 1)
                rcnn_embeds.append(rcnn_embed)
            rcnn_embeds = torch.stack(rcnn_embeds)
            rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
            embeddings = torch.cat((embeddings, rcnn_embeds), dim=1)
        return embeddings


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int]=None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def replace_return_docstrings(output_type=None, config_class=None):

    def docstring_decorator(fn):
        func_doc = fn.__doc__
        lines = func_doc.split('\n')
        i = 0
        while i < len(lines) and re.search('^\\s*Returns?:\\s*$', lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = len(_get_indent(lines[i]))
            lines[i] = _prepare_output_docstrings(output_type, config_class, min_indent=indent)
            func_doc = '\n'.join(lines)
        else:
            raise ValueError(f"The function {fn} should have an empty 'Return:' or 'Returns:' in its docstring as placeholder, current docstring is:\n{func_doc}")
        fn.__doc__ = func_doc
        return fn
    return docstring_decorator


class CLIPBaseModelOutputWithPooling(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    qks: Optional[Tuple[torch.FloatTensor]] = None


CLIP_START_DOCSTRING = """
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.CLIPConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""


def contrastive_loss(logits: torch.Tensor) ->torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) ->torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, 'position_embedding_type', 'absolute')
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:seq_length + past_key_values_length]
        if token_type_ids is None:
            if hasattr(self, 'token_type_ids'):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertPooler(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, visual_hidden_state=None, output_qks=None, current_layer=None, past_key_values=None):
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        qks = (key_layer, value_layer) if output_qks else None
        if past_key_values is not None:
            key_layer = torch.cat([past_key_values[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_values[0], value_layer], dim=2)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            bsz, nheads, length, dsize = past_key_values[0].size()
            visual_attention_mask = torch.ones((bsz, 1, 1, length))
            attention_mask = torch.cat((visual_attention_mask, attention_mask), dim=-1)
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs, qks


class BertSelfOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, visual_hidden_state=None, output_qks=None, current_layer=None, past_key_values=None):
        self_outputs, qks = self.self(hidden_states, attention_mask, head_mask, output_attentions, visual_hidden_state, output_qks, current_layer, past_key_values)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs, qks


class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.add_cross_attention = config.add_cross_attention
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, visual_hidden_state=None, output_qks=None, current_layer=None, past_key_values=None):
        self_attention_outputs, qks = self.attention(hidden_states, attention_mask, head_mask, output_attentions=output_attentions, visual_hidden_state=visual_hidden_state, output_qks=output_qks, current_layer=current_layer, past_key_values=past_key_values)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output)
        outputs = (layer_output,) + outputs
        if output_qks:
            outputs += qks,
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class IFAEncoder(nn.Module):

    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config
        self.vision_layers = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.text_layer = nn.ModuleList([BertLayer(text_config) for _ in range(text_config.num_hidden_layers)])

    def forward(self, vision_embeds=None, text_embeds=None, attention_mask=None, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers
        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None
        vision_hidden_states = vision_embeds
        text_hidden_states = text_embeds
        for idx in range(self.vision_config.num_hidden_layers):
            if output_hidden_states:
                all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states,)
                all_text_hidden_states = all_text_hidden_states + (text_hidden_states,)
            output_qks = True
            if idx == 0:
                bsz, length, dsize = text_embeds.size()
                visual_past_key_values = text_embeds.view(bsz, 12, length, dsize // 12), text_embeds.view(bsz, 12, length, dsize // 12)
            else:
                visual_past_key_values = text_layer_output[-1]
            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(vision_hidden_states, output_attentions=output_attentions, past_key_values=visual_past_key_values, current_layer=idx, output_qks=output_qks)
            vision_hidden_states = vision_layer_output[0]
            if idx == 0:
                bsz, length, dsize = vision_embeds.size()
                text_past_key_values = vision_embeds.view(bsz, 12, length, dsize // 12), vision_embeds.view(bsz, 12, length, dsize // 12)
            else:
                text_past_key_values = vision_layer_output[-1]
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_module = self.text_layer[idx]
            text_layer_output = text_layer_module(text_hidden_states, attention_mask=attention_mask, head_mask=layer_head_mask, visual_hidden_state=None, past_key_values=text_past_key_values, output_attentions=output_attentions, output_qks=output_qks, current_layer=idx)
            text_hidden_states = text_layer_output[0]
            if output_attentions:
                all_vision_attentions = all_vision_attentions + (vision_layer_output[1],)
                all_text_attentions = all_text_attentions + (text_layer_output[1],)
        if output_hidden_states:
            all_vision_hidden_states = all_vision_hidden_states + (vision_hidden_states,)
            all_text_hidden_states = all_text_hidden_states + (text_hidden_states,)
        if not return_dict:
            return tuple(v for v in [text_hidden_states, all_text_hidden_states, all_text_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=text_hidden_states, hidden_states=all_text_hidden_states, attentions=all_text_attentions)


def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device) ->Tensor:
    """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(f'Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})')
    extended_attention_mask = extended_attention_mask
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def get_head_mask(head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool=False) ->Tensor:
    """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
    head_mask = [None] * num_hidden_layers
    return head_mask


class IFAModel(nn.Module):

    def __init__(self, vision_config, text_config, add_pooling_layer=True):
        super(IFAModel, self).__init__()
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)
        self.text_config = text_config
        self.text_embeddings = BertEmbeddings(text_config)
        self.text_pooler = BertPooler(text_config) if add_pooling_layer else None
        self.encoder = IFAEncoder(vision_config, text_config)
        self.device = vision_config.device

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, pixel_values=None, aux_values=None, rcnn_values=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        vision_embedding_output = self.vision_embeddings(pixel_values, aux_values, rcnn_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            raise ValueError('token_type_ids is None!')
        extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(head_mask, self.text_config.num_hidden_layers)
        text_embedding_output = self.text_embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(vision_embeds=vision_embedding_output, text_embeds=text_embedding_output, attention_mask=extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.text_pooler(sequence_output) if self.text_pooler is not None else None
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def _get_resized_embeddings(self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int]=None) ->nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(f'Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}.You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}.')
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        self._init_text_weights(new_embeddings)
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
        return new_embeddings


class IFANERCRFModel(nn.Module):

    def __init__(self, label_list, args):
        super(IFANERCRFModel, self).__init__()
        self.args = args
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)
        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()
        None
        None
        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config)
        self.num_labels = len(label_list) + 1
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.text_config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None, rcnn_imgs=None):
        bsz = input_ids.size(0)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=images, aux_values=aux_imgs, rcnn_values=rcnn_imgs, return_dict=True)
        sequence_output = output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.fc(sequence_output)
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return logits, loss
        return logits, None


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, num_labels, embedding_dim, hidden_size, drop_out, bidirectional, num_layers):
        super().__init__()
        """ nn.Embedding: parameter size (num_words, embedding_dim) """
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, dropout=drop_out, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(hidden_size * 2, num_labels)
        """https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html"""
        self.crf = CRF(num_labels, batch_first=True)

    def _get_lstm_feature(self, input):
        out = self.embed(input)
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask)


def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1000000000000.0
    y_pred_pos = y_pred - (1 - y_true) * 1000000000000.0
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


class ATLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        loss = multilabel_categorical_crossentropy(labels, logits)
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = torch.zeros_like(logits[..., :1])
        output = torch.zeros_like(logits)
        mask = logits > th_logit
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = output[:, 1:].sum(1) == 0.0
        return output


class DoubleConv(nn.Module):
    """(conv => [BN] => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.double_conv(x)
        return x


class DownLayer(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DownLayer, self).__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(kernel_size=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpLayer(nn.Module):

    def __init__(self, in_ch, out_ch, bilinear=True):
        super(UpLayer, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet(torch.nn.Module):
    """
    UNet, down sampling & up sampling for global reasoning
    """

    def __init__(self, input_channels, class_number, **kwargs):
        super(AttentionUNet, self).__init__()
        down_channel = kwargs['down_channel']
        down_channel_2 = down_channel * 2
        up_channel_1 = down_channel_2 * 2
        up_channel_2 = down_channel * 2
        self.inc = InConv(input_channels, down_channel)
        self.down1 = DownLayer(down_channel, down_channel_2)
        self.down2 = DownLayer(down_channel_2, down_channel_2)
        self.up1 = UpLayer(up_channel_1, up_channel_1 // 4)
        self.up2 = UpLayer(up_channel_2, up_channel_2 // 4)
        self.outc = OutConv(up_channel_2 // 4, class_number)

    def forward(self, attention_channels):
        """
        Given multi-channel attention map, return the logits of every one mapping into 3-class
        :param attention_channels:
        :return:
        """
        x = attention_channels
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        output = self.outc(x)
        output = output.permute(0, 2, 3, 1).contiguous()
        return output


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens)
    end_tokens = torch.tensor(end_tokens)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, l_i - 512 + len_start:l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, l_i - 512:l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for n_s, l_i in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))
                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = att1 + att2
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention


class DocREModel(nn.Module):

    def __init__(self, config, args, model, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        self.bert_model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()
        self.head_extractor = nn.Linear(1 * config.hidden_size + args.unet_out_dim, emb_size)
        self.tail_extractor = nn.Linear(1 * config.hidden_size + args.unet_out_dim, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.bertdrop = nn.Dropout(0.6)
        self.unet_in_dim = args.unet_in_dim
        self.unet_out_dim = args.unet_in_dim
        self.liner = nn.Linear(config.hidden_size, args.unet_in_dim)
        self.min_height = args.max_height
        self.channel_type = args.channel_type
        self.segmentation_net = AttentionUNet(input_channels=args.unet_in_dim, class_number=args.unet_out_dim, down_channel=args.down_dim)

    def encode(self, input_ids, attention_mask, entity_pos):
        config = self.config
        if config.transformer_type == 'bert':
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == 'roberta':
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.bert_model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ['bert', 'roberta'] else 0
        bs, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        entity_es = []
        entity_as = []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for entity_num, e in enumerate(entity_pos[i]):
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size)
                        e_att = torch.zeros(h, c)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size)
                        e_att = torch.zeros(h, c)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)
            for _ in range(self.min_height - entity_num - 1):
                entity_atts.append(e_att)
            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)
            entity_es.append(entity_embs)
            entity_as.append(entity_atts)
            ht_i = torch.LongTensor(hts[i])
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])
            hss.append(hs)
            tss.append(ts)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        return hss, tss, entity_es, entity_as

    def get_mask(self, ents, bs, ne, run_device):
        ent_mask = torch.zeros(bs, ne, device=run_device)
        rel_mask = torch.zeros(bs, ne, ne, device=run_device)
        for _b in range(bs):
            ent_mask[_b, :len(ents[_b])] = 1
            rel_mask[_b, :len(ents[_b]), :len(ents[_b])] = 1
        return ent_mask, rel_mask

    def get_ht(self, rel_enco, hts):
        htss = []
        for i in range(len(hts)):
            ht_index = hts[i]
            for h_index, t_index in ht_index:
                htss.append(rel_enco[i, h_index, t_index])
        htss = torch.stack(htss, dim=0)
        return htss

    def get_channel_map(self, sequence_output, entity_as):
        bs, _, d = sequence_output.size()
        ne = self.min_height
        index_pair = []
        for i in range(ne):
            tmp = torch.cat((torch.ones((ne, 1), dtype=int) * i, torch.arange(0, ne).unsqueeze(1)), dim=-1)
            index_pair.append(tmp)
        index_pair = torch.stack(index_pair, dim=0).reshape(-1, 2)
        map_rss = []
        for b in range(bs):
            entity_atts = entity_as[b]
            h_att = torch.index_select(entity_atts, 0, index_pair[:, 0])
            t_att = torch.index_select(entity_atts, 0, index_pair[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-05)
            rs = contract('ld,rl->rd', sequence_output[b], ht_att)
            map_rss.append(rs)
        map_rss = torch.cat(map_rss, dim=0).reshape(bs, ne, ne, d)
        return map_rss

    def forward(self, input_ids=None, attention_mask=None, labels=None, entity_pos=None, hts=None, instance_mask=None):
        sequence_output, attention = self.encode(input_ids, attention_mask, entity_pos)
        bs, sequen_len, d = sequence_output.shape
        run_device = sequence_output.device.index
        ne = max([len(x) for x in entity_pos])
        ent_mask, rel_mask = self.get_mask(entity_pos, bs, ne, run_device)
        hs, ts, entity_embs, entity_as = self.get_hrt(sequence_output, attention, entity_pos, hts)
        if self.channel_type == 'context-based':
            feature_map = self.get_channel_map(sequence_output, entity_as)
            attn_input = self.liner(feature_map).permute(0, 3, 1, 2).contiguous()
        else:
            raise Exception('channel_type must be specify correctly')
        attn_map = self.segmentation_net(attn_input)
        h_t = self.get_ht(attn_map, hts)
        hs = torch.tanh(self.head_extractor(torch.cat([hs, h_t], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, h_t], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)
        output = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = loss, output
        return output


BATCH_SIZE = 8


NUM_WORKERS = 8


class BaseDataModule(nn.Module):
    """
    Base DataModule.
    """

    def __init__(self, args) ->None:
        super().__init__()
        self.args = args
        self.batch_size = self.args.get('batch_size', BATCH_SIZE)
        self.num_workers = self.args.get('num_workers', NUM_WORKERS)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Number of examples to operate on per forward step.')
        parser.add_argument('--num_workers', type=int, default=NUM_WORKERS, help='Number of additional processes to load data.')
        parser.add_argument('--data_dir', type=str, default='./dataset/dialogue', help='Number of additional processes to load data.')
        return parser

    def get_data_config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {'num_labels': self.num_labels}

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single GPU in distributed settings (so don't set state `self.x = y`).
        """
        pass

    def setup(self, stage=None):
        """
        Split into train, val, test, and set dims.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def train_dataloader(self):
        return DataLoader(self.data_train, shuffle=True, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.data_test, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)


LR = 5e-05


OPTIMIZER = 'AdamW'


class BaseLitModel(nn.Module):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self, model, device, args):
        super().__init__()
        self.model = model
        self.cur_model = model.module if hasattr(model, 'module') else model
        self.device = device
        self.args = args
        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.lr = self.args.get('lr', LR)

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='optimizer class from torch.optim')
        parser.add_argument('--lr', type=float, default=LR)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        return parser

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x
        logits = x
        loss = (logits - y) ** 2
        None
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x
        logits = x
        loss = (logits - y) ** 2
        None

    def test_step(self, batch, batch_idx):
        x, y = batch
        x
        logits = x
        loss = (logits - y) ** 2
        None

    def configure_optimizers(self):
        no_decay_param = ['bias', 'LayerNorm.weight']
        optimizer_group_parameters = [{'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay_param)], 'weight_decay': self.args.weight_decay}, {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay_param)], 'weight_decay': 0}]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-08)
        return optimizer
        """return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }"""

    @property
    def num_training_steps(self) ->int:
        """Total training steps inferred from datamodule and devices."""
        if isinstance(self.trainer.limit_train_batches, int) and self.trainer.limit_train_batches != 0:
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())
        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = dataset_size // effective_batch_size * self.trainer.max_epochs
        if self.trainer.max_steps and self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps


def f1_eval(logits, labels):

    def getpred(result, T1=0.5, T2=0.4):
        ret = []
        for i in range(len(result)):
            r = []
            maxl, maxj = -1, -1
            for j in range(len(result[i])):
                if result[i][j] > T1:
                    r += [j]
                if result[i][j] > maxl:
                    maxl = result[i][j]
                    maxj = j
            if len(r) == 0:
                if maxl <= T2:
                    r = [36]
                else:
                    r += [maxj]
            ret.append(r)
        return ret

    def geteval(devp, data):
        correct_sys, all_sys = 0, 0
        correct_gt = 0
        for i in range(len(data)):
            for id in data[i]:
                if id != 36:
                    correct_gt += 1
                    if id in devp[i]:
                        correct_sys += 1
            for id in devp[i]:
                if id != 36:
                    all_sys += 1
        precision = 1 if all_sys == 0 else correct_sys / all_sys
        recall = 0 if correct_gt == 0 else correct_sys / correct_gt
        f_1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return f_1
    logits = np.asarray(logits)
    logits = list(1 / (1 + np.exp(-logits)))
    temp_labels = []
    for l in labels:
        t = []
        for i in range(36):
            if l[i] == 1:
                t += [i]
        if len(t) == 0:
            t = [36]
        temp_labels.append(t)
    assert len(labels) == len(logits)
    labels = temp_labels
    bestT2 = bestf_1 = 0
    for T2 in range(51):
        devp = getpred(logits, T2=T2 / 100.0)
        f_1 = geteval(devp, labels)
        if f_1 > bestf_1:
            bestf_1 = f_1
            bestT2 = T2 / 100.0
    return bestf_1, bestT2


class BertLitModel(BaseLitModel):
    """
    use AutoModelForMaskedLM, and select the output by another layer in the lit model
    """

    def __init__(self, model, device, args, tokenizer):
        super().__init__(model, device, args)
        self.tokenizer = tokenizer
        with open(f'{args.data_dir}/rel2id.json', 'r') as file:
            rel2id = json.load(file)
        Na_num = 0
        for k, v in rel2id.items():
            if k == 'NA' or k == 'no_relation' or k == 'Other':
                Na_num = v
                break
        num_relation = len(rel2id)
        self.loss_fn = multilabel_categorical_crossentropy if 'dialogue' in args.data_dir else nn.CrossEntropyLoss()
        self.eval_fn = f1_eval if 'dialogue' in args.data_dir else partial(f1_score, rel_num=num_relation, na_num=Na_num)
        self.best_f1 = 0
        self.t_lambda = args.t_lambda
        self.label_st_id = tokenizer('[class1]', add_special_tokens=False)['input_ids'][0]
        self._init_label_word()

    def _init_label_word(self):
        args = self.args
        dataset_name = args.data_dir.split('/')[1]
        model_name_or_path = args.model_name_or_path.split('/')[-1]
        label_path = f'data/{model_name_or_path}.pt'
        if 'dialogue' in args.data_dir:
            label_word_idx = torch.load(label_path)[:-1]
        else:
            label_word_idx = torch.load(label_path)
        num_labels = len(label_word_idx)
        self.cur_model.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            word_embeddings = self.cur_model.get_input_embeddings()
            continous_label_word = [a[0] for a in self.tokenizer([f'[class{i}]' for i in range(1, num_labels + 1)], add_special_tokens=False)['input_ids']]
            for i, idx in enumerate(label_word_idx):
                word_embeddings.weight[continous_label_word[i]] = torch.mean(word_embeddings.weight[idx], dim=0)
            so_word = [a[0] for a in self.tokenizer(['[obj]', '[sub]'], add_special_tokens=False)['input_ids']]
            meaning_word = [a[0] for a in self.tokenizer(['person', 'organization', 'location', 'date', 'country'], add_special_tokens=False)['input_ids']]
            for i, idx in enumerate(so_word):
                word_embeddings.weight[so_word[i]] = torch.mean(word_embeddings.weight[meaning_word], dim=0)
            assert torch.equal(self.cur_model.get_input_embeddings().weight, word_embeddings.weight)
            assert torch.equal(self.cur_model.get_input_embeddings().weight, self.cur_model.get_output_embeddings().weight)
        self.word2label = continous_label_word

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels, so = batch
        input_ids = input_ids
        attention_mask = attention_mask
        token_type_ids = token_type_ids
        labels = labels
        so = so
        result = self.model(input_ids, attention_mask, token_type_ids, return_dict=True, output_hidden_states=True)
        logits = result.logits
        output_embedding = result.hidden_states[-1]
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels) + self.t_lambda * self.ke_loss(output_embedding, labels, so)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels, _ = batch
        input_ids = input_ids
        attention_mask = attention_mask
        token_type_ids = token_type_ids
        labels = labels
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'eval_logits': logits.detach().cpu().numpy(), 'eval_labels': labels.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs):
        logits = np.concatenate([o['eval_logits'] for o in outputs])
        labels = np.concatenate([o['eval_labels'] for o in outputs])
        f1 = self.eval_fn(logits, labels)['f1']
        best_f1 = -1
        if f1 > self.best_f1:
            self.best_f1 = f1
            best_f1 = self.best_f1
        return f1, best_f1, self.best_f1

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels, _ = batch
        input_ids = input_ids
        attention_mask = attention_mask
        token_type_ids = token_type_ids
        labels = labels
        logits = self.model(input_ids, attention_mask, token_type_ids, return_dict=True).logits
        logits = self.pvp(logits, input_ids)
        return {'test_logits': logits.detach().cpu().numpy(), 'test_labels': labels.detach().cpu().numpy()}

    def test_epoch_end(self, outputs):
        logits = np.concatenate([o['test_logits'] for o in outputs])
        labels = np.concatenate([o['test_labels'] for o in outputs])
        f1 = self.eval_fn(logits, labels)['f1']
        return f1

    @staticmethod
    def add_to_argparse(parser):
        BaseLitModel.add_to_argparse(parser)
        parser.add_argument('--t_lambda', type=float, default=0.01, help='')
        return parser

    def pvp(self, logits, input_ids):
        _, mask_idx = (input_ids == 103).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_output = logits[torch.arange(bs), mask_idx]
        assert mask_idx.shape[0] == bs, 'only one mask in sequence!'
        final_output = mask_output[:, self.word2label]
        return final_output

    def ke_loss(self, logits, labels, so):
        subject_embedding = []
        object_embedding = []
        bsz = logits.shape[0]
        for i in range(bsz):
            subject_embedding.append(torch.mean(logits[i, so[i][0]:so[i][1]], dim=0))
            object_embedding.append(torch.mean(logits[i, so[i][2]:so[i][3]], dim=0))
        subject_embedding = torch.stack(subject_embedding)
        object_embedding = torch.stack(object_embedding)
        relation_embedding = self.cur_model.get_output_embeddings().weight[labels + self.label_st_id]
        loss = torch.norm(subject_embedding + relation_embedding - object_embedding, p=2)
        return loss

    def configure_optimizers(self):
        no_decay_param = ['bias', 'LayerNorm.weight']
        if not self.args.two_steps:
            parameters = self.cur_model.named_parameters()
        else:
            parameters = [next(self.cur_model.named_parameters())]
        optimizer_group_parameters = [{'params': [p for n, p in parameters if not any(nd in n for nd in no_decay_param)], 'weight_decay': self.args.weight_decay}, {'params': [p for n, p in parameters if any(nd in n for nd in no_decay_param)], 'weight_decay': 0}]
        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-08)
        return optimizer
        """return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }"""


class IFAREModel(nn.Module):

    def __init__(self, num_labels, tokenizer, args):
        super(IFAREModel, self).__init__()
        self.args = args
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)
        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()
        None
        None
        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config)
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)
        self.model.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.text_config.hidden_size * 2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids('<s>')
        self.tail_start = tokenizer.convert_tokens_to_ids('<o>')
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None, rcnn_imgs=None):
        bsz = input_ids.size(0)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, pixel_values=images, aux_values=aux_imgs, rcnn_values=rcnn_imgs, return_dict=True)
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        entity_hidden_state = torch.Tensor(bsz, 2 * hidden_size)
        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            return loss_fn(logits, labels.view(-1)), logits
        return logits


class LabelSmoothSoftmaxCEV1(nn.Module):

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        logits = logits.float()
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1.0 - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def taylor_softmax_v1(x, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.0
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log:
        out = out.log()
    return out


class LogTaylorSoftmaxV1(nn.Module):

    def __init__(self, dim=1, n=2):
        super(LogTaylorSoftmaxV1, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        return taylor_softmax_v1(x, self.dim, self.n, use_log=True)


class TaylorCrossEntropyLossV1(nn.Module):

    def __init__(self, n=2, ignore_index=-1, reduction='mean'):
        super(TaylorCrossEntropyLossV1, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = LogTaylorSoftmaxV1(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits)
        loss = F.nll_loss(log_probs, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ATLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (BartClassificationHead,
     lambda: ([], {'input_dim': 4, 'inner_dim': 4, 'num_classes': 4, 'pooler_dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertAttention,
     lambda: ([], {'config': _mock_config(num_attention_heads=4, hidden_size=4, attention_probs_dropout_prob=0.5, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertIntermediate,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4, hidden_act=_mock_layer())}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertOutput,
     lambda: ([], {'config': _mock_config(intermediate_size=4, hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (BertPooler,
     lambda: ([], {'config': _mock_config(hidden_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (BertSelfAttention,
     lambda: ([], {'config': _mock_config(num_attention_heads=4, hidden_size=4, attention_probs_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BertSelfOutput,
     lambda: ([], {'config': _mock_config(hidden_size=4, layer_norm_eps=1, hidden_dropout_prob=0.5)}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (CLIPAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CLIPEncoderLayer,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_attention_heads=4, attention_dropout=0.5, intermediate_size=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (CLIPMLP,
     lambda: ([], {'config': _mock_config(hidden_size=4, intermediate_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CLIPVisionEmbeddings,
     lambda: ([], {'config': _mock_config(hidden_size=4, image_size=4, patch_size=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Capsule,
     lambda: ([], {'cfg': _mock_config(input_dim_capsule=4, dim_capsule=4, num_capsule=4, batch_size=4, share_weights=4, num_iterations=4)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (DotAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (DoubleConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DownLayer,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FocalLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (GCN,
     lambda: ([], {'cfg': _mock_config(num_layers=1, input_size=4, hidden_size=4, dropout=0.5)}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (GELU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (InConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LabelSmoothSoftmaxCEV1,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4], dtype=torch.int64)], {}),
     True),
    (LearnedPositionalEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4, 'padding_idx': 4, 'offset': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LogTaylorSoftmaxV1,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MultiHeadAttention,
     lambda: ([], {'embed_dim': 4, 'num_heads': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (OutConv,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TransformerAttention,
     lambda: ([], {'config': _mock_config(hidden_size=4, num_heads=4, dropout=0.5, output_attentions=4, layer_norm_eps=1)}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (UpLayer,
     lambda: ([], {'in_ch': 4, 'out_ch': 4}),
     lambda: ([torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])], {}),
     True),
]

class Test_zjunlp_DeepKE(_paritybench_base):
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

    def test_023(self):
        self._check(*TESTCASES[23])

    def test_024(self):
        self._check(*TESTCASES[24])

    def test_025(self):
        self._check(*TESTCASES[25])

    def test_026(self):
        self._check(*TESTCASES[26])

