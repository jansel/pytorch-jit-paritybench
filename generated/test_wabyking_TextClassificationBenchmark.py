import sys
_module = sys.modules[__name__]
del sys
dataHelper = _module
Dataset = _module
dataloader = _module
ag = _module
glove = _module
imdb = _module
mr = _module
sst = _module
imdb = _module
trec = _module
main = _module
BERTFast = _module
BaseModel = _module
BiBloSA = _module
CNN = _module
CNNBasic = _module
CNNInception = _module
CNNKim = _module
CNNMultiLayer = _module
CNNText = _module
CNN_Inception = _module
Capsule = _module
ConvS2S = _module
DiSAN = _module
FastText = _module
LSTM = _module
LSTMBI = _module
LSTMStack = _module
LSTMTree = _module
LSTMwithAttention = _module
MLP = _module
MemoryNetwork = _module
QuantumCNN = _module
RCNN = _module
RNN_CNN = _module
SelfAttention = _module
Transformer = _module
models = _module
opts = _module
trandition = _module
utils = _module

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


import numpy as np


import string


from collections import Counter


import random


import time


import torch


from torch.autograd import Variable


import re


from torchtext import data


from torchtext import datasets


from torchtext.vocab import GloVe


from torchtext.vocab import CharNGram


import itertools


import torch.optim as optim


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.modules.loss import NLLLoss


from torch.nn.modules.loss import MultiLabelSoftMarginLoss


from torch.nn.modules.loss import MultiLabelMarginLoss


from torch.nn.modules.loss import BCELoss


import torch as t


from torch import nn


from collections import OrderedDict


from sklearn.utils import shuffle


import torch.nn.init as init


from torchtext.vocab import Vectors


from torchtext.vocab import FastText


from functools import wraps


import logging


class BaseModel(nn.Module):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.model_name = 'BaseModel'
        self.opt = opt
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        if opt.__dict__.get('embeddings', None) is not None:
            self.encoder.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.fc = nn.Linear(opt.embedding_dim, opt.label_size)
        self.properties = {'model_name': self.__class__.__name__, 'batch_size': self.opt.batch_size, 'learning_rate': self.opt.learning_rate, 'keep_dropout': self.opt.keep_dropout}

    def forward(self, content):
        content_ = t.mean(self.encoder(content), dim=1)
        out = self.fc(content_.view(content_.size(0), -1))
        return out

    def save(self, save_dir='saved_model', metric=None):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.model_info = '__'.join([(k + '_' + str(v) if type(v) != list else k + '_' + str(v)[1:-1].replace(',', '_').replace(',', '')) for k, v in self.properties.items()])
        if metric:
            path = os.path.join(save_dir, str(metric)[2:] + '_' + self.model_info)
        else:
            path = os.path.join(save_dir, self.model_info)
        t.save(self, path)
        return path


class CNN(BaseModel):

    def __init__(self, opt):
        super(CNN, self).__init__(opt)
        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_sent_len = opt.max_sent_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.CLASS_SIZE = opt.label_size
        self.FILTERS = opt['FILTERS']
        self.FILTER_NUM = opt['FILTER_NUM']
        self.keep_dropout = opt.keep_dropout
        self.IN_CHANNEL = 1
        assert len(self.FILTERS) == len(self.FILTER_NUM)
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
        if self.embedding_type == 'static' or self.embedding_type == 'non-static' or self.embedding_type == 'multichannel':
            self.WV_MATRIX = opt['WV_MATRIX']
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
            if self.embedding_type == 'static':
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == 'multichannel':
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2
        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.embedding_dim * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, 'conv_%d' % i, conv)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.label_size)
        self.properties.update({'FILTER_NUM': self.FILTER_NUM, 'FILTERS': self.FILTERS})

    def get_conv(self, i):
        return getattr(self, 'conv_%d' % i)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.embedding_dim * self.max_sent_len)
        if self.embedding_type == 'multichannel':
            x2 = self.embedding2(inp).view(-1, 1, self.embedding_dim * self.max_sent_len)
            x = torch.cat((x, x2), 1)
        conv_results = [F.max_pool1d(F.relu(self.get_conv(i)(x)), self.max_sent_len - self.FILTERS[i] + 1).view(-1, self.FILTER_NUM[i]) for i in range(len(self.FILTERS))]
        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.keep_dropout, training=self.training)
        x = self.fc(x)
        return x


class CNN1(BaseModel):

    def __init__(self, opt):
        super(CNN1, self).__init__(opt)
        V = opt.vocab_size
        D = opt.embedding_dim
        C = opt.label_size
        Ci = 1
        Co = opt.kernel_num
        Ks = opt.kernel_sizes
        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        """
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        """
        self.dropout = nn.Dropout(opt.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)
        self.properties.update({'kernel_num': opt.kernel_num, 'kernel_sizes': opt.kernel_sizes})

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)
        if self.args.static:
            x = Variable(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        """
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        """
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


class CNN2(BaseModel):

    def __init__(self, opt):
        super(CNN2, self).__init__(opt)
        self.embed = nn.Embedding(opt.vocab_size + 1, opt.embedding_dim)
        self.conv1 = nn.Sequential(nn.Conv1d(opt.l0, 256, kernel_size=7, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3))
        self.fc = nn.Linear(256, opt.label_size)
        self.properties.update({})

    def forward(self, x_input):
        x = self.embed(x_input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x)


class CNN3(BaseModel):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, args):
        super(CNN3, self).__init__(opt)
        self.args = args
        embedding_dim = args.embed_dim
        embedding_num = args.num_features
        class_number = args.class_num
        in_channel = 1
        out_channel = args.kernel_num
        kernel_sizes = args.kernel_sizes
        self.embed = nn.Embedding(embedding_num + 1, embedding_dim)
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, embedding_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_sizes) * out_channel, class_number)
        self.properties.update({'kernel_sizes': kernel_sizes})

    def forward(self, input_x):
        """
        :param input_x: a list size having the number of batch_size elements with the same length
        :return: batch_size X num_aspects tensor
        """
        x = self.embed(input_x)
        if self.args.static:
            x = F.Variable(input_x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc(x))
        return logit


class BasicCNN1D(BaseModel):

    def __init__(self, opt):
        super(BasicCNN1D, self).__init__(opt)
        self.content_dim = opt.__dict__.get('content_dim', 256)
        self.kernel_size = opt.__dict__.get('kernel_size', 3)
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        if opt.__dict__.get('embeddings', None) is not None:
            self.encoder.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.content_conv = nn.Sequential(nn.Conv1d(in_channels=opt.embedding_dim, out_channels=self.content_dim, kernel_size=self.kernel_size), nn.ReLU(), nn.MaxPool1d(kernel_size=opt.max_seq_len - self.kernel_size + 1))
        self.fc = nn.Linear(self.content_dim, opt.label_size)
        self.properties.update({'content_dim': self.content_dim, 'kernel_size': self.kernel_size})

    def forward(self, content):
        content = self.encoder(content)
        content_out = self.content_conv(content.permute(0, 2, 1))
        reshaped = content_out.view(content_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits


class BasicCNN2D(BaseModel):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, args):
        super(BasicCNN2D, self).__init__(opt)
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.label_size = opt.label_size
        self.keep_dropout = opt.keep_dropout
        in_channel = 1
        self.kernel_nums = opt.kernel_nums
        self.kernel_sizes = opt.kernel_sizes
        self.embed = nn.Embedding(self.vocab_size + 1, self.embedding_dim)
        if opt.__dict__.get('embeddings', None) is not None:
            self.embed.weight = nn.Parameter(opt.embeddings)
        self.conv = nn.ModuleList([nn.Conv2d(in_channel, out_channel, (K, self.embedding_dim)) for K, out_channel in zip(self.kernel_sizes, self.kernel_nums)])
        self.dropout = nn.Dropout(self.keep_dropout)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.out_channel, self.label_size)
        self.properties.update({'kernel_nums': self.kernel_nums, 'kernel_sizes': self.kernel_sizes})

    def forward(self, input_x):
        """
        :param input_x: a list size having the number of batch_size elements with the same length
        :return: batch_size X num_aspects tensor
        """
        x = self.embed(input_x)
        if self.opt.static:
            x = F.Variable(input_x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = F.log_softmax(self.fc(x))
        return logit


class Inception(nn.Module):

    def __init__(self, cin, co, relu=True, norm=True):
        super(Inception, self).__init__()
        assert co % 4 == 0
        cos = [co / 4] * 4
        self.activa = nn.Sequential()
        if norm:
            self.activa.add_module('norm', nn.BatchNorm1d(co))
        if relu:
            self.activa.add_module('relu', nn.ReLU(True))
        self.branch1 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin, cos[0], 1, stride=1))]))
        self.branch2 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin, cos[1], 1)), ('norm1', nn.BatchNorm1d(cos[1])), ('relu1', nn.ReLU(inplace=True)), ('conv3', nn.Conv1d(cos[1], cos[1], 3, stride=1, padding=1))]))
        self.branch3 = nn.Sequential(OrderedDict([('conv1', nn.Conv1d(cin, cos[2], 3, padding=1)), ('norm1', nn.BatchNorm1d(cos[2])), ('relu1', nn.ReLU(inplace=True)), ('conv3', nn.Conv1d(cos[2], cos[2], 5, stride=1, padding=2))]))
        self.branch4 = nn.Sequential(OrderedDict([('conv3', nn.Conv1d(cin, cos[3], 3, stride=1, padding=1))]))

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        result = self.activa(torch.cat((branch1, branch2, branch3, branch4), 1))
        return result


class InceptionCNN(BaseModel):

    def __init__(self, opt):
        super(InceptionCNN, self).__init__(opt)
        incept_dim = getattr(opt, 'inception_dim', 512)
        self.model_name = 'CNNText_inception'
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.content_conv = nn.Sequential(Inception(opt.embedding_dim, incept_dim), Inception(incept_dim, incept_dim), nn.MaxPool1d(opt.max_seq_len))
        linear_hidden_size = getattr(opt, 'linear_hidden_size', 2000)
        self.fc = nn.Sequential(nn.Linear(incept_dim, linear_hidden_size), nn.BatchNorm1d(linear_hidden_size), nn.ReLU(inplace=True), nn.Linear(linear_hidden_size, opt.label_size))
        if opt.__dict__.get('embeddings', None) is not None:
            self.encoder.weight = nn.Parameter(opt.embeddings)
        self.properties.update({'linear_hidden_size': linear_hidden_size, 'incept_dim': incept_dim})

    def forward(self, content):
        content = self.encoder(content)
        if self.opt.embedding_type == 'static':
            content = content.detach(0)
        content_out = self.content_conv(content.permute(0, 2, 1))
        out = content_out.view(content_out.size(0), -1)
        out = self.fc(out)
        return out


class KIMCNN1D(BaseModel):

    def __init__(self, opt):
        super(KIMCNN1D, self).__init__(opt)
        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_seq_len = opt.max_seq_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.label_size = opt.label_size
        self.kernel_sizes = opt.kernel_sizes
        self.kernel_nums = opt.kernel_nums
        self.keep_dropout = opt.keep_dropout
        self.in_channel = 1
        assert len(self.kernel_sizes) == len(self.kernel_nums)
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim)
        if self.embedding_type == 'static' or self.embedding_type == 'non-static' or self.embedding_type == 'multichannel':
            self.embedding.weight = nn.Parameter(opt.embeddings)
            if self.embedding_type == 'static':
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == 'multichannel':
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
                self.embedding2.weight = nn.Parameter(opt.embeddings)
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2
            else:
                pass
        self.convs = nn.ModuleList([nn.Conv1d(self.in_channel, num, self.embedding_dim * size, stride=self.embedding_dim) for size, num in zip(opt.kernel_sizes, opt.kernel_nums)])
        self.fc = nn.Linear(sum(self.kernel_nums), self.label_size)
        self.properties.update({'kernel_sizes': self.kernel_sizes, 'kernel_nums': self.kernel_nums})

    def get_conv(self, i):
        return getattr(self, 'conv_%d' % i)

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.embedding_dim * self.max_seq_len)
        if self.embedding_type == 'multichannel':
            x2 = self.embedding2(inp).view(-1, 1, self.embedding_dim * self.max_seq_len)
            x = torch.cat((x, x2), 1)
        conv_results = [F.max_pool1d(F.relu(self.convs[i](x)), self.max_seq_len - self.kernel_sizes[i] + 1).view(-1, self.kernel_nums[i]) for i in range(len(self.convs))]
        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.keep_dropout, training=self.training)
        x = self.fc(x)
        return x


class KIMCNN2D(nn.Module):

    def __init__(self, opt):
        super(KIMCNN2D, self).__init__()
        self.opt = opt
        self.embedding_type = opt.embedding_type
        self.batch_size = opt.batch_size
        self.max_seq_len = opt.max_seq_len
        self.embedding_dim = opt.embedding_dim
        self.vocab_size = opt.vocab_size
        self.label_size = opt.label_size
        self.kernel_sizes = opt.kernel_sizes
        self.kernel_nums = opt.kernel_nums
        self.keep_dropout = opt.keep_dropout
        self.embedding = nn.Embedding(self.vocab_size + 2, self.embedding_dim)
        if self.embedding_type == 'static' or self.embedding_type == 'non-static' or self.embedding_type == 'multichannel':
            self.embedding.weight = nn.Parameter(opt.embeddings)
            if self.embedding_type == 'static':
                self.embedding.weight.requires_grad = False
            elif self.embedding_type == 'multichannel':
                self.embedding2 = nn.Embedding(self.vocab_size + 2, self.embedding_dim, padding_idx=self.vocab_size + 1)
                self.embedding2.weight = nn.Parameter(opt.embeddings)
                self.embedding2.weight.requires_grad = False
                self.in_channel = 2
            else:
                pass
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num, (size, opt.embedding_dim)) for size, num in zip(opt.kernel_sizes, opt.kernel_nums)])
        """
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        """
        self.dropout = nn.Dropout(opt.keep_dropout)
        self.fc = nn.Linear(sum(opt.kernel_nums), opt.label_size)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        """
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        """
        x = self.dropout(x)
        logit = self.fc(x)
        return logit


class MultiLayerCNN(BaseModel):

    def __init__(self, opt):
        super(MultiLayerCNN, self).__init__(opt)
        self.embed = nn.Embedding(opt.vocab_size + 1, opt.embedding_dim)
        if opt.__dict__.get('embeddings', None) is not None:
            self.embed.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.conv1 = nn.Sequential(nn.Conv1d(opt.max_seq_len, 256, kernel_size=7, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=7, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=3, stride=1), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=3))
        self.fc = nn.Linear(256 * 7, opt.label_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x)


class CNNText(BaseModel):

    def __init__(self, opt):
        super(CNNText, self).__init__(opt)
        self.content_dim = opt.__dict__.get('content_dim', 256)
        self.kernel_size = opt.__dict__.get('kernel_size', 3)
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        if opt.__dict__.get('embeddings', None) is not None:
            self.encoder.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.content_conv = nn.Sequential(nn.Conv1d(in_channels=opt.embedding_dim, out_channels=self.content_dim, kernel_size=self.kernel_size), nn.ReLU(), nn.MaxPool1d(kernel_size=opt.max_seq_len - self.kernel_size + 1))
        self.fc = nn.Linear(self.content_dim, opt.label_size)
        self.properties.update({'content_dim': self.content_dim, 'kernel_size': self.kernel_size})

    def forward(self, content):
        content = self.encoder(content)
        content_out = self.content_conv(content.permute(0, 2, 1))
        reshaped = content_out.view(content_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits


class CNNText_inception(BaseModel):

    def __init__(self, opt):
        super(CNNText_inception, self).__init__(opt)
        incept_dim = getattr(opt, 'inception_dim', 512)
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.content_conv = nn.Sequential(Inception(opt.embedding_dim, incept_dim), Inception(incept_dim, incept_dim), nn.MaxPool1d(opt.max_seq_len))
        opt.hidden_size = getattr(opt, 'linear_hidden_size', 2000)
        self.fc = nn.Sequential(nn.Linear(incept_dim, opt.hidden_size), nn.BatchNorm1d(opt.hidden_size), nn.ReLU(inplace=True), nn.Linear(opt.hidden_size, opt.label_size))
        if opt.__dict__.get('embeddings', None) is not None:
            None
            self.encoder.weight.data.copy_(t.from_numpy(opt.embeddings))
        self.properties.update({'inception_dim': incept_dim, 'hidden_size': opt.hidden_size})

    def forward(self, content):
        content = self.encoder(content)
        if self.opt.static:
            content = content.detach(0)
        content_out = self.content_conv(content.permute(0, 2, 1))
        out = content_out.view(content_out.size(0), -1)
        out = self.fc(out)
        return out


NUM_ROUTING_ITERATIONS = 3


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleLayer(nn.Module):

    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None, num_iterations=NUM_ROUTING_ITERATIONS, padding=0):
        super(CapsuleLayer, self).__init__()
        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations
        self.num_capsules = num_capsules
        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            prime = [3, 5, 7, 9, 11, 13, 17, 19, 23]
            sizes = prime[:self.num_capsules]
            self.capsules = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size=i, stride=2, padding=int((i - 1) / 2)) for i in sizes])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = torch.matmul(x[(None), :, :, (None), :], self.route_weights[:, (None), :, :, :])
            if torch.cuda.is_available():
                logits = torch.autograd.Variable(torch.zeros(priors.size()))
            else:
                logits = torch.autograd.Variable(torch.zeros(priors.size()))
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash(torch.mul(probs, priors).sum(dim=2, keepdim=True))
                if i != self.num_iterations - 1:
                    delta_logits = torch.mul(priors, outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)
        return outputs


class CapsuleNet(BaseModel):

    def __init__(self, opt):
        super(CapsuleNet, self).__init__(opt)
        self.label_size = opt.label_size
        self.embed = nn.Embedding(opt.vocab_size + 1, opt.embedding_dim)
        self.opt.cnn_dim = 1
        self.kernel_size = 3
        self.kernel_size_primary = 3
        if opt.__dict__.get('embeddings', None) is not None:
            self.embed.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32)
        self.digit_capsules = CapsuleLayer(num_capsules=opt.label_size, num_route_nodes=int(32 * opt.max_seq_len / 2), in_channels=8, out_channels=16)
        if self.opt.cnn_dim == 2:
            self.conv_2d = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(self.kernel_size, opt.embedding_dim), stride=(1, opt.embedding_dim), padding=(int((self.kernel_size - 1) / 2), 0))
        else:
            self.conv_1d = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=opt.embedding_dim * self.kernel_size, stride=opt.embedding_dim, padding=opt.embedding_dim * int((self.kernel_size - 1) / 2))
        self.decoder = nn.Sequential(nn.Linear(16 * self.label_size, 512), nn.ReLU(inplace=True), nn.Linear(512, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 784), nn.Sigmoid())

    def forward(self, x, y=None, reconstruct=False):
        x = self.embed(x)
        if self.opt.cnn_dim == 1:
            x = x.view(x.size(0), 1, x.size(-1) * x.size(-2))
            x_conv = F.relu(self.conv_1d(x), inplace=True)
        else:
            x = x.unsqueeze(1)
            x_conv = F.relu(self.conv_2d(x), inplace=True).squeeze(3)
        x = self.primary_capsules(x_conv)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes)
        if not reconstruct:
            return classes
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            if torch.cuda.is_available():
                y = Variable(torch.sparse.torch.eye(self.label_size)).index_select(dim=0, index=max_length_indices.data)
            else:
                y = Variable(torch.sparse.torch.eye(self.label_size)).index_select(dim=0, index=max_length_indices.data)
        reconstructions = self.decoder((x * y[:, :, (None)]).view(x.size(0), -1))
        return classes, reconstructions


class FastText(BaseModel):

    def __init__(self, opt):
        super(FastText, self).__init__(opt)
        linear_hidden_size = getattr(opt, 'linear_hidden_size', 2000)
        self.encoder = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        if opt.__dict__.get('embeddings', None) is not None:
            None
            self.encoder.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.content_fc = nn.Sequential(nn.Linear(opt.embedding_dim, linear_hidden_size), nn.BatchNorm1d(linear_hidden_size), nn.ReLU(inplace=True), nn.Linear(linear_hidden_size, opt.label_size))
        self.properties.update({'linear_hidden_size': linear_hidden_size})

    def forward(self, content):
        content_ = t.mean(self.encoder(content), dim=1)
        out = self.content_fc(content_.view(content_.size(0), -1))
        return out


class LSTMClassifier(BaseModel):

    def __init__(self, opt):
        super(LSTMClassifier, self).__init__(opt)
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()
        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.lsmt_reduce_by_mean = opt.__dict__.get('lstm_mean', True)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h0, c0

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.permute(1, 0, 2)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        if self.lsmt_reduce_by_mean == 'mean':
            out = lstm_out.permute(1, 0, 2)
            final = torch.mean(out, 1)
        else:
            final = lstm_out[-1]
        y = self.hidden2label(final)
        return y


class LSTMBI(BaseModel):

    def __init__(self, opt):
        super(LSTMBI, self).__init__(opt)
        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, num_layers=self.opt.lstm_layers, dropout=self.opt.keep_dropout, bidirectional=self.opt.bidirectional)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.lsmt_reduce_by_mean = opt.__dict__.get('lstm_mean', True)
        self.properties.update({'hidden_dim': self.opt.hidden_dim, 'lstm_mean': self.lsmt_reduce_by_mean, 'lstm_layers': self.opt.lstm_layers})

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.opt.batch_size
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(2 * self.opt.lstm_layers, batch_size, self.opt.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.opt.lstm_layers, batch_size, self.opt.hidden_dim // 2))
        else:
            h0 = Variable(torch.zeros(2 * self.opt.lstm_layers, batch_size, self.opt.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.opt.lstm_layers, batch_size, self.opt.hidden_dim // 2))
        return h0, c0

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.permute(1, 0, 2)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.bilstm(x, self.hidden)
        if self.lsmt_reduce_by_mean == 'mean':
            out = lstm_out.permute(1, 0, 2)
            final = torch.mean(out, 1)
        else:
            final = lstm_out[-1]
        y = self.hidden2label(final)
        return y


class LSTMAttention(torch.nn.Module):

    def __init__(self, opt):
        super(LSTMAttention, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()
        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.num_layers = opt.lstm_layers
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, batch_first=True, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.mean = opt.__dict__.get('lstm_mean', True)
        self.attn_fc = torch.nn.Linear(opt.embedding_dim, 1)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        else:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        return h0, c0

    def attention(self, rnn_out, state):
        merged_state = torch.cat([s for s in state], 1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(rnn_out, merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        return torch.bmm(torch.transpose(rnn_out, 1, 2), weights).squeeze(2)

    def forward(self, X):
        embedded = self.word_embeddings(X)
        hidden = self.init_hidden(X.size()[0])
        rnn_out, hidden = self.bilstm(embedded, hidden)
        h_n, c_n = hidden
        attn_out = self.attention(rnn_out, h_n)
        logits = self.hidden2label(attn_out)
        return logits


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_dim + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    encoding[:, (-1)] = 1.0
    return np.transpose(encoding)


class MemN2N(nn.Module):

    def __init__(self, settings):
        super(MemN2N, self).__init__()
        use_cuda = settings['use_cuda']
        num_vocab = settings['num_vocab']
        embedding_dim = settings['embedding_dim']
        sentence_size = settings['sentence_size']
        self.max_hops = settings['max_hops']
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(num_vocab, embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)
            self.add_module('C_{}'.format(hop), C)
        self.C = AttrProxy(self, 'C_')
        self.softmax = nn.Softmax()
        self.encoding = Variable(torch.FloatTensor(position_encoding(sentence_size, embedding_dim)), requires_grad=False)
        if use_cuda:
            self.encoding = self.encoding

    def forward(self, query):
        story = query
        story_size = story.size()
        u = list()
        query_embed = self.C[0](query)
        encoding = self.encoding.unsqueeze(0).expand_as(query_embed)
        u.append(torch.sum(query_embed * encoding, 1))
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.view(story.size(0), -1))
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))
            encoding = self.encoding.unsqueeze(0).unsqueeze(1).expand_as(embed_A)
            m_A = torch.sum(embed_A * encoding, 2)
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A * u_temp, 2))
            embed_C = self.C[hop + 1](story.view(story.size(0), -1))
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            m_C = torch.sum(embed_C * encoding, 2)
            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        a_hat = u[-1] @ self.C[self.max_hops].weight.transpose(0, 1)
        return a_hat, self.softmax(a_hat)


class RCNN(BaseModel):

    def __init__(self, opt):
        super(RCNN, self).__init__(opt)
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()
        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.num_layers = 1
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(input_size=opt.embedding_dim, hidden_size=opt.hidden_dim // 2, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.hidden = self.init_hidden()
        self.max_pooling = nn.MaxPool1d(kernel_size=3, stride=2)
        self.content_dim = 256
        self.hidden2label = nn.Linear(2 * opt.hidden_dim // 2 + opt.embedding_dim, opt.label_size)

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        else:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        return h0, c0

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.permute(1, 0, 2)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.bilstm(x, self.hidden)
        c_lr = lstm_out.permute(1, 0, 2)
        xi = torch.cat((c_lr[:, :, 0:int(c_lr.size()[2] / 2)], embeds, c_lr[:, :, int(c_lr.size()[2] / 2):]), 2)
        yi = torch.tanh(xi.permute(0, 2, 1))
        y = self.max_pooling(yi)
        y = y.permute(2, 0, 1)
        y = self.hidden2label(y[-1])
        return y


class RNN_CNN(BaseModel):

    def __init__(self, opt):
        super(RNN_CNN, self).__init__(opt)
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()
        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.lstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim)
        self.hidden = self.init_hidden()
        self.content_dim = 256
        self.conv = nn.Conv1d(in_channels=opt.hidden_dim, out_channels=self.content_dim, kernel_size=opt.hidden_dim * 2, stride=opt.embedding_dim)
        self.hidden2label = nn.Linear(self.content_dim, opt.label_size)
        self.properties.update({'content_dim': self.content_dim})

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        return h0, c0

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.permute(1, 0, 2)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.conv(lstm_out.permute(1, 2, 0))
        y = self.hidden2label(y.view(y.size()[0], -1))
        return y


class SelfAttention(nn.Module):

    def __init__(self, opt):
        self.opt = opt
        super(SelfAttention, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.batch_size = opt.batch_size
        self.use_gpu = torch.cuda.is_available()
        self.word_embeddings = nn.Embedding(opt.vocab_size, opt.embedding_dim)
        self.word_embeddings.weight = nn.Parameter(opt.embeddings, requires_grad=opt.embedding_training)
        self.num_layers = 1
        self.dropout = opt.keep_dropout
        self.bilstm = nn.LSTM(opt.embedding_dim, opt.hidden_dim // 2, num_layers=self.num_layers, dropout=self.dropout, bidirectional=True)
        self.hidden2label = nn.Linear(opt.hidden_dim, opt.label_size)
        self.hidden = self.init_hidden()
        self.self_attention = nn.Sequential(nn.Linear(opt.hidden_dim, 24), nn.ReLU(True), nn.Linear(24, 1))

    def init_hidden(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if self.use_gpu:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        else:
            h0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
            c0 = Variable(torch.zeros(2 * self.num_layers, batch_size, self.hidden_dim // 2))
        return h0, c0

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.permute(1, 0, 2)
        self.hidden = self.init_hidden(sentence.size()[0])
        lstm_out, self.hidden = self.bilstm(x, self.hidden)
        final = lstm_out.permute(1, 0, 2)
        attn_ene = self.self_attention(final)
        attns = F.softmax(attn_ene.view(self.batch_size, -1))
        feats = (final * attns).sum(dim=1)
        y = self.hidden2label(feats)
        return y


class Linear(nn.Module):
    """ Simple Linear layer with xavier init """

    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class Bottle(nn.Module):
    """ Perform the reshape routine before and after an operation """

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    """ Perform the reshape routine before and after a linear projection """
    pass


class BottleSoftmax(Bottle, nn.Softmax):
    """ Perform the reshape routine before and after a softmax operation"""
    pass


class LayerNormalization(nn.Module):
    """ Layer normalization module """

    def __init__(self, d_hid, eps=0.001):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, keepdim=True, dim=-1)
        sigma = torch.std(z, keepdim=True, dim=-1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out


class BatchBottle(nn.Module):
    """ Perform the reshape routine before and after an operation """

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(BatchBottle, self).forward(input)
        size = input.size()[1:]
        out = super(BatchBottle, self).forward(input.view(-1, size[0] * size[1]))
        return out.view(-1, size[0], size[1])


class BottleLayerNormalization(BatchBottle, LayerNormalization):
    """ Perform the reshape routine before and after a layer normalization"""
    pass


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax()

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), 'Attention mask shape {} mismatch with Attention logit tensor shape {}.'.format(attn_mask.size(), attn.size())
            attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_model, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_model, d_v))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head
        residual = q
        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_model)
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1)
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        return self.layer_norm(outputs + residual), attns


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_hid, d_inner_hid, 1)
        self.w_2 = nn.Conv1d(d_inner_hid, d_hid, 1)
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        output = self.relu(self.w_1(x.transpose(1, 2)))
        output = self.w_2(output).transpose(2, 1)
        output = self.dropout(output)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, attn_mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, attn_mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, attn_mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn


class ConstantsClass:

    def __init__(self):
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.PAD_WORD = '<pad>'
        self.UNK_WORD = '<unk>'
        self.BOS_WORD = '<s>'
        self.EOS_WORD = '</s>'


Constants = ConstantsClass()


def get_attn_padding_mask(seq_q, seq_k):
    """ Indicate the padding-related part to mask """
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size, len_q = seq_q.size()
    mb_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(mb_size, len_q, len_k)
    return pad_attn_mask


def position_encoding_init(n_position, d_pos_vec):
    """ Init the sinusoid position encoding table """
    position_enc = np.array([([(pos / np.power(10000, 2 * (j // 2) / d_pos_vec)) for j in range(d_pos_vec)] if pos != 0 else np.zeros(d_pos_vec)) for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, n_src_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64, d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        super(Encoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        enc_input = self.src_word_emb(src_seq)
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []
        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(src_seq, src_seq)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]
        if return_attns:
            return enc_output, enc_slf_attns
        else:
            return enc_output


def get_attn_subsequent_mask(seq):
    """ Get an attention mask to avoid using the subsequent info."""
    assert seq.dim() == 2
    attn_shape = seq.size(0), seq.size(1), seq.size(1)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    if seq.is_cuda:
        subsequent_mask = subsequent_mask
    return subsequent_mask


class Decoder(nn.Module):
    """ A decoder model with self attention mechanism. """

    def __init__(self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64, d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        super(Decoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, d_inner_hid, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        dec_input = self.tgt_word_emb(tgt_seq)
        dec_input += self.position_enc(tgt_pos)
        dec_slf_attn_pad_mask = get_attn_padding_mask(tgt_seq, tgt_seq)
        dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq)
        dec_slf_attn_mask = torch.gt(dec_slf_attn_pad_mask + dec_slf_attn_sub_mask, 0)
        dec_enc_attn_pad_mask = get_attn_padding_mask(tgt_seq, src_seq)
        if return_attns:
            dec_slf_attns, dec_enc_attns = [], []
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(dec_output, enc_output, slf_attn_mask=dec_slf_attn_mask, dec_enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]
        if return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output,


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, n_src_vocab, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_word_vec=512, d_model=512, d_inner_hid=1024, d_k=64, d_v=64, dropout=0.1, proj_share_weight=True, embs_share_weight=True):
        super(Transformer, self).__init__()
        self.encoder = Encoder(n_src_vocab, n_max_seq, n_layers=n_layers, n_head=n_head, d_word_vec=d_word_vec, d_model=d_model, d_inner_hid=d_inner_hid, dropout=dropout)
        self.decoder = Decoder(n_tgt_vocab, n_max_seq, n_layers=n_layers, n_head=n_head, d_word_vec=d_word_vec, d_model=d_model, d_inner_hid=d_inner_hid, dropout=dropout)
        self.tgt_word_proj = Linear(d_model, n_tgt_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)
        assert d_model == d_word_vec, 'To facilitate the residual connections,          the dimensions of all module output shall be the same.'
        if proj_share_weight:
            assert d_model == d_word_vec
            self.tgt_word_proj.weight = self.decoder.tgt_word_emb.weight
        if embs_share_weight:
            assert n_src_vocab == n_tgt_vocab, 'To share word embedding table, the vocabulary size of src/tgt shall be the same.'
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def get_trainable_parameters(self):
        """ Avoid updating the position encoding """
        enc_freezed_param_ids = set(map(id, self.encoder.position_enc.parameters()))
        dec_freezed_param_ids = set(map(id, self.decoder.position_enc.parameters()))
        freezed_param_ids = enc_freezed_param_ids | dec_freezed_param_ids
        return (p for p in self.parameters() if id(p) not in freezed_param_ids)

    def forward(self, src, tgt):
        src_seq, src_pos = src
        tgt_seq, tgt_pos = tgt
        tgt_seq = tgt_seq[:, :-1]
        tgt_pos = tgt_pos[:, :-1]
        enc_output, _ = self.encoder(src_seq, src_pos)
        dec_output, _ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_proj(dec_output)
        return seq_logit.view(-1, seq_logit.size(2))


class AttentionIsAllYouNeed(nn.Module):

    def __init__(self, opt, n_layers=6, n_head=8, d_word_vec=128, d_model=128, d_inner_hid=256, d_k=32, d_v=32, dropout=0.1, proj_share_weight=True, embs_share_weight=True):
        super(AttentionIsAllYouNeed, self).__init__()
        self.encoder = Encoder(opt.vocab_size, opt.max_seq_len, n_layers=n_layers, n_head=n_head, d_word_vec=d_word_vec, d_model=d_model, d_inner_hid=d_inner_hid, dropout=dropout)
        self.hidden2label = nn.Linear(opt.max_seq_len * d_model, opt.label_size)
        self.batch_size = opt.batch_size

    def forward(self, inp):
        src_seq, src_pos = inp
        enc_output = self.encoder(src_seq, src_pos)
        return self.hidden2label(enc_output.view((self.batch_size, -1)))


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BasicCNN1D,
     lambda: ([], {'opt': _mock_config(vocab_size=4, embedding_dim=4, label_size=4, batch_size=4, learning_rate=4, keep_dropout=0.5, max_seq_len=4)}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     True),
    (BottleLayerNormalization,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (BottleLinear,
     lambda: ([], {'d_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     False),
    (BottleSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CNNText,
     lambda: ([], {'opt': _mock_config(vocab_size=4, embedding_dim=4, label_size=4, batch_size=4, learning_rate=4, keep_dropout=0.5, max_seq_len=4)}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     True),
    (CapsuleLayer,
     lambda: ([], {'num_capsules': 4, 'num_route_nodes': 4, 'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (CapsuleNet,
     lambda: ([], {'opt': _mock_config(vocab_size=4, embedding_dim=4, label_size=4, batch_size=4, learning_rate=4, keep_dropout=0.5, max_seq_len=4)}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
    (Encoder,
     lambda: ([], {'n_src_vocab': 4, 'n_max_seq': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (KIMCNN1D,
     lambda: ([], {'opt': _mock_config(vocab_size=4, embedding_dim=4, label_size=4, batch_size=4, learning_rate=4, keep_dropout=0.5, embedding_type=4, max_seq_len=4, kernel_sizes=[4, 4], kernel_nums=[4, 4])}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (KIMCNN2D,
     lambda: ([], {'opt': _mock_config(embedding_type=4, batch_size=4, max_seq_len=4, embedding_dim=4, vocab_size=4, label_size=4, kernel_sizes=[4, 4], kernel_nums=[4, 4], keep_dropout=0.5)}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     True),
    (LayerNormalization,
     lambda: ([], {'d_hid': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Linear,
     lambda: ([], {'d_in': 4, 'd_out': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PositionwiseFeedForward,
     lambda: ([], {'d_hid': 4, 'd_inner_hid': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (ScaledDotProductAttention,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_wabyking_TextClassificationBenchmark(_paritybench_base):
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

