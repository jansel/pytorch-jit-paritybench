import sys
_module = sys.modules[__name__]
del sys
config = _module
DataLoader = _module
mydatasets_self_five = _module
mydatasets_self_two = _module
Common = _module
Load_Pretrained_Embed = _module
DataUtils = _module
handle_wordEmbedding2File = _module
master = _module
main = _module
models = _module
model = _module
model_BiGRU = _module
model_BiLSTM = _module
model_BiLSTM_1 = _module
model_BiLSTM_lexicon = _module
model_CBiLSTM = _module
model_CGRU = _module
model_CLSTM = _module
model_CNN = _module
model_CNN_BiGRU = _module
model_CNN_BiLSTM = _module
model_CNN_LSTM = _module
model_CNN_MUI = _module
model_DeepCNN = _module
model_DeepCNN_MUI = _module
model_GRU = _module
model_HighWay_BiLSTM_1 = _module
model_HighWay_CNN = _module
model_LSTM = _module
train_ALL_CNN = _module
train_ALL_LSTM = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn as nn


import torch.nn.init as init


from collections import OrderedDict


import numpy as np


import torch.nn.functional as F


from torch.autograd import Variable


import random


import torch.autograd as autograd


import torch.nn.utils as utils


import torch.optim.lr_scheduler as lr_scheduler


class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D)
        pretrained_weight = np.array(args.pretrained_weight)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        """
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        """
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)
        if self.args.static:
            x = Variable(x.data)
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


class BiGRU(nn.Module):

    def __init__(self, args):
        super(BiGRU, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.bigru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers, bidirectional=True)
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        embed = self.embed(input)
        embed = self.dropout(embed)
        input = embed.view(len(input), embed.size(1), -1)
        gru_out, _ = self.bigru(input)
        gru_out = torch.transpose(gru_out, 0, 1)
        gru_out = torch.transpose(gru_out, 1, 2)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        gru_out = F.tanh(gru_out)
        y = self.hidden2label(gru_out)
        logit = y
        return logit


class BiLSTM(nn.Module):

    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.bilstm = nn.LSTM(D, self.hidden_dim // 2, num_layers=1, dropout=args.dropout, bidirectional=True, bias=False)
        None
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)

    def forward(self, x):
        embed = self.embed(x)
        x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)
        logit = y
        return logit


class BiLSTM_1(nn.Module):

    def __init__(self, args):
        super(BiLSTM_1, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        if args.max_norm is not None:
            None
            self.embed = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True, padding_idx=args.paddingId)
            if args.word_Embedding:
                self.embed.weight.data.copy_(args.pretrained_weight)
        else:
            None
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
            if args.word_Embedding:
                self.embed.weight.data.copy_(args.pretrained_weight)
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, bias=True, bidirectional=True, dropout=self.args.dropout)
        None
        if args.init_weight:
            None
            init.xavier_normal(self.bilstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[1][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[1][1], gain=np.sqrt(args.init_weight_value))
        self.hidden2label = nn.Linear(self.hidden_dim * 2, C)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)
        logit = self.hidden2label(bilstm_out)
        return logit


class BiLSTM_1(nn.Module):

    def __init__(self, args):
        super(BiLSTM_1, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True, bias=True)
        self.hidden2label = nn.Linear(self.hidden_dim * 2 * 2, C, bias=True)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout(embed)
        x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(x)
        hidden = torch.cat(self.hidden, 0)
        hidden = torch.cat(hidden, 1)
        logit = self.hidden2label(F.tanh(hidden))
        return logit


class CBiLSTM(nn.Module):

    def __init__(self, args):
        super(CBiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        KK = []
        for K in Ks:
            KK.append(K + 1 if K % 2 == 0 else K)
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K // 2, 0)) for K in KK]
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True)
        self.hidden2label1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.hidden2label2 = nn.Linear(self.hidden_dim, C)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout(embed)
        cnn_x = embed
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]
        cnn_x = torch.cat(cnn_x, 0)
        cnn_x = torch.transpose(cnn_x, 1, 2)
        bilstm_out, _ = self.bilstm(cnn_x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        cnn_bilstm_out = self.hidden2label1(F.tanh(bilstm_out))
        cnn_bilstm_out = self.hidden2label2(F.tanh(cnn_bilstm_out))
        logit = self.dropout(cnn_bilstm_out)
        return logit


class CGRU(nn.Module):

    def __init__(self, args):
        super(CGRU, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        KK = []
        for K in Ks:
            KK.append(K + 1 if K % 2 == 0 else K)
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K // 2, 0)) for K in KK]
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        self.gru = nn.GRU(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout)
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        cnn_x = embed
        cnn_x = self.dropout(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]
        cnn_x = torch.cat(cnn_x, 0)
        cnn_x = torch.transpose(cnn_x, 1, 2)
        lstm_out, _ = self.gru(cnn_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        cnn_lstm_out = self.hidden2label1(F.tanh(lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))
        logit = cnn_lstm_out
        return logit


class CLSTM(nn.Module):

    def __init__(self, args):
        super(CLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        KK = []
        for K in Ks:
            KK.append(K + 1 if K % 2 == 0 else K)
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K // 2, 0)) for K in KK]
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        self.lstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout)
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        cnn_x = embed
        cnn_x = self.dropout(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]
        cnn_x = torch.cat(cnn_x, 0)
        cnn_x = torch.transpose(cnn_x, 1, 2)
        lstm_out, _ = self.lstm(cnn_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        cnn_lstm_out = self.hidden2label1(F.tanh(lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))
        logit = cnn_lstm_out
        return logit


class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        if args.max_norm is not None:
            None
            self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True, padding_idx=args.paddingId)
        else:
            None
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
            self.embed.weight.requires_grad = True
        None
        if args.wide_conv is True:
            None
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1), padding=(K // 2, 0), dilation=1, bias=False) for K in Ks]
        else:
            None
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
        None
        if args.init_weight:
            None
            for conv in self.convs1:
                init.xavier_normal(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                fan_in, fan_out = CNN_Text.calculate_fan_in_and_fan_out(conv.weight.data)
                None
                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)
        in_fea = len(Ks) * Co
        self.fc = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        if args.batch_normalizations is True:
            None
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea // 2, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            raise ValueError('Fan in and fan out can not be computed for tensor with less than 2 dimensions')
        if dimensions == 2:
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout_embed(x)
        x = x.unsqueeze(1)
        if self.args.batch_normalizations is True:
            x = [self.convs1_bn(F.tanh(conv(x))).squeeze(3) for conv in self.convs1]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        if self.args.batch_normalizations is True:
            x = self.fc1_bn(self.fc1(x))
            logit = self.fc2_bn(self.fc2(F.tanh(x)))
        else:
            logit = self.fc(x)
        return logit


class CNN_BiGRU(nn.Module):

    def __init__(self, args):
        super(CNN_BiGRU, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.C = C
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D), padding=(K // 2, 0), stride=1) for K in Ks]
        None
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        self.bigru = nn.GRU(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True, bias=True)
        L = len(Ks) * Co + self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout(embed)
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)
        bigru_x = embed.view(len(x), embed.size(1), -1)
        bigru_x, _ = self.bigru(bigru_x)
        bigru_x = torch.transpose(bigru_x, 0, 1)
        bigru_x = torch.transpose(bigru_x, 1, 2)
        bigru_x = F.max_pool1d(bigru_x, bigru_x.size(2)).squeeze(2)
        bigru_x = F.tanh(bigru_x)
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bigru_x = torch.transpose(bigru_x, 0, 1)
        cnn_bigru_out = torch.cat((cnn_x, bigru_x), 0)
        cnn_bigru_out = torch.transpose(cnn_bigru_out, 0, 1)
        cnn_bigru_out = self.hidden2label1(F.tanh(cnn_bigru_out))
        logit = self.hidden2label2(F.tanh(cnn_bigru_out))
        return logit


class CNN_BiLSTM(nn.Module):

    def __init__(self, args):
        super(CNN_BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.C = C
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D), padding=(K // 2, 0), stride=1) for K in Ks]
        None
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, dropout=args.dropout, bidirectional=True, bias=True)
        L = len(Ks) * Co + self.hidden_dim * 2
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        embed = self.embed(x)
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [conv(cnn_x).squeeze(3) for conv in self.convs1]
        cnn_x = [F.tanh(F.max_pool1d(i, i.size(2)).squeeze(2)) for i in cnn_x]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)
        bilstm_x = embed.view(len(x), embed.size(1), -1)
        bilstm_out, _ = self.bilstm(bilstm_x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        bilstm_out = F.tanh(bilstm_out)
        cnn_x = torch.transpose(cnn_x, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        cnn_bilstm_out = torch.cat((cnn_x, bilstm_out), 0)
        cnn_bilstm_out = torch.transpose(cnn_bilstm_out, 0, 1)
        cnn_bilstm_out = self.hidden2label1(F.tanh(cnn_bilstm_out))
        cnn_bilstm_out = self.hidden2label2(F.tanh(cnn_bilstm_out))
        logit = cnn_bilstm_out
        return logit


class CNN_LSTM(nn.Module):

    def __init__(self, args):
        super(CNN_LSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.C = C
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.dropout = nn.Dropout(args.dropout)
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)
        L = len(Ks) * Co + self.hidden_dim
        self.hidden2label1 = nn.Linear(L, L // 2)
        self.hidden2label2 = nn.Linear(L // 2, C)

    def forward(self, x):
        embed = self.embed(x)
        cnn_x = embed
        cnn_x = torch.transpose(cnn_x, 0, 1)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]
        cnn_x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in cnn_x]
        cnn_x = torch.cat(cnn_x, 1)
        cnn_x = self.dropout(cnn_x)
        lstm_x = embed.view(len(x), embed.size(1), -1)
        lstm_out, _ = self.lstm(lstm_x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        cnn_x = torch.transpose(cnn_x, 0, 1)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        cnn_lstm_out = torch.cat((cnn_x, lstm_out), 0)
        cnn_lstm_out = torch.transpose(cnn_lstm_out, 0, 1)
        cnn_lstm_out = self.hidden2label1(F.tanh(cnn_lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))
        logit = cnn_lstm_out
        return logit


class CNN_MUI(nn.Module):

    def __init__(self, args):
        super(CNN_MUI, self).__init__()
        self.args = args
        V = args.embed_num
        V_mui = args.embed_num_mui
        D = args.embed_dim
        C = args.class_num
        Ci = 2
        Co = args.kernel_num
        Ks = args.kernel_sizes
        if args.max_norm is not None:
            None
            self.embed_no_static = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True, padding_idx=args.paddingId)
            self.embed_static = nn.Embedding(V_mui, D, max_norm=args.max_norm, scale_grad_by_freq=True, padding_idx=args.paddingId_mui)
        else:
            None
            self.embed_no_static = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
            self.embed_static = nn.Embedding(V_mui, D, scale_grad_by_freq=True, padding_idx=args.paddingId_mui)
        if args.word_Embedding:
            self.embed_no_static.weight.data.copy_(args.pretrained_weight)
            self.embed_static.weight.data.copy_(args.pretrained_weight_static)
            self.embed_no_static.weight.requires_grad = False
        if args.wide_conv is True:
            None
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1), padding=(K // 2, 0), bias=True) for K in Ks]
        else:
            None
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
        None
        if args.init_weight:
            None
            for conv in self.convs1:
                init.xavier_normal(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                init.uniform(conv.bias, 0, 0)
        """
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        """
        self.dropout = nn.Dropout(args.dropout)
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        in_fea = len(Ks) * Co
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)
        if args.batch_normalizations is True:
            None
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea // 2, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x_no_static = self.embed_no_static(x)
        x_static = self.embed_static(x)
        x = torch.stack([x_static, x_no_static], 1)
        x = self.dropout(x)
        if self.args.batch_normalizations is True:
            x = [F.relu(self.convs1_bn(conv(x))).squeeze(3) for conv in self.convs1]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        if self.args.batch_normalizations is True:
            x = self.fc1(x)
            logit = self.fc2(F.relu(x))
        else:
            x = self.fc1(x)
            logit = self.fc2(F.relu(x))
        return logit


class DEEP_CNN(nn.Module):

    def __init__(self, args):
        super(DEEP_CNN, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        if args.max_norm is not None:
            None
            self.embed = nn.Embedding(V, D, max_norm=5, scale_grad_by_freq=True, padding_idx=args.paddingId)
        else:
            None
            self.embed = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
            self.embed.weight.requires_grad = True
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K // 2, 0), bias=True) for K in Ks]
        self.convs2 = [nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K // 2, 0), bias=True) for K in Ks]
        None
        None
        if args.init_weight:
            None
            for conv1, conv2 in zip(self.convs1, self.convs2):
                init.xavier_normal(conv1.weight.data, gain=np.sqrt(args.init_weight_value))
                init.uniform(conv1.bias, 0, 0)
                init.xavier_normal(conv2.weight.data, gain=np.sqrt(args.init_weight_value))
                init.uniform(conv2.bias, 0, 0)
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        if self.args.cuda is True:
            for conv in self.convs2:
                conv = conv
        self.dropout = nn.Dropout(args.dropout)
        in_fea = len(Ks) * Co
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)

    def forward(self, x):
        one_layer = self.embed(x)
        one_layer = one_layer.unsqueeze(1)
        one_layer = [torch.transpose(F.relu(conv(one_layer)).squeeze(3), 1, 2) for conv in self.convs1]
        two_layer = [F.relu(conv(one_layer.unsqueeze(1))).squeeze(3) for conv, one_layer in zip(self.convs2, one_layer)]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in two_layer]
        output = torch.cat(output, 1)
        output = self.dropout(output)
        output = self.fc1(F.relu(output))
        logit = self.fc2(F.relu(output))
        return logit


class DEEP_CNN_MUI(nn.Module):

    def __init__(self, args):
        super(DEEP_CNN_MUI, self).__init__()
        self.args = args
        V = args.embed_num
        V_mui = args.embed_num_mui
        D = args.embed_dim
        C = args.class_num
        Ci = 2
        Co = args.kernel_num
        Ks = args.kernel_sizes
        if args.max_norm is not None:
            None
            self.embed_no_static = nn.Embedding(V, D, max_norm=args.max_norm, scale_grad_by_freq=True, padding_idx=args.paddingId)
            self.embed_static = nn.Embedding(V_mui, D, max_norm=args.max_norm, scale_grad_by_freq=True, padding_idx=args.paddingId_mui)
        else:
            None
            self.embed_no_static = nn.Embedding(V, D, scale_grad_by_freq=True, padding_idx=args.paddingId)
            self.embed_static = nn.Embedding(V_mui, D, scale_grad_by_freq=True, padding_idx=args.paddingId_mui)
        if args.word_Embedding:
            self.embed_no_static.weight.data.copy_(args.pretrained_weight)
            self.embed_static.weight.data.copy_(args.pretrained_weight_static)
            self.embed_no_static.weight.requires_grad = False
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K // 2, 0), bias=True) for K in Ks]
        self.convs2 = [nn.Conv2d(1, Co, (K, D), stride=1, padding=(K // 2, 0), bias=True) for K in Ks]
        None
        None
        if args.init_weight:
            None
            for conv1, conv2 in zip(self.convs1, self.convs2):
                init.xavier_normal(conv1.weight.data, gain=np.sqrt(args.init_weight_value))
                init.uniform(conv1.bias, 0, 0)
                init.xavier_normal(conv2.weight.data, gain=np.sqrt(args.init_weight_value))
                init.uniform(conv2.bias, 0, 0)
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        if self.args.cuda is True:
            for conv in self.convs2:
                conv = conv
        self.dropout = nn.Dropout(args.dropout)
        in_fea = len(Ks) * Co
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea // 2, bias=True)
        self.fc2 = nn.Linear(in_features=in_fea // 2, out_features=C, bias=True)

    def forward(self, x):
        x_no_static = self.embed_no_static(x)
        x_static = self.embed_static(x)
        x_static = Variable(x_static.data)
        x = torch.stack([x_static, x_no_static], 1)
        one_layer = x
        one_layer = [torch.transpose(F.relu(conv(one_layer)).squeeze(3), 1, 2).unsqueeze(1) for conv in self.convs1]
        two_layer = [F.relu(conv(one_layer)).squeeze(3) for conv, one_layer in zip(self.convs2, one_layer)]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in two_layer]
        output = torch.cat(output, 1)
        output = self.dropout(output)
        output = self.fc1(output)
        logit = self.fc2(F.relu(output))
        return logit


class GRU(nn.Module):

    def __init__(self, args):
        super(GRU, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.gru = nn.GRU(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        embed = self.embed(input)
        input = embed.view(len(input), embed.size(1), -1)
        lstm_out, _ = self.gru(input)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)
        y = self.hidden2label(lstm_out)
        logit = y
        return logit


class HighWay_BiLSTM_1(nn.Module):

    def __init__(self, args):
        super(HighWay_BiLSTM_1, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.dropout = nn.Dropout(args.dropout)
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.bilstm = nn.LSTM(D, self.hidden_dim, num_layers=self.num_layers, bias=True, bidirectional=True, dropout=self.args.dropout)
        None
        if args.init_weight:
            None
            init.xavier_normal(self.bilstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[1][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.bilstm.all_weights[1][1], gain=np.sqrt(args.init_weight_value))
            self.bilstm.all_weights[0][3].data[20:40].fill_(1)
            self.bilstm.all_weights[0][3].data[0:20].fill_(0)
            self.bilstm.all_weights[0][3].data[40:80].fill_(0)
            self.bilstm.all_weights[0][2].data[20:40].fill_(1)
            self.bilstm.all_weights[0][2].data[0:20].fill_(0)
            self.bilstm.all_weights[0][2].data[40:80].fill_(0)
            self.bilstm.all_weights[1][3].data[20:40].fill_(1)
            self.bilstm.all_weights[1][3].data[0:20].fill_(0)
            self.bilstm.all_weights[1][3].data[40:80].fill_(0)
            self.bilstm.all_weights[1][2].data[20:40].fill_(1)
            self.bilstm.all_weights[1][2].data[0:20].fill_(0)
            self.bilstm.all_weights[1][2].data[40:80].fill_(0)
        self.hidden2label1 = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2, bias=True)
        self.gate_layer = nn.Linear(in_features=self.hidden_dim * 2, out_features=self.hidden_dim * 2, bias=True)
        self.logit_layer = nn.Linear(in_features=self.hidden_dim * 2, out_features=C, bias=True)

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2))
        bilstm_out = bilstm_out.squeeze(2)
        hidden2lable = self.hidden2label1(F.tanh(bilstm_out))
        gate_layer = F.sigmoid(self.gate_layer(bilstm_out))
        gate_hidden_layer = torch.mul(hidden2lable, gate_layer)
        gate_input = torch.mul(1 - gate_layer, bilstm_out)
        highway_output = torch.add(gate_hidden_layer, gate_input)
        logit = self.logit_layer(highway_output)
        return logit


class HighWay_CNN(nn.Module):

    def __init__(self, args):
        super(HighWay_CNN, self).__init__()
        self.args = args
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        None
        if args.wide_conv is True:
            None
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), stride=(1, 1), padding=(K // 2, 0), dilation=1, bias=True) for K in Ks]
        else:
            None
            self.convs1 = [nn.Conv2d(in_channels=Ci, out_channels=Co, kernel_size=(K, D), bias=True) for K in Ks]
        None
        if args.init_weight:
            None
            for conv in self.convs1:
                init.xavier_normal(conv.weight.data, gain=np.sqrt(args.init_weight_value))
                fan_in, fan_out = HighWay_CNN.calculate_fan_in_and_fan_out(conv.weight.data)
                None
                std = np.sqrt(args.init_weight_value) * np.sqrt(2.0 / (fan_in + fan_out))
                init.uniform(conv.bias, 0, 0)
        if self.args.cuda is True:
            for conv in self.convs1:
                conv = conv
        self.dropout = nn.Dropout(args.dropout)
        in_fea = len(Ks) * Co
        self.fc1 = nn.Linear(in_features=in_fea, out_features=in_fea, bias=True)
        self.gate_layer = nn.Linear(in_features=in_fea, out_features=in_fea, bias=True)
        self.logit_layer = nn.Linear(in_features=in_fea, out_features=C, bias=True)
        if args.batch_normalizations is True:
            None
            self.convs1_bn = nn.BatchNorm2d(num_features=Co, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_fea // 2, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)
            self.fc2_bn = nn.BatchNorm1d(num_features=C, momentum=args.bath_norm_momentum, affine=args.batch_norm_affine)

    def calculate_fan_in_and_fan_out(tensor):
        dimensions = tensor.ndimension()
        if dimensions < 2:
            raise ValueError('Fan in and fan out can not be computed for tensor with less than 2 dimensions')
        if dimensions == 2:
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        return fan_in, fan_out

    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)
        if self.args.batch_normalizations is True:
            x = [self.convs1_bn(F.tanh(conv(x))).squeeze(3) for conv in self.convs1]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        else:
            x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        if self.args.batch_normalizations is True:
            x = self.fc1_bn(self.fc1(x))
            fc = self.fc2_bn(self.fc2(F.tanh(x)))
        else:
            fc = self.fc1(x)
        gate_layer = F.sigmoid(self.gate_layer(x))
        gate_fc_layer = torch.mul(fc, gate_layer)
        gate_input = torch.mul(1 - gate_layer, x)
        highway_output = torch.add(gate_fc_layer, gate_input)
        logit = self.logit_layer(highway_output)
        return logit


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D, padding_idx=args.paddingId)
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=args.dropout, num_layers=self.num_layers)
        if args.init_weight:
            None
            init.xavier_normal(self.lstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.lstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        self.dropout = nn.Dropout(args.dropout)
        self.dropout_embed = nn.Dropout(args.dropout_embed)

    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout_embed(embed)
        x = embed.view(len(x), embed.size(1), -1)
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)
        logit = self.hidden2label(lstm_out)
        return logit

