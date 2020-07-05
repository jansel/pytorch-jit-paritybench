import sys
_module = sys.modules[__name__]
del sys
lstm = _module
shottext_lstm = _module
dataset = _module
main = _module
main_kfold = _module
TextCNN = _module
TextLSTM = _module
models = _module
runner = _module
runner_kfold = _module
data_shower = _module
dataset_split_chars = _module
lda_train = _module
model_nb = _module
prepare_dataset = _module
word2vector = _module
doc2vector = _module
read_data = _module
sample = _module
shottext = _module

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


import torch.nn.functional as F


from torch import nn


import numpy as np


import torch.autograd as autograd


class TextCNN(nn.Module):

    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args
        V = args.vocab_size
        D = args.embed_dim
        C = args.num_classes
        Cin = 1
        Cout = args.kernel_num
        Ks = args.kernel_sizes
        self.embeding = nn.Embedding(V, D)
        self.convs = nn.ModuleList([nn.Conv2d(Cin, Cout, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(Ks) * Cout, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embeding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        """
                x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
                x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
                x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
                x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        """
        x = self.dropout(x)
        out = self.fc(x)
        return out


class TextLSTM(nn.Module):
    """
    lstm model of text classify.
    """

    def __init__(self, opt):
        self.name = 'TextLstm'
        super(TextLSTM, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(opt.vocab_size, opt.embed_dim)
        self.lstm = nn.LSTM(input_size=opt.embed_dim, hidden_size=opt.hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        self.linears = nn.Sequential(nn.Linear(opt.hidden_size, opt.linear_hidden_size), nn.ReLU(), nn.Dropout(0.25), nn.Linear(opt.linear_hidden_size, opt.num_classes))
        if opt.embedding_path:
            self.embedding.weight.data.copy_(torch.from_numpy(np.load(opt.embedding_path)))

    def forward(self, x):
        x = self.embedding(x)
        None
        lstm_out, _ = self.lstm(x)
        out = self.linears(lstm_out[:, (-1), :])
        return out

