import sys
_module = sys.modules[__name__]
del sys
Config = _module
config = _module
draw_plot = _module
gen_data = _module
CNN_ATT = _module
CNN_AVE = _module
CNN_ONE = _module
Model = _module
PCNN_ATT = _module
PCNN_AVE = _module
PCNN_ONE = _module
models = _module
networks = _module
classifier = _module
embedding = _module
encoder = _module
selector = _module
test = _module
train = _module

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


from torch.autograd import Variable


import torch.optim as optim


import numpy as np


import time


import sklearn.metrics


import torch.autograd as autograd


import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.label = None
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits):
        loss = self.loss(logits, self.label)
        _, output = torch.max(logits, dim=1)
        return loss, output.data


class Embedding(nn.Module):

    def __init__(self, config):
        super(Embedding, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(self.config.data_word_vec.shape[0], self.config.data_word_vec.shape[1])
        self.pos1_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx=0)
        self.pos2_embedding = nn.Embedding(self.config.pos_num, self.config.pos_size, padding_idx=0)
        self.init_word_weights()
        self.init_pos_weights()
        self.word = None
        self.pos1 = None
        self.pos2 = None

    def init_word_weights(self):
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.config.data_word_vec))

    def init_pos_weights(self):
        nn.init.xavier_uniform(self.pos1_embedding.weight.data)
        if self.pos1_embedding.padding_idx is not None:
            self.pos1_embedding.weight.data[self.pos1_embedding.padding_idx].fill_(0)
        nn.init.xavier_uniform(self.pos2_embedding.weight.data)
        if self.pos2_embedding.padding_idx is not None:
            self.pos2_embedding.weight.data[self.pos2_embedding.padding_idx].fill_(0)

    def forward(self):
        word = self.word_embedding(self.word)
        pos1 = self.pos1_embedding(self.pos1)
        pos2 = self.pos2_embedding(self.pos2)
        embedding = torch.cat((word, pos1, pos2), dim=2)
        return embedding


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.encoder = None
        self.selector = None
        self.classifier = Classifier(config)

    def forward(self):
        embedding = self.embedding()
        sen_embedding = self.encoder(embedding)
        logits = self.selector(sen_embedding)
        return self.classifier(logits)

    def test(self):
        embedding = self.embedding()
        sen_embedding = self.encoder(embedding)
        return self.selector.test(sen_embedding)


class Selector(nn.Module):

    def __init__(self, config, relation_dim):
        super(Selector, self).__init__()
        self.config = config
        self.relation_matrix = nn.Embedding(self.config.num_classes, relation_dim)
        self.bias = nn.Parameter(torch.Tensor(self.config.num_classes))
        self.attention_matrix = nn.Embedding(self.config.num_classes, relation_dim)
        self.init_weights()
        self.scope = None
        self.attention_query = None
        self.label = None
        self.dropout = nn.Dropout(self.config.drop_prob)

    def init_weights(self):
        nn.init.xavier_uniform(self.relation_matrix.weight.data)
        nn.init.normal(self.bias)
        nn.init.xavier_uniform(self.attention_matrix.weight.data)

    def get_logits(self, x):
        logits = torch.matmul(x, torch.transpose(self.relation_matrix.weight, 0, 1)) + self.bias
        return logits

    def forward(self, x):
        raise NotImplementedError

    def test(self, x):
        raise NotImplementedError


class Attention(Selector):

    def _attention_train_logit(self, x):
        relation_query = self.relation_matrix(self.attention_query)
        attention = self.attention_matrix(self.attention_query)
        attention_logit = torch.sum(x * attention * relation_query, 1, True)
        return attention_logit

    def _attention_test_logit(self, x):
        attention_logit = torch.matmul(x, torch.transpose(self.attention_matrix.weight * self.relation_matrix.weight, 0, 1))
        return attention_logit

    def forward(self, x):
        attention_logit = self._attention_train_logit(x)
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]:self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.squeeze(torch.matmul(attention_score, sen_matrix))
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits

    def test(self, x):
        attention_logit = self._attention_test_logit(x)
        tower_output = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            attention_score = F.softmax(torch.transpose(attention_logit[self.scope[i]:self.scope[i + 1]], 0, 1), 1)
            final_repre = torch.matmul(attention_score, sen_matrix)
            logits = self.get_logits(final_repre)
            tower_output.append(torch.diag(F.softmax(logits, 1)))
        stack_output = torch.stack(tower_output)
        return list(stack_output.data.cpu().numpy())


class _CNN(nn.Module):

    def __init__(self, config):
        super(_CNN, self).__init__()
        self.config = config
        self.in_channels = 1
        self.in_height = self.config.max_length
        self.in_width = self.config.word_size + 2 * self.config.pos_size
        self.kernel_size = self.config.window_size, self.in_width
        self.out_channels = self.config.hidden_size
        self.stride = 1, 1
        self.padding = 1, 0
        self.cnn = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

    def forward(self, embedding):
        return self.cnn(embedding)


class _PiecewisePooling(nn.Module):

    def __init(self):
        super(_PiecewisePooling, self).__init__()

    def forward(self, x, mask, hidden_size):
        mask = torch.unsqueeze(mask, 1)
        x, _ = torch.max(mask + x, dim=2)
        x = x - 100
        return x.view(-1, hidden_size * 3)


class PCNN(nn.Module):

    def __init__(self, config):
        super(PCNN, self).__init__()
        self.config = config
        self.mask = None
        self.cnn = _CNN(config)
        self.pooling = _PiecewisePooling()
        self.activation = nn.ReLU()

    def forward(self, embedding):
        embedding = torch.unsqueeze(embedding, dim=1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.mask, self.config.hidden_size)
        return self.activation(x)


class PCNN_ATT(Model):

    def __init__(self, config):
        super(PCNN_ATT, self).__init__(config)
        self.encoder = PCNN(config)
        self.selector = Attention(config, config.hidden_size * 3)


class Average(Selector):

    def forward(self, x):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            final_repre = torch.mean(sen_matrix, 0)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        stack_repre = self.dropout(stack_repre)
        logits = self.get_logits(stack_repre)
        return logits

    def test(self, x):
        tower_repre = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            final_repre = torch.mean(sen_matrix, 0)
            tower_repre.append(final_repre)
        stack_repre = torch.stack(tower_repre)
        logits = self.get_logits(stack_repre)
        score = F.softmax(logits, 1)
        return list(score.data.cpu().numpy())


class PCNN_AVE(Model):

    def __init__(self, config):
        super(PCNN_AVE, self).__init__(config)
        self.encoder = PCNN(config)
        self.selector = Average(config, config.hidden_size * 3)


class One(Selector):

    def forward(self, x):
        tower_logits = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            sen_matrix = self.dropout(sen_matrix)
            logits = self.get_logits(sen_matrix)
            score = F.softmax(logits, 1)
            _, k = torch.max(score, dim=0)
            k = k[self.label[i]]
            tower_logits.append(logits[k])
        return torch.cat(tower_logits, 0)

    def test(self, x):
        tower_score = []
        for i in range(len(self.scope) - 1):
            sen_matrix = x[self.scope[i]:self.scope[i + 1]]
            logits = self.get_logits(sen_matrix)
            score = F.softmax(logits, 1)
            score, _ = torch.max(score, 0)
            tower_score.append(score)
        tower_score = torch.stack(tower_score)
        return list(tower_score.data.cpu().numpy())


class PCNN_ONE(Model):

    def __init__(self, config):
        super(PCNN_ONE, self).__init__(config)
        self.encoder = PCNN(config)
        self.selector = One(config, config.hidden_size * 3)


class _MaxPooling(nn.Module):

    def __init__(self):
        super(_MaxPooling, self).__init__()

    def forward(self, x, hidden_size):
        x, _ = torch.max(x, dim=2)
        return x.view(-1, hidden_size)


class CNN(nn.Module):

    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.cnn = _CNN(config)
        self.pooling = _MaxPooling()
        self.activation = nn.ReLU()

    def forward(self, embedding):
        embedding = torch.unsqueeze(embedding, dim=1)
        x = self.cnn(embedding)
        x = self.pooling(x, self.config.hidden_size)
        return self.activation(x)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (_CNN,
     lambda: ([], {'config': _mock_config(max_length=4, word_size=4, pos_size=4, window_size=4, hidden_size=4)}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (_PiecewisePooling,
     lambda: ([], {}),
     lambda: ([4, torch.rand([4, 4, 3, 3]), 4], {}),
     True),
]

class Test_ShulinCao_OpenNRE_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

