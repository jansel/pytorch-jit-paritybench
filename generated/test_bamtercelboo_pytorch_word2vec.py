import sys
_module = sys.modules[__name__]
del sys
CBOW = _module
huffman = _module
input_data = _module
model = _module
over_heap = _module
pytorch_word2vec = _module
test = _module
torch_CBOW = _module
word2vec = _module

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


from torch.autograd import Variable


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import numpy


import torch.optim as optim


import torch.optim as opt


class CBOWModel(nn.Module):

    def __init__(self, emb_size, emb_dimension, window_size):
        super(CBOWModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.window_size = window_size
        self.u_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        emb_v = []
        for i in range(len(pos_v)):
            emb_v_v = self.u_embeddings(Variable(torch.LongTensor(pos_v[i])))
            emb_v_v_numpy = emb_v_v.data.numpy()
            emb_v_v_numpy = np.sum(emb_v_v_numpy, axis=0)
            emb_v_v_list = emb_v_v_numpy.tolist()
            emb_v.append(emb_v_v_list)
        emb_v = Variable(torch.FloatTensor(emb_v))
        emb_u = self.v_embeddings(Variable(torch.LongTensor(pos_u)))
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))
        neg_emb_v = []
        for i in range(len(neg_v)):
            neg_emb_v_v = self.u_embeddings(Variable(torch.LongTensor(neg_v[i])))
            neg_emb_v_v_numpy = neg_emb_v_v.data.numpy()
            neg_emb_v_v_numpy = np.sum(neg_emb_v_v_numpy, axis=0)
            neg_emb_v_v_list = neg_emb_v_v_numpy.tolist()
            neg_emb_v.append(neg_emb_v_v_list)
        neg_emb_v = Variable(torch.FloatTensor(neg_emb_v))
        neg_emb_u = self.v_embeddings(Variable(torch.LongTensor(neg_u)))
        neg_score = torch.mul(neg_emb_u, neg_emb_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding='utf-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        emb_u = self.u_embeddings(Variable(torch.LongTensor(pos_u)))
        emb_v = self.v_embeddings(Variable(torch.LongTensor(pos_v)))
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))
        neg_emb_u = self.u_embeddings(Variable(torch.LongTensor(neg_u)))
        neg_emb_v = self.v_embeddings(Variable(torch.LongTensor(neg_v)))
        neg_score = torch.mul(neg_emb_u, neg_emb_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding='UTF-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


class CBOW(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(CBOW, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(2 * emb_size - 1, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        emb_u = []
        for i in range(len(pos_u)):
            emb_ui = self.u_embeddings(Variable(torch.LongTensor(pos_u[i])))
            emb_u.append(np.sum(emb_ui.data.numpy(), axis=0).tolist())
        emb_u = Variable(torch.FloatTensor(emb_u))
        emb_v = self.v_embeddings(Variable(torch.LongTensor(pos_v)))
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))
        neg_emb_u = []
        for i in range(len(neg_u)):
            neg_emb_ui = self.u_embeddings(Variable(torch.LongTensor(neg_u[i])))
            neg_emb_u.append(np.sum(neg_emb_ui.data.numpy(), axis=0).tolist())
        neg_emb_u = Variable(torch.FloatTensor(neg_emb_u))
        neg_emb_v = self.v_embeddings(Variable(torch.LongTensor(neg_v)))
        neg_score = torch.mul(neg_emb_u, neg_emb_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)

    def forwards(self, pos_u, pos_v, neg_u, neg_v):
        losses = []
        emb_v = []
        for i in range(len(pos_v)):
            emb_v_v = self.u_embeddings(Variable(torch.LongTensor(pos_v[i])))
            emb_v_v_numpy = emb_v_v.data.numpy()
            emb_v_v_numpy = np.sum(emb_v_v_numpy, axis=0)
            emb_v_v_list = emb_v_v_numpy.tolist()
            emb_v.append(emb_v_v_list)
        emb_v = Variable(torch.FloatTensor(emb_v))
        emb_u = self.v_embeddings(Variable(torch.LongTensor(pos_u)))
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)
        losses.append(sum(score))
        neg_emb_v = []
        for i in range(len(neg_v)):
            neg_emb_v_v = self.u_embeddings(Variable(torch.LongTensor(neg_v[i])))
            neg_emb_v_v_numpy = neg_emb_v_v.data.numpy()
            neg_emb_v_v_numpy = np.sum(neg_emb_v_v_numpy, axis=0)
            neg_emb_v_v_list = neg_emb_v_v_numpy.tolist()
            neg_emb_v.append(neg_emb_v_v_list)
        neg_emb_v = Variable(torch.FloatTensor(neg_emb_v))
        neg_emb_u = self.v_embeddings(Variable(torch.LongTensor(neg_u)))
        neg_score = torch.mul(neg_emb_u, neg_emb_v)
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)
        losses.append(sum(neg_score))
        return -1 * sum(losses)

    def save_embedding(self, id2word, file_name):
        embedding = self.v_embeddings.weight.data.numpy()
        fout = open(file_name + 'v', 'w', encoding='UTF-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))
        embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name + 'u', 'w', encoding='UTF-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


class CBOW(nn.Module):

    def __init__(self, vocab_size, context_size, embed_dim, hidden_size):
        super(CBOW, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embed_dim)
        self.linear_1 = nn.Linear(2 * context_size * embed_dim, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_data):
        embeds = self.embed_layer(input_data).view((1, -1))
        output = F.relu(self.linear_1(embeds))
        output = F.log_softmax(self.linear_2(output))
        return output

