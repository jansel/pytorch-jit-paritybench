import sys
_module = sys.modules[__name__]
del sys
demo_server = _module
eval_res = _module
extract = _module
train = _module
visu = _module
vqa = _module
datasets = _module
coco = _module
features = _module
images = _module
utils = _module
vgenome = _module
vgenome_interim = _module
vgenome_processed = _module
vqa = _module
vqa2_interim = _module
vqa_interim = _module
vqa_processed = _module
lib = _module
criterions = _module
dataloader = _module
engine = _module
logger = _module
sampler = _module
utils = _module
models = _module
att = _module
convnets = _module
fusion = _module
noatt = _module
seq2vec = _module
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
xrange = range
wraps = functools.wraps


import time


import re


import torch


from torch.autograd import Variable


import torchvision.transforms as transforms


import numpy


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torchvision.datasets as datasets


import numpy as np


import torch.utils.data as data


import copy


import torch.multiprocessing as multiprocessing


import collections


import math


import itertools


import torch.nn.functional as F


import torchvision.models as pytorch_models


import torchvision.models as models


class AbstractAtt(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(AbstractAtt, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'], self.opt['attention']['dim_v'], 1, 1)
        self.linear_q_att = nn.Linear(self.opt['dim_q'], self.opt['attention']['dim_q'])
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_mm'], self.opt['attention']['nb_glimpses'], 1, 1)
        self.list_linear_v_fusion = None
        self.linear_q_fusion = None
        self.linear_classif = None

    def _fusion_att(self, x_v, x_q):
        raise NotImplementedError

    def _fusion_classif(self, x_v, x_q):
        raise NotImplementedError

    def _attention(self, input_v, x_q_vec):
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)
        x_v = input_v
        x_v = F.dropout(x_v, p=self.opt['attention']['dropout_v'], training=self.training)
        x_v = self.conv_v_att(x_v)
        if 'activation_v' in self.opt['attention']:
            x_v = getattr(F, self.opt['attention']['activation_v'])(x_v)
        x_v = x_v.view(batch_size, self.opt['attention']['dim_v'], width * height)
        x_v = x_v.transpose(1, 2)
        x_q = F.dropout(x_q_vec, p=self.opt['attention']['dropout_q'], training=self.training)
        x_q = self.linear_q_att(x_q)
        if 'activation_q' in self.opt['attention']:
            x_q = getattr(F, self.opt['attention']['activation_q'])(x_q)
        x_q = x_q.view(batch_size, 1, self.opt['attention']['dim_q'])
        x_q = x_q.expand(batch_size, width * height, self.opt['attention']['dim_q'])
        x_att = self._fusion_att(x_v, x_q)
        if 'activation_mm' in self.opt['attention']:
            x_att = getattr(F, self.opt['attention']['activation_mm'])(x_att)
        x_att = F.dropout(x_att, p=self.opt['attention']['dropout_mm'], training=self.training)
        x_att = x_att.view(batch_size, width, height, self.opt['attention']['dim_mm'])
        x_att = x_att.transpose(2, 3).transpose(1, 2)
        x_att = self.conv_att(x_att)
        x_att = x_att.view(batch_size, self.opt['attention']['nb_glimpses'], width * height)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width * height)
            x_att = F.softmax(x_att)
            list_att.append(x_att)
        self.list_att = [x_att.data for x_att in list_att]
        x_v = input_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v = x_v.transpose(1, 2)
        list_v_att = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size, width * height, 1)
            x_att = x_att.expand(batch_size, width * height, self.opt['dim_v'])
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(batch_size, self.opt['dim_v'])
            list_v_att.append(x_v_att)
        return list_v_att

    def _fusion_glimpses(self, list_v_att, x_q_vec):
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att, p=self.opt['fusion']['dropout_v'], training=self.training)
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            if 'activation_v' in self.opt['fusion']:
                x_v = getattr(F, self.opt['fusion']['activation_v'])(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)
        x_q = F.dropout(x_q_vec, p=self.opt['fusion']['dropout_q'], training=self.training)
        x_q = self.linear_q_fusion(x_q)
        if 'activation_q' in self.opt['fusion']:
            x_q = getattr(F, self.opt['fusion']['activation_q'])(x_q)
        x = self._fusion_classif(x_v, x_q)
        return x

    def _classif(self, x):
        if 'activation' in self.opt['classif']:
            x = getattr(F, self.opt['classif']['activation'])(x)
        x = F.dropout(x, p=self.opt['classif']['dropout'], training=self.training)
        x = self.linear_classif(x)
        return x

    def forward(self, input_v, input_q):
        if input_v.dim() != 4 and input_q.dim() != 2:
            raise ValueError
        x_q_vec = self.seq2vec(input_q)
        list_v_att = self._attention(input_v, x_q_vec)
        x = self._fusion_glimpses(list_v_att, x_q_vec)
        x = self._classif(x)
        return x


class MLBAtt(AbstractAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['attention']['dim_v'] = opt['attention']['dim_h']
        opt['attention']['dim_q'] = opt['attention']['dim_h']
        opt['attention']['dim_mm'] = opt['attention']['dim_h']
        super(MLBAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.list_linear_v_fusion = nn.ModuleList([nn.Linear(self.opt['dim_v'], self.opt['fusion']['dim_h']) for i in range(self.opt['attention']['nb_glimpses'])])
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'], self.opt['fusion']['dim_h'] * self.opt['attention']['nb_glimpses'])
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_h'] * self.opt['attention']['nb_glimpses'], self.num_classes)

    def _fusion_att(self, x_v, x_q):
        x_att = torch.mul(x_v, x_q)
        return x_att

    def _fusion_classif(self, x_v, x_q):
        x_mm = torch.mul(x_v, x_q)
        return x_mm


class MutanAtt(AbstractAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['attention']['dim_v'] = opt['attention']['dim_hv']
        opt['attention']['dim_q'] = opt['attention']['dim_hq']
        super(MutanAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion_att = fusion.MutanFusion2d(self.opt['attention'], visual_embedding=False, question_embedding=False)
        self.list_linear_v_fusion = nn.ModuleList([nn.Linear(self.opt['dim_v'], int(self.opt['fusion']['dim_hv'] / opt['attention']['nb_glimpses'])) for i in range(self.opt['attention']['nb_glimpses'])])
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'], self.opt['fusion']['dim_hq'])
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_mm'], self.num_classes)
        self.fusion_classif = fusion.MutanFusion(self.opt['fusion'], visual_embedding=False, question_embedding=False)

    def _fusion_att(self, x_v, x_q):
        return self.fusion_att(x_v, x_q)

    def _fusion_classif(self, x_v, x_q):
        return self.fusion_classif(x_v, x_q)


class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError


class MLBFusion(AbstractFusion):

    def __init__(self, opt):
        super(MLBFusion, self).__init__(opt)
        if 'dim_v' in self.opt:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_h'])
        else:
            None
        if 'dim_q' in self.opt:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_h'])
        else:
            None

    def forward(self, input_v, input_q):
        if 'dim_v' in self.opt:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v
        if 'dim_q' in self.opt:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q
        x_mm = torch.mul(x_q, x_v)
        return x_mm


class MutanFusion(AbstractFusion):

    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion, self).__init__(opt)
        self.visual_embedding = visual_embedding
        self.question_embedding = question_embedding
        if self.visual_embedding:
            self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_hv'])
        else:
            None
        if self.question_embedding:
            self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_hq'])
        else:
            None
        self.list_linear_hv = nn.ModuleList([nn.Linear(self.opt['dim_hv'], self.opt['dim_mm']) for i in range(self.opt['R'])])
        self.list_linear_hq = nn.ModuleList([nn.Linear(self.opt['dim_hq'], self.opt['dim_mm']) for i in range(self.opt['R'])])

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 2:
            raise ValueError
        batch_size = input_v.size(0)
        if self.visual_embedding:
            x_v = F.dropout(input_v, p=self.opt['dropout_v'], training=self.training)
            x_v = self.linear_v(x_v)
            if 'activation_v' in self.opt:
                x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            x_v = input_v
        if self.question_embedding:
            x_q = F.dropout(input_q, p=self.opt['dropout_q'], training=self.training)
            x_q = self.linear_q(x_q)
            if 'activation_q' in self.opt:
                x_q = getattr(F, self.opt['activation_q'])(x_q)
        else:
            x_q = input_q
        x_mm = []
        for i in range(self.opt['R']):
            x_hv = F.dropout(x_v, p=self.opt['dropout_hv'], training=self.training)
            x_hv = self.list_linear_hv[i](x_hv)
            if 'activation_hv' in self.opt:
                x_hv = getattr(F, self.opt['activation_hv'])(x_hv)
            x_hq = F.dropout(x_q, p=self.opt['dropout_hq'], training=self.training)
            x_hq = self.list_linear_hq[i](x_hq)
            if 'activation_hq' in self.opt:
                x_hq = getattr(F, self.opt['activation_hq'])(x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.opt['dim_mm'])
        if 'activation_mm' in self.opt:
            x_mm = getattr(F, self.opt['activation_mm'])(x_mm)
        return x_mm


class MutanFusion2d(MutanFusion):

    def __init__(self, opt, visual_embedding=True, question_embedding=True):
        super(MutanFusion2d, self).__init__(opt, visual_embedding, question_embedding)

    def forward(self, input_v, input_q):
        if input_v.dim() != input_q.dim() and input_v.dim() != 3:
            raise ValueError
        batch_size = input_v.size(0)
        weight_height = input_v.size(1)
        dim_hv = input_v.size(2)
        dim_hq = input_q.size(2)
        if not input_v.is_contiguous():
            input_v = input_v.contiguous()
        if not input_q.is_contiguous():
            input_q = input_q.contiguous()
        x_v = input_v.view(batch_size * weight_height, self.opt['dim_hv'])
        x_q = input_q.view(batch_size * weight_height, self.opt['dim_hq'])
        x_mm = super().forward(x_v, x_q)
        x_mm = x_mm.view(batch_size, weight_height, self.opt['dim_mm'])
        return x_mm


class AbstractNoAtt(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(AbstractNoAtt, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_h'], self.num_classes)

    def _fusion(self, input_v, input_q):
        raise NotImplementedError

    def _classif(self, x):
        if 'activation' in self.opt['classif']:
            x = getattr(F, self.opt['classif']['activation'])(x)
        x = F.dropout(x, p=self.opt['classif']['dropout'], training=self.training)
        x = self.linear_classif(x)
        return x

    def forward(self, input_v, input_q):
        x_q = self.seq2vec(input_q)
        x = self._fusion(input_v, x_q)
        x = self._classif(x)
        return x


class MLBNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(MLBNoAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.MLBFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x


class MutanNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['fusion']['dim_h'] = opt['fusion']['dim_mm']
        super(MutanNoAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.MutanFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x


def process_lengths(input):
    max_length = input.size(1)
    lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
    return lengths


def select_last(x, lengths):
    batch_size = x.size(0)
    seq_length = x.size(1)
    mask = x.data.new().resize_as_(x.data).fill_(0)
    for i in range(batch_size):
        mask[i][lengths[i] - 1].fill_(1)
    mask = Variable(mask)
    x = x.mul(mask)
    x = x.sum(1).view(batch_size, x.size(2))
    return x


class LSTM(nn.Module):

    def __init__(self, vocab, emb_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.vocab = vocab
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 1, embedding_dim=emb_size, padding_idx=0)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, input):
        lengths = process_lengths(input)
        x = self.embedding(input)
        output, hn = self.rnn(x)
        output = select_last(output, lengths)
        return output


class TwoLSTM(nn.Module):

    def __init__(self, vocab, emb_size, hidden_size):
        super(TwoLSTM, self).__init__()
        self.vocab = vocab
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 1, embedding_dim=emb_size, padding_idx=0)
        self.rnn_0 = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=1)
        self.rnn_1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)

    def forward(self, input):
        lengths = process_lengths(input)
        x = self.embedding(input)
        x = getattr(F, 'tanh')(x)
        x_0, hn = self.rnn_0(x)
        vec_0 = select_last(x_0, lengths)
        x_1, hn = self.rnn_1(x_0)
        vec_1 = select_last(x_1, lengths)
        vec_0 = F.dropout(vec_0, p=0.3, training=self.training)
        vec_1 = F.dropout(vec_1, p=0.3, training=self.training)
        output = torch.cat((vec_0, vec_1), 1)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LSTM,
     lambda: ([], {'vocab': [4, 4], 'emb_size': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
    (TwoLSTM,
     lambda: ([], {'vocab': [4, 4], 'emb_size': 4, 'hidden_size': 4}),
     lambda: ([torch.zeros([4, 4], dtype=torch.int64)], {}),
     False),
]

class Test_Cadene_vqa_pytorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

