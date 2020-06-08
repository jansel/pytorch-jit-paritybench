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
vqa2_interim = _module
vqa_interim = _module
vqa_processed = _module
lib = _module
criterions = _module
dataloader = _module
engine = _module
logger = _module
sampler = _module
models = _module
att = _module
convnets = _module
fusion = _module
noatt = _module
seq2vec = _module
utils = _module

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


import re


import torch


from torch.autograd import Variable


import numpy


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.nn.functional as F


import copy


class AbstractAtt(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(AbstractAtt, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'], self.opt['attention'
            ]['dim_v'], 1, 1)
        self.linear_q_att = nn.Linear(self.opt['dim_q'], self.opt[
            'attention']['dim_q'])
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_mm'], self.opt
            ['attention']['nb_glimpses'], 1, 1)
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
        x_v = F.dropout(x_v, p=self.opt['attention']['dropout_v'], training
            =self.training)
        x_v = self.conv_v_att(x_v)
        if 'activation_v' in self.opt['attention']:
            x_v = getattr(F, self.opt['attention']['activation_v'])(x_v)
        x_v = x_v.view(batch_size, self.opt['attention']['dim_v'], width *
            height)
        x_v = x_v.transpose(1, 2)
        x_q = F.dropout(x_q_vec, p=self.opt['attention']['dropout_q'],
            training=self.training)
        x_q = self.linear_q_att(x_q)
        if 'activation_q' in self.opt['attention']:
            x_q = getattr(F, self.opt['attention']['activation_q'])(x_q)
        x_q = x_q.view(batch_size, 1, self.opt['attention']['dim_q'])
        x_q = x_q.expand(batch_size, width * height, self.opt['attention'][
            'dim_q'])
        x_att = self._fusion_att(x_v, x_q)
        if 'activation_mm' in self.opt['attention']:
            x_att = getattr(F, self.opt['attention']['activation_mm'])(x_att)
        x_att = F.dropout(x_att, p=self.opt['attention']['dropout_mm'],
            training=self.training)
        x_att = x_att.view(batch_size, width, height, self.opt['attention']
            ['dim_mm'])
        x_att = x_att.transpose(2, 3).transpose(1, 2)
        x_att = self.conv_att(x_att)
        x_att = x_att.view(batch_size, self.opt['attention']['nb_glimpses'],
            width * height)
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
            x_v = F.dropout(x_v_att, p=self.opt['fusion']['dropout_v'],
                training=self.training)
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            if 'activation_v' in self.opt['fusion']:
                x_v = getattr(F, self.opt['fusion']['activation_v'])(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)
        x_q = F.dropout(x_q_vec, p=self.opt['fusion']['dropout_q'],
            training=self.training)
        x_q = self.linear_q_fusion(x_q)
        if 'activation_q' in self.opt['fusion']:
            x_q = getattr(F, self.opt['fusion']['activation_q'])(x_q)
        x = self._fusion_classif(x_v, x_q)
        return x

    def _classif(self, x):
        if 'activation' in self.opt['classif']:
            x = getattr(F, self.opt['classif']['activation'])(x)
        x = F.dropout(x, p=self.opt['classif']['dropout'], training=self.
            training)
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


class AbstractFusion(nn.Module):

    def __init__(self, opt={}):
        super(AbstractFusion, self).__init__()
        self.opt = opt

    def forward(self, input_v, input_q):
        raise NotImplementedError


class AbstractNoAtt(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(AbstractNoAtt, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_h'], self.
            num_classes)

    def _fusion(self, input_v, input_q):
        raise NotImplementedError

    def _classif(self, x):
        if 'activation' in self.opt['classif']:
            x = getattr(F, self.opt['classif']['activation'])(x)
        x = F.dropout(x, p=self.opt['classif']['dropout'], training=self.
            training)
        x = self.linear_classif(x)
        return x

    def forward(self, input_v, input_q):
        x_q = self.seq2vec(input_q)
        x = self._fusion(input_v, x_q)
        x = self._classif(x)
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
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 1,
            embedding_dim=emb_size, padding_idx=0)
        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
            num_layers=num_layers)

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
        self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 1,
            embedding_dim=emb_size, padding_idx=0)
        self.rnn_0 = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
            num_layers=1)
        self.rnn_1 = nn.LSTM(input_size=hidden_size, hidden_size=
            hidden_size, num_layers=1)

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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Cadene_vqa_pytorch(_paritybench_base):
    pass
    @_fails_compile()

    def test_000(self):
        self._check(LSTM(*[], **{'vocab': [4, 4], 'emb_size': 4, 'hidden_size': 4, 'num_layers': 1}), [torch.zeros([4, 4], dtype=torch.int64)], {})
    @_fails_compile()

    def test_001(self):
        self._check(TwoLSTM(*[], **{'vocab': [4, 4], 'emb_size': 4, 'hidden_size': 4}), [torch.zeros([4, 4], dtype=torch.int64)], {})
