import sys
_module = sys.modules[__name__]
del sys
pytorch = _module
setup = _module
skipthoughts = _module
dropout = _module
gru = _module
skipthoughts = _module
version = _module
test = _module
dump_features = _module
dump_grus = _module
dump_hashmaps = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import numpy as np


from torch import nn


from torch.autograd import Variable


from itertools import repeat


import torch.nn as nn


import torch.nn.functional as F


import numpy


from collections import OrderedDict


class SequentialDropout(nn.Module):

    def __init__(self, p=0.5):
        super(SequentialDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                'dropout probability has to be between 0 and 1, but got {}'
                .format(p))
        self.p = p
        self.restart = True

    def _make_noise(self, input):
        return Variable(input.data.new().resize_as_(input.data))

    def forward(self, input):
        if self.p > 0 and self.training:
            if self.restart:
                self.noise = self._make_noise(input)
                self.noise.data.bernoulli_(1 - self.p).div_(1 - self.p)
                if self.p == 1:
                    self.noise.data.fill_(0)
                self.noise = self.noise.expand_as(input)
                self.restart = False
            return input.mul(self.noise)
        return input

    def end_of_sequence(self):
        self.restart = True

    def backward(self, grad_output):
        self.end_of_sequence()
        if self.p > 0 and self.training:
            return grad_output.mul(self.noise)
        else:
            return grad_output

    def __repr__(self):
        return type(self).__name__ + '({:.4f})'.format(self.p)


class AbstractGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False):
        super(AbstractGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.weight_ir = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_ii = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_in = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.weight_hr = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hi = nn.Linear(hidden_size, hidden_size, bias=bias_hh)
        self.weight_hn = nn.Linear(hidden_size, hidden_size, bias=bias_hh)

    def forward(self, x, hx=None):
        raise NotImplementedError


class AbstractGRU(nn.Module):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False):
        super(AbstractGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self._load_gru_cell()

    def _load_gru_cell(self):
        raise NotImplementedError

    def forward(self, x, hx=None, max_length=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(x[:, (i), :], hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        output = torch.cat(output, 1)
        return output, hx


urls = {}


class AbstractSkipThoughts(nn.Module):

    def __init__(self, dir_st, vocab, save=False, dropout=0, fixed_emb=False):
        super(AbstractSkipThoughts, self).__init__()
        self.dir_st = dir_st
        self.vocab = vocab
        self.save = save
        self.dropout = dropout
        self.fixed_emb = fixed_emb
        self.embedding = self._load_embedding()
        if fixed_emb:
            self.embedding.weight.requires_grad = False
        self.rnn = self._load_rnn()

    def _get_table_name(self):
        raise NotImplementedError

    def _get_skip_name(self):
        raise NotImplementedError

    def _load_dictionary(self):
        path_dico = os.path.join(self.dir_st, 'dictionary.txt')
        if not os.path.exists(path_dico):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls['dictionary'], self.dir_st))
        with open(path_dico, 'r') as handle:
            dico_list = handle.readlines()
        dico = {word.strip(): idx for idx, word in enumerate(dico_list)}
        return dico

    def _load_emb_params(self):
        table_name = self._get_table_name()
        path_params = os.path.join(self.dir_st, table_name + '.npy')
        if not os.path.exists(path_params):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls[table_name], self.dir_st))
        params = numpy.load(path_params, encoding='latin1', allow_pickle=True)
        return params

    def _load_rnn_params(self):
        skip_name = self._get_skip_name()
        path_params = os.path.join(self.dir_st, skip_name + '.npz')
        if not os.path.exists(path_params):
            os.system('mkdir -p ' + self.dir_st)
            os.system('wget {} -P {}'.format(urls[skip_name], self.dir_st))
        params = numpy.load(path_params, encoding='latin1', allow_pickle=True)
        return params

    def _load_embedding(self):
        if self.save:
            hash_id = hashlib.sha256(pickle.dumps(self.vocab, -1)).hexdigest()
            path = '/tmp/uniskip_embedding_' + str(hash_id) + '.pth'
        if self.save and os.path.exists(path):
            self.embedding = torch.load(path)
        else:
            self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 
                1, embedding_dim=620, padding_idx=0, sparse=False)
            dictionary = self._load_dictionary()
            parameters = self._load_emb_params()
            state_dict = self._make_emb_state_dict(dictionary, parameters)
            self.embedding.load_state_dict(state_dict)
            if self.save:
                torch.save(self.embedding, path)
        return self.embedding

    def _make_emb_state_dict(self, dictionary, parameters):
        weight = torch.zeros(len(self.vocab) + 1, 620)
        unknown_params = parameters[dictionary['UNK']]
        nb_unknown = 0
        for id_weight, word in enumerate(self.vocab):
            if word in dictionary:
                id_params = dictionary[word]
                params = parameters[id_params]
            else:
                params = unknown_params
                nb_unknown += 1
            weight[id_weight + 1] = torch.from_numpy(params)
        state_dict = OrderedDict({'weight': weight})
        if nb_unknown > 0:
            None
        return state_dict

    def _select_last(self, x, lengths):
        batch_size = x.size(0)
        seq_length = x.size(1)
        mask = x.data.new().resize_as_(x.data).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i] - 1].fill_(1)
        mask = Variable(mask)
        x = x.mul(mask)
        x = x.sum(1).view(batch_size, -1)
        return x

    def _select_last_old(self, input, lengths):
        batch_size = input.size(0)
        x = []
        for i in range(batch_size):
            x.append(input[i, lengths[i] - 1].view(1, -1))
        output = torch.cat(x, 0)
        return output

    def _process_lengths(self, input):
        max_length = input.size(1)
        lengths = list(max_length - input.data.eq(0).sum(1).squeeze())
        return lengths

    def _load_rnn(self):
        raise NotImplementedError

    def _make_rnn_state_dict(self, p):
        raise NotImplementedError

    def forward(self, input, lengths=None):
        raise NotImplementedError


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Cadene_skip_thoughts_torch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(SequentialDropout(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

