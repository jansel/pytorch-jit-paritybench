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
            raise ValueError('dropout probability has to be between 0 and 1, but got {}'.format(p))
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


class GRUCell(AbstractGRUCell):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False):
        super(GRUCell, self).__init__(input_size, hidden_size, bias_ih, bias_hh)

    def forward(self, x, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        r = F.sigmoid(self.weight_ir(x) + self.weight_hr(hx))
        i = F.sigmoid(self.weight_ii(x) + self.weight_hi(hx))
        n = F.tanh(self.weight_in(x) + r * self.weight_hn(hx))
        hx = (1 - i) * n + i * hx
        return hx


class BayesianGRUCell(AbstractGRUCell):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False, dropout=0.25):
        super(BayesianGRUCell, self).__init__(input_size, hidden_size, bias_ih, bias_hh)
        self.set_dropout(dropout)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.drop_ir = SequentialDropout(p=dropout)
        self.drop_ii = SequentialDropout(p=dropout)
        self.drop_in = SequentialDropout(p=dropout)
        self.drop_hr = SequentialDropout(p=dropout)
        self.drop_hi = SequentialDropout(p=dropout)
        self.drop_hn = SequentialDropout(p=dropout)

    def end_of_sequence(self):
        self.drop_ir.end_of_sequence()
        self.drop_ii.end_of_sequence()
        self.drop_in.end_of_sequence()
        self.drop_hr.end_of_sequence()
        self.drop_hi.end_of_sequence()
        self.drop_hn.end_of_sequence()

    def forward(self, x, hx=None):
        if hx is None:
            hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        x_ir = self.drop_ir(x)
        x_ii = self.drop_ii(x)
        x_in = self.drop_in(x)
        x_hr = self.drop_hr(hx)
        x_hi = self.drop_hi(hx)
        x_hn = self.drop_hn(hx)
        r = F.sigmoid(self.weight_ir(x_ir) + self.weight_hr(x_hr))
        i = F.sigmoid(self.weight_ii(x_ii) + self.weight_hi(x_hi))
        n = F.tanh(self.weight_in(x_in) + r * self.weight_hn(x_hn))
        hx = (1 - i) * n + i * hx
        return hx


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


class GRU(AbstractGRU):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False):
        super(GRU, self).__init__(input_size, hidden_size, bias_ih, bias_hh)

    def _load_gru_cell(self):
        self.gru_cell = GRUCell(self.input_size, self.hidden_size, self.bias_ih, self.bias_hh)


class BayesianGRU(AbstractGRU):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False, dropout=0.25):
        self.dropout = dropout
        super(BayesianGRU, self).__init__(input_size, hidden_size, bias_ih, bias_hh)

    def _load_gru_cell(self):
        self.gru_cell = BayesianGRUCell(self.input_size, self.hidden_size, self.bias_ih, self.bias_hh, dropout=self.dropout)

    def set_dropout(self, dropout):
        self.dropout = dropout
        self.gru_cell.set_dropout(dropout)

    def forward(self, x, hx=None, max_length=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        if max_length is None:
            max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(x[:, (i), :], hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        self.gru_cell.end_of_sequence()
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
            self.embedding = nn.Embedding(num_embeddings=len(self.vocab) + 1, embedding_dim=620, padding_idx=0, sparse=False)
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


class AbstractUniSkip(AbstractSkipThoughts):

    def __init__(self, dir_st, vocab, save=False, dropout=0, fixed_emb=False):
        super(AbstractUniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)

    def _get_table_name(self):
        return 'utable'

    def _get_skip_name(self):
        return 'uni_skip'


class UniSkip(AbstractUniSkip):

    def __init__(self, dir_st, vocab, save=False, dropout=0.25, fixed_emb=False):
        super(UniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)

    def _load_rnn(self):
        self.rnn = nn.GRU(input_size=620, hidden_size=2400, batch_first=True, dropout=self.dropout)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih_l0'] = torch.zeros(7200)
        s['bias_hh_l0'] = torch.zeros(7200)
        s['weight_ih_l0'] = torch.zeros(7200, 620)
        s['weight_hh_l0'] = torch.zeros(7200, 2400)
        s['weight_ih_l0'][:4800] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih_l0'][4800:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_ih_l0'][:4800] = torch.from_numpy(p['encoder_b'])
        s['bias_ih_l0'][4800:] = torch.from_numpy(p['encoder_bx'])
        s['weight_hh_l0'][:4800] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh_l0'][4800:] = torch.from_numpy(p['encoder_Ux']).t()
        return s

    def forward(self, input, lengths=None):
        if lengths is None:
            lengths = self._process_lengths(input)
        x = self.embedding(input)
        x, hn = self.rnn(x)
        if lengths:
            x = self._select_last(x, lengths)
        return x


class DropUniSkip(AbstractUniSkip):

    def __init__(self, dir_st, vocab, save=False, dropout=0.25, fixed_emb=False):
        super(DropUniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)
        self.seq_drop_x = SequentialDropout(p=self.dropout)
        self.seq_drop_h = SequentialDropout(p=self.dropout)

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih'] = torch.zeros(7200)
        s['bias_hh'] = torch.zeros(7200)
        s['weight_ih'] = torch.zeros(7200, 620)
        s['weight_hh'] = torch.zeros(7200, 2400)
        s['weight_ih'][:4800] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih'][4800:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_ih'][:4800] = torch.from_numpy(p['encoder_b'])
        s['bias_ih'][4800:] = torch.from_numpy(p['encoder_bx'])
        s['weight_hh'][:4800] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh'][4800:] = torch.from_numpy(p['encoder_Ux']).t()
        return s

    def _load_rnn(self):
        self.rnn = nn.GRUCell(620, 2400)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def forward(self, input, lengths=None):
        batch_size = input.size(0)
        seq_length = input.size(1)
        if lengths is None:
            lengths = self._process_lengths(input)
        x = self.embedding(input)
        hx = Variable(x.data.new().resize_((batch_size, 2400)).fill_(0))
        output = []
        for i in range(seq_length):
            if self.dropout > 0:
                input_gru_cell = self.seq_drop_x(x[:, (i), :])
                hx = self.seq_drop_h(hx)
            else:
                input_gru_cell = x[:, (i), :]
            hx = self.rnn(input_gru_cell, hx)
            output.append(hx.view(batch_size, 1, 2400))
        output = torch.cat(output, 1)
        if lengths:
            output = self._select_last(output, lengths)
        return output


class BayesianUniSkip(AbstractUniSkip):

    def __init__(self, dir_st, vocab, save=False, dropout=0.25, fixed_emb=False):
        super(BayesianUniSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['gru_cell.weight_ir.weight'] = torch.from_numpy(p['encoder_W']).t()[:2400]
        s['gru_cell.weight_ii.weight'] = torch.from_numpy(p['encoder_W']).t()[2400:]
        s['gru_cell.weight_in.weight'] = torch.from_numpy(p['encoder_Wx']).t()
        s['gru_cell.weight_ir.bias'] = torch.from_numpy(p['encoder_b'])[:2400]
        s['gru_cell.weight_ii.bias'] = torch.from_numpy(p['encoder_b'])[2400:]
        s['gru_cell.weight_in.bias'] = torch.from_numpy(p['encoder_bx'])
        s['gru_cell.weight_hr.weight'] = torch.from_numpy(p['encoder_U']).t()[:2400]
        s['gru_cell.weight_hi.weight'] = torch.from_numpy(p['encoder_U']).t()[2400:]
        s['gru_cell.weight_hn.weight'] = torch.from_numpy(p['encoder_Ux']).t()
        return s

    def _load_rnn(self):
        self.rnn = BayesianGRU(620, 2400, dropout=self.dropout)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def forward(self, input, lengths=None):
        if lengths is None:
            lengths = self._process_lengths(input)
        max_length = max(lengths)
        x = self.embedding(input)
        x, hn = self.rnn(x, max_length=max_length)
        if lengths:
            x = self._select_last(x, lengths)
        return x


class AbstractBiSkip(AbstractSkipThoughts):

    def __init__(self, dir_st, vocab, save=False, dropout=0, fixed_emb=False):
        super(AbstractBiSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)

    def _get_table_name(self):
        return 'btable'

    def _get_skip_name(self):
        return 'bi_skip'


class BiSkip(AbstractBiSkip):

    def __init__(self, dir_st, vocab, save=False, dropout=0.25, fixed_emb=False):
        super(BiSkip, self).__init__(dir_st, vocab, save, dropout, fixed_emb)

    def _load_rnn(self):
        self.rnn = nn.GRU(input_size=620, hidden_size=1200, batch_first=True, dropout=self.dropout, bidirectional=True)
        parameters = self._load_rnn_params()
        state_dict = self._make_rnn_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def _make_rnn_state_dict(self, p):
        s = OrderedDict()
        s['bias_ih_l0'] = torch.zeros(3600)
        s['bias_hh_l0'] = torch.zeros(3600)
        s['weight_ih_l0'] = torch.zeros(3600, 620)
        s['weight_hh_l0'] = torch.zeros(3600, 1200)
        s['bias_ih_l0_reverse'] = torch.zeros(3600)
        s['bias_hh_l0_reverse'] = torch.zeros(3600)
        s['weight_ih_l0_reverse'] = torch.zeros(3600, 620)
        s['weight_hh_l0_reverse'] = torch.zeros(3600, 1200)
        s['weight_ih_l0'][:2400] = torch.from_numpy(p['encoder_W']).t()
        s['weight_ih_l0'][2400:] = torch.from_numpy(p['encoder_Wx']).t()
        s['bias_ih_l0'][:2400] = torch.from_numpy(p['encoder_b'])
        s['bias_ih_l0'][2400:] = torch.from_numpy(p['encoder_bx'])
        s['weight_hh_l0'][:2400] = torch.from_numpy(p['encoder_U']).t()
        s['weight_hh_l0'][2400:] = torch.from_numpy(p['encoder_Ux']).t()
        s['weight_ih_l0_reverse'][:2400] = torch.from_numpy(p['encoder_r_W']).t()
        s['weight_ih_l0_reverse'][2400:] = torch.from_numpy(p['encoder_r_Wx']).t()
        s['bias_ih_l0_reverse'][:2400] = torch.from_numpy(p['encoder_r_b'])
        s['bias_ih_l0_reverse'][2400:] = torch.from_numpy(p['encoder_r_bx'])
        s['weight_hh_l0_reverse'][:2400] = torch.from_numpy(p['encoder_r_U']).t()
        s['weight_hh_l0_reverse'][2400:] = torch.from_numpy(p['encoder_r_Ux']).t()
        return s

    def _argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def forward(self, input, lengths=None):
        batch_size = input.size(0)
        if lengths is None:
            lengths = self._process_lengths(input)
        sorted_lengths = sorted(lengths)
        sorted_lengths = sorted_lengths[::-1]
        idx = self._argsort(lengths)
        idx = idx[::-1]
        inverse_idx = self._argsort(idx)
        idx = Variable(torch.LongTensor(idx))
        inverse_idx = Variable(torch.LongTensor(inverse_idx))
        if input.data.is_cuda:
            idx = idx
            inverse_idx = inverse_idx
        x = torch.index_select(input, 0, idx)
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, sorted_lengths, batch_first=True)
        x, hn = self.rnn(x)
        hn = hn.transpose(0, 1)
        hn = hn.contiguous()
        hn = hn.view(batch_size, 2 * hn.size(2))
        hn = torch.index_select(hn, 0, inverse_idx)
        return hn


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BayesianGRU,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (BayesianGRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GRU,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GRUCell,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SequentialDropout,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Cadene_skip_thoughts_torch(_paritybench_base):
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

