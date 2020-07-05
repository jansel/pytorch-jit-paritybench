import sys
_module = sys.modules[__name__]
del sys
corenlp = _module
pipeline_demo = _module
lang2code = _module
setup = _module
stanza = _module
_version = _module
models = _module
_training_logging = _module
charlm = _module
common = _module
beam = _module
biaffine = _module
char_model = _module
chuliu_edmonds = _module
constant = _module
crf = _module
data = _module
doc = _module
dropout = _module
hlstm = _module
loss = _module
packed_lstm = _module
pretrain = _module
seq2seq_constant = _module
seq2seq_model = _module
seq2seq_modules = _module
seq2seq_utils = _module
trainer = _module
utils = _module
vocab = _module
depparse = _module
model = _module
scorer = _module
trainer = _module
identity_lemmatizer = _module
lemma = _module
edit = _module
trainer = _module
lemmatizer = _module
mwt = _module
trainer = _module
mwt_expander = _module
ner = _module
model = _module
trainer = _module
ner_tagger = _module
parser = _module
pos = _module
build_xpos_vocab_factory = _module
model = _module
trainer = _module
xpos_vocab_factory = _module
tagger = _module
tokenize = _module
model = _module
trainer = _module
tokenizer = _module
pipeline = _module
_constants = _module
core = _module
demo_server = _module
depparse_processor = _module
lemma_processor = _module
mwt_processor = _module
ner_processor = _module
pos_processor = _module
processor = _module
tokenize_processor = _module
CoreNLP_pb2 = _module
protobuf = _module
server = _module
annotator = _module
client = _module
main = _module
avg_sent_len = _module
conll = _module
conll18_ud_eval = _module
contract_mwt = _module
helper_func = _module
jieba = _module
max_mwt_length = _module
postprocess_vietnamese_tokenizer_data = _module
prepare_ner_data = _module
prepare_resources = _module
prepare_tokenizer_data = _module
resources = _module
select_backoff = _module
spacy = _module
tests = _module
test_client = _module
test_depparse = _module
test_english_pipeline = _module
test_lemmatizer = _module
test_mwt_expander = _module
test_ner_tagger = _module
test_protobuf = _module
test_requirements = _module
test_run_pipeline = _module
test_server_misc = _module
test_server_request = _module
test_server_start = _module
test_tagger = _module
test_tokenizer = _module

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


import random


from copy import copy


from collections import Counter


import numpy as np


import torch


import math


import logging


import time


import torch.nn as nn


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import PackedSequence


from numbers import Number


from torch import nn


import torch.nn.init as init


import torch.optim as optim


class PairwiseBilinear(nn.Module):
    """ A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)"""

    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()
        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.Tensor(input1_size, input2_size, output_size))
        self.bias = nn.Parameter(torch.Tensor(output_size)) if bias else 0

    def forward(self, input1, input2):
        input1_size = list(input1.size())
        input2_size = list(input2.size())
        output_size = [input1_size[0], input1_size[1], input2_size[1], self.output_size]
        intermediate = torch.mm(input1.view(-1, input1_size[-1]), self.weight.view(-1, self.input2_size * self.output_size))
        input2 = input2.transpose(1, 2)
        output = intermediate.view(input1_size[0], input1_size[1] * self.output_size, input2_size[2]).bmm(input2)
        output = output.view(input1_size[0], input1_size[1], self.output_size, input2_size[1]).transpose(2, 3)
        return output


class BiaffineScorer(nn.Module):

    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = nn.Bilinear(input1_size + 1, input2_size + 1, output_size)
        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class PairwiseBiaffineScorer(nn.Module):

    def __init__(self, input1_size, input2_size, output_size):
        super().__init__()
        self.W_bilin = PairwiseBilinear(input1_size + 1, input2_size + 1, output_size)
        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(*input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(*input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class DeepBiaffineScorer(nn.Module):

    def __init__(self, input1_size, input2_size, hidden_size, output_size, hidden_func=F.relu, dropout=0, pairwise=True):
        super().__init__()
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)
        self.hidden_func = hidden_func
        if pairwise:
            self.scorer = PairwiseBiaffineScorer(hidden_size, hidden_size, output_size)
        else:
            self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))), self.dropout(self.hidden_func(self.W2(input2))))


def tensor_unsort(sorted_tensor, oidx):
    """
    Unsort a sorted tensor on its 0-th dimension, based on the original idx.
    """
    assert sorted_tensor.size(0) == len(oidx), 'Number of list elements must match with original indices.'
    backidx = [x[0] for x in sorted(enumerate(oidx), key=lambda x: x[1])]
    return sorted_tensor[backidx]


class CharacterModel(nn.Module):

    def __init__(self, args, vocab, pad=False, bidirectional=False, attention=True):
        super().__init__()
        self.args = args
        self.pad = pad
        self.num_dir = 2 if bidirectional else 1
        self.attn = attention
        self.char_emb = nn.Embedding(len(vocab['char']), self.args['char_emb_dim'], padding_idx=0)
        if self.attn:
            self.char_attn = nn.Linear(self.num_dir * self.args['char_hidden_dim'], 1, bias=False)
            self.char_attn.weight.data.zero_()
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True, dropout=0 if self.args['char_num_layers'] == 1 else args['dropout'], rec_dropout=self.args['char_rec_dropout'], bidirectional=bidirectional)
        self.charlstm_h_init = nn.Parameter(torch.zeros(self.num_dir * self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.charlstm_c_init = nn.Parameter(torch.zeros(self.num_dir * self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, chars, chars_mask, word_orig_idx, sentlens, wordlens):
        embs = self.dropout(self.char_emb(chars))
        batch_size = embs.size(0)
        embs = pack_padded_sequence(embs, wordlens, batch_first=True)
        output = self.charlstm(embs, wordlens, hx=(self.charlstm_h_init.expand(self.num_dir * self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous(), self.charlstm_c_init.expand(self.num_dir * self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous()))
        if self.attn:
            char_reps = output[0]
            weights = torch.sigmoid(self.char_attn(self.dropout(char_reps.data)))
            char_reps = PackedSequence(char_reps.data * weights, char_reps.batch_sizes)
            char_reps, _ = pad_packed_sequence(char_reps, batch_first=True)
            res = char_reps.sum(1)
        else:
            h, c = output[1]
            res = h[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        res = tensor_unsort(res, word_orig_idx)
        res = pack_sequence(res.split(sentlens))
        if self.pad:
            res = pad_packed_sequence(res, batch_first=True)[0]
        return res


UNK = '<UNK>'


class BaseVocab:
    """ A base class for common vocabulary operations. Each subclass should at least 
    implement its own build_vocab() function."""

    def __init__(self, data=None, lang='', idx=0, cutoff=0, lower=False):
        self.data = data
        self.lang = lang
        self.idx = idx
        self.cutoff = cutoff
        self.lower = lower
        if data is not None:
            self.build_vocab()
        self.state_attrs = ['lang', 'idx', 'cutoff', 'lower', '_unit2id', '_id2unit']

    def build_vocab(self):
        raise NotImplementedError()

    def state_dict(self):
        """ Returns a dictionary containing all states that are necessary to recover
        this vocab. Useful for serialization."""
        state = OrderedDict()
        for attr in self.state_attrs:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        """ Returns a new Vocab instance constructed from a state dict. """
        new = cls()
        for attr, value in state_dict.items():
            setattr(new, attr, value)
        return new

    def normalize_unit(self, unit):
        if self.lower:
            return unit.lower()
        return unit

    def unit2id(self, unit):
        unit = self.normalize_unit(unit)
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            return self._unit2id[UNK]

    def id2unit(self, id):
        return self._id2unit[id]

    def map(self, units):
        return [self.unit2id(x) for x in units]

    def unmap(self, ids):
        return [self.id2unit(x) for x in ids]

    def __len__(self):
        return len(self._id2unit)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.unit2id(key)
        elif isinstance(key, int) or isinstance(key, list):
            return self.id2unit(key)
        else:
            raise TypeError('Vocab key must be one of str, list, or int')

    def __contains__(self, key):
        return key in self._unit2id

    @property
    def size(self):
        return len(self)


EOS = '<EOS>'


PAD = '<PAD>'


SOS = '<SOS>'


VOCAB_PREFIX = [PAD, UNK, SOS, EOS]


class CharVocab(BaseVocab):

    def build_vocab(self):
        if type(self.data[0][0]) is list:
            counter = Counter([c for sent in self.data for w in sent for c in w[self.idx]])
            for k in list(counter.keys()):
                if counter[k] < self.cutoff:
                    del counter[k]
        else:
            counter = Counter([c for sent in self.data for c in sent])
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: (counter[k], k), reverse=True))
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}


UNK_ID = 1


def unsort(sorted_list, oidx):
    """
    Unsort a sorted list, based on the original idx.
    """
    assert len(sorted_list) == len(oidx), 'Number of list elements must match with original indices.'
    _, unsorted = [list(t) for t in zip(*sorted(zip(oidx, sorted_list)))]
    return unsorted


class CharacterLanguageModel(nn.Module):

    def __init__(self, args, vocab, pad=False, is_forward_lm=True):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.is_forward_lm = is_forward_lm
        self.pad = pad
        self.finetune = True
        self.char_emb = nn.Embedding(len(self.vocab['char']), self.args['char_emb_dim'], padding_idx=None)
        self.charlstm = PackedLSTM(self.args['char_emb_dim'], self.args['char_hidden_dim'], self.args['char_num_layers'], batch_first=True, dropout=0 if self.args['char_num_layers'] == 1 else args['char_dropout'], rec_dropout=self.args['char_rec_dropout'], bidirectional=False)
        self.charlstm_h_init = nn.Parameter(torch.zeros(self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.charlstm_c_init = nn.Parameter(torch.zeros(self.args['char_num_layers'], 1, self.args['char_hidden_dim']))
        self.decoder = nn.Linear(self.args['char_hidden_dim'], len(self.vocab['char']))
        self.dropout = nn.Dropout(args['char_dropout'])
        self.char_dropout = SequenceUnitDropout(args.get('char_unit_dropout', 0), UNK_ID)

    def forward(self, chars, charlens, hidden=None):
        chars = self.char_dropout(chars)
        embs = self.dropout(self.char_emb(chars))
        batch_size = embs.size(0)
        embs = pack_padded_sequence(embs, charlens, batch_first=True)
        if hidden is None:
            hidden = self.charlstm_h_init.expand(self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous(), self.charlstm_c_init.expand(self.args['char_num_layers'], batch_size, self.args['char_hidden_dim']).contiguous()
        output, hidden = self.charlstm(embs, charlens, hx=hidden)
        output = self.dropout(pad_packed_sequence(output, batch_first=True)[0])
        decoded = self.decoder(output)
        return output, hidden, decoded

    def get_representation(self, chars, charoffsets, charlens, char_orig_idx):
        with torch.no_grad():
            output, _, _ = self.forward(chars, charlens)
            res = [output[i, offsets] for i, offsets in enumerate(charoffsets)]
            res = unsort(res, char_orig_idx)
            res = pack_sequence(res)
            if self.pad:
                res = pad_packed_sequence(res, batch_first=True)[0]
        return res

    def train(self, mode=True):
        """
        Override the default train() function, so that when self.finetune == False, the training mode 
        won't be impacted by the parent models' status change.
        """
        if not mode:
            super().train(mode)
        elif self.finetune:
            super().train(mode)

    def save(self, filename):
        state = {'vocab': self.vocab['char'].state_dict(), 'args': self.args, 'state_dict': self.state_dict(), 'pad': self.pad, 'is_forward_lm': self.is_forward_lm}
        torch.save(state, filename)

    @classmethod
    def load(cls, filename, finetune=False):
        state = torch.load(filename, lambda storage, loc: storage)
        vocab = {'char': CharVocab.load_state_dict(state['vocab'])}
        model = cls(state['args'], vocab, state['pad'], state['is_forward_lm'])
        model.load_state_dict(state['state_dict'])
        model.eval()
        model.finetune = finetune
        return model


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


def set_cuda(var, cuda):
    if cuda:
        return var.cuda()
    return var


class CRFLoss(nn.Module):
    """
    Calculate log-space crf loss, given unary potentials, a transition matrix
    and gold tag sequences.
    """

    def __init__(self, num_tag, batch_average=True):
        super().__init__()
        self._transitions = nn.Parameter(torch.zeros(num_tag, num_tag))
        self._batch_average = batch_average

    def forward(self, inputs, masks, tag_indices):
        """
        inputs: batch_size x seq_len x num_tags
        masks: batch_size x seq_len
        tag_indices: batch_size x seq_len
        
        @return:
            loss: CRF negative log likelihood on all instances.
            transitions: the transition matrix
        """
        self.bs, self.sl, self.nc = inputs.size()
        unary_scores = self.crf_unary_score(inputs, masks, tag_indices)
        binary_scores = self.crf_binary_score(inputs, masks, tag_indices)
        log_norm = self.crf_log_norm(inputs, masks, tag_indices)
        log_likelihood = unary_scores + binary_scores - log_norm
        loss = torch.sum(-log_likelihood)
        if self._batch_average:
            loss = loss / self.bs
        else:
            total = masks.eq(0).sum()
            loss = loss / (total + 1e-08)
        return loss, self._transitions

    def crf_unary_score(self, inputs, masks, tag_indices):
        """
        @return:
            unary_scores: batch_size
        """
        flat_inputs = inputs.view(self.bs, -1)
        flat_tag_indices = tag_indices + set_cuda(torch.arange(self.sl).long().unsqueeze(0) * self.nc, tag_indices.is_cuda)
        unary_scores = torch.gather(flat_inputs, 1, flat_tag_indices).view(self.bs, -1)
        unary_scores.masked_fill_(masks, 0)
        return unary_scores.sum(dim=1)

    def crf_binary_score(self, inputs, masks, tag_indices):
        """
        @return:
            binary_scores: batch_size
        """
        nt = tag_indices.size(-1) - 1
        start_indices = tag_indices[:, :nt]
        end_indices = tag_indices[:, 1:]
        flat_transition_indices = start_indices * self.nc + end_indices
        flat_transition_indices = flat_transition_indices.view(-1)
        flat_transition_matrix = self._transitions.view(-1)
        binary_scores = torch.gather(flat_transition_matrix, 0, flat_transition_indices).view(self.bs, -1)
        score_masks = masks[:, 1:]
        binary_scores.masked_fill_(score_masks, 0)
        return binary_scores.sum(dim=1)

    def crf_log_norm(self, inputs, masks, tag_indices):
        """
        Calculate the CRF partition in log space for each instance, following:
            http://www.cs.columbia.edu/~mcollins/fb.pdf
        @return:
            log_norm: batch_size
        """
        start_inputs = inputs[:, (0), :]
        rest_inputs = inputs[:, 1:, :]
        rest_masks = masks[:, 1:]
        alphas = start_inputs
        trans = self._transitions.unsqueeze(0)
        for i in range(rest_inputs.size(1)):
            transition_scores = alphas.unsqueeze(2) + trans
            new_alphas = rest_inputs[:, (i), :] + log_sum_exp(transition_scores, dim=1)
            m = rest_masks[:, (i)].unsqueeze(1).expand_as(new_alphas)
            new_alphas.masked_scatter_(m, alphas.masked_select(m))
            alphas = new_alphas
        log_norm = log_sum_exp(alphas, dim=1)
        return log_norm


class WordDropout(nn.Module):
    """ A word dropout layer that's designed for embedded inputs (e.g., any inputs to an LSTM layer).
    Given a batch of embedded inputs, this layer randomly set some of them to be a replacement state.
    Note that this layer assumes the last dimension of the input to be the hidden dimension of a unit.
    """

    def __init__(self, dropprob):
        super().__init__()
        self.dropprob = dropprob

    def forward(self, x, replacement=None):
        if not self.training or self.dropprob == 0:
            return x
        masksize = [y for y in x.size()]
        masksize[-1] = 1
        dropmask = torch.rand(*masksize, device=x.device) < self.dropprob
        res = x.masked_fill(dropmask, 0)
        if replacement is not None:
            res = res + dropmask.float() * replacement
        return res

    def extra_repr(self):
        return 'p={}'.format(self.dropprob)


class LockedDropout(nn.Module):
    """
    A variant of dropout layer that consistently drops out the same parameters over time. Also known as the variational dropout. 
    This implentation was modified from the LockedDropout implementation in the flair library (https://github.com/zalandoresearch/flair).
    """

    def __init__(self, dropprob, batch_first=True):
        super().__init__()
        self.dropprob = dropprob
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training or self.dropprob == 0:
            return x
        if not self.batch_first:
            m = x.new_empty(1, x.size(1), x.size(2), requires_grad=False).bernoulli_(1 - self.dropprob)
        else:
            m = x.new_empty(x.size(0), 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropprob)
        mask = m.div(1 - self.dropprob).expand_as(x)
        return mask * x

    def extra_repr(self):
        return 'p={}'.format(self.dropprob)


class SequenceUnitDropout(nn.Module):
    """ A unit dropout layer that's designed for input of sequence units (e.g., word sequence, char sequence, etc.).
    Given a sequence of unit indices, this layer randomly set some of them to be a replacement id (usually set to be <UNK>).
    """

    def __init__(self, dropprob, replacement_id):
        super().__init__()
        self.dropprob = dropprob
        self.replacement_id = replacement_id

    def forward(self, x):
        """ :param: x must be a LongTensor of unit indices. """
        if not self.training or self.dropprob == 0:
            return x
        masksize = [y for y in x.size()]
        dropmask = torch.rand(*masksize, device=x.device) < self.dropprob
        res = x.masked_fill(dropmask, self.replacement_id)
        return res

    def extra_repr(self):
        return 'p={}, replacement_id={}'.format(self.dropprob, self.replacement_id)


class HLSTMCell(nn.modules.rnn.RNNCellBase):
    """
    A Highway LSTM Cell as proposed in Zhang et al. (2018) Highway Long Short-Term Memory RNNs for 
    Distant Speech Recognition.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(HLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.Wg = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.gate = nn.Linear(input_size + 2 * hidden_size, hidden_size, bias=bias)

    def forward(self, input, c_l_minus_one=None, hx=None):
        self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            hx = hx, hx
        if c_l_minus_one is None:
            c_l_minus_one = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        self.check_forward_hidden(input, hx[0], '[0]')
        self.check_forward_hidden(input, hx[1], '[1]')
        self.check_forward_hidden(input, c_l_minus_one, 'c_l_minus_one')
        rec_input = torch.cat([input, hx[0]], 1)
        i = F.sigmoid(self.Wi(rec_input))
        f = F.sigmoid(self.Wf(rec_input))
        o = F.sigmoid(self.Wo(rec_input))
        g = F.tanh(self.Wg(rec_input))
        gate = F.sigmoid(self.gate(torch.cat([c_l_minus_one, hx[1], input], 1)))
        c = gate * c_l_minus_one + f * hx[1] + i * g
        h = o * F.tanh(c)
        return h, c


class HighwayLSTM(nn.Module):
    """
    A Highway LSTM network, as used in the original Tensorflow version of the Dozat parser. Note that this
    is independent from the HLSTMCell above.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0, bidirectional=False, rec_dropout=0, highway_func=None, pad=False):
        super(HighwayLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.dropout_state = {}
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.highway_func = highway_func
        self.pad = pad
        self.lstm = nn.ModuleList()
        self.highway = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.drop = nn.Dropout(dropout, inplace=True)
        in_size = input_size
        for l in range(num_layers):
            self.lstm.append(PackedLSTM(in_size, hidden_size, num_layers=1, bias=bias, batch_first=batch_first, dropout=0, bidirectional=bidirectional, rec_dropout=rec_dropout))
            self.highway.append(nn.Linear(in_size, hidden_size * self.num_directions))
            self.gate.append(nn.Linear(in_size, hidden_size * self.num_directions))
            self.highway[-1].bias.data.zero_()
            self.gate[-1].bias.data.zero_()
            in_size = hidden_size * self.num_directions

    def forward(self, input, seqlens, hx=None):
        highway_func = (lambda x: x) if self.highway_func is None else self.highway_func
        hs = []
        cs = []
        if not isinstance(input, PackedSequence):
            input = pack_padded_sequence(input, seqlens, batch_first=self.batch_first)
        for l in range(self.num_layers):
            if l > 0:
                input = PackedSequence(self.drop(input.data), input.batch_sizes)
            layer_hx = (hx[0][l * self.num_directions:(l + 1) * self.num_directions], hx[1][l * self.num_directions:(l + 1) * self.num_directions]) if hx is not None else None
            h, (ht, ct) = self.lstm[l](input, seqlens, layer_hx)
            hs.append(ht)
            cs.append(ct)
            input = PackedSequence(h.data + torch.sigmoid(self.gate[l](input.data)) * highway_func(self.highway[l](input.data)), input.batch_sizes)
        if self.pad:
            input = pad_packed_sequence(input, batch_first=self.batch_first)[0]
        return input, (torch.cat(hs, 0), torch.cat(cs, 0))


def SequenceLoss(vocab_size):
    weight = torch.ones(vocab_size)
    weight[constant.PAD_ID] = 0
    crit = nn.NLLLoss(weight)
    return crit


class MixLoss(nn.Module):
    """
    A mixture of SequenceLoss and CrossEntropyLoss.
    Loss = SequenceLoss + alpha * CELoss
    """

    def __init__(self, vocab_size, alpha):
        super().__init__()
        self.seq_loss = SequenceLoss(vocab_size)
        self.ce_loss = nn.CrossEntropyLoss()
        assert alpha >= 0
        self.alpha = alpha

    def forward(self, seq_inputs, seq_targets, class_inputs, class_targets):
        sl = self.seq_loss(seq_inputs, seq_targets)
        cel = self.ce_loss(class_inputs, class_targets)
        loss = sl + self.alpha * cel
        return loss


class MaxEntropySequenceLoss(nn.Module):
    """
    A max entropy loss that encourage the model to have large entropy,
    therefore giving more diverse outputs.

    Loss = NLLLoss + alpha * EntropyLoss
    """

    def __init__(self, vocab_size, alpha):
        super().__init__()
        weight = torch.ones(vocab_size)
        weight[constant.PAD_ID] = 0
        self.nll = nn.NLLLoss(weight)
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        inputs: [N, C]
        targets: [N]
        """
        assert inputs.size(0) == targets.size(0)
        nll_loss = self.nll(inputs, targets)
        mask = targets.eq(constant.PAD_ID).unsqueeze(1).expand_as(inputs)
        masked_inputs = inputs.clone().masked_fill_(mask, 0.0)
        p = torch.exp(masked_inputs)
        ent_loss = p.mul(masked_inputs).sum() / inputs.size(0)
        loss = nll_loss + self.alpha * ent_loss
        return loss


class PackedLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, pad=False, rec_dropout=0):
        super().__init__()
        self.batch_first = batch_first
        self.pad = pad
        if rec_dropout == 0:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        else:
            self.lstm = LSTMwRecDropout(input_size, hidden_size, num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, rec_dropout=rec_dropout)

    def forward(self, input, lengths, hx=None):
        if not isinstance(input, PackedSequence):
            input = pack_padded_sequence(input, lengths, batch_first=self.batch_first)
        res = self.lstm(input, hx)
        if self.pad:
            res = pad_packed_sequence(res[0], batch_first=self.batch_first)[0], res[1]
        return res


class LSTMwRecDropout(nn.Module):
    """ An LSTM implementation that supports recurrent dropout """

    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False, dropout=0, bidirectional=False, pad=False, rec_dropout=0):
        super().__init__()
        self.batch_first = batch_first
        self.pad = pad
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.drop = nn.Dropout(dropout, inplace=True)
        self.rec_drop = nn.Dropout(rec_dropout, inplace=True)
        self.num_directions = 2 if bidirectional else 1
        self.cells = nn.ModuleList()
        for l in range(num_layers):
            in_size = input_size if l == 0 else self.num_directions * hidden_size
            for d in range(self.num_directions):
                self.cells.append(nn.LSTMCell(in_size, hidden_size, bias=bias))

    def forward(self, input, hx=None):

        def rnn_loop(x, batch_sizes, cell, inits, reverse=False):
            batch_size = batch_sizes[0].item()
            states = [list(init.split([1] * batch_size)) for init in inits]
            h_drop_mask = x.new_ones(batch_size, self.hidden_size)
            h_drop_mask = self.rec_drop(h_drop_mask)
            resh = []
            if not reverse:
                st = 0
                for bs in batch_sizes:
                    s1 = cell(x[st:st + bs], (torch.cat(states[0][:bs], 0) * h_drop_mask[:bs], torch.cat(states[1][:bs], 0)))
                    resh.append(s1[0])
                    for j in range(bs):
                        states[0][j] = s1[0][j].unsqueeze(0)
                        states[1][j] = s1[1][j].unsqueeze(0)
                    st += bs
            else:
                en = x.size(0)
                for i in range(batch_sizes.size(0) - 1, -1, -1):
                    bs = batch_sizes[i]
                    s1 = cell(x[en - bs:en], (torch.cat(states[0][:bs], 0) * h_drop_mask[:bs], torch.cat(states[1][:bs], 0)))
                    resh.append(s1[0])
                    for j in range(bs):
                        states[0][j] = s1[0][j].unsqueeze(0)
                        states[1][j] = s1[1][j].unsqueeze(0)
                    en -= bs
                resh = list(reversed(resh))
            return torch.cat(resh, 0), tuple(torch.cat(s, 0) for s in states)
        all_states = [[], []]
        inputdata, batch_sizes = input.data, input.batch_sizes
        for l in range(self.num_layers):
            new_input = []
            if self.dropout > 0 and l > 0:
                inputdata = self.drop(inputdata)
            for d in range(self.num_directions):
                idx = l * self.num_directions + d
                cell = self.cells[idx]
                out, states = rnn_loop(inputdata, batch_sizes, cell, (hx[i][idx] for i in range(2)) if hx is not None else (input.data.new_zeros(input.batch_sizes[0].item(), self.hidden_size, requires_grad=False) for _ in range(2)), reverse=d == 1)
                new_input.append(out)
                all_states[0].append(states[0].unsqueeze(0))
                all_states[1].append(states[1].unsqueeze(0))
            if self.num_directions > 1:
                inputdata = torch.cat(new_input, 1)
            else:
                inputdata = new_input[0]
        input = PackedSequence(inputdata, batch_sizes)
        return input, tuple(torch.cat(x, 0) for x in all_states)


class Beam(object):

    def __init__(self, size, cuda=False):
        self.size = size
        self.done = False
        self.tt = torch.cuda if cuda else torch
        self.scores = self.tt.FloatTensor(size).zero_()
        self.allScores = []
        self.prevKs = []
        self.nextYs = [self.tt.LongTensor(size).fill_(constant.PAD_ID)]
        self.nextYs[0][0] = constant.SOS_ID
        self.copy = []

    def get_current_state(self):
        """Get the outputs for the current timestep."""
        return self.nextYs[-1]

    def get_current_origin(self):
        """Get the backpointers for the current timestep."""
        return self.prevKs[-1]

    def advance(self, wordLk, copy_indices=None):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `copy_indices` - copy indices (K x ctx_len)

        Returns: True if beam search is complete.
        """
        if self.done:
            return True
        numWords = wordLk.size(1)
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.allScores.append(self.scores)
        self.scores = bestScores
        prevK = bestScoresId / numWords
        self.prevKs.append(prevK)
        self.nextYs.append(bestScoresId - prevK * numWords)
        if copy_indices is not None:
            self.copy.append(copy_indices.index_select(0, prevK))
        if self.nextYs[-1][0] == constant.EOS_ID:
            self.done = True
            self.allScores.append(self.scores)
        return self.done

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        """Get the score of the best in the beam."""
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    def get_hyp(self, k):
        """
        Walk back to construct the full hypothesis.

        Parameters:

             * `k` - the position in the beam to construct.

         Returns: The hypothesis
        """
        hyp = []
        cpy = []
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            if len(self.copy) > 0:
                cpy.append(self.copy[j][k])
            k = self.prevKs[j][k]
        hyp = hyp[::-1]
        cpy = cpy[::-1]
        for i, cidx in enumerate(cpy):
            if cidx >= 0:
                hyp[i] = -(cidx + 1)
        return hyp


logger = logging.getLogger('stanza')


class Seq2SeqModel(nn.Module):
    """
    A complete encoder-decoder model, with optional attention.
    """

    def __init__(self, args, emb_matrix=None, use_cuda=False):
        super().__init__()
        self.vocab_size = args['vocab_size']
        self.emb_dim = args['emb_dim']
        self.hidden_dim = args['hidden_dim']
        self.nlayers = args['num_layers']
        self.emb_dropout = args.get('emb_dropout', 0.0)
        self.dropout = args['dropout']
        self.pad_token = constant.PAD_ID
        self.max_dec_len = args['max_dec_len']
        self.use_cuda = use_cuda
        self.top = args.get('top', 10000000000.0)
        self.args = args
        self.emb_matrix = emb_matrix
        logger.debug('Building an attentional Seq2Seq model...')
        logger.debug('Using a Bi-LSTM encoder')
        self.num_directions = 2
        self.enc_hidden_dim = self.hidden_dim // 2
        self.dec_hidden_dim = self.hidden_dim
        self.use_pos = args.get('pos', False)
        self.pos_dim = args.get('pos_dim', 0)
        self.pos_vocab_size = args.get('pos_vocab_size', 0)
        self.pos_dropout = args.get('pos_dropout', 0)
        self.edit = args.get('edit', False)
        self.num_edit = args.get('num_edit', 0)
        self.emb_drop = nn.Dropout(self.emb_dropout)
        self.drop = nn.Dropout(self.dropout)
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, self.pad_token)
        self.encoder = nn.LSTM(self.emb_dim, self.enc_hidden_dim, self.nlayers, bidirectional=True, batch_first=True, dropout=self.dropout if self.nlayers > 1 else 0)
        self.decoder = LSTMAttention(self.emb_dim, self.dec_hidden_dim, batch_first=True, attn_type=self.args['attn_type'])
        self.dec2vocab = nn.Linear(self.dec_hidden_dim, self.vocab_size)
        if self.use_pos and self.pos_dim > 0:
            logger.debug('Using POS in encoder')
            self.pos_embedding = nn.Embedding(self.pos_vocab_size, self.pos_dim, self.pad_token)
            self.pos_drop = nn.Dropout(self.pos_dropout)
        if self.edit:
            edit_hidden = self.hidden_dim // 2
            self.edit_clf = nn.Sequential(nn.Linear(self.hidden_dim, edit_hidden), nn.ReLU(), nn.Linear(edit_hidden, self.num_edit))
        self.SOS_tensor = torch.LongTensor([constant.SOS_ID])
        self.SOS_tensor = self.SOS_tensor if self.use_cuda else self.SOS_tensor
        self.init_weights()

    def init_weights(self):
        init_range = constant.EMB_INIT_RANGE
        if self.emb_matrix is not None:
            if isinstance(self.emb_matrix, np.ndarray):
                self.emb_matrix = torch.from_numpy(self.emb_matrix)
            assert self.emb_matrix.size() == (self.vocab_size, self.emb_dim), 'Input embedding matrix must match size: {} x {}'.format(self.vocab_size, self.emb_dim)
            self.embedding.weight.data.copy_(self.emb_matrix)
        else:
            self.embedding.weight.data.uniform_(-init_range, init_range)
        if self.top <= 0:
            logger.debug('Do not finetune embedding layer.')
            self.embedding.weight.requires_grad = False
        elif self.top < self.vocab_size:
            logger.debug('Finetune top {} embeddings.'.format(self.top))
            self.embedding.weight.register_hook(lambda x: utils.keep_partial_grad(x, self.top))
        else:
            logger.debug('Finetune all embeddings.')
        if self.use_pos:
            self.pos_embedding.weight.data.uniform_(-init_range, init_range)

    def cuda(self):
        super()
        self.use_cuda = True

    def cpu(self):
        super().cpu()
        self.use_cuda = False

    def zero_state(self, inputs):
        batch_size = inputs.size(0)
        h0 = torch.zeros(self.encoder.num_layers * 2, batch_size, self.enc_hidden_dim, requires_grad=False)
        c0 = torch.zeros(self.encoder.num_layers * 2, batch_size, self.enc_hidden_dim, requires_grad=False)
        if self.use_cuda:
            return h0, c0
        return h0, c0

    def encode(self, enc_inputs, lens):
        """ Encode source sequence. """
        self.h0, self.c0 = self.zero_state(enc_inputs)
        packed_inputs = nn.utils.rnn.pack_padded_sequence(enc_inputs, lens, batch_first=True)
        packed_h_in, (hn, cn) = self.encoder(packed_inputs, (self.h0, self.c0))
        h_in, _ = nn.utils.rnn.pad_packed_sequence(packed_h_in, batch_first=True)
        hn = torch.cat((hn[-1], hn[-2]), 1)
        cn = torch.cat((cn[-1], cn[-2]), 1)
        return h_in, (hn, cn)

    def decode(self, dec_inputs, hn, cn, ctx, ctx_mask=None):
        """ Decode a step, based on context encoding and source context states."""
        dec_hidden = hn, cn
        h_out, dec_hidden = self.decoder(dec_inputs, dec_hidden, ctx, ctx_mask)
        h_out_reshape = h_out.contiguous().view(h_out.size(0) * h_out.size(1), -1)
        decoder_logits = self.dec2vocab(h_out_reshape)
        decoder_logits = decoder_logits.view(h_out.size(0), h_out.size(1), -1)
        log_probs = self.get_log_prob(decoder_logits)
        return log_probs, dec_hidden

    def forward(self, src, src_mask, tgt_in, pos=None):
        batch_size = src.size(0)
        enc_inputs = self.emb_drop(self.embedding(src))
        dec_inputs = self.emb_drop(self.embedding(tgt_in))
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(0).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        log_probs, _ = self.decode(dec_inputs, hn, cn, h_in, src_mask)
        return log_probs, edit_logits

    def get_log_prob(self, logits):
        logits_reshape = logits.view(-1, self.vocab_size)
        log_probs = F.log_softmax(logits_reshape, dim=1)
        if logits.dim() == 2:
            return log_probs
        return log_probs.view(logits.size(0), logits.size(1), logits.size(2))

    def predict_greedy(self, src, src_mask, pos=None):
        """ Predict with greedy decoding. """
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(constant.PAD_ID).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        dec_inputs = self.embedding(self.SOS_tensor)
        dec_inputs = dec_inputs.expand(batch_size, dec_inputs.size(0), dec_inputs.size(1))
        done = [(False) for _ in range(batch_size)]
        total_done = 0
        max_len = 0
        output_seqs = [[] for _ in range(batch_size)]
        while total_done < batch_size and max_len < self.max_dec_len:
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask)
            assert log_probs.size(1) == 1, 'Output must have 1-step of output.'
            _, preds = log_probs.squeeze(1).max(1, keepdim=True)
            dec_inputs = self.embedding(preds)
            max_len += 1
            for i in range(batch_size):
                if not done[i]:
                    token = preds.data[i][0].item()
                    if token == constant.EOS_ID:
                        done[i] = True
                        total_done += 1
                    else:
                        output_seqs[i].append(token)
        return output_seqs, edit_logits

    def predict(self, src, src_mask, pos=None, beam_size=5):
        """ Predict with beam search. """
        if beam_size == 1:
            return self.predict_greedy(src, src_mask, pos=pos)
        enc_inputs = self.embedding(src)
        batch_size = enc_inputs.size(0)
        if self.use_pos:
            assert pos is not None, 'Missing POS input for seq2seq lemmatizer.'
            pos_inputs = self.pos_drop(self.pos_embedding(pos))
            enc_inputs = torch.cat([pos_inputs.unsqueeze(1), enc_inputs], dim=1)
            pos_src_mask = src_mask.new_zeros([batch_size, 1])
            src_mask = torch.cat([pos_src_mask, src_mask], dim=1)
        src_lens = list(src_mask.data.eq(constant.PAD_ID).long().sum(1))
        h_in, (hn, cn) = self.encode(enc_inputs, src_lens)
        if self.edit:
            edit_logits = self.edit_clf(hn)
        else:
            edit_logits = None
        with torch.no_grad():
            h_in = h_in.data.repeat(beam_size, 1, 1)
            src_mask = src_mask.repeat(beam_size, 1)
            hn = hn.data.repeat(beam_size, 1)
            cn = cn.data.repeat(beam_size, 1)
        beam = [Beam(beam_size, self.use_cuda) for _ in range(batch_size)]

        def update_state(states, idx, positions, beam_size):
            """ Select the states according to back pointers. """
            for e in states:
                br, d = e.size()
                s = e.contiguous().view(beam_size, br // beam_size, d)[:, (idx)]
                s.data.copy_(s.data.index_select(0, positions))
        for i in range(self.max_dec_len):
            dec_inputs = torch.stack([b.get_current_state() for b in beam]).t().contiguous().view(-1, 1)
            dec_inputs = self.embedding(dec_inputs)
            log_probs, (hn, cn) = self.decode(dec_inputs, hn, cn, h_in, src_mask)
            log_probs = log_probs.view(beam_size, batch_size, -1).transpose(0, 1).contiguous()
            done = []
            for b in range(batch_size):
                is_done = beam[b].advance(log_probs.data[b])
                if is_done:
                    done += [b]
                update_state((hn, cn), b, beam[b].get_current_origin(), beam_size)
            if len(done) == batch_size:
                break
        all_hyp, all_scores = [], []
        for b in range(batch_size):
            scores, ks = beam[b].sort_best()
            all_scores += [scores[0]]
            k = ks[0]
            hyp = beam[b].get_hyp(k)
            hyp = utils.prune_hyp(hyp)
            hyp = [i.item() for i in hyp]
            all_hyp += [hyp]
        return all_hyp, edit_logits


class BasicAttention(nn.Module):
    """
    A basic MLP attention layer.
    """

    def __init__(self, dim):
        super(BasicAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_c = nn.Linear(dim, dim)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        target = self.linear_in(input)
        source = self.linear_c(context.contiguous().view(-1, dim)).view(batch_size, source_len, dim)
        attn = target.unsqueeze(1).expand_as(context) + source
        attn = self.tanh(attn)
        attn = self.linear_v(attn.view(-1, dim)).view(batch_size, source_len)
        if mask is not None:
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        weighted_context = torch.bmm(attn.unsqueeze(1), context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)
        attn = torch.bmm(context, target).squeeze(2)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(attn.size(0), 1, attn.size(1))
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LinearAttention(nn.Module):
    """ A linear attention form, inspired by BiDAF:
        a = W (u; v; u o v)
    """

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.linear = nn.Linear(dim * 3, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)
        v = context.contiguous().view(-1, dim)
        attn_in = torch.cat((u, v, u.mul(v)), 1)
        attn = self.linear(attn_in).view(batch_size, source_len)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(batch_size, 1, source_len)
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class DeepAttention(nn.Module):
    """ A deep attention form, invented by Robert:
        u = ReLU(Wx)
        v = ReLU(Wy)
        a = V.(u o v)
    """

    def __init__(self, dim):
        super(DeepAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, 1, bias=False)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.relu = nn.ReLU()
        self.sm = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context, mask=None, attn_only=False):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        batch_size = context.size(0)
        source_len = context.size(1)
        dim = context.size(2)
        u = input.unsqueeze(1).expand_as(context).contiguous().view(-1, dim)
        u = self.relu(self.linear_in(u))
        v = self.relu(self.linear_in(context.contiguous().view(-1, dim)))
        attn = self.linear_v(u.mul(v)).view(batch_size, source_len)
        if mask is not None:
            assert mask.size() == attn.size(), 'Mask size must match the attention size!'
            attn.masked_fill_(mask, -constant.INFINITY_NUMBER)
        attn = self.sm(attn)
        if attn_only:
            return attn
        attn3 = attn.view(batch_size, 1, source_len)
        weighted_context = torch.bmm(attn3, context).squeeze(1)
        h_tilde = torch.cat((weighted_context, input), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn


class LSTMAttention(nn.Module):
    """A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, batch_first=True, attn_type='soft'):
        """Initialize params."""
        super(LSTMAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        if attn_type == 'soft':
            self.attention_layer = SoftDotAttention(hidden_size)
        elif attn_type == 'mlp':
            self.attention_layer = BasicAttention(hidden_size)
        elif attn_type == 'linear':
            self.attention_layer = LinearAttention(hidden_size)
        elif attn_type == 'deep':
            self.attention_layer = DeepAttention(hidden_size)
        else:
            raise Exception('Unsupported LSTM attention type: {}'.format(attn_type))
        logger.debug('Using {} attention for LSTM.'.format(attn_type))

    def forward(self, input, hidden, ctx, ctx_mask=None):
        """Propogate input through the network."""
        if self.batch_first:
            input = input.transpose(0, 1)
        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = self.lstm_cell(input[i], hidden)
            hy, cy = hidden
            h_tilde, alpha = self.attention_layer(hy, ctx, mask=ctx_mask)
            output.append(h_tilde)
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, hidden


EMPTY_ID = 2


class CompositeVocab(BaseVocab):
    """ Vocabulary class that handles parsing and printing composite values such as
    compositional XPOS and universal morphological features (UFeats).

    Two key options are `keyed` and `sep`. `sep` specifies the separator used between
    different parts of the composite values, which is `|` for UFeats, for example.
    If `keyed` is `True`, then the incoming value is treated similarly to UFeats, where
    each part is a key/value pair separated by an equal sign (`=`). There are no inherit
    order to the keys, and we sort them alphabetically for serialization and deserialization.
    Whenever a part is absent, its internal value is a special `<EMPTY>` symbol that will
    be treated accordingly when generating the output. If `keyed` is `False`, then the parts
    are treated as positioned values, and `<EMPTY>` is used to pad parts at the end when the
    incoming value is not long enough."""

    def __init__(self, data=None, lang='', idx=0, sep='', keyed=False):
        self.sep = sep
        self.keyed = keyed
        super().__init__(data, lang, idx=idx)
        self.state_attrs += ['sep', 'keyed']

    def unit2parts(self, unit):
        if self.sep == '':
            parts = [x for x in unit]
        else:
            parts = unit.split(self.sep)
        if self.keyed:
            if len(parts) == 1 and parts[0] == '_':
                return dict()
            parts = [x.split('=') for x in parts]
            parts = dict(parts)
        elif unit == '_':
            parts = []
        return parts

    def unit2id(self, unit):
        parts = self.unit2parts(unit)
        if self.keyed:
            return [(self._unit2id[k].get(parts[k], UNK_ID) if k in parts else EMPTY_ID) for k in self._unit2id]
        else:
            return [(self._unit2id[i].get(parts[i], UNK_ID) if i < len(parts) else EMPTY_ID) for i in range(len(self._unit2id))]

    def id2unit(self, id):
        items = []
        for v, k in zip(id, self._id2unit.keys()):
            if v == EMPTY_ID:
                continue
            if self.keyed:
                items.append('{}={}'.format(k, self._id2unit[k][v]))
            else:
                items.append(self._id2unit[k][v])
        res = self.sep.join(items)
        if res == '':
            res = '_'
        return res

    def build_vocab(self):
        allunits = [w[self.idx] for sent in self.data for w in sent]
        if self.keyed:
            self._id2unit = dict()
            for u in allunits:
                parts = self.unit2parts(u)
                for key in parts:
                    if key not in self._id2unit:
                        self._id2unit[key] = copy(VOCAB_PREFIX)
                    if parts[key] not in self._id2unit[key]:
                        self._id2unit[key].append(parts[key])
            if len(self._id2unit) == 0:
                self._id2unit['_'] = copy(VOCAB_PREFIX)
        else:
            self._id2unit = dict()
            allparts = [self.unit2parts(u) for u in allunits]
            maxlen = max([len(p) for p in allparts])
            for parts in allparts:
                for i, p in enumerate(parts):
                    if i not in self._id2unit:
                        self._id2unit[i] = copy(VOCAB_PREFIX)
                    if i < len(parts) and p not in self._id2unit[i]:
                        self._id2unit[i].append(p)
            if len(self._id2unit) == 0:
                self._id2unit[0] = copy(VOCAB_PREFIX)
        self._id2unit = OrderedDict([(k, self._id2unit[k]) for k in sorted(self._id2unit.keys())])
        self._unit2id = {k: {w: i for i, w in enumerate(self._id2unit[k])} for k in self._id2unit}

    def lens(self):
        return [len(self._unit2id[k]) for k in self._unit2id]


class Parser(nn.Module):

    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            self.lemma_emb = nn.Embedding(len(vocab['lemma']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim'] * 2
        if self.args['tag_emb_dim'] > 0:
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)
            if not isinstance(vocab['xpos'], CompositeVocab):
                self.xpos_emb = nn.Embedding(len(vocab['xpos']), self.args['tag_emb_dim'], padding_idx=0)
            else:
                self.xpos_emb = nn.ModuleList()
                for l in vocab['xpos'].lens():
                    self.xpos_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))
            self.ufeats_emb = nn.ModuleList()
            for l in vocab['feats'].lens():
                self.ufeats_emb.append(nn.Embedding(l, self.args['tag_emb_dim'], padding_idx=0))
            input_size += self.args['tag_emb_dim'] * 2
        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']
        if self.args['pretrain']:
            add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']
        self.parserlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.parserlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.parserlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.unlabeled = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        self.deprel = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], len(vocab['deprel']), pairwise=True, dropout=args['dropout'])
        if args['linearization']:
            self.linearization = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        if args['distance']:
            self.distance = DeepBiaffineScorer(2 * self.args['hidden_dim'], 2 * self.args['hidden_dim'], self.args['deep_biaff_hidden_dim'], 1, pairwise=True, dropout=args['dropout'])
        self.crit = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, lemma, head, deprel, word_orig_idx, sentlens, wordlens):

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        inputs = []
        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            lemma_emb = self.lemma_emb(lemma)
            lemma_emb = pack(lemma_emb)
            inputs += [word_emb, lemma_emb]
        if self.args['tag_emb_dim'] > 0:
            pos_emb = self.upos_emb(upos)
            if isinstance(self.vocab['xpos'], CompositeVocab):
                for i in range(len(self.vocab['xpos'])):
                    pos_emb += self.xpos_emb[i](xpos[:, :, (i)])
            else:
                pos_emb += self.xpos_emb(xpos)
            pos_emb = pack(pos_emb)
            feats_emb = 0
            for i in range(len(self.vocab['feats'])):
                feats_emb += self.ufeats_emb[i](ufeats[:, :, (i)])
            feats_emb = pack(feats_emb)
            inputs += [pos_emb, feats_emb]
        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]
        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        lstm_outputs, _ = self.parserlstm(lstm_inputs, sentlens, hx=(self.parserlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.parserlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)
        unlabeled_scores = self.unlabeled(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
        deprel_scores = self.deprel(self.drop(lstm_outputs), self.drop(lstm_outputs))
        if self.args['linearization'] or self.args['distance']:
            head_offset = torch.arange(word.size(1), device=head.device).view(1, 1, -1).expand(word.size(0), -1, -1) - torch.arange(word.size(1), device=head.device).view(1, -1, 1).expand(word.size(0), -1, -1)
        if self.args['linearization']:
            lin_scores = self.linearization(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            unlabeled_scores += F.logsigmoid(lin_scores * torch.sign(head_offset).float()).detach()
        if self.args['distance']:
            dist_scores = self.distance(self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            dist_pred = 1 + F.softplus(dist_scores)
            dist_target = torch.abs(head_offset)
            dist_kld = -torch.log((dist_target.float() - dist_pred) ** 2 / 2 + 1)
            unlabeled_scores += dist_kld.detach()
        diag = torch.eye(head.size(-1) + 1, dtype=torch.bool, device=head.device).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))
        preds = []
        if self.training:
            unlabeled_scores = unlabeled_scores[:, 1:, :]
            unlabeled_scores = unlabeled_scores.masked_fill(word_mask.unsqueeze(1), -float('inf'))
            unlabeled_target = head.masked_fill(word_mask[:, 1:], -1)
            loss = self.crit(unlabeled_scores.contiguous().view(-1, unlabeled_scores.size(2)), unlabeled_target.view(-1))
            deprel_scores = deprel_scores[:, 1:]
            deprel_scores = torch.gather(deprel_scores, 2, head.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, len(self.vocab['deprel']))).view(-1, len(self.vocab['deprel']))
            deprel_target = deprel.masked_fill(word_mask[:, 1:], -1)
            loss += self.crit(deprel_scores.contiguous(), deprel_target.view(-1))
            if self.args['linearization']:
                lin_scores = torch.gather(lin_scores[:, 1:], 2, head.unsqueeze(2)).view(-1)
                lin_scores = torch.cat([-lin_scores.unsqueeze(1) / 2, lin_scores.unsqueeze(1) / 2], 1)
                lin_target = torch.gather((head_offset[:, 1:] > 0).long(), 2, head.unsqueeze(2))
                loss += self.crit(lin_scores.contiguous(), lin_target.view(-1))
            if self.args['distance']:
                dist_kld = torch.gather(dist_kld[:, 1:], 2, head.unsqueeze(2))
                loss -= dist_kld.sum()
            loss /= wordchars.size(0)
        else:
            loss = 0
            preds.append(F.log_softmax(unlabeled_scores, 2).detach().cpu().numpy())
            preds.append(deprel_scores.max(3)[1].detach().cpu().numpy())
        return loss, preds


PAD_ID = 0


class NERTagger(nn.Module):

    def __init__(self, args, vocab, emb_matrix=None):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            self.word_emb = nn.Embedding(len(self.vocab['word']), self.args['word_emb_dim'], PAD_ID)
            if emb_matrix is not None:
                self.init_emb(emb_matrix)
            if not self.args.get('emb_finetune', True):
                self.word_emb.weight.detach_()
            input_size += self.args['word_emb_dim']
        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args['charlm']:
                add_unsaved_module('charmodel_forward', CharacterLanguageModel.load(args['charlm_forward_file'], finetune=False))
                add_unsaved_module('charmodel_backward', CharacterLanguageModel.load(args['charlm_backward_file'], finetune=False))
            else:
                self.charmodel = CharacterModel(args, vocab, bidirectional=True, attention=False)
            input_size += self.args['char_hidden_dim'] * 2
        if self.args.get('input_transform', False):
            self.input_transform = nn.Linear(input_size, input_size)
        else:
            self.input_transform = None
        self.taggerlstm = PackedLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=0 if self.args['num_layers'] == 1 else self.args['dropout'])
        self.drop_replacement = None
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']), requires_grad=False)
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']), requires_grad=False)
        num_tag = len(self.vocab['tag'])
        self.tag_clf = nn.Linear(self.args['hidden_dim'] * 2, num_tag)
        self.tag_clf.bias.data.zero_()
        self.crit = CRFLoss(num_tag)
        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])
        self.lockeddrop = LockedDropout(args['locked_dropout'])

    def init_emb(self, emb_matrix):
        if isinstance(emb_matrix, np.ndarray):
            emb_matrix = torch.from_numpy(emb_matrix)
        vocab_size = len(self.vocab['word'])
        dim = self.args['word_emb_dim']
        assert emb_matrix.size() == (vocab_size, dim), 'Input embedding matrix must match size: {} x {}'.format(vocab_size, dim)
        self.word_emb.weight.data.copy_(emb_matrix)

    def forward(self, word, word_mask, wordchars, wordchars_mask, tags, word_orig_idx, sentlens, wordlens, chars, charoffsets, charlens, char_orig_idx):

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]
        if self.args['char'] and self.args['char_emb_dim'] > 0:
            if self.args.get('charlm', None):
                char_reps_forward = self.charmodel_forward.get_representation(chars[0], charoffsets[0], charlens, char_orig_idx)
                char_reps_forward = PackedSequence(char_reps_forward.data, char_reps_forward.batch_sizes)
                char_reps_backward = self.charmodel_backward.get_representation(chars[1], charoffsets[1], charlens, char_orig_idx)
                char_reps_backward = PackedSequence(char_reps_backward.data, char_reps_backward.batch_sizes)
                inputs += [char_reps_forward, char_reps_backward]
            else:
                char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
                char_reps = PackedSequence(char_reps.data, char_reps.batch_sizes)
                inputs += [char_reps]
        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        if self.args['word_dropout'] > 0:
            lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = pad(lstm_inputs)
        lstm_inputs = self.lockeddrop(lstm_inputs)
        lstm_inputs = pack(lstm_inputs).data
        if self.input_transform:
            lstm_inputs = self.input_transform(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data
        lstm_outputs = self.drop(lstm_outputs)
        lstm_outputs = pad(lstm_outputs)
        lstm_outputs = self.lockeddrop(lstm_outputs)
        lstm_outputs = pack(lstm_outputs).data
        logits = pad(self.tag_clf(lstm_outputs)).contiguous()
        loss, trans = self.crit(logits, word_mask, tags)
        return loss, logits, trans


class Tagger(nn.Module):

    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()
        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']
        if not share_hid:
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)
        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']
        if self.args['pretrain']:
            add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']
        self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))
        self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()
        if share_hid:
            clf_constructor = lambda insize, outsize: nn.Linear(insize, outsize)
        else:
            self.xpos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'] if not isinstance(vocab['xpos'], CompositeVocab) else self.args['composite_deep_biaff_hidden_dim'])
            self.ufeats_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['composite_deep_biaff_hidden_dim'])
            clf_constructor = lambda insize, outsize: BiaffineScorer(insize, self.args['tag_emb_dim'], outsize)
        if isinstance(vocab['xpos'], CompositeVocab):
            self.xpos_clf = nn.ModuleList()
            for l in vocab['xpos'].lens():
                self.xpos_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))
        else:
            self.xpos_clf = clf_constructor(self.args['deep_biaff_hidden_dim'], len(vocab['xpos']))
            if share_hid:
                self.xpos_clf.weight.data.zero_()
                self.xpos_clf.bias.data.zero_()
        self.ufeats_clf = nn.ModuleList()
        for l in vocab['feats'].lens():
            if share_hid:
                self.ufeats_clf.append(clf_constructor(self.args['deep_biaff_hidden_dim'], l))
                self.ufeats_clf[-1].weight.data.zero_()
                self.ufeats_clf[-1].bias.data.zero_()
            else:
                self.ufeats_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))
        self.crit = nn.CrossEntropyLoss(ignore_index=0)
        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens):

        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]
        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]
        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]
        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)
        lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
        lstm_outputs = lstm_outputs.data
        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))
        preds = [pad(upos_pred).max(2)[1]]
        upos = pack(upos).data
        loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))
        if self.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid
            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
            ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))
            if self.training:
                upos_emb = self.upos_emb(upos)
            else:
                upos_emb = self.upos_emb(upos_pred.max(1)[1])
            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(upos_emb))
        xpos = pack(xpos).data
        if isinstance(self.vocab['xpos'], CompositeVocab):
            xpos_preds = []
            for i in range(len(self.vocab['xpos'])):
                xpos_pred = clffunc(self.xpos_clf[i], xpos_hid)
                loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos[:, (i)].view(-1))
                xpos_preds.append(pad(xpos_pred).max(2, keepdim=True)[1])
            preds.append(torch.cat(xpos_preds, 2))
        else:
            xpos_pred = clffunc(self.xpos_clf, xpos_hid)
            loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))
            preds.append(pad(xpos_pred).max(2)[1])
        ufeats_preds = []
        ufeats = pack(ufeats).data
        for i in range(len(self.vocab['feats'])):
            ufeats_pred = clffunc(self.ufeats_clf[i], ufeats_hid)
            loss += self.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, (i)].view(-1))
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])
        preds.append(torch.cat(ufeats_preds, 2))
        return loss, preds


class Tokenizer(nn.Module):

    def __init__(self, args, nchars, emb_dim, hidden_dim, N_CLASSES=5, dropout=0):
        super().__init__()
        self.args = args
        feat_dim = args['feat_dim']
        self.embeddings = nn.Embedding(nchars, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + feat_dim, hidden_dim, num_layers=self.args['rnn_layers'], bidirectional=True, batch_first=True, dropout=dropout if self.args['rnn_layers'] > 1 else 0)
        if self.args['conv_res'] is not None:
            self.conv_res = nn.ModuleList()
            self.conv_sizes = [int(x) for x in self.args['conv_res'].split(',')]
            for si, size in enumerate(self.conv_sizes):
                l = nn.Conv1d(emb_dim + feat_dim, hidden_dim * 2, size, padding=size // 2, bias=self.args.get('hier_conv_res', False) or si == 0)
                self.conv_res.append(l)
            if self.args.get('hier_conv_res', False):
                self.conv_res2 = nn.Conv1d(hidden_dim * 2 * len(self.conv_sizes), hidden_dim * 2, 1)
        self.tok_clf = nn.Linear(hidden_dim * 2, 1)
        self.sent_clf = nn.Linear(hidden_dim * 2, 1)
        self.mwt_clf = nn.Linear(hidden_dim * 2, 1)
        if args['hierarchical']:
            in_dim = hidden_dim * 2
            self.rnn2 = nn.LSTM(in_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.tok_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.sent_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
            self.mwt_clf2 = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.toknoise = nn.Dropout(self.args['tok_noise'])

    def forward(self, x, feats):
        emb = self.embeddings(x)
        emb = self.dropout(emb)
        emb = torch.cat([emb, feats], 2)
        inp, _ = self.rnn(emb)
        if self.args['conv_res'] is not None:
            conv_input = emb.transpose(1, 2).contiguous()
            if not self.args.get('hier_conv_res', False):
                for l in self.conv_res:
                    inp = inp + l(conv_input).transpose(1, 2).contiguous()
            else:
                hid = []
                for l in self.conv_res:
                    hid += [l(conv_input)]
                hid = torch.cat(hid, 1)
                hid = F.relu(hid)
                hid = self.dropout(hid)
                inp = inp + self.conv_res2(hid).transpose(1, 2).contiguous()
        inp = self.dropout(inp)
        tok0 = self.tok_clf(inp)
        sent0 = self.sent_clf(inp)
        mwt0 = self.mwt_clf(inp)
        if self.args['hierarchical']:
            if self.args['hier_invtemp'] > 0:
                inp2, _ = self.rnn2(inp * (1 - self.toknoise(torch.sigmoid(-tok0 * self.args['hier_invtemp']))))
            else:
                inp2, _ = self.rnn2(inp)
            inp2 = self.dropout(inp2)
            tok0 = tok0 + self.tok_clf2(inp2)
            sent0 = sent0 + self.sent_clf2(inp2)
            mwt0 = mwt0 + self.mwt_clf2(inp2)
        nontok = F.logsigmoid(-tok0)
        tok = F.logsigmoid(tok0)
        nonsent = F.logsigmoid(-sent0)
        sent = F.logsigmoid(sent0)
        nonmwt = F.logsigmoid(-mwt0)
        mwt = F.logsigmoid(mwt0)
        pred = torch.cat([nontok, tok + nonsent + nonmwt, tok + sent + nonmwt, tok + nonsent + mwt, tok + sent + mwt], 2)
        return pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BiaffineScorer,
     lambda: ([], {'input1_size': 4, 'input2_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (DeepBiaffineScorer,
     lambda: ([], {'input1_size': 4, 'input2_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (HighwayLSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (LockedDropout,
     lambda: ([], {'dropprob': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PackedLSTM,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'num_layers': 1}),
     lambda: ([torch.rand([4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (PairwiseBiaffineScorer,
     lambda: ([], {'input1_size': 4, 'input2_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (PairwiseBilinear,
     lambda: ([], {'input1_size': 4, 'input2_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (SequenceUnitDropout,
     lambda: ([], {'dropprob': 4, 'replacement_id': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (WordDropout,
     lambda: ([], {'dropprob': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_stanfordnlp_stanza(_paritybench_base):
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

