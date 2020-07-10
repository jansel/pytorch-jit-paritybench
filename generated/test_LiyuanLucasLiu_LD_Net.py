import sys
_module = sys.modules[__name__]
del sys
conf = _module
model_seq = _module
crf = _module
dataset = _module
elmo = _module
evaluator = _module
seqlabel = _module
seqlm = _module
sparse_lm = _module
utils = _module
LM = _module
model_word_ada = _module
adaptive = _module
basic = _module
dataset = _module
densenet = _module
ldnet = _module
utils = _module
encode_data = _module
gene_map = _module
encode_data2folder = _module
prune_sparse_seq = _module
train_lm = _module
train_seq = _module
train_seq_elmo = _module

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


import torch.nn as nn


import torch.optim as optim


import torch.sparse as sparse


import torch.nn.functional as F


import random


import functools


import itertools


import time


import numpy as np


from torch.autograd import Variable


import torch.nn.init


from torch import nn


from math import sqrt


from torch.utils.data import Dataset


import torch.autograd as autograd


import math


import logging


class CRF(nn.Module):
    """
    Conditional Random Field Module

    Parameters
    ----------
    hidden_dim : ``int``, required.
        the dimension of the input features.
    tagset_size : ``int``, required.
        the size of the target labels.
    if_bias: ``bool``, optional, (default=True).
        whether the linear transformation has the bias term.
    """

    def __init__(self, hidden_dim: int, tagset_size: int, if_bias: bool=True):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size, bias=if_bias)
        self.transitions = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))

    def rand_init(self):
        """
        random initialization
        """
        utils.init_linear(self.hidden2tag)
        self.transitions.data.zero_()

    def forward(self, feats):
        """
        calculate the potential score for the conditional random field.

        Parameters
        ----------
        feats: ``torch.FloatTensor``, required.
            the input features for the conditional random field, of shape (*, hidden_dim).

        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (ins_num, from_tag_size, to_tag_size)
        """
        scores = self.hidden2tag(feats).view(-1, 1, self.tagset_size)
        ins_num = scores.size(0)
        crf_scores = scores.expand(ins_num, self.tagset_size, self.tagset_size) + self.transitions.view(1, self.tagset_size, self.tagset_size).expand(ins_num, self.tagset_size, self.tagset_size)
        return crf_scores


class CRFLoss(nn.Module):
    """
    
    The negative loss for the Conditional Random Field Module

    Parameters
    ----------
    y_map : ``dict``, required.
        a ``dict`` maps from tag string to tag index.
    average_batch : ``bool``, optional, (default=True).
        whether the return score would be averaged per batch.
    """

    def __init__(self, y_map: dict, average_batch: bool=True):
        super(CRFLoss, self).__init__()
        self.tagset_size = len(y_map)
        self.start_tag = y_map['<s>']
        self.end_tag = y_map['<eof>']
        self.average_batch = average_batch

    def forward(self, scores, target, mask):
        """
        calculate the negative log likehood for the conditional random field.

        Parameters
        ----------
        scores: ``torch.FloatTensor``, required.
            the potential score for the conditional random field, of shape (seq_len, batch_size, from_tag_size, to_tag_size).
        target: ``torch.LongTensor``, required.
            the positive path for the conditional random field, of shape (seq_len, batch_size).
        mask: ``torch.ByteTensor``, required.
            the mask for the unpadded sentence parts, of shape (seq_len, batch_size).

        Returns
        -------
        loss: ``torch.FloatTensor``.
            The NLL loss.
        """
        seq_len = scores.size(0)
        bat_size = scores.size(1)
        tg_energy = torch.gather(scores.view(seq_len, bat_size, -1), 2, target.unsqueeze(2)).view(seq_len, bat_size)
        tg_energy = tg_energy.masked_select(mask).sum()
        seq_iter = enumerate(scores)
        _, inivalues = seq_iter.__next__()
        partition = inivalues[:, (self.start_tag), :].squeeze(1).clone()
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.unsqueeze(2).expand(bat_size, self.tagset_size, self.tagset_size)
            cur_partition = utils.log_sum_exp(cur_values)
            mask_idx = mask[(idx), :].view(bat_size, 1).expand(bat_size, self.tagset_size)
            partition.masked_scatter_(mask_idx, cur_partition.masked_select(mask_idx))
        partition = partition[:, (self.end_tag)].sum()
        if self.average_batch:
            return (partition - tg_energy) / bat_size
        else:
            return partition - tg_energy


class EBUnit(nn.Module):
    """
    The basic recurrent unit for the ELMo RNNs wrapper.

    Parameters
    ----------
    ori_unit : ``torch.nn.Module``, required.
        The original module of rnn unit.
    droprate : ``float``, required.
        The dropout ratrio.
    fix_rate: ``bool``, required.
        Whether to fix the rqtio.
    """

    def __init__(self, ori_unit, droprate, fix_rate):
        super(EBUnit, self).__init__()
        self.layer = ori_unit.layer
        self.droprate = droprate
        self.output_dim = ori_unit.output_dim

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            The input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        out, _ = self.layer(x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return out


class ERNN(nn.Module):
    """
    The multi-layer recurrent networks for the ELMo RNNs wrapper.

    Parameters
    ----------
    ori_drnn : ``torch.nn.Module``, required.
        The original module of rnn networks.
    droprate : ``float``, required.
        The dropout ratrio.
    fix_rate: ``bool``, required.
        Whether to fix the rqtio.
    """

    def __init__(self, ori_drnn, droprate, fix_rate):
        super(ERNN, self).__init__()
        self.layer_list = [EBUnit(ori_unit, droprate, fix_rate) for ori_unit in ori_drnn.layer._modules.values()]
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))
        self.weight_list = nn.Parameter(torch.FloatTensor([0.0] * len(self.layer_list)))
        self.layer = nn.ModuleList(self.layer_list)
        for param in self.layer.parameters():
            param.requires_grad = False
        if fix_rate:
            self.gamma.requires_grad = False
            self.weight_list.requires_grad = False
        self.output_dim = self.layer_list[-1].output_dim

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        The regularization term.
        """
        srd_weight = self.weight_list - 1.0 / len(self.layer_list)
        return (srd_weight ** 2).sum()

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        out = 0
        nw = self.gamma * F.softmax(self.weight_list, dim=0)
        for ind in range(len(self.layer_list)):
            x = self.layer[ind](x)
            out += x * nw[ind]
        return out


class ElmoLM(nn.Module):
    """
    The language model for the ELMo RNNs wrapper.

    Parameters
    ----------
    ori_lm : ``torch.nn.Module``, required.
        the original module of language model.
    backward : ``bool``, required.
        whether the language model is backward.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """

    def __init__(self, ori_lm, backward, droprate, fix_rate):
        super(ElmoLM, self).__init__()
        self.rnn = ERNN(ori_lm.rnn, droprate, fix_rate)
        self.w_num = ori_lm.w_num
        self.w_dim = ori_lm.w_dim
        self.word_embed = ori_lm.word_embed
        self.word_embed.weight.requires_grad = False
        self.output_dim = ori_lm.rnn_output
        self.backward = backward

    def init_hidden(self):
        """
        initialize hidden states.
        """
        return

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg: ``list``.
            The list of regularization terms.
        """
        return self.rnn.regularizer()

    def prox(self, lambda0):
        """
        the proximal calculator.
        """
        return 0.0

    def forward(self, w_in, ind=None):
        """
        Calculate the output.

        Parameters
        ----------
        w_in : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size).
        ind : ``torch.LongTensor``, optional, (default=None).
            the index tensor for the backward language model, of shape (seq_len, batch_size).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        w_emb = self.word_embed(w_in)
        out = self.rnn(w_emb)
        if self.backward:
            out_size = out.size()
            out = out.view(out_size[0] * out_size[1], out_size[2]).index_select(0, ind).contiguous().view(out_size)
        return out


class SeqLabel(nn.Module):
    """
    Sequence Labeling model augumented with language model.

    Parameters
    ----------
    f_lm : ``torch.nn.Module``, required.
        The forward language modle for contextualized representations.
    b_lm : ``torch.nn.Module``, required.
        The backward language modle for contextualized representations.
    c_num : ``int`` , required.
        The number of characters.
    c_dim : ``int`` , required.
        The dimension of character embedding.
    c_hidden : ``int`` , required.
        The dimension of character hidden states.
    c_layer : ``int`` , required.
        The number of character lstms.
    w_num : ``int`` , required.
        The number of words.
    w_dim : ``int`` , required.
        The dimension of word embedding.
    w_hidden : ``int`` , required.
        The dimension of word hidden states.
    w_layer : ``int`` , required.
        The number of word lstms.
    y_num : ``int`` , required.
        The number of tags types.
    droprate : ``float`` , required
        The dropout ratio.
    unit : "str", optional, (default = 'lstm')
        The type of the recurrent unit.
    """

    def __init__(self, f_lm, b_lm, c_num: int, c_dim: int, c_hidden: int, c_layer: int, w_num: int, w_dim: int, w_hidden: int, w_layer: int, y_num: int, droprate: float, unit: str='lstm'):
        super(SeqLabel, self).__init__()
        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        self.f_lm = f_lm
        self.b_lm = b_lm
        self.unit_type = unit
        self.char_embed = nn.Embedding(c_num, c_dim)
        self.word_embed = nn.Embedding(w_num, w_dim)
        self.char_seq = nn.Linear(c_hidden * 2, w_dim)
        self.lm_seq = nn.Linear(f_lm.output_dim + b_lm.output_dim, w_dim)
        self.relu = nn.ReLU()
        self.c_hidden = c_hidden
        tmp_rnn_dropout = droprate if c_layer > 1 else 0
        self.char_fw = rnnunit_map[unit](c_dim, c_hidden, c_layer, dropout=tmp_rnn_dropout)
        self.char_bw = rnnunit_map[unit](c_dim, c_hidden, c_layer, dropout=tmp_rnn_dropout)
        tmp_rnn_dropout = droprate if w_layer > 1 else 0
        self.word_rnn = rnnunit_map[unit](w_dim * 3, w_hidden // 2, w_layer, dropout=tmp_rnn_dropout, bidirectional=True)
        self.y_num = y_num
        self.crf = CRF(w_hidden, y_num)
        self.drop = nn.Dropout(p=droprate)

    def to_params(self):
        """
        To parameters.
        """
        return {'model_type': 'char-lstm-crf', 'forward_lm': self.f_lm.to_params(), 'backward_lm': self.b_lm.to_params(), 'word_embed_num': self.word_embed.num_embeddings, 'word_embed_dim': self.word_embed.embedding_dim, 'char_embed_num': self.char_embed.num_embeddings, 'char_embed_dim': self.char_embed.embedding_dim, 'char_hidden': self.c_hidden, 'char_layers': self.char_fw.num_layers, 'word_hidden': self.word_rnn.hidden_size, 'word_layers': self.word_rnn.num_layers, 'droprate': self.drop.p, 'y_num': self.y_num, 'label_schema': 'iobes', 'unit_type': self.unit_type}

    def prune_dense_rnn(self):
        """
        Prune dense rnn to be smaller by delecting layers.
        """
        f_prune_mask = self.f_lm.prune_dense_rnn()
        b_prune_mask = self.b_lm.prune_dense_rnn()
        prune_mask = torch.cat([f_prune_mask, b_prune_mask], dim=0)
        mask_index = prune_mask.nonzero().squeeze(1)
        self.lm_seq.weight = nn.Parameter(self.lm_seq.weight.data.index_select(1, mask_index).contiguous())
        self.lm_seq.in_features = self.lm_seq.weight.size(1)

    def set_batch_seq_size(self, sentence):
        """
        Set the batch size and sequence length.
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        Load pre-trained word embedding.
        """
        self.word_embed.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self):
        """
        Random initialization.
        """
        utils.init_embedding(self.char_embed.weight)
        utils.init_lstm(self.char_fw)
        utils.init_lstm(self.char_bw)
        utils.init_lstm(self.word_rnn)
        utils.init_linear(self.char_seq)
        utils.init_linear(self.lm_seq)
        self.crf.rand_init()

    def forward(self, f_c, f_p, b_c, b_p, flm_w, blm_w, blm_ind, f_w):
        """
        Calculate the output (crf potentials).

        Parameters
        ----------
        f_c : ``torch.LongTensor``, required.
            Character-level inputs in the forward direction.
        f_p : ``torch.LongTensor``, required.
            Ouput position of character-level inputs in the forward direction.
        b_c : ``torch.LongTensor``, required.
            Character-level inputs in the backward direction.
        b_p : ``torch.LongTensor``, required.
            Ouput position of character-level inputs in the backward direction.
        flm_w : ``torch.LongTensor``, required.
            Word-level inputs for the forward language model.
        blm_w : ``torch.LongTensor``, required.
            Word-level inputs for the backward language model.
        blm_ind : ``torch.LongTensor``, required.
            Ouput position of word-level inputs for the backward language model.
        f_w: ``torch.LongTensor``, required.
            Word-level inputs for the sequence labeling model.

        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (sequence_len, batch_size, from_tag_size, to_tag_size)
        """
        self.set_batch_seq_size(f_w)
        f_c_e = self.drop(self.char_embed(f_c))
        b_c_e = self.drop(self.char_embed(b_c))
        f_c_e, _ = self.char_fw(f_c_e)
        b_c_e, _ = self.char_bw(b_c_e)
        f_c_e = f_c_e.view(-1, self.c_hidden).index_select(0, f_p).view(self.word_seq_length, self.batch_size, self.c_hidden)
        b_c_e = b_c_e.view(-1, self.c_hidden).index_select(0, b_p).view(self.word_seq_length, self.batch_size, self.c_hidden)
        c_o = self.drop(torch.cat([f_c_e, b_c_e], dim=2))
        c_o = self.char_seq(c_o)
        self.f_lm.init_hidden()
        self.b_lm.init_hidden()
        f_lm_e = self.f_lm(flm_w)
        b_lm_e = self.b_lm(blm_w, blm_ind)
        lm_o = self.drop(torch.cat([f_lm_e, b_lm_e], dim=2))
        lm_o = self.relu(self.lm_seq(lm_o))
        w_e = self.word_embed(f_w)
        rnn_in = self.drop(torch.cat([c_o, lm_o, w_e], dim=2))
        rnn_out, _ = self.word_rnn(rnn_in)
        crf_out = self.crf(self.drop(rnn_out)).view(self.word_seq_length, self.batch_size, self.y_num, self.y_num)
        return crf_out


class Vanilla_SeqLabel(nn.Module):
    """
    Sequence Labeling model augumented without language model.

    Parameters
    ----------
    f_lm : ``torch.nn.Module``, required.
        forward language modle for contextualized representations.
    b_lm : ``torch.nn.Module``, required.
        backward language modle for contextualized representations.
    c_num : ``int`` , required.
        number of characters.
    c_dim : ``int`` , required.
        dimension of character embedding.
    c_hidden : ``int`` , required.
        dimension of character hidden states.
    c_layer : ``int`` , required.
        number of character lstms.
    w_num : ``int`` , required.
        number of words.
    w_dim : ``int`` , required.
        dimension of word embedding.
    w_hidden : ``int`` , required.
        dimension of word hidden states.
    w_layer : ``int`` , required.
        number of word lstms.
    y_num : ``int`` , required.
        number of tags types.
    droprate : ``float`` , required
        dropout ratio.
    unit : "str", optional, (default = 'lstm')
        type of the recurrent unit.
    """

    def __init__(self, f_lm, b_lm, c_num, c_dim, c_hidden, c_layer, w_num, w_dim, w_hidden, w_layer, y_num, droprate, unit='lstm'):
        super(Vanilla_SeqLabel, self).__init__()
        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        self.char_embed = nn.Embedding(c_num, c_dim)
        self.word_embed = nn.Embedding(w_num, w_dim)
        self.char_seq = nn.Linear(c_hidden * 2, w_dim)
        self.c_hidden = c_hidden
        self.char_fw = rnnunit_map[unit](c_dim, c_hidden, c_layer, dropout=droprate)
        self.char_bw = rnnunit_map[unit](c_dim, c_hidden, c_layer, dropout=droprate)
        self.word_rnn = rnnunit_map[unit](w_dim + w_dim, w_hidden // 2, w_layer, dropout=droprate, bidirectional=True)
        self.y_num = y_num
        self.crf = CRF(w_hidden, y_num)
        self.drop = nn.Dropout(p=droprate)

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_word_embedding(self, pre_word_embeddings):
        """
        Load pre-trained word embedding.
        """
        self.word_embed.weight = nn.Parameter(pre_word_embeddings)

    def rand_init(self):
        """
        Random initialization.
        """
        utils.init_embedding(self.char_embed.weight)
        utils.init_lstm(self.char_fw)
        utils.init_lstm(self.char_bw)
        utils.init_lstm(self.word_rnn)
        utils.init_linear(self.char_seq)
        self.crf.rand_init()

    def forward(self, f_c, f_p, b_c, b_p, flm_w, blm_w, blm_ind, f_w):
        """
        Calculate the output (crf potentials).

        Parameters
        ----------
        f_c : ``torch.LongTensor``, required.
            Character-level inputs in the forward direction.
        f_p : ``torch.LongTensor``, required.
            Ouput position of character-level inputs in the forward direction.
        b_c : ``torch.LongTensor``, required.
            Character-level inputs in the backward direction.
        b_p : ``torch.LongTensor``, required.
            Ouput position of character-level inputs in the backward direction.
        flm_w : ``torch.LongTensor``, required.
            Word-level inputs for the forward language model.
        blm_w : ``torch.LongTensor``, required.
            Word-level inputs for the backward language model.
        blm_ind : ``torch.LongTensor``, required.
            Ouput position of word-level inputs for the backward language model.
        f_w: ``torch.LongTensor``, required.
            Word-level inputs for the sequence labeling model.

        Returns
        -------
        output: ``torch.FloatTensor``.
            A float tensor of shape (sequence_len, batch_size, from_tag_size, to_tag_size)
        """
        self.set_batch_seq_size(f_w)
        f_c_e = self.drop(self.char_embed(f_c))
        b_c_e = self.drop(self.char_embed(b_c))
        f_c_e, _ = self.char_fw(f_c_e)
        b_c_e, _ = self.char_bw(b_c_e)
        f_c_e = f_c_e.view(-1, self.c_hidden).index_select(0, f_p).view(self.word_seq_length, self.batch_size, self.c_hidden)
        b_c_e = b_c_e.view(-1, self.c_hidden).index_select(0, b_p).view(self.word_seq_length, self.batch_size, self.c_hidden)
        c_o = self.drop(torch.cat([f_c_e, b_c_e], dim=2))
        c_o = self.char_seq(c_o)
        w_e = self.word_embed(f_w)
        rnn_in = self.drop(torch.cat([c_o, w_e], dim=2))
        rnn_out, _ = self.word_rnn(rnn_in)
        crf_out = self.crf(self.drop(rnn_out)).view(self.word_seq_length, self.batch_size, self.y_num, self.y_num)
        return crf_out


class BasicSeqLM(nn.Module):
    """
    The language model for the dense rnns.

    Parameters
    ----------
    ori_lm : ``torch.nn.Module``, required.
        the original module of language model.
    backward : ``bool``, required.
        whether the language model is backward.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """

    def __init__(self, ori_lm, backward, droprate, fix_rate):
        super(BasicSeqLM, self).__init__()
        self.rnn = ori_lm.rnn
        for param in self.rnn.parameters():
            param.requires_grad = False
        self.w_num = ori_lm.w_num
        self.w_dim = ori_lm.w_dim
        self.word_embed = ori_lm.word_embed
        self.word_embed.weight.requires_grad = False
        self.output_dim = ori_lm.rnn_output
        self.backward = backward

    def to_params(self):
        """
        To parameters.
        """
        return {'rnn_params': self.rnn.to_params(), 'word_embed_num': self.word_embed.num_embeddings, 'word_embed_dim': self.word_embed.embedding_dim}

    def init_hidden(self):
        """
        initialize hidden states.
        """
        self.rnn.init_hidden()

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg: ``list``.
            The list of regularization terms.
        """
        return self.rnn.regularizer()

    def forward(self, w_in, ind=None):
        """
        Calculate the output.

        Parameters
        ----------
        w_in : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size).
        ind : ``torch.LongTensor``, optional, (default=None).
            the index tensor for the backward language model, of shape (seq_len, batch_size).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        w_emb = self.word_embed(w_in)
        out = self.rnn(w_emb)
        if self.backward:
            out_size = out.size()
            out = out.view(out_size[0] * out_size[1], out_size[2]).index_select(0, ind).contiguous().view(out_size)
        return out


class SBUnit(nn.Module):
    """
    The basic recurrent unit for the dense-RNNs wrapper.

    Parameters
    ----------
    ori_unit : ``torch.nn.Module``, required.
        the original module of rnn unit.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """

    def __init__(self, ori_unit, droprate, fix_rate):
        super(SBUnit, self).__init__()
        self.unit_type = ori_unit.unit_type
        self.layer = ori_unit.layer
        self.droprate = droprate
        self.input_dim = ori_unit.input_dim
        self.increase_rate = ori_unit.increase_rate
        self.output_dim = ori_unit.input_dim + ori_unit.increase_rate

    def prune_rnn(self, mask):
        """
        Prune dense rnn to be smaller by delecting layers.

        Parameters
        ----------
        mask : ``torch.ByteTensor``, required.
            The selection tensor for the input matrix.
        """
        mask_index = mask.nonzero().squeeze(1)
        self.layer.weight_ih_l0 = nn.Parameter(self.layer.weight_ih_l0.data.index_select(1, mask_index).contiguous())
        self.layer.input_size = self.layer.weight_ih_l0.size(1)

    def forward(self, x, weight=1):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            The input tensor, of shape (seq_len, batch_size, input_dim).
        weight : ``torch.FloatTensor``, required.
            The selection variable.

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        if self.droprate > 0:
            new_x = F.dropout(x, p=self.droprate, training=self.training)
        else:
            new_x = x
        out, _ = self.layer(new_x)
        out = weight * out
        return torch.cat([x, out], 2)


class SDRNN(nn.Module):
    """
    The multi-layer recurrent networks for the dense-RNNs wrapper.

    Parameters
    ----------
    ori_unit : ``torch.nn.Module``, required.
        the original module of rnn unit.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """

    def __init__(self, ori_drnn, droprate, fix_rate):
        super(SDRNN, self).__init__()
        if ori_drnn.layer:
            self.layer_list = [SBUnit(ori_unit, droprate, fix_rate) for ori_unit in ori_drnn.layer._modules.values()]
            self.weight_list = nn.Parameter(torch.FloatTensor([1.0] * len(self.layer_list)))
            self.weight_list.requires_grad = not fix_rate
            self.layer = nn.ModuleList(self.layer_list)
            for param in self.layer.parameters():
                param.requires_grad = False
        else:
            self.layer_list = list()
            self.weight_list = list()
            self.layer = None
        self.emb_dim = ori_drnn.emb_dim
        self.output_dim = ori_drnn.output_dim
        self.unit_type = ori_drnn.unit_type

    def to_params(self):
        """
        To parameters.
        """
        return {'rnn_type': 'LDRNN', 'unit_type': self.unit_type, 'layer_num': 0 if not self.layer else len(self.layer), 'emb_dim': self.emb_dim, 'hid_dim': -1 if not self.layer else self.layer[0].increase_rate, 'droprate': -1 if not self.layer else self.layer[0].droprate, 'after_pruned': True}

    def prune_dense_rnn(self):
        """
        Prune dense rnn to be smaller by delecting layers.
        """
        prune_mask = torch.ones(self.layer_list[0].input_dim)
        increase_mask_one = torch.ones(self.layer_list[0].increase_rate)
        increase_mask_zero = torch.zeros(self.layer_list[0].increase_rate)
        new_layer_list = list()
        new_weight_list = list()
        for ind in range(0, len(self.layer_list)):
            if self.weight_list.data[ind] > 0:
                new_weight_list.append(self.weight_list.data[ind])
                self.layer_list[ind].prune_rnn(prune_mask)
                new_layer_list.append(self.layer_list[ind])
                prune_mask = torch.cat([prune_mask, increase_mask_one], dim=0)
            else:
                prune_mask = torch.cat([prune_mask, increase_mask_zero], dim=0)
        if not new_layer_list:
            self.output_dim = self.layer_list[0].input_dim
            self.layer = None
            self.weight_list = None
            self.layer_list = None
        else:
            self.layer_list = new_layer_list
            self.layer = nn.ModuleList(self.layer_list)
            self.weight_list = nn.Parameter(torch.FloatTensor(new_weight_list))
            self.weight_list.requires_grad = False
            for param in self.layer.parameters():
                param.requires_grad = False
        return prune_mask

    def prox(self):
        """
        the proximal calculator.
        """
        self.weight_list.data.masked_fill_(self.weight_list.data < 0, 0)
        self.weight_list.data.masked_fill_(self.weight_list.data > 1, 1)
        none_zero_count = (self.weight_list.data > 0).sum()
        return none_zero_count

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg0: ``torch.FloatTensor``.
            The value of reg0.
        reg1: ``torch.FloatTensor``.
            The value of reg1.
        reg2: ``torch.FloatTensor``.
            The value of reg2.
        """
        reg3 = (self.weight_list * (1 - self.weight_list)).sum()
        none_zero = self.weight_list.data > 0
        none_zero_count = none_zero.sum()
        reg0 = none_zero_count
        reg1 = self.weight_list[none_zero].sum()
        return reg0, reg1, reg3

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.FloatTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        if self.layer_list is not None:
            for ind in range(len(self.layer_list)):
                x = self.layer[ind](x, self.weight_list[ind])
        return x


class SparseSeqLM(nn.Module):
    """
    The language model for the dense rnns with layer-wise selection.

    Parameters
    ----------
    ori_lm : ``torch.nn.Module``, required.
        the original module of language model.
    backward : ``bool``, required.
        whether the language model is backward.
    droprate : ``float``, required.
        the dropout ratrio.
    fix_rate: ``bool``, required.
        whether to fix the rqtio.
    """

    def __init__(self, ori_lm, backward, droprate, fix_rate):
        super(SparseSeqLM, self).__init__()
        self.rnn = SDRNN(ori_lm.rnn, droprate, fix_rate)
        self.w_num = ori_lm.w_num
        self.w_dim = ori_lm.w_dim
        self.word_embed = ori_lm.word_embed
        self.word_embed.weight.requires_grad = False
        self.output_dim = ori_lm.rnn_output
        self.backward = backward

    def to_params(self):
        """
        To parameters.
        """
        return {'backward': self.backward, 'rnn_params': self.rnn.to_params(), 'word_embed_num': self.word_embed.num_embeddings, 'word_embed_dim': self.word_embed.embedding_dim}

    def prune_dense_rnn(self):
        """
        Prune dense rnn to be smaller by delecting layers.
        """
        prune_mask = self.rnn.prune_dense_rnn()
        self.output_dim = self.rnn.output_dim
        return prune_mask

    def init_hidden(self):
        """
        initialize hidden states.
        """
        return

    def regularizer(self):
        """
        Calculate the regularization term.

        Returns
        ----------
        reg: ``list``.
            The list of regularization terms.
        """
        return self.rnn.regularizer()

    def prox(self):
        """
        the proximal calculator.
        """
        return self.rnn.prox()

    def forward(self, w_in, ind=None):
        """
        Calculate the output.

        Parameters
        ----------
        w_in : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size).
        ind : ``torch.LongTensor``, optional, (default=None).
            the index tensor for the backward language model, of shape (seq_len, batch_size).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The ELMo outputs.
        """
        w_emb = self.word_embed(w_in)
        out = self.rnn(w_emb)
        if self.backward:
            out_size = out.size()
            out = out.view(out_size[0] * out_size[1], out_size[2]).index_select(0, ind).contiguous().view(out_size)
        return out


class LM(nn.Module):
    """
    The language model model.
    
    Parameters
    ----------
    rnn : ``torch.nn.Module``, required.
        The RNNs network.
    soft_max : ``torch.nn.Module``, required.
        The softmax layer.
    w_num : ``int`` , required.
        The number of words.
    w_dim : ``int`` , required.
        The dimension of word embedding.
    droprate : ``float`` , required
        The dropout ratio.
    label_dim : ``int`` , required.
        The input dimension of softmax.    
    """

    def __init__(self, rnn, soft_max, w_num, w_dim, droprate, label_dim=-1, add_relu=False):
        super(LM, self).__init__()
        self.rnn = rnn
        self.soft_max = soft_max
        self.w_num = w_num
        self.w_dim = w_dim
        self.word_embed = nn.Embedding(w_num, w_dim)
        self.rnn_output = self.rnn.output_dim
        self.add_proj = label_dim > 0
        if self.add_proj:
            self.project = nn.Linear(self.rnn_output, label_dim)
            if add_relu:
                self.relu = nn.ReLU()
            else:
                self.relu = lambda x: x
        self.drop = nn.Dropout(p=droprate)

    def load_embed(self, origin_lm):
        """
        Load embedding from another language model.
        """
        self.word_embed = origin_lm.word_embed
        self.soft_max = origin_lm.soft_max

    def rand_ini(self):
        """
        Random initialization.
        """
        self.rnn.rand_ini()
        self.soft_max.rand_ini()
        utils.init_embedding(self.word_embed.weight)
        if self.add_proj:
            utils.init_linear(self.project)

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.rnn.init_hidden()

    def forward(self, w_in, target):
        """
        Calculate the loss.

        Parameters
        ----------
        w_in : ``torch.FloatTensor``, required.
            the input tensor, of shape (word_num, input_dim).
        target : ``torch.FloatTensor``, required.
            the target of the language model, of shape (word_num).
        
        Returns
        ----------
        loss: ``torch.FloatTensor``.
            The NLL loss.
        """
        w_emb = self.word_embed(w_in)
        w_emb = self.drop(w_emb)
        out = self.rnn(w_emb).contiguous().view(-1, self.rnn_output)
        if self.add_proj:
            out = self.drop(self.relu(self.project(out)))
        out = self.soft_max(out, target)
        return out

    def log_prob(self, w_in):
        """
        Calculate log-probability for the whole dictionary.
        
        Parameters
        ----------
        w_in : ``torch.FloatTensor``, required.
            the input tensor, of shape (word_num, input_dim).
        
        Returns
        ----------
        prob: ``torch.FloatTensor``.
            The full log-probability.
        """
        w_emb = self.word_embed(w_in)
        out = self.rnn(w_emb).contiguous().view(-1, self.rnn_output)
        if self.add_proj:
            out = self.relu(self.project(out))
        out = self.soft_max.log_prob(out, w_emb.device)
        return out


class AdaptiveSoftmax(nn.Module):
    """
    The adaptive softmax layer.
    Modified from: https://github.com/rosinality/adaptive-softmax-pytorch/blob/master/adasoft.py

    Parameters
    ----------
    input_size : ``int``, required.
        The input dimension.
    cutoff : ``list``, required.
        The list of cutoff values.
    """

    def __init__(self, input_size, cutoff):
        super().__init__()
        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1
        self.head = nn.Linear(input_size, self.output_size)
        self.tail = nn.ModuleList()
        self.cross_entropy = nn.CrossEntropyLoss(size_average=False)
        for i in range(len(self.cutoff) - 1):
            seq = nn.Sequential(nn.Linear(input_size, input_size // 4 ** i, False), nn.Linear(input_size // 4 ** i, cutoff[i + 1] - cutoff[i], False))
            self.tail.append(seq)

    def rand_ini(self):
        """
        Random Initialization.
        """
        nn.init.xavier_normal_(self.head.weight)
        for tail in self.tail:
            nn.init.xavier_normal_(tail[0].weight)
            nn.init.xavier_normal_(tail[1].weight)

    def log_prob(self, w_in, device):
        """
        Calculate log-probability for the whole dictionary.
        
        Parameters
        ----------
        w_in : ``torch.FloatTensor``, required.
            the input tensor, of shape (word_num, input_dim).
        device: ``torch.device``, required.
            the target device for calculation.

        Returns
        ----------
        prob: ``torch.FloatTensor``.
            The full log-probability.
        """
        lsm = nn.LogSoftmax(dim=1)
        head_out = self.head(w_in)
        batch_size = head_out.size(0)
        prob = torch.zeros(batch_size, self.cutoff[-1])
        lsm_head = lsm(head_out)
        prob.narrow(1, 0, self.output_size).add_(lsm_head.narrow(1, 0, self.output_size).data)
        for i in range(len(self.tail)):
            pos = self.cutoff[i]
            i_size = self.cutoff[i + 1] - pos
            buffer = lsm_head.narrow(1, self.cutoff[0] + i, 1)
            buffer = buffer.expand(batch_size, i_size)
            lsm_tail = lsm(self.tail[i](w_in))
            prob.narrow(1, pos, i_size).copy_(buffer.data).add_(lsm_tail.data)
        return prob

    def forward(self, w_in, target):
        """
        Calculate the log-likihood w.o. calculate the full distribution.

        Parameters
        ----------
        w_in : ``torch.FloatTensor``, required.
            the input tensor, of shape (word_num, input_dim).
        target : ``torch.FloatTensor``, required.
            the target of the language model, of shape (word_num).
        
        Returns
        ----------
        loss: ``torch.FloatTensor``.
            The NLL loss.
        """
        batch_size = w_in.size(0)
        output = 0.0
        first_target = target.clone()
        for i in range(len(self.cutoff) - 1):
            mask = target.ge(self.cutoff[i]).mul(target.lt(self.cutoff[i + 1]))
            if mask.sum() > 0:
                first_target[mask] = self.cutoff[0] + i
                second_target = target[mask].add(-self.cutoff[i])
                second_input = w_in.index_select(0, mask.nonzero().squeeze())
                second_output = self.tail[i](second_input)
                output += self.cross_entropy(second_output, second_target)
        output += self.cross_entropy(self.head(w_in), first_target)
        output /= batch_size
        return output


class BasicUnit(nn.Module):
    """
    The basic recurrent unit for the densely connected RNNs with layer-wise dropout.

    Parameters
    ----------
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``float``, required.
        The input dimension fo the unit.
    increase_rate : ``float``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    layer_dropout : ``float``, required.
        The layer-wise dropout ratrio.
    """

    def __init__(self, unit, input_dim, increase_rate, droprate, layer_drop=0):
        super(BasicUnit, self).__init__()
        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}
        self.unit_type = unit
        self.layer = rnnunit_map[unit](input_dim, increase_rate, 1)
        if 'lstm' == self.unit_type:
            utils.init_lstm(self.layer)
        self.layer_drop = layer_drop
        self.droprate = droprate
        self.input_dim = input_dim
        self.increase_rate = increase_rate
        self.output_dim = input_dim + increase_rate
        self.init_hidden()

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        self.hidden_state = None

    def rand_ini(self):
        """
        Random Initialization.
        """
        return

    def forward(self, x, p_out):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).
        p_out : ``torch.LongTensor``, required.
            the final output tensor for the softmax, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        out: ``torch.FloatTensor``.
            The undropped outputs of RNNs to the softmax.
        p_out: ``torch.FloatTensor``.
            The dropped outputs of RNNs to the next_layer.
        """
        if self.droprate > 0:
            new_x = F.dropout(x, p=self.droprate, training=self.training)
        else:
            new_x = x
        out, new_hidden = self.layer(new_x, self.hidden_state)
        self.hidden_state = utils.repackage_hidden(new_hidden)
        out = out.contiguous()
        if self.training and random.uniform(0, 1) < self.layer_drop:
            deep_out = torch.autograd.Variable(torch.zeros(x.size(0), x.size(1), self.increase_rate))
        else:
            deep_out = out
        o_out = torch.cat([p_out, out], 2)
        d_out = torch.cat([x, deep_out], 2)
        return d_out, o_out


class BasicRNN(nn.Module):
    """
    The multi-layer recurrent networks for the vanilla stacked RNNs.

    Parameters
    ----------
    layer_num: ``int``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``int``, required.
        The input dimension fo the unit.
    hid_dim : ``int``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """

    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
        super(BasicRNN, self).__init__()
        layer_list = [BasicUnit(unit, emb_dim, hid_dim, droprate)] + [BasicUnit(unit, hid_dim, hid_dim, droprate) for i in range(layer_num - 1)]
        self.layer = nn.Sequential(*layer_list)
        self.output_dim = layer_list[-1].output_dim
        self.unit_type = unit
        self.init_hidden()

    def to_params(self):
        """
        To parameters.
        """
        return {'rnn_type': 'Basic', 'unit_type': self.layer[0].unit_type, 'layer_num': len(self.layer), 'emb_dim': self.layer[0].layer.input_size, 'hid_dim': self.layer[0].layer.hidden_size, 'droprate': self.layer[0].droprate}

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        for tup in self.layer.children():
            tup.init_hidden()

    def rand_ini(self):
        """
        Random Initialization.
        """
        for tup in self.layer.children():
            tup.rand_ini()

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        return self.layer(x)


class DenseRNN(nn.Module):
    """
    The multi-layer recurrent networks for the densely connected RNNs.

    Parameters
    ----------
    layer_num: ``float``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``float``, required.
        The input dimension fo the unit.
    hid_dim : ``float``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    """

    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate):
        super(DenseRNN, self).__init__()
        self.unit_type = unit
        self.layer_list = [BasicUnit(unit, emb_dim + i * hid_dim, hid_dim, droprate) for i in range(layer_num)]
        self.layer = nn.Sequential(*self.layer_list) if layer_num > 0 else None
        self.output_dim = self.layer_list[-1].output_dim if layer_num > 0 else emb_dim
        self.emb_dim = emb_dim
        self.init_hidden()

    def to_params(self):
        """
        To parameters.
        """
        return {'rnn_type': 'DenseRNN', 'unit_type': self.layer[0].unit_type, 'layer_num': len(self.layer), 'emb_dim': self.layer[0].input_dim, 'hid_dim': self.layer[0].increase_rate, 'droprate': self.layer[0].droprate}

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        for tup in self.layer_list:
            tup.init_hidden()

    def rand_ini(self):
        """
        Random Initialization.
        """
        for tup in self.layer_list:
            tup.rand_ini()

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs.
        """
        return self.layer(x)


class LDRNN(nn.Module):
    """
    The multi-layer recurrent networks for the densely connected RNNs with layer-wise dropout.

    Parameters
    ----------
    layer_num: ``float``, required.
        The number of layers. 
    unit : ``torch.nn.Module``, required.
        The type of rnn unit.
    input_dim : ``float``, required.
        The input dimension fo the unit.
    hid_dim : ``float``, required.
        The hidden dimension fo the unit.
    droprate : ``float``, required.
        The dropout ratrio.
    layer_dropout : ``float``, required.
        The layer-wise dropout ratrio.
    """

    def __init__(self, layer_num, unit, emb_dim, hid_dim, droprate, layer_drop):
        super(LDRNN, self).__init__()
        self.unit_type = unit
        self.layer_list = [BasicUnit(unit, emb_dim + i * hid_dim, hid_dim, droprate, layer_drop) for i in range(layer_num)]
        self.layer_num = layer_num
        self.layer = nn.ModuleList(self.layer_list) if layer_num > 0 else None
        self.output_dim = self.layer_list[-1].output_dim if layer_num > 0 else emb_dim
        self.emb_dim = emb_dim
        self.init_hidden()

    def to_params(self):
        """
        To parameters.
        """
        return {'rnn_type': 'LDRNN', 'unit_type': self.layer[0].unit_type, 'layer_num': len(self.layer), 'emb_dim': self.layer[0].input_dim, 'hid_dim': self.layer[0].increase_rate, 'droprate': self.layer[0].droprate, 'after_pruned': False}

    def init_hidden(self):
        """
        Initialize hidden states.
        """
        for tup in self.layer_list:
            tup.init_hidden()

    def rand_ini(self):
        """
        Random Initialization.
        """
        for tup in self.layer_list:
            tup.rand_ini()

    def forward(self, x):
        """
        Calculate the output.

        Parameters
        ----------
        x : ``torch.LongTensor``, required.
            the input tensor, of shape (seq_len, batch_size, input_dim).

        Returns
        ----------
        output: ``torch.FloatTensor``.
            The output of RNNs to the Softmax.
        """
        output = x
        for ind in range(self.layer_num):
            x, output = self.layer_list[ind](x, output)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (CRF,
     lambda: ([], {'hidden_dim': 4, 'tagset_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_LiyuanLucasLiu_LD_Net(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

