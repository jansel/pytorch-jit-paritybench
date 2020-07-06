import sys
_module = sys.modules[__name__]
del sys
master = _module
RetrieveSentences = _module
api = _module
common = _module
dataset = _module
evaluation = _module
evaluators = _module
evaluator = _module
msrvid_evaluator = _module
pit2015_evaluator = _module
qa_evaluator = _module
quora_evaluator = _module
sick_evaluator = _module
snli_evaluator = _module
sst_evaluator = _module
sts2014_evaluator = _module
trecqa_evaluator = _module
wikiqa_evaluator = _module
train = _module
trainers = _module
msrvid_trainer = _module
pit2015_trainer = _module
qa_trainer = _module
quora_trainer = _module
sick_trainer = _module
snli_trainer = _module
sst_trainer = _module
sts2014_trainer = _module
trainer = _module
trecqa_trainer = _module
wikiqa_trainer = _module
data = _module
model = _module
test = _module
train = _module
datasets = _module
castor_dataset = _module
idf_utils = _module
msrvid = _module
pit2015 = _module
quora = _module
sick = _module
snli = _module
sst = _module
sts2014 = _module
trecqa = _module
wikiqa = _module
decatt = _module
model = _module
esim = _module
model = _module
experimental_settings = _module
mp_cnn = _module
lite_model = _module
model = _module
nce_pairwise_mp = _module
main = _module
train_script = _module
qa_trainer = _module
args = _module
main = _module
model = _module
overlap_features = _module
train = _module
setup = _module
bridge = _module
external_features = _module
main = _module
model = _module
train = _module
trec_dataset = _module
wiki_dataset = _module
sse = _module
model = _module
utils = _module
build_w2v = _module
nce_neighbors = _module
relevancy_metrics = _module
serialization = _module
torch_util = _module
data = _module
model = _module
log = _module
preprocess = _module
tune = _module

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


from scipy.stats import pearsonr


import torch.nn.functional as F


from scipy.stats import spearmanr


import math


import time


from torch.optim.lr_scheduler import ReduceLROnPlateau


import re


import numpy as np


import torch.utils.data as data


import random


import torch.nn.utils.rnn as rnn_utils


import torch.utils as utils


from torch import utils


from abc import ABCMeta


from abc import abstractmethod


from torchtext.data.dataset import Dataset


from torchtext.data.example import Example


from torchtext.data.field import Field


from torchtext.data.field import RawField


from torchtext.data.iterator import BucketIterator


from torchtext.data.pipeline import Pipeline


from torchtext.vocab import Vectors


from torchtext.data import Field


from torchtext.data import TabularDataset


import logging


import torch.optim as optim


from torch.autograd import Variable


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torchtext import data


from torch.nn import functional as F


from collections import Counter


import torch.onnx


class ConvRNNModel(nn.Module):

    def __init__(self, word_model, **config):
        super().__init__()
        embedding_dim = word_model.dim
        self.word_model = word_model
        self.hidden_size = config['hidden_size']
        fc_size = config['fc_size']
        self.batch_size = config['mbatch_size']
        n_fmaps = config['n_feature_maps']
        self.rnn_type = config['rnn_type']
        self.no_cuda = config['no_cuda']
        if self.rnn_type.upper() == 'LSTM':
            self.bi_rnn = nn.LSTM(embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)
        elif self.rnn_type.upper() == 'GRU':
            self.bi_rnn = nn.GRU(embedding_dim, self.hidden_size, 1, batch_first=True, bidirectional=True)
        else:
            raise ValueError('RNN type must be one of LSTM or GRU')
        self.conv = nn.Conv2d(1, n_fmaps, (1, self.hidden_size * 2))
        self.fc1 = nn.Linear(n_fmaps + 2 * self.hidden_size, fc_size)
        self.fc2 = nn.Linear(fc_size, config['n_labels'])

    def convert_dataset(self, dataset):
        dataset = np.stack(dataset)
        model_in = dataset[:, (1)].reshape(-1)
        model_out = dataset[:, (0)].flatten().astype(np.int)
        model_out = torch.from_numpy(model_out)
        indices, lengths = self.preprocess(model_in)
        if not self.no_cuda:
            model_out = model_out
            indices = indices
            lengths = lengths
        lengths, sort_idx = torch.sort(lengths, descending=True)
        indices = indices[sort_idx]
        model_out = model_out[sort_idx]
        return (indices, lengths), model_out

    def preprocess(self, sentences):
        indices, lengths = self.word_model.lookup(sentences)
        return torch.LongTensor(indices), torch.LongTensor(lengths)

    def forward(self, x, lengths):
        x = self.word_model(x)
        x = rnn_utils.pack_padded_sequence(x, lengths, batch_first=True)
        rnn_seq, rnn_out = self.bi_rnn(x)
        if self.rnn_type.upper() == 'LSTM':
            rnn_out = rnn_out[0]
        rnn_seq, _ = rnn_utils.pad_packed_sequence(rnn_seq, batch_first=True)
        rnn_out.data = rnn_out.data.permute(1, 0, 2)
        x = self.conv(rnn_seq.unsqueeze(1)).squeeze(3)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2))
        out = [t.squeeze(1) for t in rnn_out.chunk(2, 1)]
        out.append(x.squeeze(-1))
        x = torch.cat(out, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class WordEmbeddingModel(nn.Module):

    def __init__(self, id_dict, weights, unknown_vocab=[], static=True, padding_idx=0):
        super().__init__()
        vocab_size = len(id_dict) + len(unknown_vocab)
        self.lookup_table = id_dict
        last_id = max(id_dict.values())
        for word in unknown_vocab:
            last_id += 1
            self.lookup_table[word] = last_id
        self.dim = weights.shape[1]
        self.weights = np.concatenate((weights, np.random.rand(len(unknown_vocab), self.dim) / 2 - 0.25))
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, self.dim, padding_idx=padding_idx)
        self.embedding.weight.data.copy_(torch.from_numpy(self.weights))
        if static:
            self.embedding.weight.requires_grad = False

    @classmethod
    def make_random_model(cls, id_dict, unknown_vocab=[], dim=300):
        weights = np.random.rand(len(id_dict), dim) - 0.5
        return cls(id_dict, weights, unknown_vocab, static=False)

    def forward(self, x):
        return self.embedding(x)

    def lookup(self, sentences):
        raise NotImplementedError


class SSTWordEmbeddingModel(WordEmbeddingModel):

    def __init__(self, id_dict, weights, unknown_vocab=[]):
        super().__init__(id_dict, weights, unknown_vocab, padding_idx=16259)

    def lookup(self, sentences):
        indices_list = []
        max_len = 0
        for sentence in sentences:
            indices = []
            for word in data.sst_tokenize(sentence):
                try:
                    index = self.lookup_table[word]
                    indices.append(index)
                except KeyError:
                    continue
            indices_list.append(indices)
            if len(indices) > max_len:
                max_len = len(indices)
        lengths = [len(x) for x in indices_list]
        for indices in indices_list:
            indices.extend([self.padding_idx] * (max_len - len(indices)))
        return indices_list, lengths


class DecAtt(nn.Module):

    def __init__(self, num_units, num_classes, embedding_size, dropout, device=0, training=True, project_input=True, use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
        Create the model based on MLP networks.

        :param num_units: size of the networks
        :param num_classes: number of classes in the problem
        :param embedding_size: size of each word embedding
        :param use_intra_attention: whether to use intra-attention model
        :param training: whether to create training tensors (optimizer)
        :p/word_embeddingaram project_input: whether to project input embeddings to a
            different dimensionality
        :param distance_biases: number of different distances with biases used
            in the intra-attention model
        """
        super().__init__()
        self.arch = 'DecAtt'
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input
        self.embedding_size = embedding_size
        self.distance_biases = distance_biases
        self.intra_attention = False
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.bias_embedding = nn.Embedding(max_sentence_length, 1)
        self.linear_layer_project = nn.Linear(embedding_size, num_units, bias=False)
        self.linear_layer_attend = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(num_units, num_units), nn.ReLU())
        self.linear_layer_compare = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(num_units, num_units), nn.ReLU())
        self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units, num_classes), nn.LogSoftmax())
        self.init_weight()

    def init_weight(self):
        self.linear_layer_project.weight.data.normal_(0, 0.01)
        self.linear_layer_attend[1].weight.data.normal_(0, 0.01)
        self.linear_layer_attend[1].bias.data.fill_(0)
        self.linear_layer_attend[4].weight.data.normal_(0, 0.01)
        self.linear_layer_attend[4].bias.data.fill_(0)
        self.linear_layer_compare[1].weight.data.normal_(0, 0.01)
        self.linear_layer_compare[1].bias.data.fill_(0)
        self.linear_layer_compare[4].weight.data.normal_(0, 0.01)
        self.linear_layer_compare[4].bias.data.fill_(0)
        self.linear_layer_aggregate[1].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[1].bias.data.fill_(0)
        self.linear_layer_aggregate[4].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[4].bias.data.fill_(0)

    def attention_softmax3d(self, raw_attentions):
        reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
        out = nn.functional.softmax(reshaped_attentions, dim=1)
        return out.view(raw_attentions.size(0), raw_attentions.size(1), raw_attentions.size(2))

    def _transformation_input(self, embed_sent):
        embed_sent = self.linear_layer_project(embed_sent)
        result = embed_sent
        if self.intra_attention:
            f_intra = self.linear_layer_intra(embed_sent)
            f_intra_t = torch.transpose(f_intra, 1, 2)
            raw_attentions = torch.matmul(f_intra, f_intra_t)
            time_steps = embed_sent.size(1)
            r = torch.arange(0, time_steps)
            r_matrix = r.view(1, -1).expand(time_steps, time_steps)
            raw_index = r_matrix - r.view(-1, 1)
            clipped_index = torch.clamp(raw_index, 0, self.distance_biases - 1)
            clipped_index = Variable(clipped_index.long())
            if torch.cuda.is_available():
                clipped_index = clipped_index
            bias = self.bias_embedding(clipped_index)
            bias = torch.squeeze(bias)
            raw_attentions += bias
            attentions = self.attention_softmax3d(raw_attentions)
            attended = torch.matmul(attentions, embed_sent)
            result = torch.cat([embed_sent, attended], 2)
        return result

    def attend(self, sent1, sent2, lsize_list, rsize_list):
        """
        Compute inter-sentence attention. This is step 1 (attend) in the paper

        :param sent1: tensor in shape (batch, time_steps, num_units),
            the projected sentence 1
        :param sent2: tensor in shape (batch, time_steps, num_units)
        :return: a tuple of 3-d tensors, alfa and beta.
        """
        repr1 = self.linear_layer_attend(sent1)
        repr2 = self.linear_layer_attend(sent2)
        repr2 = torch.transpose(repr2, 1, 2)
        raw_attentions = torch.matmul(repr1, repr2)
        att_sent1 = self.attention_softmax3d(raw_attentions)
        beta = torch.matmul(att_sent1, sent2)
        raw_attentions_t = torch.transpose(raw_attentions, 1, 2).contiguous()
        att_sent2 = self.attention_softmax3d(raw_attentions_t)
        alpha = torch.matmul(att_sent2, sent1)
        return alpha, beta

    def compare(self, sentence, soft_alignment):
        """
        Apply a feed forward network to compare o   ne sentence to its
        soft alignment with the other.

        :param sentence: embedded and projected sentence,
            shape (batch, time_steps, num_units)
        :param soft_alignment: tensor with shape (batch, time_steps, num_units)
        :return: a tensor (batch, time_steps, num_units)
        """
        sent_alignment = torch.cat([sentence, soft_alignment], 2)
        out = self.linear_layer_compare(sent_alignment)
        return out

    def aggregate(self, v1, v2):
        """
        Aggregate the representations induced from both sentences and their
        representations

        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
        """
        v1_sum = torch.sum(v1, 1)
        v2_sum = torch.sum(v2, 1)
        out = self.linear_layer_aggregate(torch.cat([v1_sum, v2_sum], 1))
        return out

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        lsize_list = [len(s.split(' ')) for s in raw_sent1]
        rsize_list = [len(s.split(' ')) for s in raw_sent2]
        sent1 = sent1.permute(0, 2, 1)
        sent2 = sent2.permute(0, 2, 1)
        sent1 = self._transformation_input(sent1)
        sent2 = self._transformation_input(sent2)
        alpha, beta = self.attend(sent1, sent2, lsize_list, rsize_list)
        v1 = self.compare(sent1, beta)
        v2 = self.compare(sent2, alpha)
        logits = self.aggregate(v1, v2)
        return logits


def ortho_weight(ndim):
    """
    Random orthogonal weights
    Used by norm_weights(below), in which case, we
    are ensuring that the rows are orthogonal
    (i.e W = U \\Sigma V, U has the same
    # of rows, V has the same # of cols)
    """
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


class LSTM_Cell(nn.Module):

    def __init__(self, device, in_dim, mem_dim):
        super(LSTM_Cell, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            h = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
            h.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            return h

        def new_W():
            w = nn.Linear(self.in_dim, self.mem_dim)
            w.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            return w
        self.ih = new_gate()
        self.fh = new_gate()
        self.oh = new_gate()
        self.ch = new_gate()
        self.cx = new_W()
        self.ox = new_W()
        self.fx = new_W()
        self.ix = new_W()

    def forward(self, input, h, c):
        u = F.tanh(self.cx(input) + self.ch(h))
        i = F.sigmoid(self.ix(input) + self.ih(h))
        f = F.sigmoid(self.fx(input) + self.fh(h))
        c = i * u + f * c
        o = F.sigmoid(self.ox(input) + self.oh(h))
        h = o * F.tanh(c)
        return c, h


class LSTM(nn.Module):

    def __init__(self, device, in_dim, mem_dim):
        super(LSTM, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.TreeCell = LSTM_Cell(device, in_dim, mem_dim)
        self.output_module = None

    def forward(self, x, x_mask):
        """
        :param x: #step x #sample x dim_emb
        :param x_mask: #step x #sample
        :param x_left_mask: #step x #sample x #step
        :param x_right_mask: #step x #sample x #step
        :return:
        """
        h = Variable(torch.zeros(x.size(1), x.size(2)))
        c = Variable(torch.zeros(x.size(1), x.size(2)))
        if torch.cuda.is_available():
            h = h
            c = c
        all_hidden = []
        for step in range(x.size(0)):
            input = x[step]
            step_c, step_h = self.TreeCell(input, h, c)
            h = x_mask[step][:, (None)] * step_h + (1.0 - x_mask[step])[:, (None)] * h
            c = x_mask[step][:, (None)] * step_c + (1.0 - x_mask[step])[:, (None)] * c
            all_hidden.append(torch.unsqueeze(h, 0))
        return torch.cat(all_hidden, 0)


class ESIM(nn.Module):
    """
        Implementation of the multi feed forward network model described in
        the paper "A Decomposable Attention Model for Natural Language
        Inference" by Parikh et al., 2016.
        It applies feedforward MLPs to combinations of parts of the two sentences,
        without any recurrent structure.
    """

    def __init__(self, num_units, num_classes, embedding_size, dropout, device=0, training=True, project_input=True, use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
        Create the model based on MLP networks.
        :param num_units: size of the networks
        :param num_classes: number of classes in the problem
        :param embedding_size: size of each word embedding
        :param use_intra_attention: whether to use intra-attention model
        :param training: whether to create training tensors (optimizer)
        :param project_input: whether to project input embeddings to a
            different dimensionality
        :param distance_biases: number of different distances with biases used
            in the intra-attention model
        """
        super(ESIM, self).__init__()
        self.arch = 'ESIM'
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input
        self.embedding_size = embedding_size
        self.distance_biases = distance_biases
        self.max_sentence_length = max_sentence_length
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.lstm_intra = LSTM(device, embedding_size, num_units)
        self.linear_layer_compare = nn.Sequential(nn.Linear(4 * num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=dropout))
        self.lstm_compare = LSTM(device, embedding_size, num_units)
        self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(4 * num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(num_units, num_classes))
        self.init_weight()

    def ortho_weight(self):
        """
        Random orthogonal weights
        Used by norm_weights(below), in which case, we
        are ensuring that the rows are orthogonal
        (i.e W = U \\Sigma V, U has the same
        # of rows, V has the same # of cols)
        """
        ndim = self.num_units
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype('float32')

    def initialize_lstm(self):
        if torch.cuda.is_available():
            init = torch.Tensor(np.concatenate([self.ortho_weight(), self.ortho_weight(), self.ortho_weight(), self.ortho_weight()], 0))
        else:
            init = torch.Tensor(np.concatenate([self.ortho_weight(), self.ortho_weight(), self.ortho_weight(), self.ortho_weight()], 0))
        return init

    def init_weight(self):
        self.linear_layer_compare[0].weight.data.normal_(0, 0.01)
        self.linear_layer_compare[0].bias.data.fill_(0)
        self.linear_layer_aggregate[1].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[1].bias.data.fill_(0)
        self.linear_layer_aggregate[4].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[4].bias.data.fill_(0)

    def attention_softmax3d(self, raw_attentions):
        reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
        out = nn.functional.softmax(reshaped_attentions, dim=1)
        return out.view(raw_attentions.size(0), raw_attentions.size(1), raw_attentions.size(2))

    def _transformation_input(self, embed_sent, x1_mask):
        embed_sent = self.word_embedding(embed_sent)
        embed_sent = self.dropout(embed_sent)
        hidden = self.lstm_intra(embed_sent, x1_mask)
        return hidden

    def aggregate(self, v1, v2):
        """
        Aggregate the representations induced from both sentences and their
        representations
        :param v1: tensor with shape (batch, time_steps, num_units)
        :param v2: tensor with shape (batch, time_steps, num_units)
        :return: logits over classes, shape (batch, num_classes)
        """
        v1_mean = torch.mean(v1, 0)
        v2_mean = torch.mean(v2, 0)
        v1_max, _ = torch.max(v1, 0)
        v2_max, _ = torch.max(v2, 0)
        out = self.linear_layer_aggregate(torch.cat((v1_mean, v1_max, v2_mean, v2_max), 1))
        return out

    def cosine_interaction(self, tensor1, tensor2):
        """
        :param tensor1: #step1 * dim
        :param tensor2: #step2 * dim
        :return: #step1 * #step2
        """
        simCube_0 = tensor1[0].view(1, -1)
        simCube_1 = tensor2[0].view(1, -1)
        for i in range(tensor1.size(0)):
            for j in range(tensor2.size(0)):
                if not (i == 0 and j == 0):
                    simCube_0 = torch.cat((simCube_0, tensor1[i].view(1, -1)))
                    simCube_1 = torch.cat((simCube_1, tensor2[j].view(1, -1)))
        simCube = F.cosine_similarity(simCube_0, simCube_1)
        return simCube.view(tensor1.size(0), tensor2.size(0))

    def create_mask(self, sent):
        masks = []
        sent_lengths = [len(s.split(' ')) for s in sent]
        max_len = max(sent_lengths)
        for s_length in sent_lengths:
            pad_mask = np.zeros(max_len)
            pad_mask[:s_length] = 1
            masks.append(pad_mask)
        masks = np.array(masks)
        return torch.from_numpy(masks).float()

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None, visualize=False):
        sent1 = sent1.permute(2, 0, 1)
        sent2 = sent2.permute(2, 0, 1)
        x1_mask = self.create_mask(raw_sent1)
        x2_mask = self.create_mask(raw_sent2)
        x1_mask = x1_mask.permute(1, 0)
        x2_mask = x2_mask.permute(1, 0)
        x1 = self.dropout(sent1)
        x2 = self.dropout(sent2)
        idx_1 = [i for i in range(x1.size(0) - 1, -1, -1)]
        idx_1 = Variable(torch.LongTensor(idx_1))
        if torch.cuda.is_available():
            idx_1 = idx_1
        x1_r = torch.index_select(x1, 0, idx_1)
        x1_mask_r = torch.index_select(x1_mask, 0, idx_1)
        idx_2 = [i for i in range(x2.size(0) - 1, -1, -1)]
        idx_2 = Variable(torch.LongTensor(idx_2))
        if torch.cuda.is_available():
            idx_2 = Variable(torch.LongTensor(idx_2))
        x2_r = torch.index_select(x2, 0, idx_2)
        x2_mask_r = torch.index_select(x2_mask, 0, idx_2)
        proj1 = self.lstm_intra(x1, x1_mask)
        proj1_r = self.lstm_intra(x1_r, x1_mask_r)
        proj2 = self.lstm_intra(x2, x2_mask)
        proj2_r = self.lstm_intra(x2_r, x2_mask_r)
        ctx1 = torch.cat((proj1, torch.index_select(proj1_r, 0, idx_1)), 2)
        ctx2 = torch.cat((proj2, torch.index_select(proj2_r, 0, idx_2)), 2)
        ctx1 = ctx1 * x1_mask[:, :, (None)]
        ctx2 = ctx2 * x2_mask[:, :, (None)]
        weight_matrix = torch.matmul(ctx1.permute(1, 0, 2), ctx2.permute(1, 2, 0))
        if visualize:
            return weight_matrix
        weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1, keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2, keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_1 = weight_matrix_1 * x1_mask[:, (None), :]
        weight_matrix_2 = weight_matrix_2 * x2_mask[(None), :, :]
        alpha = weight_matrix_1 / weight_matrix_1.sum(0, keepdim=True)
        beta = weight_matrix_2 / weight_matrix_2.sum(1, keepdim=True)
        self.alpha = alpha
        self.beta = beta
        ctx2_ = (torch.unsqueeze(ctx1, 1) * torch.unsqueeze(alpha, 3)).sum(0)
        ctx1_ = (torch.unsqueeze(ctx2, 0) * torch.unsqueeze(beta, 3)).sum(1)
        """
        tmp_result=[]
        for batch_i in range(ctx1.size(1)):
            tmp_result.append(torch.unsqueeze(self.cosine_interaction(ctx1[:,batch_i,:], ctx2[:,batch_i,:]), 0))
        weight_matrix=torch.cat(tmp_result)
        weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1, keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2, keepdim=True)[0]).permute(1, 2, 0)
        # weight_matrix_1: #step1 x #step2 x #sample
        weight_matrix_1 = weight_matrix_1 * x1_mask[:, None, :]
        weight_matrix_2 = weight_matrix_2 * x2_mask[None, :, :]
        alpha = weight_matrix_1 / weight_matrix_1.sum(0, keepdim=True)
        beta = weight_matrix_2 / weight_matrix_2.sum(1, keepdim=True)
        ctx2_cos_ = (torch.unsqueeze(ctx1, 1) * torch.unsqueeze(alpha, 3)).sum(0)
        ctx1_cos_ = (torch.unsqueeze(ctx2, 0) * torch.unsqueeze(beta, 3)).sum(1)
        """
        inp1 = torch.cat([ctx1, ctx1_, ctx1 * ctx1_, ctx1 - ctx1_], 2)
        inp2 = torch.cat([ctx2, ctx2_, ctx2 * ctx2_, ctx2 - ctx2_], 2)
        inp1 = self.dropout(self.linear_layer_compare(inp1))
        inp2 = self.dropout(self.linear_layer_compare(inp2))
        inp1_r = torch.index_select(inp1, 0, idx_1)
        inp2_r = torch.index_select(inp2, 0, idx_2)
        v1 = self.lstm_compare(inp1, x1_mask)
        v2 = self.lstm_compare(inp2, x2_mask)
        v1_r = self.lstm_compare(inp1_r, x1_mask)
        v2_r = self.lstm_compare(inp2_r, x2_mask)
        v1 = torch.cat((v1, torch.index_select(v1_r, 0, idx_1)), 2)
        v2 = torch.cat((v2, torch.index_select(v2_r, 0, idx_2)), 2)
        out = self.aggregate(v1, v2)
        out = F.log_softmax(out, dim=1)
        return out


class MPCNN(nn.Module):

    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units, num_classes, dropout, ext_feats, attention, wide_conv):
        super().__init__()
        self.arch = 'mpcnn'
        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        self.ext_feats = ext_feats
        self.attention = attention
        self.wide_conv = wide_conv
        self.in_channels = n_word_dim if attention == 'none' else 2 * n_word_dim
        self._add_layers()
        n_feats = self._get_n_feats()
        self.final_layers = nn.Sequential(nn.Linear(n_feats, hidden_layer_units), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden_layer_units, num_classes), nn.LogSoftmax(1))

    def _add_layers(self):
        holistic_conv_layers_max = []
        holistic_conv_layers_min = []
        holistic_conv_layers_mean = []
        per_dim_conv_layers_max = []
        per_dim_conv_layers_min = []
        for ws in self.filter_widths:
            if np.isinf(ws):
                continue
            padding = ws - 1 if self.wide_conv else 0
            holistic_conv_layers_max.append(nn.Sequential(nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding), nn.Tanh()))
            holistic_conv_layers_min.append(nn.Sequential(nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding), nn.Tanh()))
            holistic_conv_layers_mean.append(nn.Sequential(nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding), nn.Tanh()))
            per_dim_conv_layers_max.append(nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, ws, padding=padding, groups=self.in_channels), nn.Tanh()))
            per_dim_conv_layers_min.append(nn.Sequential(nn.Conv1d(self.in_channels, self.in_channels * self.n_per_dim_filters, ws, padding=padding, groups=self.in_channels), nn.Tanh()))
        self.holistic_conv_layers_max = nn.ModuleList(holistic_conv_layers_max)
        self.holistic_conv_layers_min = nn.ModuleList(holistic_conv_layers_min)
        self.holistic_conv_layers_mean = nn.ModuleList(holistic_conv_layers_mean)
        self.per_dim_conv_layers_max = nn.ModuleList(per_dim_conv_layers_max)
        self.per_dim_conv_layers_min = nn.ModuleList(per_dim_conv_layers_min)

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_1_COMPONENTS_PER_DIM, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2 + self.in_channels, 2
        n_feats_h = 3 * self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = 3 * (len(self.filter_widths) - 1) ** 2 * COMP_1_COMPONENTS_HOLISTIC + 3 * 3 + 2 * (len(self.filter_widths) - 1) * self.n_per_dim_filters * COMP_1_COMPONENTS_PER_DIM
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        block_b = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(1) * sent.size(2)
                block_a[ws] = {'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1), 'min': F.max_pool1d(-1 * sent_flattened, sent_flattened_size).view(sent.size(0), -1), 'mean': F.avg_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)}
                continue
            holistic_conv_out_max = self.holistic_conv_layers_max[ws - 1](sent)
            holistic_conv_out_min = self.holistic_conv_layers_min[ws - 1](sent)
            holistic_conv_out_mean = self.holistic_conv_layers_mean[ws - 1](sent)
            block_a[ws] = {'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1, self.n_holistic_filters), 'min': F.max_pool1d(-1 * holistic_conv_out_min, holistic_conv_out_min.size(2)).contiguous().view(-1, self.n_holistic_filters), 'mean': F.avg_pool1d(holistic_conv_out_mean, holistic_conv_out_mean.size(2)).contiguous().view(-1, self.n_holistic_filters)}
            per_dim_conv_out_max = self.per_dim_conv_layers_max[ws - 1](sent)
            per_dim_conv_out_min = self.per_dim_conv_layers_min[ws - 1](sent)
            block_b[ws] = {'max': F.max_pool1d(per_dim_conv_out_max, per_dim_conv_out_max.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters), 'min': F.max_pool1d(-1 * per_dim_conv_out_min, per_dim_conv_out_min.size(2)).contiguous().view(-1, self.in_channels, self.n_per_dim_filters)}
        return block_a, block_b

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for pool in ('max', 'min', 'mean'):
            regM1, regM2 = [], []
            for ws in self.filter_widths:
                x1 = sent1_block_a[ws][pool].unsqueeze(2)
                x2 = sent2_block_a[ws][pool].unsqueeze(2)
                if np.isinf(ws):
                    x1 = x1.expand(-1, self.n_holistic_filters, -1)
                    x2 = x2.expand(-1, self.n_holistic_filters, -1)
                regM1.append(x1)
                regM2.append(x2)
            regM1 = torch.cat(regM1, dim=2)
            regM2 = torch.cat(regM2, dim=2)
            comparison_feats.append(F.cosine_similarity(regM1, regM2, dim=2))
            pairwise_distances = []
            for x1, x2 in zip(regM1, regM2):
                dist = F.pairwise_distance(x1, x2).view(1, -1)
                pairwise_distances.append(dist)
            comparison_feats.append(torch.cat(pairwise_distances))
        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b):
        comparison_feats = []
        ws_no_inf = [w for w in self.filter_widths if not np.isinf(w)]
        for pool in ('max', 'min', 'mean'):
            for ws1 in self.filter_widths:
                x1 = sent1_block_a[ws1][pool]
                for ws2 in self.filter_widths:
                    x2 = sent2_block_a[ws2][pool]
                    if not np.isinf(ws1) and not np.isinf(ws2) or np.isinf(ws1) and np.isinf(ws2):
                        comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                        comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                        comparison_feats.append(torch.abs(x1 - x2))
        for pool in ('max', 'min'):
            for ws in ws_no_inf:
                oG_1B = sent1_block_b[ws][pool]
                oG_2B = sent2_block_b[ws][pool]
                for i in range(0, self.n_per_dim_filters):
                    x1 = oG_1B[:, :, (i)]
                    x2 = oG_2B[:, :, (i)]
                    comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                    comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                    comparison_feats.append(torch.abs(x1 - x2))
        return torch.cat(comparison_feats, dim=1)

    def concat_attention(self, sent1, sent2, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        sent1_transposed = sent1.transpose(1, 2)
        attention_dot = torch.bmm(sent1_transposed, sent2)
        sent1_norms = torch.norm(sent1_transposed, p=2, dim=2, keepdim=True)
        sent2_norms = torch.norm(sent2, p=2, dim=1, keepdim=True)
        attention_norms = torch.bmm(sent1_norms, sent2_norms)
        attention_matrix = attention_dot / attention_norms
        if self.attention == 'idf' and word_to_doc_count is not None:
            idf_matrix1 = sent1.data.new_ones(sent1.size(0), sent1.size(2))
            for i, sent in enumerate(raw_sent1):
                for j, word in enumerate(sent.split(' ')):
                    idf_matrix1[i, j] /= word_to_doc_count.get(word, 1)
            idf_matrix2 = sent2.data.new_ones(sent2.size(0), sent2.size(2)).fill_(1)
            for i, sent in enumerate(raw_sent2):
                for j, word in enumerate(sent.split(' ')):
                    idf_matrix2[i, j] /= word_to_doc_count.get(word, 1)
            sum_row = (attention_matrix * idf_matrix2.unsqueeze(1)).sum(2)
            sum_col = (attention_matrix * idf_matrix1.unsqueeze(2)).sum(1)
        else:
            sum_row = attention_matrix.sum(2)
            sum_col = attention_matrix.sum(1)
        if self.attention == 'idf' and word_to_doc_count is not None:
            for i, sent in enumerate(raw_sent1):
                for j, word in enumerate(sent.split(' ')):
                    sum_row[i, j] /= word_to_doc_count.get(word, 1)
            for i, sent in enumerate(raw_sent2):
                for j, word in enumerate(sent.split(' ')):
                    sum_col[i, j] /= word_to_doc_count.get(word, 1)
        attention_weight_vec1 = F.softmax(sum_row, 1)
        attention_weight_vec2 = F.softmax(sum_col, 1)
        attention_weighted_sent1 = attention_weight_vec1.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent1
        attention_weighted_sent2 = attention_weight_vec2.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent2
        attention_emb1 = torch.cat((attention_weighted_sent1, sent1), dim=1)
        attention_emb2 = torch.cat((attention_weighted_sent2, sent2), dim=1)
        return attention_emb1, attention_emb2

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        if self.attention != 'none':
            sent1, sent2 = self.concat_attention(sent1, sent2, word_to_doc_count, raw_sent1, raw_sent2)
        sent1_block_a, sent1_block_b = self._get_blocks_for_sentence(sent1)
        sent2_block_a, sent2_block_b = self._get_blocks_for_sentence(sent2)
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a, sent1_block_b, sent2_block_b)
        combined_feats = [feat_h, feat_v, ext_feats] if self.ext_feats else [feat_h, feat_v]
        feat_all = torch.cat(combined_feats, dim=1)
        preds = self.final_layers(feat_all)
        return preds


class PairwiseConv(nn.Module):
    """docstring for PairwiseConv"""

    def __init__(self, model):
        super(PairwiseConv, self).__init__()
        self.convModel = model
        self.dropout = nn.Dropout(self.convModel.dropout)
        self.linearLayer = nn.Linear(model.n_hidden, 1)
        self.posModel = self.convModel
        self.negModel = self.convModel

    def forward(self, input):
        pos = self.posModel(input[0])
        neg = self.negModel(input[1])
        pos = self.dropout(pos)
        neg = self.dropout(neg)
        pos = self.linearLayer(pos)
        neg = self.linearLayer(neg)
        combine = torch.cat([pos, neg], 1)
        return combine


class SmPlusPlus(nn.Module):

    def __init__(self, config):
        super(SmPlusPlus, self).__init__()
        output_channel = config.output_channel
        questions_num = config.questions_num
        answers_num = config.answers_num
        words_dim = config.words_dim
        filter_width = config.filter_width
        self.mode = config.mode
        n_classes = config.target_class
        ext_feats_size = 4
        if self.mode == 'multichannel':
            input_channel = 2
        else:
            input_channel = 1
        self.question_embed = nn.Embedding(questions_num, words_dim)
        self.answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed = nn.Embedding(questions_num, words_dim)
        self.nonstatic_question_embed = nn.Embedding(questions_num, words_dim)
        self.static_answer_embed = nn.Embedding(answers_num, words_dim)
        self.nonstatic_answer_embed = nn.Embedding(answers_num, words_dim)
        self.static_question_embed.weight.requires_grad = False
        self.static_answer_embed.weight.requires_grad = False
        self.conv_q = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))
        self.conv_a = nn.Conv2d(input_channel, output_channel, (filter_width, words_dim), padding=(filter_width - 1, 0))
        self.dropout = nn.Dropout(config.dropout)
        n_hidden = 2 * output_channel + ext_feats_size
        self.combined_feature_vector = nn.Linear(n_hidden, n_hidden)
        self.hidden = nn.Linear(n_hidden, n_classes)

    def _unsqueeze(self, tensor):
        dim = tensor.size()
        return tensor.view(dim[0], 1, dim[1], dim[2])

    def forward(self, x_question, x_answer, x_ext):
        if self.mode == 'rand':
            question = self._unsqueeze(self.question_embed(x_question))
            answer = self._unsqueeze(self.answer_embed(x_answer))
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        elif self.mode == 'static':
            question = self._unsqueeze(self.static_question_embed(x_question))
            answer = self._unsqueeze(self.static_answer_embed(x_answer))
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        elif self.mode == 'non-static':
            question = self._unsqueeze(self.nonstatic_question_embed(x_question))
            answer = self._unsqueeze(self.nonstatic_answer_embed(x_answer))
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        elif self.mode == 'multichannel':
            question_static = self.static_question_embed(x_question)
            answer_static = self.static_answer_embed(x_answer)
            question_nonstatic = self.nonstatic_question_embed(x_question)
            answer_nonstatic = self.nonstatic_answer_embed(x_answer)
            question = torch.stack([question_static, question_nonstatic], dim=1)
            answer = torch.stack([answer_static, answer_nonstatic], dim=1)
            x = [F.tanh(self.conv_q(question)).squeeze(3), F.tanh(self.conv_a(answer)).squeeze(3)]
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        else:
            None
            exit()
        x.append(x_ext)
        x = torch.cat(x, 1)
        x = F.tanh(self.combined_feature_vector(x))
        x = self.dropout(x)
        x = self.hidden(x)
        return x


class StackBiLSTMMaxout(nn.Module):

    def __init__(self, h_size=[512, 1024, 2048], d=300, mlp_d=1600, dropout_r=0.1, max_l=60, num_classes=3):
        super().__init__()
        self.arch = 'SSE'
        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size[0], num_layers=1, bidirectional=True)
        self.lstm_1 = nn.LSTM(input_size=d + h_size[0] * 2, hidden_size=h_size[1], num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=d + (h_size[0] + h_size[1]) * 2, hidden_size=h_size[2], num_layers=1, bidirectional=True)
        self.max_l = max_l
        self.h_size = h_size
        self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, num_classes)
        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout(dropout_r), self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r), self.sm])

    def display(self):
        for param in self.parameters():
            None

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        sent1 = sent1.permute(2, 0, 1)
        sent2 = sent2.permute(2, 0, 1)
        sent1_lengths = torch.tensor([len(s.split(' ')) for s in raw_sent1])
        sent2_lengths = torch.tensor([len(s.split(' ')) for s in raw_sent2])
        if self.max_l:
            sent1_lengths = sent1_lengths.clamp(max=self.max_l)
            sent2_lengths = sent2_lengths.clamp(max=self.max_l)
            if sent1.size(0) > self.max_l:
                sent1 = sent1[:self.max_l, :]
            if sent2.size(0) > self.max_l:
                sent2 = sent2[:self.max_l, :]
        sent1_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, sent1, sent1_lengths)
        sent2_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, sent2, sent2_lengths)
        len1 = sent1_layer1_out.size(0)
        len2 = sent2_layer1_out.size(0)
        p_sent1 = sent1[:len1, :, :]
        p_sent2 = sent2[:len2, :, :]
        sent1_layer2_in = torch.cat([p_sent1, sent1_layer1_out], dim=2)
        sent2_layer2_in = torch.cat([p_sent2, sent2_layer1_out], dim=2)
        sent1_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, sent1_layer2_in, sent1_lengths)
        sent2_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1, sent2_layer2_in, sent2_lengths)
        sent1_layer3_in = torch.cat([p_sent1, sent1_layer1_out, sent1_layer2_out], dim=2)
        sent2_layer3_in = torch.cat([p_sent2, sent2_layer1_out, sent2_layer2_out], dim=2)
        sent1_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, sent1_layer3_in, sent1_lengths)
        sent2_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2, sent2_layer3_in, sent2_lengths)
        sent1_layer3_maxout = torch_util.max_along_time(sent1_layer3_out, sent1_lengths)
        sent2_layer3_maxout = torch_util.max_along_time(sent2_layer3_out, sent2_lengths)
        features = torch.cat([sent1_layer3_maxout, sent2_layer3_maxout, torch.abs(sent1_layer3_maxout - sent2_layer3_maxout), sent1_layer3_maxout * sent2_layer3_maxout], dim=1)
        out = self.classifier(features)
        out = F.log_softmax(out, dim=1)
        return out


class ResNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        n_layers = config['res_layers']
        n_maps = config['res_fmaps']
        n_labels = config['n_labels']
        self.conv0 = nn.Conv2d(12, n_maps, (3, 3), padding=1)
        self.convs = nn.ModuleList([nn.Conv2d(n_maps, n_maps, (3, 3), padding=1) for _ in range(n_layers)])
        self.output = nn.Linear(n_maps, n_labels)
        self.input_len = None

    def forward(self, x):
        x = F.relu(self.conv0(x))
        old_x = x
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i % 2 == 1:
                x += old_x
                old_x = x
        x = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
        return F.log_softmax(self.output(x), 1)


def hard_pad2d(x, pad):

    def pad_side(idx):
        pad_len = max(pad - x.size(idx), 0)
        return [0, pad_len]
    padding = pad_side(3)
    padding.extend(pad_side(2))
    x = F.pad(x, padding)
    return x[:, :, :pad, :pad]


class VDPWIConvNet(nn.Module):

    def __init__(self, config):
        super().__init__()

        def make_conv(n_in, n_out):
            conv = nn.Conv2d(n_in, n_out, 3, padding=1)
            conv.bias.data.zero_()
            nn.init.xavier_normal_(conv.weight)
            return conv
        self.conv1 = make_conv(12, 128)
        self.conv2 = make_conv(128, 164)
        self.conv3 = make_conv(164, 192)
        self.conv4 = make_conv(192, 192)
        self.conv5 = make_conv(192, 128)
        self.maxpool2 = nn.MaxPool2d(2, ceil_mode=True)
        self.dnn = nn.Linear(128, 128)
        self.output = nn.Linear(128, config['n_labels'])
        self.input_len = 32

    def forward(self, x):
        x = hard_pad2d(x, self.input_len)
        pool_final = nn.MaxPool2d(2, ceil_mode=True) if x.size(2) == 32 else nn.MaxPool2d(3, 1, ceil_mode=True)
        x = self.maxpool2(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool2(F.relu(self.conv3(x)))
        x = self.maxpool2(F.relu(self.conv4(x)))
        x = pool_final(F.relu(self.conv5(x)))
        x = F.relu(self.dnn(x.view(x.size(0), -1)))
        return F.log_softmax(self.output(x), 1)


class VDPWIModel(nn.Module):

    def __init__(self, dim, config):
        super().__init__()
        self.arch = 'vdpwi'
        self.hidden_dim = config['rnn_hidden_dim']
        self.rnn = nn.LSTM(dim, self.hidden_dim, 1, batch_first=True)
        self.device = config['device']
        if config['classifier'] == 'vdpwi':
            self.classifier_net = VDPWIConvNet(config)
        elif config['classifier'] == 'resnet':
            self.classifier_net = ResNet(config)

    def create_pad_cube(self, sent1, sent2):
        pad_cube = []
        sent1_lengths = [len(s.split(' ')) for s in sent1]
        sent2_lengths = [len(s.split(' ')) for s in sent2]
        max_len1 = max(sent1_lengths)
        max_len2 = max(sent2_lengths)
        for s1_length, s2_length in zip(sent1_lengths, sent2_lengths):
            pad_mask = np.ones((max_len1, max_len2))
            pad_mask[:s1_length, :s2_length] = 0
            pad_cube.append(pad_mask)
        pad_cube = np.array(pad_cube)
        return torch.from_numpy(pad_cube).float().unsqueeze(0)

    def compute_sim_cube(self, seq1, seq2):

        def compute_sim(prism1, prism2):
            prism1_len = prism1.norm(dim=3)
            prism2_len = prism2.norm(dim=3)
            dot_prod = torch.matmul(prism1.unsqueeze(3), prism2.unsqueeze(4))
            dot_prod = dot_prod.squeeze(3).squeeze(3)
            cos_dist = dot_prod / (prism1_len * prism2_len + 1e-08)
            l2_dist = (prism1 - prism2).norm(dim=3)
            return torch.stack([dot_prod, cos_dist, l2_dist], 1)

        def compute_prism(seq1, seq2):
            prism1 = seq1.repeat(seq2.size(1), 1, 1, 1)
            prism2 = seq2.repeat(seq1.size(1), 1, 1, 1)
            prism1 = prism1.permute(1, 2, 0, 3).contiguous()
            prism2 = prism2.permute(1, 0, 2, 3).contiguous()
            return compute_sim(prism1, prism2)
        sim_cube = torch.Tensor(seq1.size(0), 12, seq1.size(1), seq2.size(1))
        sim_cube = sim_cube
        seq1_f = seq1[:, :, :self.hidden_dim]
        seq1_b = seq1[:, :, self.hidden_dim:]
        seq2_f = seq2[:, :, :self.hidden_dim]
        seq2_b = seq2[:, :, self.hidden_dim:]
        sim_cube[:, 0:3] = compute_prism(seq1, seq2)
        sim_cube[:, 3:6] = compute_prism(seq1_f, seq2_f)
        sim_cube[:, 6:9] = compute_prism(seq1_b, seq2_b)
        sim_cube[:, 9:12] = compute_prism(seq1_f + seq1_b, seq2_f + seq2_b)
        return sim_cube

    def compute_focus_cube(self, sim_cube, pad_cube):
        neg_magic = -10000
        pad_cube = pad_cube.repeat(12, 1, 1, 1)
        pad_cube = pad_cube.permute(1, 0, 2, 3).contiguous()
        sim_cube = neg_magic * pad_cube + sim_cube
        mask = torch.Tensor(*sim_cube.size())
        mask[:, :, :, :] = 0.1

        def build_mask(index):
            max_mask = sim_cube[:, (index)].clone()
            for _ in range(min(sim_cube.size(2), sim_cube.size(3))):
                values, indices = torch.max(max_mask.view(sim_cube.size(0), -1), 1)
                row_indices = indices / sim_cube.size(3)
                col_indices = indices % sim_cube.size(3)
                row_indices = row_indices.unsqueeze(1)
                col_indices = col_indices.unsqueeze(1).unsqueeze(1)
                for i, (row_i, col_i, val) in enumerate(zip(row_indices, col_indices, values)):
                    if val < neg_magic / 2:
                        continue
                    mask[(i), :, (row_i), (col_i)] = 1
                    max_mask[(i), (row_i), :] = neg_magic
                    max_mask[(i), :, (col_i)] = neg_magic
        build_mask(9)
        build_mask(10)
        focus_cube = mask * sim_cube * (1 - pad_cube)
        return focus_cube

    def forward(self, sent1, sent2, ext_feats=None, word_to_doc_count=None, raw_sent1=None, raw_sent2=None):
        pad_cube = self.create_pad_cube(raw_sent1, raw_sent2)
        sent1 = sent1.permute(0, 2, 1).contiguous()
        sent2 = sent2.permute(0, 2, 1).contiguous()
        seq1f, _ = self.rnn(sent1)
        seq2f, _ = self.rnn(sent2)
        seq1b, _ = self.rnn(torch.cat(sent1.split(1, 1)[::-1], 1))
        seq2b, _ = self.rnn(torch.cat(sent2.split(1, 1)[::-1], 1))
        seq1 = torch.cat([seq1f, seq1b], 2)
        seq2 = torch.cat([seq2f, seq2b], 2)
        sim_cube = self.compute_sim_cube(seq1, seq2)
        truncate = self.classifier_net.input_len
        sim_cube = sim_cube[:, :, :pad_cube.size(2), :pad_cube.size(3)].contiguous()
        if truncate is not None:
            sim_cube = sim_cube[:, :, :truncate, :truncate].contiguous()
            pad_cube = pad_cube[:, :, :sim_cube.size(2), :sim_cube.size(3)].contiguous()
        focus_cube = self.compute_focus_cube(sim_cube, pad_cube)
        log_prob = self.classifier_net(focus_cube)
        return log_prob


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (LSTM,
     lambda: ([], {'device': 0, 'in_dim': 4, 'mem_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LSTM_Cell,
     lambda: ([], {'device': 0, 'in_dim': 4, 'mem_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResNet,
     lambda: ([], {'config': _mock_config(res_layers=1, res_fmaps=4, n_labels=4)}),
     lambda: ([torch.rand([4, 12, 64, 64])], {}),
     True),
    (VDPWIConvNet,
     lambda: ([], {'config': _mock_config(n_labels=4)}),
     lambda: ([torch.rand([4, 12, 4, 4])], {}),
     False),
]

class Test_castorini_castor(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

