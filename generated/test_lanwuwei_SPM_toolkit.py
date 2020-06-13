import sys
_module = sys.modules[__name__]
del sys
main_mnli = _module
main_quora = _module
main_snli = _module
main_sts = _module
main_trecqa = _module
main_url = _module
main_wikiqa = _module
model = _module
preprocess = _module
util = _module
Constants = _module
binary_tree = _module
data_iterator = _module
main = _module
main_batch = _module
main_mnli = _module
main_pit = _module
main_quora = _module
main_snli = _module
main_sts = _module
main_trecqa = _module
main_url = _module
main_wikiqa = _module
model = _module
model_batch = _module
tree = _module
vocab = _module
main_batch_mnli = _module
main_batch_pit = _module
main_batch_quora = _module
main_batch_snli = _module
main_batch_trecqa = _module
main_batch_url = _module
main_batch_wikiqa = _module
model_batch = _module
main = _module
main_sts = _module
model = _module
data_loader = _module
main_mnli = _module
main_pit = _module
main_quora = _module
main_snli = _module
main_sts = _module
main_trecqa = _module
main_url = _module
main_wikiqa = _module
mnli = _module
model = _module
test_quora = _module
torch_util = _module
train_quora = _module

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


from torch.autograd import Variable


import torch


import random


import logging


import numpy as np


import itertools


import math


import torch.nn as nn


import torch.nn.functional as F


import numpy


from numpy import linalg as LA


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


from torch import optim


from collections import Counter


class DecAtt(nn.Module):
    """
	    Implementation of the multi feed forward network model described in
	    the paper "A Decomposable Attention Model for Natural Language
	    Inference" by Parikh et al., 2016.

	    It applies feedforward MLPs to combinations of parts of the two sentences,
	    without any recurrent structure.
	"""

    def __init__(self, num_units, num_classes, vocab_size, embedding_size,
        pretrained_emb, training=True, project_input=True,
        use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
		Create the model based on MLP networks.

		:param num_units: size of the networks
		:param num_classes: number of classes in the problem
		:param vocab_size: size of the vocabulary
		:param embedding_size: size of each word embedding
		:param use_intra_attention: whether to use intra-attention model
		:param training: whether to create training tensors (optimizer)
		:param project_input: whether to project input embeddings to a
		    different dimensionality
		:param distance_biases: number of different distances with biases used
		    in the intra-attention model
		"""
        super(DecAtt, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input
        self.embedding_size = embedding_size
        self.distance_biases = distance_biases
        self.intra_attention = False
        self.max_sentence_length = max_sentence_length
        self.pretrained_emb = pretrained_emb
        self.bias_embedding = nn.Embedding(max_sentence_length, 1)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear_layer_project = nn.Linear(embedding_size, num_units,
            bias=False)
        self.linear_layer_attend = nn.Sequential(nn.Dropout(p=0.2), nn.
            Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=0.2), nn.
            Linear(num_units, num_units), nn.ReLU())
        self.linear_layer_compare = nn.Sequential(nn.Dropout(p=0.2), nn.
            Linear(num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(num_units, num_units), nn.ReLU())
        self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=0.2), nn.
            Linear(num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units,
            num_classes), nn.LogSoftmax())
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
        return out.view(raw_attentions.size(0), raw_attentions.size(1),
            raw_attentions.size(2))

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

    def forward(self, sent1, sent2, lsize_list, rsize_list):
        sent1 = self._transformation_input(sent1)
        sent2 = self._transformation_input(sent2)
        alpha, beta = self.attend(sent1, sent2, lsize_list, rsize_list)
        v1 = self.compare(sent1, beta)
        v2 = self.compare(sent2, alpha)
        logits = self.aggregate(v1, v2)
        return logits


class DecAtt(nn.Module):
    """
	    Implementation of the multi feed forward network model described in
	    the paper "A Decomposable Attention Model for Natural Language
	    Inference" by Parikh et al., 2016.

	    It applies feedforward MLPs to combinations of parts of the two sentences,
	    without any recurrent structure.
	"""

    def __init__(self, num_units, num_classes, vocab_size, embedding_size,
        pretrained_emb, training=True, project_input=True,
        use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
		Create the model based on MLP networks.

		:param num_units: size of the networks
		:param num_classes: number of classes in the problem
		:param vocab_size: size of the vocabulary
		:param embedding_size: size of each word embedding
		:param use_intra_attention: whether to use intra-attention model
		:param training: whether to create training tensors (optimizer)
		:param project_input: whether to project input embeddings to a
		    different dimensionality
		:param distance_biases: number of different distances with biases used
		    in the intra-attention model
		"""
        super(DecAtt, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input
        self.embedding_size = embedding_size
        self.distance_biases = distance_biases
        self.intra_attention = False
        self.max_sentence_length = max_sentence_length
        self.pretrained_emb = pretrained_emb
        self.bias_embedding = nn.Embedding(max_sentence_length, 1)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.linear_layer_project = nn.Linear(embedding_size, num_units,
            bias=False)
        self.linear_layer_attend = nn.Sequential(nn.Dropout(p=0.2), nn.
            Linear(num_units, num_units), nn.ReLU(), nn.Dropout(p=0.2), nn.
            Linear(num_units, num_units), nn.ReLU())
        self.linear_layer_compare = nn.Sequential(nn.Dropout(p=0.2), nn.
            Linear(num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(num_units, num_units), nn.ReLU())
        self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=0.2), nn.
            Linear(num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(num_units, num_units), nn.ReLU(), nn.Linear(num_units,
            num_classes), nn.LogSoftmax())
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
        self.word_embedding.weight.data.copy_(self.pretrained_emb)

    def attention_softmax3d(self, raw_attentions):
        reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
        out = nn.functional.softmax(reshaped_attentions, dim=1)
        return out.view(raw_attentions.size(0), raw_attentions.size(1),
            raw_attentions.size(2))

    def _transformation_input(self, embed_sent):
        embed_sent = self.word_embedding(embed_sent)
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

    def forward(self, sent1, sent2, lsize_list=None, rsize_list=None):
        sent1 = self._transformation_input(sent1)
        sent2 = self._transformation_input(sent2)
        alpha, beta = self.attend(sent1, sent2, lsize_list, rsize_list)
        v1 = self.compare(sent1, beta)
        v2 = self.compare(sent2, alpha)
        logits = self.aggregate(v1, v2)
        return logits


class BinaryTreeLeafModule(nn.Module):

    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)
        if self.cudaFlag:
            self.cx = self.cx
            self.ox = self.ox

    def forward(self, input):
        c = self.cx(input)
        o = F.sigmoid(self.ox(input))
        h = o * F.tanh(c)
        return c, h


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


class BinaryTreeComposer(nn.Module):

    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
            rh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
            lh.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            rh.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            return lh, rh

        def new_W():
            w = nn.Linear(self.in_dim, self.mem_dim)
            w.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            return w
        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()
        self.olh, self.orh = new_gate()
        self.cx = new_W()
        self.ox = new_W()
        self.fx = new_W()
        self.ix = new_W()
        if self.cudaFlag:
            self.ilh = self.ilh
            self.irh = self.irh
            self.lflh = self.lflh
            self.lfrh = self.lfrh
            self.rflh = self.rflh
            self.rfrh = self.rfrh
            self.ulh = self.ulh
            self.urh = self.urh
            self.olh = self.olh
            self.orh = self.orh

    def forward(self, input, lc, lh, rc, rh):
        u = F.tanh(self.cx(input) + self.ulh(lh) + self.urh(rh))
        i = F.sigmoid(self.ix(input) + self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.fx(input) + self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.fx(input) + self.rflh(lh) + self.rfrh(rh))
        c = i * u + lf * lc + rf * rc
        o = F.sigmoid(self.ox(input) + self.olh(lh) + self.orh(rh))
        h = o * F.tanh(c)
        return c, h


class BinaryTreeLSTM(nn.Module):

    def __init__(self, cuda, in_dim, mem_dim, word_embedding, num_words):
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.word_embedding = word_embedding
        self.num_words = num_words
        self.composer = BinaryTreeComposer(cuda, in_dim, mem_dim)
        self.output_module = None
        self.all_ststes = []
        self.all_words = []

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
		Get flatParameters
		note that getParameters and parameters is not equal in this case
		getParameters do not get parameters of output module
		:return: 1d tensor
		"""
        params = []
        for m in [self.ix, self.ih, self.fx, self.fh, self.ox, self.oh,
            self.ux, self.uh]:
            l = list(m.parameters())
            params.extend(l)
        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def forward(self, tree, embs, PAD):
        if tree.num_children == 0:
            lc = Variable(torch.zeros(1, self.mem_dim))
            lh = Variable(torch.zeros(1, self.mem_dim))
            rc = Variable(torch.zeros(1, self.mem_dim))
            rh = Variable(torch.zeros(1, self.mem_dim))
            if torch.cuda.is_available():
                lc = lc
                lh = lh
                rc = rc
                rh = rh
            tree.state = self.composer.forward(embs[tree.idx - 1], lc, lh,
                rc, rh)
            self.all_ststes.append(tree.state[1].view(1, self.mem_dim))
        else:
            for idx in xrange(tree.num_children):
                _ = self.forward(tree.children[idx], embs, PAD)
            lc, lh, rc, rh = self.get_child_state(tree)
            if PAD:
                index = Variable(torch.LongTensor([self.num_words - 1]))
                if torch.cuda.is_available():
                    index = index
                tree.state = self.composer.forward(self.word_embedding(
                    index), lc, lh, rc, rh)
            else:
                tree.state = self.composer.forward(embs[tree.idx - 1], lc,
                    lh, rc, rh)
            self.all_ststes.append(tree.state[1].view(1, self.mem_dim))
        return tree.state

    def get_child_state(self, tree):
        lc, lh = tree.children[0].state
        rc, rh = tree.children[1].state
        return lc, lh, rc, rh


class ESIM(nn.Module):
    """
		Implementation of the multi feed forward network model described in
		the paper "A Decomposable Attention Model for Natural Language
		Inference" by Parikh et al., 2016.

		It applies feedforward MLPs to combinations of parts of the two sentences,
		without any recurrent structure.
	"""

    def __init__(self, num_units, num_classes, vocab_size, embedding_size,
        pretrained_emb, num_words):
        super(ESIM, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.pretrained_emb = pretrained_emb
        self.dropout = nn.Dropout(p=0.5)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.tree_lstm_intra = BinaryTreeLSTM(torch.cuda.is_available(),
            embedding_size, num_units, self.word_embedding, num_words)
        self.linear_layer_compare = nn.Sequential(nn.Linear(4 * num_units,
            num_units), nn.ReLU(), nn.Dropout(p=0.5))
        self.tree_lstm_compare = BinaryTreeLSTM(torch.cuda.is_available(),
            embedding_size, num_units, self.word_embedding, num_words)
        self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=0.5), nn.
            Linear(4 * num_units, num_units), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(num_units, num_classes))
        self.init_weight()

    def init_weight(self):
        self.linear_layer_compare[0].weight.data.normal_(0, 0.01)
        self.linear_layer_compare[0].bias.data.fill_(0)
        self.linear_layer_aggregate[1].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[1].bias.data.fill_(0)
        self.linear_layer_aggregate[4].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[4].bias.data.fill_(0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.
            pretrained_emb))

    def attention_softmax3d(self, raw_attentions):
        reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
        out = nn.functional.softmax(reshaped_attentions, dim=1)
        return out.view(raw_attentions.size(0), raw_attentions.size(1),
            raw_attentions.size(2))

    def _transformation_input(self, embed_sent, tree, PAD=True):
        embed_sent = self.word_embedding(embed_sent)
        embed_sent = self.dropout(embed_sent)
        _ = self.tree_lstm_intra(tree, embed_sent, PAD)
        output = torch.cat(self.tree_lstm_intra.all_ststes, 0)
        del self.tree_lstm_intra.all_ststes[:]
        return output

    def attend(self, sent1, sent2):
        repr2 = torch.transpose(sent2, 1, 2)
        self.raw_attentions = torch.matmul(sent1, repr2)
        att_sent1 = self.attention_softmax3d(self.raw_attentions)
        beta = torch.matmul(att_sent1, sent2)
        raw_attentions_t = torch.transpose(self.raw_attentions, 1, 2
            ).contiguous()
        att_sent2 = self.attention_softmax3d(raw_attentions_t)
        alpha = torch.matmul(att_sent2, sent1)
        return alpha, beta

    def compare(self, sentence, soft_alignment, tree, PAD=False):
        sent_alignment = torch.cat([sentence, soft_alignment, sentence -
            soft_alignment, sentence * soft_alignment], 2)
        sent_alignment = self.linear_layer_compare(sent_alignment)
        sent_alignment = self.dropout(sent_alignment)
        sent_alignment = sent_alignment[0]
        _ = self.tree_lstm_compare(tree, sent_alignment, PAD)
        output = torch.cat(self.tree_lstm_compare.all_ststes, 0)
        del self.tree_lstm_compare.all_ststes[:]
        return output

    def aggregate(self, v1, v2):
        v1_mean = torch.mean(v1, 1)
        v2_mean = torch.mean(v2, 1)
        v1_max, _ = torch.max(v1, 1)
        v2_max, _ = torch.max(v2, 1)
        out = self.linear_layer_aggregate(torch.cat((v1_mean, v1_max,
            v2_mean, v2_max), 1))
        return out

    def forward(self, sent1, sent2, tree1, tree2):
        sent1 = self._transformation_input(sent1, tree1)
        sent2 = self._transformation_input(sent2, tree2)
        sent1 = torch.unsqueeze(sent1, 0)
        sent2 = torch.unsqueeze(sent2, 0)
        alpha, beta = self.attend(sent1, sent2)
        v1 = self.compare(sent1, beta, tree1)
        v2 = self.compare(sent2, alpha, tree2)
        v1 = torch.unsqueeze(v1, 0)
        v2 = torch.unsqueeze(v2, 0)
        logits = self.aggregate(v1, v2)
        return logits


class BinaryTreeCell(nn.Module):

    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeCell, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
            rh = nn.Linear(self.mem_dim, self.mem_dim, bias=False)
            lh.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            rh.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            return lh, rh

        def new_W():
            w = nn.Linear(self.in_dim, self.mem_dim)
            w.weight.data.copy_(torch.from_numpy(ortho_weight(self.mem_dim)))
            return w
        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()
        self.olh, self.orh = new_gate()
        self.cx = new_W()
        self.ox = new_W()
        self.fx = new_W()
        self.ix = new_W()

    def forward(self, input, lc, lh, rc, rh):
        u = F.tanh(self.cx(input) + self.ulh(lh) + self.urh(rh))
        i = F.sigmoid(self.ix(input) + self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.fx(input) + self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.fx(input) + self.rflh(lh) + self.rfrh(rh))
        c = i * u + lf * lc + rf * rc
        o = F.sigmoid(self.ox(input) + self.olh(lh) + self.orh(rh))
        h = o * F.tanh(c)
        return c, h


class BinaryTreeLSTM(nn.Module):

    def __init__(self, cuda, in_dim, mem_dim):
        super(BinaryTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.TreeCell = BinaryTreeCell(cuda, in_dim, mem_dim)
        self.output_module = None
        self.all_ststes = []
        self.all_words = []

    def forward(self, x, x_mask, x_left_mask, x_right_mask):
        """

		:param x: #step x #sample x dim_emb
		:param x_mask: #step x #sample
		:param x_left_mask: #step x #sample x #step
		:param x_right_mask: #step x #sample x #step
		:return:
		"""
        h = Variable(torch.zeros(x.size(1), x.size(0), x.size(2)))
        c = Variable(torch.zeros(x.size(1), x.size(0), x.size(2)))
        if torch.cuda.is_available():
            h = h
            c = c
        for step in range(x.size(0)):
            input = x[step]
            lh = torch.sum(x_left_mask[step][:, :, (None)] * h, 1)
            rh = torch.sum(x_right_mask[step][:, :, (None)] * h, 1)
            lc = torch.sum(x_left_mask[step][:, :, (None)] * c, 1)
            rc = torch.sum(x_right_mask[step][:, :, (None)] * c, 1)
            step_c, step_h = self.TreeCell(input, lc, lh, rc, rh)
            if step == 0:
                new_h = torch.cat((torch.unsqueeze(step_h, 1), h[:, step + 
                    1:, :]), 1)
                new_c = torch.cat((torch.unsqueeze(step_c, 1), c[:, step + 
                    1:, :]), 1)
            elif step == x.size(0) - 1:
                new_h = torch.cat((h[:, :step, :], torch.unsqueeze(step_h, 
                    1)), 1)
                new_c = torch.cat((c[:, :step, :], torch.unsqueeze(step_c, 
                    1)), 1)
            else:
                new_h = torch.cat((h[:, :step, :], torch.unsqueeze(step_h, 
                    1), h[:, step + 1:, :]), 1)
                new_c = torch.cat((c[:, :step, :], torch.unsqueeze(step_c, 
                    1), c[:, step + 1:, :]), 1)
            h = x_mask[step][:, (None), (None)] * new_h + (1 - x_mask[step]
                [:, (None), (None)]) * h
            c = x_mask[step][:, (None), (None)] * new_c + (1 - x_mask[step]
                [:, (None), (None)]) * c
        return h


class ESIM(nn.Module):
    """
		Implementation of the multi feed forward network model described in
		the paper "A Decomposable Attention Model for Natural Language
		Inference" by Parikh et al., 2016.

		It applies feedforward MLPs to combinations of parts of the two sentences,
		without any recurrent structure.
	"""

    def __init__(self, num_units, num_classes, vocab_size, embedding_size,
        pretrained_emb, training=True, project_input=True,
        use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
		Create the model based on MLP networks.

		:param num_units: size of the networks
		:param num_classes: number of classes in the problem
		:param vocab_size: size of the vocabulary
		:param embedding_size: size of each word embedding
		:param use_intra_attention: whether to use intra-attention model
		:param training: whether to create training tensors (optimizer)
		:param project_input: whether to project input embeddings to a
			different dimensionality
		:param distance_biases: number of different distances with biases used
			in the intra-attention model
		"""
        super(ESIM, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input
        self.embedding_size = embedding_size
        self.distance_biases = distance_biases
        self.max_sentence_length = max_sentence_length
        self.pretrained_emb = pretrained_emb
        self.dropout = nn.Dropout(p=0.5)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.tree_lstm_intra = BinaryTreeLSTM(torch.cuda.is_available(),
            embedding_size, num_units)
        self.linear_layer_compare = nn.Sequential(nn.Linear(4 * num_units,
            num_units), nn.ReLU(), nn.Dropout(p=0.5))
        self.tree_lstm_compare = BinaryTreeLSTM(torch.cuda.is_available(),
            embedding_size, num_units)
        self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=0.5), nn.
            Linear(4 * num_units, num_units), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(num_units, num_classes))
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
            init = torch.Tensor(np.concatenate([self.ortho_weight(), self.
                ortho_weight(), self.ortho_weight(), self.ortho_weight()], 0))
        else:
            init = torch.Tensor(np.concatenate([self.ortho_weight(), self.
                ortho_weight(), self.ortho_weight(), self.ortho_weight()], 0))
        return init

    def init_weight(self):
        self.linear_layer_compare[0].weight.data.normal_(0, 0.01)
        self.linear_layer_compare[0].bias.data.fill_(0)
        self.linear_layer_aggregate[1].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[1].bias.data.fill_(0)
        self.linear_layer_aggregate[4].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[4].bias.data.fill_(0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.
            pretrained_emb))

    def attention_softmax3d(self, raw_attentions):
        reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
        out = nn.functional.softmax(reshaped_attentions, dim=1)
        return out.view(raw_attentions.size(0), raw_attentions.size(1),
            raw_attentions.size(2))

    def _transformation_input(self, embed_sent, x1_mask, x1_left_mask,
        x1_right_mask):
        embed_sent = self.word_embedding(embed_sent)
        embed_sent = self.dropout(embed_sent)
        hidden = self.tree_lstm_intra(embed_sent, x1_mask, x1_left_mask,
            x1_right_mask)
        return hidden

    def aggregate(self, v1, v2):
        """
		Aggregate the representations induced from both sentences and their
		representations

		:param v1: tensor with shape (batch, time_steps, num_units)
		:param v2: tensor with shape (batch, time_steps, num_units)
		:return: logits over classes, shape (batch, num_classes)
		"""
        v1_mean = torch.mean(v1, 1)
        v2_mean = torch.mean(v2, 1)
        v1_max, _ = torch.max(v1, 1)
        v2_max, _ = torch.max(v2, 1)
        out = self.linear_layer_aggregate(torch.cat((v1_mean, v1_max,
            v2_mean, v2_max), 1))
        return out

    def forward(self, x1, x1_mask, x1_left_mask, x1_right_mask, x2, x2_mask,
        x2_left_mask, x2_right_mask):
        sent1 = self._transformation_input(x1, x1_mask, x1_left_mask,
            x1_right_mask)
        sent2 = self._transformation_input(x2, x2_mask, x2_left_mask,
            x2_right_mask)
        ctx1 = torch.transpose(sent1, 0, 1)
        ctx2 = torch.transpose(sent2, 0, 1)
        ctx1 = ctx1 * x1_mask[:, :, (None)]
        ctx2 = ctx2 * x2_mask[:, :, (None)]
        weight_matrix = torch.matmul(ctx1.permute(1, 0, 2), ctx2.permute(1,
            2, 0))
        weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1,
            keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2,
            keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_1 = weight_matrix_1 * x1_mask[:, (None), :]
        weight_matrix_2 = weight_matrix_2 * x2_mask[(None), :, :]
        alpha = weight_matrix_1 / weight_matrix_1.sum(0, keepdim=True)
        beta = weight_matrix_2 / weight_matrix_2.sum(1, keepdim=True)
        ctx2_ = (torch.unsqueeze(ctx1, 1) * torch.unsqueeze(alpha, 3)).sum(0)
        ctx1_ = (torch.unsqueeze(ctx2, 0) * torch.unsqueeze(beta, 3)).sum(1)
        inp1 = torch.cat([ctx1, ctx1_, ctx1 * ctx1_, ctx1 - ctx1_], 2)
        inp2 = torch.cat([ctx2, ctx2_, ctx2 * ctx2_, ctx2 - ctx2_], 2)
        inp1 = self.dropout(self.linear_layer_compare(inp1))
        inp2 = self.dropout(self.linear_layer_compare(inp2))
        v1 = self.tree_lstm_compare(inp1, x1_mask, x1_left_mask, x1_right_mask)
        v2 = self.tree_lstm_compare(inp2, x2_mask, x2_left_mask, x2_right_mask)
        logits = self.aggregate(v1, v2)
        return logits


class LSTM_Cell(nn.Module):

    def __init__(self, cuda, in_dim, mem_dim):
        super(LSTM_Cell, self).__init__()
        self.cudaFlag = cuda
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

    def __init__(self, cuda, in_dim, mem_dim):
        super(LSTM, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.TreeCell = LSTM_Cell(cuda, in_dim, mem_dim)
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
            h = x_mask[step][:, (None)] * step_h + (1.0 - x_mask[step])[:,
                (None)] * h
            c = x_mask[step][:, (None)] * step_c + (1.0 - x_mask[step])[:,
                (None)] * c
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

    def __init__(self, num_units, num_classes, vocab_size, embedding_size,
        pretrained_emb, training=True, project_input=True,
        use_intra_attention=False, distance_biases=10, max_sentence_length=30):
        """
		Create the model based on MLP networks.

		:param num_units: size of the networks
		:param num_classes: number of classes in the problem
		:param vocab_size: size of the vocabulary
		:param embedding_size: size of each word embedding
		:param use_intra_attention: whether to use intra-attention model
		:param training: whether to create training tensors (optimizer)
		:param project_input: whether to project input embeddings to a
			different dimensionality
		:param distance_biases: number of different distances with biases used
			in the intra-attention model
		"""
        super(ESIM, self).__init__()
        self.vocab_size = vocab_size
        self.num_units = num_units
        self.num_classes = num_classes
        self.project_input = project_input
        self.embedding_size = embedding_size
        self.distance_biases = distance_biases
        self.max_sentence_length = max_sentence_length
        self.pretrained_emb = pretrained_emb
        self.dropout = nn.Dropout(p=0.5)
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm_intra = LSTM(torch.cuda.is_available(), embedding_size,
            num_units)
        self.linear_layer_compare = nn.Sequential(nn.Linear(4 * num_units *
            2, num_units), nn.ReLU(), nn.Dropout(p=0.5))
        self.lstm_compare = LSTM(torch.cuda.is_available(), embedding_size,
            num_units)
        self.linear_layer_aggregate = nn.Sequential(nn.Dropout(p=0.5), nn.
            Linear(4 * num_units * 2, num_units), nn.ReLU(), nn.Dropout(p=
            0.5), nn.Linear(num_units, num_classes))
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
            init = torch.Tensor(np.concatenate([self.ortho_weight(), self.
                ortho_weight(), self.ortho_weight(), self.ortho_weight()], 0))
        else:
            init = torch.Tensor(np.concatenate([self.ortho_weight(), self.
                ortho_weight(), self.ortho_weight(), self.ortho_weight()], 0))
        return init

    def init_weight(self):
        self.linear_layer_compare[0].weight.data.normal_(0, 0.01)
        self.linear_layer_compare[0].bias.data.fill_(0)
        self.linear_layer_aggregate[1].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[1].bias.data.fill_(0)
        self.linear_layer_aggregate[4].weight.data.normal_(0, 0.01)
        self.linear_layer_aggregate[4].bias.data.fill_(0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(self.
            pretrained_emb))

    def attention_softmax3d(self, raw_attentions):
        reshaped_attentions = raw_attentions.view(-1, raw_attentions.size(2))
        out = nn.functional.softmax(reshaped_attentions, dim=1)
        return out.view(raw_attentions.size(0), raw_attentions.size(1),
            raw_attentions.size(2))

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
        out = self.linear_layer_aggregate(torch.cat((v1_mean, v1_max,
            v2_mean, v2_max), 1))
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

    def forward_old(self, x1, x1_mask, x2, x2_mask):
        x1 = self.word_embedding(x1)
        x1 = self.dropout(x1)
        x2 = self.word_embedding(x2)
        x2 = self.dropout(x2)
        idx_1 = [i for i in range(x1.size(0) - 1, -1, -1)]
        if torch.cuda.is_available():
            idx_1 = Variable(torch.cuda.LongTensor(idx_1))
        else:
            idx_1 = Variable(torch.LongTensor(idx_1))
        x1_r = torch.index_select(x1, 0, idx_1)
        x1_mask_r = torch.index_select(x1_mask, 0, idx_1)
        idx_2 = [i for i in range(x2.size(0) - 1, -1, -1)]
        if torch.cuda.is_available():
            idx_2 = Variable(torch.cuda.LongTensor(idx_2))
        else:
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
        weight_matrix = torch.matmul(ctx1.permute(1, 0, 2), ctx2.permute(1,
            2, 0))
        weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1,
            keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2,
            keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_1 = weight_matrix_1 * x1_mask[:, (None), :]
        weight_matrix_2 = weight_matrix_2 * x2_mask[(None), :, :]
        alpha = weight_matrix_1 / weight_matrix_1.sum(0, keepdim=True)
        beta = weight_matrix_2 / weight_matrix_2.sum(1, keepdim=True)
        ctx2_ = (torch.unsqueeze(ctx1, 1) * torch.unsqueeze(alpha, 3)).sum(0)
        ctx1_ = (torch.unsqueeze(ctx2, 0) * torch.unsqueeze(beta, 3)).sum(1)
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
        logits = self.aggregate(v1, v2)
        return logits

    def forward(self, x1, x1_mask, x2, x2_mask):
        x1 = self.word_embedding(x1)
        x1 = self.dropout(x1)
        x2 = self.word_embedding(x2)
        x2 = self.dropout(x2)
        idx_1 = [i for i in range(x1.size(0) - 1, -1, -1)]
        if torch.cuda.is_available():
            idx_1 = Variable(torch.cuda.LongTensor(idx_1))
        else:
            idx_1 = Variable(torch.LongTensor(idx_1))
        x1_r = torch.index_select(x1, 0, idx_1)
        x1_mask_r = torch.index_select(x1_mask, 0, idx_1)
        idx_2 = [i for i in range(x2.size(0) - 1, -1, -1)]
        if torch.cuda.is_available():
            idx_2 = Variable(torch.cuda.LongTensor(idx_2))
        else:
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
        weight_matrix = torch.matmul(ctx1.permute(1, 0, 2), ctx2.permute(1,
            2, 0))
        weight_matrix_1 = torch.exp(weight_matrix - weight_matrix.max(1,
            keepdim=True)[0]).permute(1, 2, 0)
        weight_matrix_2 = torch.exp(weight_matrix - weight_matrix.max(2,
            keepdim=True)[0]).permute(1, 2, 0)
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
        logits = self.aggregate(v1, v2)
        return logits


def splitclusters(s):
    """Generate the grapheme clusters for the string s. (Not the full
    Unicode text segmentation algorithm, but probably good enough for
    Devanagari.)

    """
    virama = '्'
    cluster = ''
    last = None
    for c in s:
        cat = unicodedata.category(c)[0]
        if cat == 'M' or cat == 'L' and last == virama:
            cluster += c
        else:
            if cluster:
                yield cluster
            cluster = c
        last = c
    if cluster:
        yield cluster


class DeepPairWiseWord(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, num_layers, task,
        granularity, num_class, dict, fake_dict, dict_char_ngram, oov,
        tokens, word_freq, feature_maps, kernels, charcnn_embedding_size,
        charcnn_max_word_length, character_ngrams, c2w_mode,
        character_ngrams_overlap, word_mode, combine_mode, lm_mode, deep_CNN):
        super(DeepPairWiseWord, self).__init__()
        self.task = task
        if task == 'pit':
            self.limit = 32
        else:
            self.limit = 48
        if lm_mode:
            self.lm_loss = nn.NLLLoss()
            self.lm_softmax = nn.LogSoftmax()
            self.lm_tanh = nn.Tanh()
            self.lm_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                bidirectional=True)
            self.lm_Wm_forward = Variable(torch.rand(hidden_dim, hidden_dim))
            self.lm_Wm_backward = Variable(torch.rand(hidden_dim, hidden_dim))
            self.lm_Wq_forward = Variable(torch.rand(hidden_dim, len(tokens)))
            self.lm_Wq_backword = Variable(torch.rand(hidden_dim, len(tokens)))
        self.granularity = granularity
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.dict = dict
        self.fake_dict = fake_dict
        self.dict_char_ngram = dict_char_ngram
        self.word_freq = word_freq
        self.oov = oov
        self.tokens = tokens
        word2id = {}
        index = 0
        for word in tokens:
            word2id[word] = index
            index += 1
        self.word2id = word2id
        self.feature_maps = feature_maps
        self.kernels = kernels
        self.charcnn_max_word_length = charcnn_max_word_length
        self.character_ngrams = character_ngrams
        self.c2w_mode = c2w_mode
        self.character_ngrams_overlap = character_ngrams_overlap
        self.word_mode = word_mode
        self.combine_mode = combine_mode
        self.lm_mode = lm_mode
        self.deep_CNN = deep_CNN
        if granularity == 'char':
            self.df = Variable(torch.rand(embedding_dim, embedding_dim))
            self.db = Variable(torch.rand(embedding_dim, embedding_dim))
            self.bias = Variable(torch.rand(embedding_dim))
            self.Wx = Variable(torch.rand(embedding_dim, embedding_dim))
            self.W1 = Variable(torch.rand(embedding_dim, embedding_dim))
            self.W2 = Variable(torch.rand(embedding_dim, embedding_dim))
            self.W3 = Variable(torch.rand(embedding_dim, embedding_dim))
            self.vg = Variable(torch.rand(embedding_dim, 1))
            self.bg = Variable(torch.rand(1, 1))
            if torch.cuda.is_available():
                self.df = self.df
                self.db = self.db
                self.bias = self.bias
                self.Wx = self.Wx
                self.W1 = self.W1
                self.W2 = self.W2
                self.W3 = self.W3
                self.vg = self.vg
                self.bg = self.bg
                if lm_mode:
                    self.lm_Wm_forward = self.lm_Wm_forward
                    self.lm_Wm_backward = self.lm_Wm_backward
                    self.lm_Wq_forward = self.lm_Wq_forward
                    self.lm_Wq_backword = self.lm_Wq_backword
                pass
            self.c2w_embedding = nn.Embedding(len(dict_char_ngram), 50)
            self.char_cnn_embedding = nn.Embedding(len(dict_char_ngram),
                charcnn_embedding_size)
            self.lstm_c2w = nn.LSTM(50, embedding_dim, 1, bidirectional=True)
            self.charCNN_filter1 = nn.Sequential(nn.Conv2d(1, feature_maps[
                0], (kernels[0], charcnn_embedding_size)), nn.Tanh(), nn.
                MaxPool2d((charcnn_max_word_length - kernels[0] + 1, 1),
                stride=1))
            self.charCNN_filter2 = nn.Sequential(nn.Conv2d(1, feature_maps[
                1], (kernels[1], charcnn_embedding_size)), nn.Tanh(), nn.
                MaxPool2d((charcnn_max_word_length - kernels[1] + 1, 1),
                stride=1))
            self.charCNN_filter3 = nn.Sequential(nn.Conv2d(1, feature_maps[
                2], (kernels[2], charcnn_embedding_size)), nn.Tanh(), nn.
                MaxPool2d((charcnn_max_word_length - kernels[2] + 1, 1),
                stride=1))
            self.charCNN_filter4 = nn.Sequential(nn.Conv2d(1, feature_maps[
                3], (kernels[3], charcnn_embedding_size)), nn.Tanh(), nn.
                MaxPool2d((charcnn_max_word_length - kernels[3] + 1, 1),
                stride=1))
            self.charCNN_filter5 = nn.Sequential(nn.Conv2d(1, feature_maps[
                4], (kernels[4], charcnn_embedding_size)), nn.Tanh(), nn.
                MaxPool2d((charcnn_max_word_length - kernels[4] + 1, 1),
                stride=1))
            self.charCNN_filter6 = nn.Sequential(nn.Conv2d(1, feature_maps[
                5], (kernels[5], charcnn_embedding_size)), nn.Tanh(), nn.
                MaxPool2d((charcnn_max_word_length - kernels[5] + 1, 1),
                stride=1))
            self.charCNN_filter7 = nn.Sequential(nn.Conv2d(1, feature_maps[
                6], (kernels[6], charcnn_embedding_size)), nn.Tanh(), nn.
                MaxPool2d((charcnn_max_word_length - kernels[6] + 1, 1),
                stride=1))
            self.transform_gate = nn.Sequential(nn.Linear(1100, 1100), nn.
                Sigmoid())
            self.char_cnn_mlp = nn.Sequential(nn.Linear(1100, 1100), nn.Tanh())
            self.down_sampling_200 = nn.Linear(1100, 200)
            self.down_sampling_300 = nn.Linear(1100, 300)
        elif granularity == 'word':
            """"""
            self.word_embedding = nn.Embedding(len(tokens), embedding_dim)
            self.copied_word_embedding = nn.Embedding(len(tokens),
                embedding_dim)
            pretrained_weight = numpy.zeros(shape=(len(self.tokens), self.
                embedding_dim))
            for word in self.tokens:
                pretrained_weight[self.tokens.index(word)] = self.dict[word
                    ].numpy()
            self.copied_word_embedding.weight.data.copy_(torch.from_numpy(
                pretrained_weight))
            """"""
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
            bidirectional=True)
        if not deep_CNN:
            self.mlp_layer = nn.Sequential(nn.Linear(self.limit * self.
                limit * 13, 16), nn.Linear(16, num_class), nn.LogSoftmax())
        else:
            self.layer1 = nn.Sequential(nn.Conv2d(13, 128, kernel_size=3,
                stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                stride=2, ceil_mode=True))
            self.layer1_ = nn.Sequential(nn.Conv2d(26, 128, kernel_size=3,
                stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                stride=2, ceil_mode=True))
            self.layer2 = nn.Sequential(nn.Conv2d(128, 164, kernel_size=3,
                stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                stride=2, ceil_mode=True))
            self.layer3 = nn.Sequential(nn.Conv2d(164, 192, kernel_size=3,
                stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                stride=2, ceil_mode=True))
            self.layer4 = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3,
                stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                stride=2, ceil_mode=True))
            self.layer5 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3,
                stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                stride=2, ceil_mode=True))
            self.layer5_0 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=3,
                stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=3,
                stride=3, ceil_mode=True))
            self.fc1 = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True)
                )
            self.fc2 = nn.Sequential(nn.Linear(128, num_class), nn.LogSoftmax()
                )
        self.init_weight()

    def init_weight(self):
        if self.deep_CNN:
            self.layer1[0].weight.data.normal_(0, math.sqrt(2.0 / (3 * 3 * 
                128)))
            self.layer1[0].bias.data.fill_(0)
            self.layer2[0].weight.data.normal_(0, math.sqrt(2.0 / (3 * 3 * 
                164)))
            self.layer2[0].bias.data.fill_(0)
            self.layer3[0].weight.data.normal_(0, math.sqrt(2.0 / (3 * 3 * 
                192)))
            self.layer3[0].bias.data.fill_(0)
            self.layer4[0].weight.data.normal_(0, math.sqrt(2.0 / (3 * 3 * 
                192)))
            self.layer4[0].bias.data.fill_(0)
            self.layer5[0].weight.data.normal_(0, math.sqrt(2.0 / (3 * 3 * 
                128)))
            self.layer5[0].bias.data.fill_(0)
            self.fc1[0].weight.data.uniform_(-0.1, 0.1)
            self.fc1[0].bias.data.fill_(0)
            self.fc2[0].weight.data.uniform_(-0.1, 0.1)
            self.fc2[0].bias.data.fill_(0)
        if self.granularity == 'word':
            pretrained_weight = numpy.zeros(shape=(len(self.tokens), self.
                embedding_dim))
            for word in self.tokens:
                pretrained_weight[self.tokens.index(word)] = self.dict[word
                    ].numpy()
            self.copied_word_embedding.weight.data.copy_(torch.from_numpy(
                pretrained_weight))

    def unpack(self, bi_hidden, half_dim):
        for i in range(bi_hidden.size(0)):
            vec = bi_hidden[i][:]
            if i == 0:
                h_fw = vec[:half_dim].view(1, -1)
                h_bw = vec[half_dim:].view(1, -1)
            else:
                h_fw_new = vec[:half_dim].view(1, -1)
                h_bw_new = vec[half_dim:].view(1, -1)
                h_fw = torch.cat((h_fw, h_fw_new), 0)
                h_bw = torch.cat((h_bw, h_bw_new), 0)
        return h_fw, h_bw

    def pairwise_word_interaction(self, out0, out1, target_A, target_B):
        extra_loss = 0
        h_fw_0, h_bw_0 = self.unpack(out0.view(out0.size(0), out0.size(2)),
            half_dim=self.hidden_dim)
        h_fw_1, h_bw_1 = self.unpack(out1.view(out1.size(0), out1.size(2)),
            half_dim=self.hidden_dim)
        h_bi_0 = out0.view(out0.size(0), out0.size(2))
        h_bi_1 = out1.view(out1.size(0), out1.size(2))
        h_sum_0 = h_fw_0 + h_bw_0
        h_sum_1 = h_fw_1 + h_bw_1
        len0 = h_fw_0.size(0)
        len1 = h_fw_1.size(0)
        i = 0
        j = 0
        simCube5_0 = h_fw_0[i].view(1, -1)
        simCube5_1 = h_fw_1[j].view(1, -1)
        simCube6_0 = h_bw_0[i].view(1, -1)
        simCube6_1 = h_bw_1[j].view(1, -1)
        simCube7_0 = h_bi_0[i].view(1, -1)
        simCube7_1 = h_bi_1[j].view(1, -1)
        simCube8_0 = h_sum_0[i].view(1, -1)
        simCube8_1 = h_sum_1[j].view(1, -1)
        for i in range(len0):
            for j in range(len1):
                if not (i == 0 and j == 0):
                    simCube5_0 = torch.cat((simCube5_0, h_fw_0[i].view(1, -1)))
                    simCube5_1 = torch.cat((simCube5_1, h_fw_1[j].view(1, -1)))
                    simCube6_0 = torch.cat((simCube6_0, h_bw_0[i].view(1, -1)))
                    simCube6_1 = torch.cat((simCube6_1, h_bw_1[j].view(1, -1)))
                    simCube7_0 = torch.cat((simCube7_0, h_bi_0[i].view(1, -1)))
                    simCube7_1 = torch.cat((simCube7_1, h_bi_1[j].view(1, -1)))
                    simCube8_0 = torch.cat((simCube8_0, h_sum_0[i].view(1, -1))
                        )
                    simCube8_1 = torch.cat((simCube8_1, h_sum_1[j].view(1, -1))
                        )
        simCube1 = torch.unsqueeze(torch.mm(h_fw_0, torch.transpose(h_fw_1,
            0, 1)), 0)
        simCube2 = torch.unsqueeze(torch.mm(h_bw_0, torch.transpose(h_bw_1,
            0, 1)), 0)
        simCube3 = torch.unsqueeze(torch.mm(h_bi_0, torch.transpose(h_bi_1,
            0, 1)), 0)
        simCube4 = torch.unsqueeze(torch.mm(h_sum_0, torch.transpose(
            h_sum_1, 0, 1)), 0)
        simCube5 = torch.neg(F.pairwise_distance(simCube5_0, simCube5_1))
        simCube5 = torch.unsqueeze(simCube5.view(len0, len1), 0)
        simCube6 = torch.neg(F.pairwise_distance(simCube6_0, simCube6_1))
        simCube6 = torch.unsqueeze(simCube6.view(len0, len1), 0)
        simCube7 = torch.neg(F.pairwise_distance(simCube7_0, simCube7_1))
        simCube7 = torch.unsqueeze(simCube7.view(len0, len1), 0)
        simCube8 = torch.neg(F.pairwise_distance(simCube8_0, simCube8_1))
        simCube8 = torch.unsqueeze(simCube8.view(len0, len1), 0)
        simCube9 = F.cosine_similarity(simCube5_0, simCube5_1)
        simCube9 = torch.unsqueeze(simCube9.view(len0, len1), 0)
        simCube10 = F.cosine_similarity(simCube6_0, simCube6_1)
        simCube10 = torch.unsqueeze(simCube10.view(len0, len1), 0)
        simCube11 = F.cosine_similarity(simCube7_0, simCube7_1)
        simCube11 = torch.unsqueeze(simCube11.view(len0, len1), 0)
        simCube12 = F.cosine_similarity(simCube8_0, simCube8_1)
        simCube12 = torch.unsqueeze(simCube12.view(len0, len1), 0)
        """"""
        if torch.cuda.is_available():
            simCube13 = torch.unsqueeze(Variable(torch.zeros(len0, len1)) +
                1, 0)
        else:
            simCube13 = torch.unsqueeze(Variable(torch.zeros(len0, len1)) +
                1, 0)
        simCube = torch.cat((simCube9, simCube5, simCube1, simCube10,
            simCube6, simCube2, simCube12, simCube8, simCube4, simCube11,
            simCube7, simCube3, simCube13), 0)
        return simCube, extra_loss

    def similarity_focus(self, simCube):
        if torch.cuda.is_available():
            mask = torch.mul(torch.ones(simCube.size(0), simCube.size(1),
                simCube.size(2)), 0.1)
        else:
            mask = torch.mul(torch.ones(simCube.size(0), simCube.size(1),
                simCube.size(2)), 0.1)
        s1tag = torch.zeros(simCube.size(1))
        s2tag = torch.zeros(simCube.size(2))
        sorted, indices = torch.sort(simCube[6].view(1, -1), descending=True)
        record = []
        for indix in indices[0]:
            pos1 = torch.div(indix, simCube.size(2)).data[0]
            pos2 = (indix - simCube.size(2) * pos1).data[0]
            if s1tag[pos1] + s2tag[pos2] <= 0:
                s1tag[pos1] = 1
                s2tag[pos2] = 1
                record.append((pos1, pos2))
                mask[0][pos1][pos2] = mask[0][pos1][pos2] + 0.9
                mask[1][pos1][pos2] = mask[1][pos1][pos2] + 0.9
                mask[2][pos1][pos2] = mask[2][pos1][pos2] + 0.9
                mask[3][pos1][pos2] = mask[3][pos1][pos2] + 0.9
                mask[4][pos1][pos2] = mask[4][pos1][pos2] + 0.9
                mask[5][pos1][pos2] = mask[5][pos1][pos2] + 0.9
                mask[6][pos1][pos2] = mask[6][pos1][pos2] + 0.9
                mask[7][pos1][pos2] = mask[7][pos1][pos2] + 0.9
                mask[8][pos1][pos2] = mask[8][pos1][pos2] + 0.9
                mask[9][pos1][pos2] = mask[9][pos1][pos2] + 0.9
                mask[10][pos1][pos2] = mask[10][pos1][pos2] + 0.9
                mask[11][pos1][pos2] = mask[11][pos1][pos2] + 0.9
            mask[12][pos1][pos2] = mask[12][pos1][pos2] + 0.9
        s1tag = torch.zeros(simCube.size(1))
        s2tag = torch.zeros(simCube.size(2))
        sorted, indices = torch.sort(simCube[7].view(1, -1), descending=True)
        counter = 0
        for indix in indices[0]:
            pos1 = torch.div(indix, simCube.size(2)).data[0]
            pos2 = (indix - simCube.size(2) * pos1).data[0]
            if s1tag[pos1] + s2tag[pos2] <= 0:
                counter += 1
                if (pos1, pos2) in record:
                    continue
                else:
                    s1tag[pos1] = 1
                    s2tag[pos2] = 1
                    mask[0][pos1][pos2] = mask[0][pos1][pos2] + 0.9
                    mask[1][pos1][pos2] = mask[1][pos1][pos2] + 0.9
                    mask[2][pos1][pos2] = mask[2][pos1][pos2] + 0.9
                    mask[3][pos1][pos2] = mask[3][pos1][pos2] + 0.9
                    mask[4][pos1][pos2] = mask[4][pos1][pos2] + 0.9
                    mask[5][pos1][pos2] = mask[5][pos1][pos2] + 0.9
                    mask[6][pos1][pos2] = mask[6][pos1][pos2] + 0.9
                    mask[7][pos1][pos2] = mask[7][pos1][pos2] + 0.9
                    mask[8][pos1][pos2] = mask[8][pos1][pos2] + 0.9
                    mask[9][pos1][pos2] = mask[9][pos1][pos2] + 0.9
                    mask[10][pos1][pos2] = mask[10][pos1][pos2] + 0.9
                    mask[11][pos1][pos2] = mask[11][pos1][pos2] + 0.9
            if counter >= len(record):
                break
        focusCube = torch.mul(simCube, Variable(mask))
        return focusCube

    def deep_cnn(self, focusCube):
        simCube = torch.unsqueeze(focusCube, 0)
        focusCube = F.pad(simCube, (0, self.limit - simCube.size(3), 0, 
            self.limit - simCube.size(2)))[0]
        focusCube = torch.unsqueeze(focusCube, 0)
        out = self.layer1(focusCube)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.limit == 16:
            out = self.layer5(out)
        elif self.limit == 32:
            out = self.layer4(out)
            out = self.layer5(out)
        elif self.limit == 48:
            out = self.layer4(out)
            out = self.layer5_0(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def mlp(self, focusCube):
        simCube = torch.unsqueeze(focusCube, 0)
        focusCube = F.pad(simCube, (0, self.limit - simCube.size(3), 0, 
            self.limit - simCube.size(2)))[0]
        result = self.mlp_layer(focusCube.view(-1))
        return result

    def language_model(self, out0, out1, target_A, target_B):
        extra_loss = 0
        h_fw_0, h_bw_0 = self.unpack(out0.view(out0.size(0), out0.size(2)),
            half_dim=self.hidden_dim)
        h_fw_1, h_bw_1 = self.unpack(out1.view(out1.size(0), out1.size(2)),
            half_dim=self.hidden_dim)
        """"""
        m_fw_0 = self.lm_tanh(torch.mm(h_fw_0, self.lm_Wm_forward))
        m_bw_0 = self.lm_tanh(torch.mm(h_bw_0, self.lm_Wm_backward))
        m_fw_1 = self.lm_tanh(torch.mm(h_fw_1, self.lm_Wm_forward))
        m_bw_1 = self.lm_tanh(torch.mm(h_bw_1, self.lm_Wm_backward))
        q_fw_0 = self.lm_softmax(torch.mm(m_fw_0, self.lm_Wq_forward))
        q_bw_0 = self.lm_softmax(torch.mm(m_bw_0, self.lm_Wq_backword))
        q_fw_1 = self.lm_softmax(torch.mm(m_fw_1, self.lm_Wq_forward))
        q_bw_1 = self.lm_softmax(torch.mm(m_bw_1, self.lm_Wq_backword))
        target_fw_0 = Variable(torch.LongTensor(target_A[1:] + [self.tokens
            .index('</s>')]))
        target_bw_0 = Variable(torch.LongTensor([self.tokens.index('<s>')] +
            target_A[:-1]))
        target_fw_1 = Variable(torch.LongTensor(target_B[1:] + [self.tokens
            .index('</s>')]))
        target_bw_1 = Variable(torch.LongTensor([self.tokens.index('<s>')] +
            target_B[:-1]))
        if torch.cuda.is_available():
            target_fw_0 = target_fw_0
            target_bw_0 = target_bw_0
            target_fw_1 = target_fw_1
            target_bw_1 = target_bw_1
        loss1 = self.lm_loss(q_fw_0, target_fw_0)
        loss2 = self.lm_loss(q_bw_0, target_bw_0)
        loss3 = self.lm_loss(q_fw_1, target_fw_1)
        loss4 = self.lm_loss(q_bw_1, target_bw_1)
        extra_loss = loss1 + loss2 + loss3 + loss4
        """"""
        return extra_loss

    def word_layer(self, lsents, rsents):
        glove_mode = self.word_mode[0]
        update_inv_mode = self.word_mode[1]
        update_oov_mode = self.word_mode[2]
        if (glove_mode == True and update_inv_mode == False and 
            update_oov_mode == False):
            try:
                sentA = torch.cat([self.dict[word].view(1, self.
                    embedding_dim) for word in lsents], 0)
                sentA = Variable(sentA)
                sentB = torch.cat([self.dict[word].view(1, self.
                    embedding_dim) for word in rsents], 0)
                sentB = Variable(sentB)
            except:
                None
                None
                sys.exit()
            if torch.cuda.is_available():
                sentA = sentA
                sentB = sentB
        elif glove_mode == True and update_inv_mode == False and update_oov_mode == True:
            firstFlag = True
            for word in lsents:
                if firstFlag:
                    if word in self.oov:
                        indice = Variable(torch.LongTensor([self.tokens.
                            index(word)]))
                        if torch.cuda.is_available():
                            indice = indice
                        output = self.word_embedding(indice)
                        firstFlag = False
                    else:
                        output = Variable(self.dict[word].view(1, self.
                            embedding_dim))
                        if torch.cuda.is_available():
                            output = output
                        firstFlag = False
                elif word in self.oov:
                    indice = Variable(torch.LongTensor([self.tokens.index(
                        word)]))
                    if torch.cuda.is_available():
                        indice = indice
                    output_new = self.word_embedding(indice)
                    output = torch.cat((output, output_new), 0)
                else:
                    output_new = Variable(self.dict[word].view(1, self.
                        embedding_dim))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output = torch.cat((output, output_new), 0)
            sentA = output
            firstFlag = False
            for word in rsents:
                if firstFlag:
                    if word in self.oov:
                        indice = Variable(torch.LongTensor([self.tokens.
                            index(word)]))
                        if torch.cuda.is_available():
                            indice = indice
                        output = self.word_embedding(indice)
                        firstFlag = False
                    else:
                        output = Variable(self.dict[word].view(1, self.
                            embedding_dim))
                        if torch.cuda.is_available():
                            output = output
                        firstFlag = False
                elif word in self.oov:
                    indice = Variable(torch.LongTensor([self.tokens.index(
                        word)]))
                    if torch.cuda.is_available():
                        indice = indice
                    output_new = self.word_embedding(indice)
                    output = torch.cat((output, output_new), 0)
                else:
                    output_new = Variable(self.dict[word].view(1, self.
                        embedding_dim))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output = torch.cat((output, output_new), 0)
            sentB = output
        elif glove_mode == True and update_inv_mode == True and update_oov_mode == False:
            firstFlag = True
            for word in lsents:
                if firstFlag:
                    if word in self.oov:
                        output = Variable(self.fake_dict[word].view(1, self
                            .embedding_dim))
                        if torch.cuda.is_available():
                            output = output
                        output = output.view(1, -1)
                        firstFlag = False
                    else:
                        indice = Variable(torch.LongTensor([self.tokens.
                            index(word)]))
                        if torch.cuda.is_available():
                            indice = indice
                        output = self.copied_word_embedding(indice)
                        firstFlag = False
                elif word in self.oov:
                    output_new = Variable(self.fake_dict[word].view(1, self
                        .embedding_dim))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output = torch.cat((output, output_new.view(1, -1)), 0)
                else:
                    indice = Variable(torch.LongTensor([self.tokens.index(
                        word)]))
                    if torch.cuda.is_available():
                        indice = indice
                    output_new = self.copied_word_embedding(indice)
                    output = torch.cat((output, output_new), 0)
            sentA = output
            firstFlag = True
            for word in rsents:
                if firstFlag:
                    if word in self.oov:
                        output = Variable(torch.Tensor([random.uniform(-
                            0.05, 0.05) for i in range(self.embedding_dim)]))
                        if torch.cuda.is_available():
                            output = output
                        output = output.view(1, -1)
                        firstFlag = False
                    else:
                        indice = Variable(torch.LongTensor([self.tokens.
                            index(word)]))
                        if torch.cuda.is_available():
                            indice = indice
                        output = self.copied_word_embedding(indice)
                        firstFlag = False
                elif word in self.oov:
                    output_new = Variable(torch.Tensor([random.uniform(-
                        0.05, 0.05) for i in range(self.embedding_dim)]))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output = torch.cat((output, output_new.view(1, -1)), 0)
                else:
                    indice = Variable(torch.LongTensor([self.tokens.index(
                        word)]))
                    if torch.cuda.is_available():
                        indice = indice
                    output_new = self.copied_word_embedding(indice)
                    output = torch.cat((output, output_new), 0)
            sentB = output
        elif glove_mode == True and update_inv_mode == True and update_oov_mode == True:
            tmp = []
            for word in lsents:
                try:
                    tmp.append(self.word2id[word])
                except:
                    tmp.append(self.word2id['oov'])
            indices = Variable(torch.LongTensor(tmp))
            if torch.cuda.is_available():
                indices = indices
            sentA = self.copied_word_embedding(indices)
            tmp = []
            for word in rsents:
                try:
                    tmp.append(self.word2id[word])
                except:
                    tmp.append(self.word2id['oov'])
            indices = Variable(torch.LongTensor(tmp))
            if torch.cuda.is_available():
                indices = indices
            sentB = self.copied_word_embedding(indices)
        elif glove_mode == False and update_inv_mode == False and update_oov_mode == False:
            firstFlag = True
            for word in lsents:
                if firstFlag:
                    output = Variable(self.fake_dict[word].view(1, self.
                        embedding_dim))
                    if torch.cuda.is_available():
                        output = output
                    output = output.view(1, -1)
                    firstFlag = False
                else:
                    output_new = Variable(self.fake_dict[word].view(1, self
                        .embedding_dim))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output = torch.cat((output, output_new.view(1, -1)), 0)
            sentA = output
            firstFlag = True
            for word in rsents:
                if firstFlag:
                    output = Variable(self.fake_dict[word].view(1, self.
                        embedding_dim))
                    if torch.cuda.is_available():
                        output = output
                    output = output.view(1, -1)
                    firstFlag = False
                else:
                    output_new = Variable(self.fake_dict[word].view(1, self
                        .embedding_dim))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output = torch.cat((output, output_new.view(1, -1)), 0)
            sentB = output
        elif glove_mode == False and update_inv_mode == False and update_oov_mode == True:
            firstFlag = True
            for word in lsents:
                if firstFlag:
                    if word in self.oov:
                        indice = Variable(torch.LongTensor([self.tokens.
                            index(word)]))
                        if torch.cuda.is_available():
                            indice = indice
                        output = self.word_embedding(indice)
                        firstFlag = False
                    else:
                        output = Variable(self.fake_dict[word].view(1, self
                            .embedding_dim))
                        if torch.cuda.is_available():
                            output = output
                        output = output.view(1, -1)
                        firstFlag = False
                elif word in self.oov:
                    indice = Variable(torch.LongTensor([self.tokens.index(
                        word)]))
                    if torch.cuda.is_available():
                        indice = indice
                    output_new = self.word_embedding(indice)
                    output = torch.cat((output, output_new), 0)
                else:
                    output_new = Variable(self.fake_dict[word].view(1, self
                        .embedding_dim))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output_new = output_new.view(1, -1)
                    output = torch.cat((output, output_new), 0)
            sentA = output
            firstFlag = True
            for word in rsents:
                if firstFlag:
                    if word in self.oov:
                        indice = Variable(torch.LongTensor([self.tokens.
                            index(word)]))
                        if torch.cuda.is_available():
                            indice = indice
                        output = self.word_embedding(indice)
                        firstFlag = False
                    else:
                        output = Variable(self.fake_dict[word].view(1, self
                            .embedding_dim))
                        if torch.cuda.is_available():
                            output = output
                        output = output.view(1, -1)
                        firstFlag = False
                elif word in self.oov:
                    indice = Variable(torch.LongTensor([self.tokens.index(
                        word)]))
                    if torch.cuda.is_available():
                        indice = indice
                    output_new = self.word_embedding(indice)
                    output = torch.cat((output, output_new), 0)
                else:
                    output_new = Variable(self.fake_dict[word].view(1, self
                        .embedding_dim))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output_new = output_new.view(1, -1)
                    output = torch.cat((output, output_new), 0)
            sentB = output
        elif glove_mode == False and update_inv_mode == True and update_oov_mode == False:
            firstFlag = True
            for word in lsents:
                if firstFlag:
                    if word in self.oov:
                        output = Variable(self.dict[word].view(1, self.
                            embedding_dim))
                        if torch.cuda.is_available():
                            output = output
                        output = output.view(1, -1)
                        firstFlag = False
                    else:
                        indice = Variable(torch.LongTensor([self.tokens.
                            index(word)]))
                        if torch.cuda.is_available():
                            indice = indice
                        output = self.word_embedding(indice)
                        firstFlag = False
                elif word in self.oov:
                    output_new = Variable(torch.Tensor(self.dict[word].view
                        (1, self.embedding_dim)))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output = torch.cat((output, output_new.view(1, -1)), 0)
                else:
                    indice = Variable(torch.LongTensor([self.tokens.index(
                        word)]))
                    if torch.cuda.is_available():
                        indice = indice
                    output_new = self.word_embedding(indice)
                    output = torch.cat((output, output_new.view(1, -1)), 0)
            sentA = output
            firstFlag = True
            for word in rsents:
                if firstFlag:
                    if word in self.oov:
                        output = Variable(torch.Tensor([random.uniform(-
                            0.05, 0.05) for i in range(self.embedding_dim)]))
                        if torch.cuda.is_available():
                            output = output
                        output = output.view(1, -1)
                        firstFlag = False
                    else:
                        indice = Variable(torch.LongTensor([self.tokens.
                            index(word)]))
                        if torch.cuda.is_available():
                            indice = indice
                        output = self.word_embedding(indice)
                        firstFlag = False
                elif word in self.oov:
                    output_new = Variable(torch.Tensor([random.uniform(-
                        0.05, 0.05) for i in range(self.embedding_dim)]))
                    if torch.cuda.is_available():
                        output_new = output_new
                    output = torch.cat((output, output_new.view(1, -1)), 0)
                else:
                    indice = Variable(torch.LongTensor([self.tokens.index(
                        word)]))
                    if torch.cuda.is_available():
                        indice = indice
                    output_new = self.word_embedding(indice)
                    output = torch.cat((output, output_new.view(1, -1)), 0)
            sentB = output
        elif glove_mode == False and update_inv_mode == True and update_oov_mode == True:
            indices = Variable(torch.LongTensor([self.tokens.index(word) for
                word in lsents]))
            if torch.cuda.is_available():
                indices = indices
            sentA = self.word_embedding(indices)
            indices = Variable(torch.LongTensor([self.tokens.index(word) for
                word in rsents]))
            if torch.cuda.is_available():
                indices = indices
            sentB = self.word_embedding(indices)
        sentA = torch.unsqueeze(sentA, 0).view(-1, 1, self.embedding_dim)
        sentB = torch.unsqueeze(sentB, 0).view(-1, 1, self.embedding_dim)
        return sentA, sentB

    def c2w_cell(self, indices, h, c):
        input = Variable(torch.LongTensor(indices))
        if torch.cuda.is_available():
            input = input
        input = self.c2w_embedding(input)
        input = input.view(-1, 1, 50)
        out, (state, _) = self.lstm_c2w(input, (h, c))
        output_char = torch.mm(self.df, state[0][0][:].view(-1, 1)) + torch.mm(
            self.db, state[1][0][:].view(-1, 1)) + self.bias.view(-1, 1)
        output_char = output_char.view(1, -1)
        return output_char

    def charCNN_cell(self, indices):
        input = Variable(torch.LongTensor(indices))
        if torch.cuda.is_available():
            input = input
        input = self.char_cnn_embedding(input)
        input = torch.unsqueeze(input, 0)
        out1 = self.charCNN_filter1(input)
        out2 = self.charCNN_filter2(input)
        out3 = self.charCNN_filter3(input)
        out4 = self.charCNN_filter4(input)
        out5 = self.charCNN_filter5(input)
        out6 = self.charCNN_filter6(input)
        out7 = self.charCNN_filter7(input)
        final_output = torch.cat([torch.squeeze(out1), torch.squeeze(out2),
            torch.squeeze(out3), torch.squeeze(out4), torch.squeeze(out5),
            torch.squeeze(out6), torch.squeeze(out7)])
        final_output = final_output.view(1, -1)
        transform_gate = self.transform_gate(final_output)
        final_output = transform_gate * self.char_cnn_mlp(final_output) + (
            1 - transform_gate) * final_output
        return final_output

    def generate_word_indices(self, word):
        if self.task == 'hindi':
            char_gram = list(splitclusters(word))
            indices = []
            if self.character_ngrams == 1:
                indices = [self.dict_char_ngram[char] for char in char_gram]
            elif self.character_ngrams == 2:
                if self.character_ngrams_overlap:
                    if len(char_gram) <= 2:
                        indices = [self.dict_char_ngram[word]]
                    else:
                        for i in range(len(char_gram) - 1):
                            indices.append(self.dict_char_ngram[char_gram[i
                                ] + char_gram[i + 1]])
                elif len(char_gram) <= 2:
                    indices = [self.dict_char_ngram[word]]
                else:
                    for i in range(0, len(char_gram) - 1, 2):
                        indices.append(self.dict_char_ngram[char_gram[i] +
                            char_gram[i + 1]])
                    if len(char_gram) % 2 == 1:
                        indices.append(self.dict_char_ngram[char_gram[len(
                            char_gram) - 1]])
        else:
            indices = []
            if self.character_ngrams == 1:
                for char in word:
                    try:
                        indices.append(self.dict_char_ngram[char])
                    except:
                        continue
            elif self.character_ngrams == 2:
                if self.character_ngrams_overlap:
                    if len(word) <= 2:
                        try:
                            indices = [self.dict_char_ngram[word]]
                        except:
                            indices = [self.dict_char_ngram[' ']]
                    else:
                        for i in range(len(word) - 1):
                            try:
                                indices.append(self.dict_char_ngram[word[i:
                                    i + 2]])
                            except:
                                indices.append(self.dict_char_ngram[' '])
                elif len(word) <= 2:
                    indices = [self.dict_char_ngram[word]]
                else:
                    for i in range(0, len(word) - 1, 2):
                        indices.append(self.dict_char_ngram[word[i:i + 2]])
                    if len(word) % 2 == 1:
                        indices.append(self.dict_char_ngram[word[len(word) -
                            1]])
            elif self.character_ngrams == 3:
                if self.character_ngrams_overlap:
                    if len(word) <= 3:
                        indices = [self.dict_char_ngram[word]]
                    else:
                        for i in range(len(word) - 2):
                            indices.append(self.dict_char_ngram[word[i:i + 3]])
                elif len(word) <= 3:
                    indices = [self.dict_char_ngram[word]]
                else:
                    for i in range(0, len(word) - 2, 3):
                        indices.append(self.dict_char_ngram[word[i:i + 3]])
                    if len(word) % 3 == 1:
                        indices.append(self.dict_char_ngram[word[len(word) -
                            1]])
                    elif len(word) % 3 == 2:
                        indices.append(self.dict_char_ngram[word[len(word) -
                            2:]])
        return indices

    def c2w_or_cnn_layer(self, lsents, rsents):
        h = Variable(torch.zeros(2, 1, self.embedding_dim))
        c = Variable(torch.zeros(2, 1, self.embedding_dim))
        if torch.cuda.is_available():
            h = h
            c = c
        firstFlag = True
        for word in lsents:
            indices = self.generate_word_indices(word)
            if not self.c2w_mode:
                if len(indices) < 20:
                    indices = indices + [(0) for i in range(self.
                        charcnn_max_word_length - len(indices))]
                else:
                    indices = indices[0:20]
            if firstFlag:
                if self.c2w_mode:
                    output = self.c2w_cell([indices], h, c)
                else:
                    output = self.charCNN_cell([indices])
                firstFlag = False
            else:
                if self.c2w_mode:
                    output_new = self.c2w_cell([indices], h, c)
                else:
                    output_new = self.charCNN_cell([indices])
                output = torch.cat((output, output_new), 0)
        sentA = output
        firstFlag = True
        for word in rsents:
            indices = self.generate_word_indices(word)
            if not self.c2w_mode:
                if len(indices) < 20:
                    indices = indices + [(0) for i in range(self.
                        charcnn_max_word_length - len(indices))]
                else:
                    indices = indices[0:20]
            if firstFlag:
                if self.c2w_mode:
                    output = self.c2w_cell([indices], h, c)
                else:
                    output = self.charCNN_cell([indices])
                firstFlag = False
            else:
                if self.c2w_mode:
                    output_new = self.c2w_cell([indices], h, c)
                else:
                    output_new = self.charCNN_cell([indices])
                output = torch.cat((output, output_new), 0)
        sentB = output
        sentA = torch.unsqueeze(sentA, 0).view(-1, 1, self.embedding_dim)
        sentB = torch.unsqueeze(sentB, 0).view(-1, 1, self.embedding_dim)
        return sentA, sentB

    def mix_cell(self, word, output_word, output_char):
        result = None
        extra_loss = 0
        indices_reduce_dim = Variable(torch.LongTensor([(i * 2) for i in
            range(self.embedding_dim)]))
        if torch.cuda.is_available():
            indices_reduce_dim = indices_reduce_dim
        if self.combine_mode == 'concat':
            result = torch.cat((output_word, output_char), 1)
            result = torch.index_select(result, 1, indices_reduce_dim)
        elif self.combine_mode == 'g_0.25':
            result = 0.25 * output_word + 0.75 * output_char
        elif self.combine_mode == 'g_0.50':
            result = 0.5 * output_word + 0.5 * output_char
        elif self.combine_mode == 'g_0.75':
            result = 0.75 * output_word + 0.25 * output_char
        elif self.combine_mode == 'adaptive':
            gate = self.sigmoid(torch.mm(output_word, self.vg) + self.bg)
            gate = gate.expand(1, self.embedding_dim)
            result = (1 - gate) * output_word + gate * output_char
        elif self.combine_mode == 'attention':
            gate = self.sigmoid(torch.mm(self.tanh(torch.mm(output_word,
                self.W1) + torch.mm(output_char, self.W2)), self.W3))
            result = gate * output_word + (1 - gate) * output_char
            if word not in self.oov:
                extra_loss += 1 - F.cosine_similarity(output_word, output_char)
        elif self.combine_mode == 'backoff':
            if word in self.oov:
                result = output_char
            else:
                result = output_word
        return result, extra_loss

    def mix_layer(self, lsents, rsents):
        h = Variable(torch.zeros(2, 1, self.embedding_dim))
        c = Variable(torch.zeros(2, 1, self.embedding_dim))
        if torch.cuda.is_available():
            h = h
            c = c
        firstFlag = True
        extra_loss = 0
        for word in lsents:
            indices = self.generate_word_indices(word)
            if self.c2w_mode:
                output_char = self.c2w_cell([indices], h, c)
            else:
                if len(indices) < 20:
                    indices = indices + [(0) for i in range(self.
                        charcnn_max_word_length - len(indices))]
                else:
                    indices = indices[0:20]
                output_char = self.charCNN_cell([indices])
                if self.task == 'sts':
                    output_char = self.down_sampling_300(output_char)
                else:
                    output_char = self.down_sampling_200(output_char)
            output_word = Variable(torch.Tensor(self.dict[word])).view(1, -1)
            if torch.cuda.is_available():
                output_word = output_word
            if firstFlag:
                output, extra_loss = self.mix_cell(word, output_word,
                    output_char)
                output2 = output_char
                firstFlag = False
            else:
                output_new, extra_loss = self.mix_cell(word, output_word,
                    output_char)
                output_new2 = output_char
                output = torch.cat((output, output_new), 0)
                output2 = torch.cat((output2, output_new2), 0)
        sentA = output
        sentA2 = output2
        firstFlag = True
        for word in rsents:
            indices = self.generate_word_indices(word)
            if self.c2w_mode:
                output_char = self.c2w_cell([indices], h, c)
            else:
                if len(indices) < 20:
                    indices = indices + [(0) for i in range(self.
                        charcnn_max_word_length - len(indices))]
                else:
                    indices = indices[0:20]
                output_char = self.charCNN_cell([indices])
                if self.task == 'sts':
                    output_char = self.down_sampling_300(output_char)
                else:
                    output_char = self.down_sampling_200(output_char)
            output_word = Variable(torch.Tensor(self.dict[word])).view(1, -1)
            if torch.cuda.is_available():
                output_word = output_word
            if firstFlag:
                output, extra_loss = self.mix_cell(word, output_word,
                    output_char)
                output2 = output_char
                firstFlag = False
            else:
                output_new, extra_loss = self.mix_cell(word, output_word,
                    output_char)
                output_new2 = output_char
                output = torch.cat((output, output_new), 0)
                output2 = torch.cat((output2, output_new2), 0)
        sentB = output
        sentB2 = output2
        sentA = torch.unsqueeze(sentA, 0).view(-1, 1, self.embedding_dim)
        sentB = torch.unsqueeze(sentB, 0).view(-1, 1, self.embedding_dim)
        sentA2 = torch.unsqueeze(sentA2, 0).view(-1, 1, self.embedding_dim)
        sentB2 = torch.unsqueeze(sentB2, 0).view(-1, 1, self.embedding_dim)
        return sentA, sentA2, sentB, sentB2, extra_loss

    def forward(self, input_A, input_B, index):
        extra_loss1 = 0
        extra_loss2 = 0
        raw_input_A = input_A
        raw_input_B = input_B
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim))
        else:
            h0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim))
        if self.granularity == 'word':
            input_A, input_B = self.word_layer(input_A, input_B)
        elif self.granularity == 'char':
            input_A, input_B = self.c2w_or_cnn_layer(input_A, input_B)
            if self.lm_mode:
                target_A = []
                for word in raw_input_A:
                    if self.word_freq[word] >= 4:
                        target_A.append(self.tokens.index(word))
                    else:
                        target_A.append(self.tokens.index('oov'))
                target_B = []
                for word in raw_input_B:
                    if self.word_freq[word] >= 4:
                        target_B.append(self.tokens.index(word))
                    else:
                        target_B.append(self.tokens.index('oov'))
                lm_out0, _ = self.lm_lstm(input_A, (h0, c0))
                lm_out1, _ = self.lm_lstm(input_B, (h0, c0))
                extra_loss2 = self.language_model(lm_out0, lm_out1,
                    target_A, target_B)
        elif self.granularity == 'mix':
            input_A, input_A2, input_B, input_B2, extra_loss1 = self.mix_layer(
                input_A, input_B)
            if self.lm_mode:
                target_A = []
                for word in raw_input_A:
                    if self.word_freq[word] >= 4:
                        target_A.append(self.tokens.index(word))
                    else:
                        target_A.append(self.tokens.index('oov'))
                target_B = []
                for word in raw_input_B:
                    if self.word_freq[word] >= 4:
                        target_B.append(self.tokens.index(word))
                    else:
                        target_B.append(self.tokens.index('oov'))
                lm_out0, _ = self.lm_lstm(input_A2, (h0, c0))
                lm_out1, _ = self.lm_lstm(input_B2, (h0, c0))
                extra_loss2 = self.language_model(lm_out0, lm_out1,
                    target_A, target_B)
        out0, (state0, _) = self.lstm(input_A, (h0, c0))
        out1, (state1, _) = self.lstm(input_B, (h0, c0))
        simCube, _ = self.pairwise_word_interaction(out0, out1, target_A=
            None, target_B=None)
        focusCube = self.similarity_focus(simCube)
        if self.deep_CNN:
            output = self.deep_cnn(focusCube)
        else:
            output = self.mlp(focusCube)
        output = output.view(1, 2)
        return output, extra_loss1 + extra_loss2


class StackBiLSTMMaxout(nn.Module):

    def __init__(self, h_size, v_size=10, d=300, mlp_d=1600, dropout_r=0.1,
        max_l=60, num_class=3):
        super(StackBiLSTMMaxout, self).__init__()
        self.Embd = nn.Embedding(v_size, d)
        self.lstm = nn.LSTM(input_size=d, hidden_size=h_size[0], num_layers
            =1, bidirectional=True)
        self.lstm_1 = nn.LSTM(input_size=d + h_size[0] * 2, hidden_size=
            h_size[1], num_layers=1, bidirectional=True)
        self.lstm_2 = nn.LSTM(input_size=d + (h_size[0] + h_size[1]) * 2,
            hidden_size=h_size[2], num_layers=1, bidirectional=True)
        self.max_l = max_l
        self.h_size = h_size
        self.mlp_1 = nn.Linear(h_size[2] * 2 * 4, mlp_d)
        self.mlp_2 = nn.Linear(mlp_d, mlp_d)
        self.sm = nn.Linear(mlp_d, num_class)
        self.classifier = nn.Sequential(*[self.mlp_1, nn.ReLU(), nn.Dropout
            (dropout_r), self.mlp_2, nn.ReLU(), nn.Dropout(dropout_r), self.sm]
            )

    def display(self):
        for param in self.parameters():
            None

    def forward(self, s1, l1, s2, l2):
        if self.max_l:
            l1 = l1.clamp(max=self.max_l)
            l2 = l2.clamp(max=self.max_l)
            if s1.size(0) > self.max_l:
                s1 = s1[:self.max_l, :]
            if s2.size(0) > self.max_l:
                s2 = s2[:self.max_l, :]
        p_s1 = self.Embd(s1)
        p_s2 = self.Embd(s2)
        s1_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s1, l1)
        s2_layer1_out = torch_util.auto_rnn_bilstm(self.lstm, p_s2, l2)
        len1 = s1_layer1_out.size(0)
        len2 = s2_layer1_out.size(0)
        p_s1 = p_s1[:len1, :, :]
        p_s2 = p_s2[:len2, :, :]
        s1_layer2_in = torch.cat([p_s1, s1_layer1_out], dim=2)
        s2_layer2_in = torch.cat([p_s2, s2_layer1_out], dim=2)
        s1_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1,
            s1_layer2_in, l1)
        s2_layer2_out = torch_util.auto_rnn_bilstm(self.lstm_1,
            s2_layer2_in, l2)
        s1_layer3_in = torch.cat([p_s1, s1_layer1_out, s1_layer2_out], dim=2)
        s2_layer3_in = torch.cat([p_s2, s2_layer1_out, s2_layer2_out], dim=2)
        s1_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2,
            s1_layer3_in, l1)
        s2_layer3_out = torch_util.auto_rnn_bilstm(self.lstm_2,
            s2_layer3_in, l2)
        s1_layer3_maxout = torch_util.max_along_time(s1_layer3_out, l1)
        s2_layer3_maxout = torch_util.max_along_time(s2_layer3_out, l2)
        features = torch.cat([s1_layer3_maxout, s2_layer3_maxout, torch.abs
            (s1_layer3_maxout - s2_layer3_maxout), s1_layer3_maxout *
            s2_layer3_maxout], dim=1)
        out = self.classifier(features)
        return out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lanwuwei_SPM_toolkit(_paritybench_base):
    pass
    def test_000(self):
        self._check(BinaryTreeCell(*[], **{'cuda': 4, 'in_dim': 4, 'mem_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(BinaryTreeComposer(*[], **{'cuda': 4, 'in_dim': 4, 'mem_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(BinaryTreeLSTM(*[], **{'cuda': 4, 'in_dim': 4, 'mem_dim': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_003(self):
        self._check(BinaryTreeLeafModule(*[], **{'cuda': 4, 'in_dim': 4, 'mem_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(LSTM(*[], **{'cuda': 4, 'in_dim': 4, 'mem_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(LSTM_Cell(*[], **{'cuda': 4, 'in_dim': 4, 'mem_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

