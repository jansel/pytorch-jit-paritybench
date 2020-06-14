import sys
_module = sys.modules[__name__]
del sys
layers = _module
model = _module
rnn_reader = _module
utils = _module
prepro = _module
train = _module
dataloader = _module
modules = _module
train_classifier = _module
train_enwik8 = _module
train_lm = _module
compare_cpu_speed_sru_gru = _module
compare_gpu_speed_sru_gru = _module
test_backward_with_transpose = _module
test_impl = _module
test_mm = _module
test_multigpu = _module
test_sru = _module
setup = _module
sru = _module
cuda_functional = _module
sru_functional = _module
version = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import torch.optim as optim


import numpy as np


import logging


import time


import random


import math


import copy


import warnings


class StackedBRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0,
        dropout_output=False, rnn_type=nn.LSTM, concat_layers=False,
        padding=False):
        super(StackedBRNN, self).__init__()
        self.padding = padding
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(sru.SRUCell(input_size, hidden_size, dropout=
                dropout_rate, rnn_dropout=dropout_rate, use_tanh=0,
                bidirectional=True))

    def forward(self, x, x_mask):
        """Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        """
        if x_mask.data.sum() == 0:
            return self._forward_unpadded(x, x_mask)
        return self._forward_unpadded(x, x_mask)

    def _forward_unpadded(self, x, x_mask):
        """Faster encoding that ignores any padding."""
        x = x.transpose(0, 1)
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.
                training)
        return output.contiguous()

    def _forward_padded(self, x, x_mask):
        """Slower (significantly), but more precise,
        encoding that handles padding."""
        lengths = x_mask.data.eq(0).long().sum(1)
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        lengths = list(lengths[idx_sort])
        idx_sort = Variable(idx_sort)
        idx_unsort = Variable(idx_unsort)
        x = x.index_select(0, idx_sort)
        x = x.transpose(0, 1)
        rnn_input = nn.utils.rnn.pack_padded_sequence(x, lengths)
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                dropout_input = F.dropout(rnn_input.data, p=self.
                    dropout_rate, training=self.training)
                rnn_input = nn.utils.rnn.PackedSequence(dropout_input,
                    rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])
        for i, o in enumerate(outputs[1:], 1):
            outputs[i] = nn.utils.rnn.pad_packed_sequence(o)[0]
        if self.concat_layers:
            output = torch.cat(outputs[1:], 2)
        else:
            output = outputs[-1]
        output = output.transpose(0, 1)
        output = output.index_select(0, idx_unsort)
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output, p=self.dropout_rate, training=self.
                training)
        return output


class SeqAttnMatch(nn.Module):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            self.linear = nn.Linear(input_size, input_size)
        else:
            self.linear = None

    def forward(self, x, y, y_mask):
        """Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj = F.relu(x_proj)
            y_proj = self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))
        alpha_flat = F.softmax(scores.view(-1, y.size(1)))
        alpha = alpha_flat.view(-1, x.size(1), y.size(1))
        matched_seq = alpha.bmm(y)
        return matched_seq


class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.training:
            alpha = F.log_softmax(xWy)
        else:
            alpha = F.softmax(xWy)
        return alpha


class LinearSeqAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        x_flat = x.view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores)
        return alpha


def normalize_emb_(data):
    print(data.size(), data[:10].norm(2, 1))
    norms = data.norm(2, 1) + 1e-08
    if norms.dim() == 1:
        norms = norms.unsqueeze(1)
    data.div_(norms.expand_as(data))
    print(data.size(), data[:10].norm(2, 1))


class RnnDocReader(nn.Module):
    """Network for the Document Reader module of DrQA."""
    RNN_TYPES = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}

    def __init__(self, opt, padding_idx=0, embedding=None, normalize_emb=False
        ):
        super(RnnDocReader, self).__init__()
        self.opt = opt
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = nn.Embedding(embedding.size(0), embedding.size
                (1), padding_idx=padding_idx)
            if normalize_emb:
                normalize_emb_(embedding)
            self.embedding.weight.data = embedding
            if opt['fix_embeddings']:
                assert opt['tune_partial'] == 0
                for p in self.embedding.parameters():
                    p.requires_grad = False
            elif opt['tune_partial'] > 0:
                assert opt['tune_partial'] + 2 < embedding.size(0)
                fixed_embedding = embedding[opt['tune_partial'] + 2:]
                self.register_buffer('fixed_embedding', fixed_embedding)
                self.fixed_embedding = fixed_embedding
        else:
            self.embedding = nn.Embedding(opt['vocab_size'], opt[
                'embedding_dim'], padding_idx=padding_idx)
        if opt['pos']:
            self.pos_embedding = nn.Embedding(opt['pos_size'], opt['pos_dim'])
            if normalize_emb:
                normalize_emb_(self.pos_embedding.weight.data)
        if opt['ner']:
            self.ner_embedding = nn.Embedding(opt['ner_size'], opt['ner_dim'])
            if normalize_emb:
                normalize_emb_(self.ner_embedding.weight.data)
        if opt['use_qemb']:
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'])
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_dim']
        if opt['ner']:
            doc_input_size += opt['ner_dim']
        self.doc_rnn = layers.StackedBRNN(input_size=doc_input_size,
            hidden_size=opt['hidden_size'], num_layers=opt['doc_layers'],
            dropout_rate=opt['dropout_rnn'], dropout_output=opt[
            'dropout_rnn_output'], concat_layers=opt['concat_rnn_layers'],
            rnn_type=self.RNN_TYPES[opt['rnn_type']], padding=opt[
            'rnn_padding'])
        self.question_rnn = layers.StackedBRNN(input_size=opt[
            'embedding_dim'], hidden_size=opt['hidden_size'], num_layers=
            opt['question_layers'], dropout_rate=opt['dropout_rnn'],
            dropout_output=opt['dropout_rnn_output'], concat_layers=opt[
            'concat_rnn_layers'], rnn_type=self.RNN_TYPES[opt['rnn_type']],
            padding=opt['rnn_padding'])
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']
        if opt['question_merge'] not in ['avg', 'self_attn']:
            raise NotImplementedError('question_merge = %s' % opt[
                'question_merge'])
        if opt['question_merge'] == 'self_attn':
            self.self_attn = layers.LinearSeqAttn(question_hidden_size)
        self.start_attn = layers.BilinearSeqAttn(doc_hidden_size,
            question_hidden_size)
        self.end_attn = layers.BilinearSeqAttn(doc_hidden_size,
            question_hidden_size)

    def forward(self, x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)
        if self.opt['dropout_emb'] > 0:
            x1_emb = nn.functional.dropout(x1_emb, p=self.opt['dropout_emb'
                ], training=self.training)
            x2_emb = nn.functional.dropout(x2_emb, p=self.opt['dropout_emb'
                ], training=self.training)
        drnn_input_list = [x1_emb, x1_f]
        if self.opt['use_qemb']:
            x2_weighted_emb = self.qemb_match(x1_emb, x2_emb, x2_mask)
            drnn_input_list.append(x2_weighted_emb)
        if self.opt['pos']:
            x1_pos_emb = self.pos_embedding(x1_pos)
            if self.opt['dropout_emb'] > 0:
                x1_pos_emb = nn.functional.dropout(x1_pos_emb, p=self.opt[
                    'dropout_emb'], training=self.training)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = self.ner_embedding(x1_ner)
            if self.opt['dropout_emb'] > 0:
                x1_ner_emb = nn.functional.dropout(x1_ner_emb, p=self.opt[
                    'dropout_emb'], training=self.training)
            drnn_input_list.append(x1_ner_emb)
        drnn_input = torch.cat(drnn_input_list, 2)
        doc_hiddens = self.doc_rnn(drnn_input, x1_mask)
        question_hiddens = self.question_rnn(x2_emb, x2_mask)
        if self.opt['question_merge'] == 'avg':
            q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
        elif self.opt['question_merge'] == 'self_attn':
            q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights
            )
        start_scores = self.start_attn(doc_hiddens, question_hidden, x1_mask)
        end_scores = self.end_attn(doc_hiddens, question_hidden, x1_mask)
        return start_scores, end_scores


class CNN_Text(nn.Module):

    def __init__(self, n_in, widths=[3, 4, 5], filters=100):
        super(CNN_Text, self).__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths]
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x


def deep_iter(x):
    if isinstance(x, list) or isinstance(x, tuple):
        for u in x:
            for v in deep_iter(u):
                yield v
    else:
        yield x


class EmbeddingLayer(nn.Module):

    def __init__(self, n_d, words, embs=None, fix_emb=True, oov='<oov>',
        pad='<pad>', normalize=True):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        if embs is not None:
            embwords, embvecs = embs
            for word in embwords:
                assert word not in word2id, 'Duplicate words in pre-trained embeddings'
                word2id[word] = len(word2id)
            sys.stdout.write('{} pre-trained word embeddings loaded.\n'.
                format(len(word2id)))
            if n_d != len(embvecs[0]):
                sys.stdout.write(
                    """[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.
"""
                    .format(n_d, len(embvecs[0]), len(embvecs[0])))
                n_d = len(embvecs[0])
        for w in deep_iter(words):
            if w not in word2id:
                word2id[w] = len(word2id)
        if oov not in word2id:
            word2id[oov] = len(word2id)
        if pad not in word2id:
            word2id[pad] = len(word2id)
        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.oovid = word2id[oov]
        self.padid = word2id[pad]
        self.embedding = nn.Embedding(self.n_V, n_d)
        self.embedding.weight.data.uniform_(-0.25, 0.25)
        if embs is not None:
            weight = self.embedding.weight
            weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
            sys.stdout.write('embedding shape: {}\n'.format(weight.size()))
        if normalize:
            weight = self.embedding.weight
            norms = weight.data.norm(2, 1)
            if norms.dim() == 1:
                norms = norms.unsqueeze(1)
            weight.data.div_(norms.expand_as(weight.data))
        if fix_emb:
            self.embedding.weight.requires_grad = False

    def forward(self, input):
        return self.embedding(input)


class Model(nn.Module):

    def __init__(self, args, emb_layer, nclasses=2):
        super(Model, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.dropout)
        self.emb_layer = emb_layer
        if args.cnn:
            self.encoder = modules.CNN_Text(emb_layer.n_d, widths=[3, 4, 5])
            d_out = 300
        elif args.lstm:
            self.encoder = nn.LSTM(emb_layer.n_d, args.d, args.depth,
                dropout=args.dropout)
            d_out = args.d
        else:
            self.encoder = SRU(emb_layer.n_d, args.d, args.depth, dropout=
                args.dropout)
            d_out = args.d
        self.out = nn.Linear(d_out, nclasses)

    def forward(self, input):
        if self.args.cnn:
            input = input.t()
        emb = self.emb_layer(input)
        emb = self.drop(emb)
        if self.args.cnn:
            output = self.encoder(emb)
        else:
            output, hidden = self.encoder(emb)
            output = output[-1]
        output = self.drop(output)
        return self.out(output)


class Model(nn.Module):

    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args
        if args.n_e:
            self.n_e = args.n_e
        else:
            self.n_e = len(words) if len(words) < args.n_d else args.n_d
        self.n_d = args.n_d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = nn.Embedding(len(words), self.n_e)
        self.n_V = len(words)
        if args.lstm:
            self.rnn = nn.LSTM(self.n_e, self.n_d, self.depth, dropout=args
                .dropout)
        else:
            self.rnn = sru.SRU(self.n_e, self.n_d, self.depth, dropout=args
                .dropout, n_proj=args.n_proj, highway_bias=args.bias,
                layer_norm=args.layer_norm)
        self.output_layer = nn.Linear(self.n_d, self.n_V)
        self.init_weights()

    def init_weights(self, val_range=None):
        params = list(self.embedding_layer.parameters()) + list(self.
            output_layer.parameters()) + (list(self.rnn.parameters()) if
            self.args.lstm else [])
        for p in params:
            if p.dim() > 1:
                val = val_range or (3.0 / p.size(0)) ** 0.5
                p.data.uniform_(-val, val)
            else:
                p.data.zero_()

    def forward(self, x, hidden):
        emb = self.drop(self.embedding_layer(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.depth, batch_size, self.n_d).zero_())
        if self.args.lstm:
            return zeros, zeros
        else:
            return zeros


class EmbeddingLayer(nn.Module):

    def __init__(self, n_d, words, fix_emb=False):
        super(EmbeddingLayer, self).__init__()
        word2id = {}
        for w in words:
            if w not in word2id:
                word2id[w] = len(word2id)
        self.word2id = word2id
        self.n_V, self.n_d = len(word2id), n_d
        self.embedding = nn.Embedding(self.n_V, n_d)

    def forward(self, x):
        return self.embedding(x)

    def map_to_ids(self, text):
        return np.asarray([self.word2id[x] for x in text], dtype='int64')


class Model(nn.Module):

    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args
        self.n_d = args.d
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.embedding_layer = EmbeddingLayer(self.n_d, words)
        self.n_V = self.embedding_layer.n_V
        if args.lstm:
            self.rnn = nn.LSTM(self.n_d, self.n_d, self.depth, dropout=args
                .rnn_dropout)
        else:
            self.rnn = sru.SRU(self.n_d, self.n_d, self.depth, dropout=args
                .rnn_dropout, rnn_dropout=args.rnn_dropout, use_tanh=0,
                rescale=False, v1=True, highway_bias=args.bias)
        self.output_layer = nn.Linear(self.n_d, self.n_V)
        self.output_layer.weight = self.embedding_layer.embedding.weight
        self.init_weights()

    def init_weights(self):
        val_range = (3.0 / self.n_d) ** 0.5
        params = list(self.embedding_layer.parameters()) + list(self.
            output_layer.parameters()) + (list(self.rnn.parameters()) if
            self.args.lstm else [])
        for p in params:
            if p.dim() > 1:
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()

    def forward(self, x, hidden):
        emb = self.drop(self.embedding_layer(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        zeros = Variable(weight.new(self.depth, batch_size, self.n_d).zero_())
        if self.args.lstm:
            return zeros, zeros
        else:
            return zeros

    def print_pnorm(self):
        norms = ['{:.0f}'.format(x.norm().item()) for x in self.parameters()]
        sys.stdout.write('\tp_norm: {}\n'.format(norms))


class Model(nn.Module):

    def __init__(self, rnn):
        super(Model, self).__init__()
        self.rnn = rnn

    def forward(self, x):
        out, state = self.rnn(x)
        return out[-1:]


def _lazy_load_cpu_kernel():
    global SRU_CPU_kernel
    if SRU_CPU_kernel is not None:
        return SRU_CPU_kernel
    try:
        from torch.utils.cpp_extension import load
        cpu_source = os.path.join(os.path.dirname(__file__), 'sru_cpu_impl.cpp'
            )
        SRU_CPU_kernel = load(name='sru_cpu_impl', sources=[cpu_source],
            extra_cflags=['-O3'], verbose=False)
    except:
        SRU_CPU_kernel = False
    return SRU_CPU_kernel


class SRU_Compute_CPU:
    """CPU version of the core SRU computation.

    Has the same interface as SRU_Compute_GPU() but is a regular Python function
    instead of a torch.autograd.Function because we don't implement backward()
    explicitly.
    """

    @staticmethod
    def apply(u, x, weight_c, bias, init, activation_type, d, bidirectional,
        has_skip_term, scale_x, mask_c=None, mask_pad=None):
        """
        An SRU is a recurrent neural network cell comprised of 5 equations, described
        in "Simple Recurrent Units for Highly Parallelizable Recurrence."

        The first 3 of these equations each require a matrix-multiply component,
        i.e. the input vector x_t dotted with a weight matrix W_i, where i is in
        {0, 1, 2}.

        As each weight matrix W is dotted with the same input x_t, we can fuse these
        computations into a single matrix-multiply, i.e. `x_t <dot> stack([W_0, W_1, W_2])`.
        We call the result of this computation `U`.

        sru_compute_cpu() accepts 'u' and 'x' (along with a tensor of biases,
        an initial memory cell `c0`, and an optional dropout mask) and computes
        equations (3) - (7). It returns a tensor containing all `t` hidden states
        (where `t` is the number of elements in our input sequence) and the final
        memory cell `c_T`.
        """
        bidir = 2 if bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        k = u.size(-1) // d // bidir
        is_custom = len(weight_c.size()) > 1
        sru_cpu_impl = _lazy_load_cpu_kernel()
        if sru_cpu_impl is not None and sru_cpu_impl != False:
            if not torch.is_grad_enabled():
                assert mask_c is None
                cpu_forward = (sru_cpu_impl.cpu_bi_forward if bidirectional
                     else sru_cpu_impl.cpu_forward)
                mask_pad_ = torch.FloatTensor(
                    ) if mask_pad is None else mask_pad.float()
                return cpu_forward(u.contiguous(), x.contiguous(), weight_c
                    .contiguous(), bias, init, mask_pad_, length, batch, d,
                    k, activation_type, has_skip_term, scale_x.item() if 
                    scale_x is not None else 1.0, is_custom)
            else:
                warnings.warn(
                    'Running SRU on CPU with grad_enabled=True. Are you sure?')
        else:
            warnings.warn(
                'C++ kernel for SRU CPU inference was not loaded. Use Python version instead.'
                )
        mask_pad_ = mask_pad.view(length, batch, 1).float(
            ) if mask_pad is not None else mask_pad
        u = u.contiguous().view(length, batch, bidir, d, k)
        if is_custom:
            weight_c = weight_c.view(length, batch, bidir, d, 2)
            forget_wc = weight_c[..., 0]
            reset_wc = weight_c[..., 1]
        else:
            forget_wc, reset_wc = weight_c.view(2, bidir, d)
        forget_bias, reset_bias = bias.view(2, bidir, d)
        if not has_skip_term:
            x_prime = None
        elif k == 3:
            x_prime = x.view(length, batch, bidir, d)
            x_prime = x_prime * scale_x if scale_x is not None else x_prime
        else:
            x_prime = u[..., 3]
        h = x.new_zeros(length, batch, bidir, d)
        if init is None:
            c_init = x.new_zeros(size=(batch, bidir, d))
        else:
            c_init = init.view(batch, bidir, d)
        c_final = []
        for di in range(bidir):
            if di == 0:
                time_seq = range(length)
            else:
                time_seq = range(length - 1, -1, -1)
            mask_c_ = 1 if mask_c is None else mask_c.view(batch, bidir, d)[:,
                (di), :]
            c_prev = c_init[:, (di), :]
            fb, rb = forget_bias[di], reset_bias[di]
            if is_custom:
                fw = forget_wc[:, :, (di), :].chunk(length)
                rw = reset_wc[:, :, (di), :].chunk(length)
            else:
                fw = forget_wc[di].expand(batch, d)
                rw = reset_wc[di].expand(batch, d)
            u0 = u[:, :, (di), :, (0)].chunk(length)
            u1 = (u[:, :, (di), :, (1)] + fb).chunk(length)
            u2 = (u[:, :, (di), :, (2)] + rb).chunk(length)
            if x_prime is not None:
                xp = x_prime[:, :, (di), :].chunk(length)
            for t in time_seq:
                if is_custom:
                    forget_t = (u1[t] + c_prev * fw[t]).sigmoid()
                    reset_t = (u2[t] + c_prev * rw[t]).sigmoid()
                else:
                    forget_t = (u1[t] + c_prev * fw).sigmoid()
                    reset_t = (u2[t] + c_prev * rw).sigmoid()
                c_t = u0[t] + (c_prev - u0[t]) * forget_t
                if mask_pad_ is not None:
                    c_t = c_t * (1 - mask_pad_[t]) + c_prev * mask_pad_[t]
                c_prev = c_t
                if activation_type == 0:
                    g_c_t = c_t
                elif activation_type == 1:
                    g_c_t = c_t.tanh()
                else:
                    raise ValueError('Activation type must be 0 or 1, not {}'
                        .format(activation_type))
                if x_prime is not None:
                    h_t = xp[t] + (g_c_t - xp[t]) * mask_c_ * reset_t
                else:
                    h_t = g_c_t * mask_c_ * reset_t
                if mask_pad_ is not None:
                    h_t = h_t * (1 - mask_pad_[t])
                h[(t), :, (di), :] = h_t
            c_final.append(c_t.view(batch, d))
        return h.view(length, batch, -1), torch.stack(c_final, dim=1).view(
            batch, -1)


def _lazy_load_cuda_kernel():
    try:
        from .cuda_functional import SRU_Compute_GPU
    except:
        from cuda_functional import SRU_Compute_GPU
    return SRU_Compute_GPU


class SRUCell(nn.Module):
    """
    An SRU cell, i.e. a single recurrent neural network cell,
    as per `LSTMCell`, `GRUCell` and `RNNCell` in PyTorch.

    Args:
        input_size (int) : the number of dimensions in a single
            input sequence element. For example, if the input sequence
            is a sequence of word embeddings, `input_size` is the
            dimensionality of a single word embedding, e.g. 300.
        hidden_size (int) : the dimensionality of the hidden state
            of this cell.
        dropout (float) : a number between 0.0 and 1.0. The amount of dropout
            applied to `g(c_t)` internally in this cell.
        rnn_dropout (float) : the amount of dropout applied to the input of
            this cell.
        use_tanh (bool) : use tanh activation
        is_input_normalized (bool) : whether the input is normalized (e.g. batch norm / layer norm)
        bidirectional (bool) : whether or not to employ a bidirectional cell.
    """

    def __init__(self, input_size, hidden_size, dropout=0, rnn_dropout=0,
        bidirectional=False, n_proj=0, use_tanh=0, highway_bias=0,
        has_skip_term=True, layer_norm=False, rescale=True, v1=False,
        custom_m=None):
        super(SRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.has_skip_term = has_skip_term
        self.highway_bias = highway_bias
        self.v1 = v1
        self.rescale = rescale
        self.activation_type = 0
        self.activation = 'none'
        self.custom_m = custom_m
        if use_tanh:
            self.activation_type = 1
            self.activation = 'tanh'
        self.projection_size = 0
        if (n_proj > 0 and n_proj < self.input_size and n_proj < self.
            output_size):
            self.projection_size = n_proj
        self.num_matrices = 3
        if has_skip_term and self.input_size != self.output_size:
            self.num_matrices = 4
        if self.custom_m is None:
            if self.projection_size == 0:
                self.weight = nn.Parameter(torch.Tensor(input_size, self.
                    output_size * self.num_matrices))
            else:
                self.weight_proj = nn.Parameter(torch.Tensor(input_size,
                    self.projection_size))
                self.weight = nn.Parameter(torch.Tensor(self.
                    projection_size, self.output_size * self.num_matrices))
        self.weight_c = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.output_size))
        self.register_buffer('scale_x', torch.FloatTensor([0]))
        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.input_size)
        else:
            self.layer_norm = None
        self.reset_parameters()

    def reset_parameters(self):
        """
        Properly initialize the weights of SRU, following the same recipe as:
            Xavier init:  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
            Kaiming init: https://arxiv.org/abs/1502.01852

        """
        self.bias.data.zero_()
        bias_val, output_size = self.highway_bias, self.output_size
        self.bias.data[output_size:].zero_().add_(bias_val)
        self.scale_x.data[0] = 1
        if self.rescale and self.has_skip_term:
            scale_val = (1 + math.exp(bias_val) * 2) ** 0.5
            self.scale_x.data[0] = scale_val
        if self.custom_m is None:
            d = self.weight.size(0)
            val_range = (3.0 / d) ** 0.5
            self.weight.data.uniform_(-val_range, val_range)
            if self.projection_size > 0:
                val_range = (3.0 / self.weight_proj.size(0)) ** 0.5
                self.weight_proj.data.uniform_(-val_range, val_range)
            w = self.weight.data.view(d, -1, self.hidden_size, self.
                num_matrices)
            if self.dropout > 0:
                w[:, :, :, (0)].mul_((1 - self.dropout) ** 0.5)
            if self.rnn_dropout > 0:
                w.mul_((1 - self.rnn_dropout) ** 0.5)
            if self.layer_norm:
                w.mul_(0.1)
            if self.rescale and self.has_skip_term and self.num_matrices == 4:
                scale_val = (1 + math.exp(bias_val) * 2) ** 0.5
                w[:, :, :, (3)].mul_(scale_val)
        elif hasattr(self.custom_m, 'reset_parameters'):
            self.custom_m.reset_parameters()
        else:
            warnings.warn(
                'Unable to reset parameters for custom module. reset_parameters() method not found for custom module.'
                )
        if not self.v1:
            self.weight_c.data.uniform_(-3.0 ** 0.5, 3.0 ** 0.5)
            if self.custom_m is None:
                w[:, :, :, (1)].mul_(0.5 ** 0.5)
                w[:, :, :, (2)].mul_(0.5 ** 0.5)
            self.weight_c.data.mul_(0.5 ** 0.5)
        else:
            self.weight_c.data.zero_()
            self.weight_c.requires_grad = False

    def forward(self, input, c0=None, mask_pad=None, **kwargs):
        """
        This method computes `U`. In addition, it computes the remaining components
        in `SRU_Compute_GPU` or `SRU_Compute_CPU` and return the results.
        """
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('Input must be 2 or 3 dimensional')
        input_size, hidden_size = self.input_size, self.hidden_size
        batch_size = input.size(-2)
        if c0 is None:
            c0 = input.new_zeros(batch_size, self.output_size)
        residual = input
        if self.layer_norm:
            input = self.layer_norm(input)
        if self.training and self.rnn_dropout > 0:
            mask = self.get_dropout_mask_((batch_size, input.size(-1)),
                self.rnn_dropout)
            input = input * mask.expand_as(input)
        if self.custom_m is None:
            U = self.compute_U(input)
            V = self.weight_c
        else:
            ret = self.custom_m(input, c0=c0, mask_pad=mask_pad, **kwargs)
            if isinstance(ret, tuple) or isinstance(ret, list):
                if len(ret) > 2:
                    raise Exception(
                        'Custom module must return 1 or 2 tensors but got {}.'
                        .format(len(ret)))
                U, V = ret[0], ret[1] + self.weight_c
            else:
                U, V = ret, self.weight_c
            if U.size(-1) != self.output_size * self.num_matrices:
                raise ValueError(
                    'U must have a last dimension of {} but got {}.'.format
                    (self.output_size * self.num_matrices, U.size(-1)))
            if V.size(-1) != self.output_size * 2:
                raise ValueError(
                    'V must have a last dimension of {} but got {}.'.format
                    (self.output_size * 2, V.size(-1)))
        scale_val = self.scale_x if self.rescale else None
        if self.training and self.dropout > 0:
            mask_c = self.get_dropout_mask_((batch_size, self.output_size),
                self.dropout)
        else:
            mask_c = None
        SRU_Compute = _lazy_load_cuda_kernel(
            ) if input.is_cuda else SRU_Compute_CPU
        h, c = SRU_Compute.apply(U, residual, V, self.bias, c0, self.
            activation_type, hidden_size, self.bidirectional, self.
            has_skip_term, scale_val, mask_c, mask_pad)
        return h, c

    def compute_U(self, input):
        """
        SRU performs grouped matrix multiplication to transform
        the input (length, batch_size, input_size) into a tensor
        U of size (length * batch_size, output_size * num_matrices)
        """
        x = input if input.dim() == 2 else input.contiguous().view(-1, self
            .input_size)
        if self.projection_size > 0:
            x_projected = x.mm(self.weight_proj)
            U = x_projected.mm(self.weight)
        else:
            U = x.mm(self.weight)
        return U

    def get_dropout_mask_(self, size, p):
        """
        Composes the dropout mask for the `SRUCell`.
        """
        b = self.bias.data
        return b.new(*size).bernoulli_(1 - p).div_(1 - p)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.projection_size > 0:
            s += ', projection_size={projection_size}'
        if self.dropout > 0:
            s += ', dropout={dropout}'
        if self.rnn_dropout > 0:
            s += ', rnn_dropout={rnn_dropout}'
        if self.bidirectional:
            s += ', bidirectional={bidirectional}'
        if self.highway_bias != 0:
            s += ', highway_bias={highway_bias}'
        if self.activation_type != 0:
            s += ', activation={activation}'
        if self.v1:
            s += ', v1={v1}'
        s += ', rescale={rescale}'
        if not self.has_skip_term:
            s += ', has_skip_term={has_skip_term}'
        if self.layer_norm:
            s += ', layer_norm=True'
        if self.custom_m is not None:
            s += ',\n  custom_m=' + str(self.custom_m)
        return s.format(**self.__dict__)

    def __repr__(self):
        s = self.extra_repr()
        if len(s.split('\n')) == 1:
            return '{}({})'.format(self.__class__.__name__, s)
        else:
            return '{}({}\n)'.format(self.__class__.__name__, s)


class SRU(nn.Module):
    """
    PyTorch SRU model. In effect, simply wraps an arbitrary number of contiguous `SRUCell`s, and
    returns the matrix and hidden states , as well as final memory cell (`c_t`), from the last of
    these `SRUCell`s.

    Args:
        input_size (int) : the number of dimensions in a single input sequence element. For example,
            if the input sequence is a sequence of word embeddings, `input_size` is the dimensionality
            of a single word embedding, e.g. 300.
        hidden_size (int) : the dimensionality of the hidden state of the SRU cell.
        num_layers (int) : number of `SRUCell`s to use in the model.
        dropout (float) : a number between 0.0 and 1.0. The amount of dropout applied to `g(c_t)`
            internally in each `SRUCell`.
        rnn_dropout (float) : the amount of dropout applied to the input of each `SRUCell`.
        use_tanh (bool) : use tanh activation
        layer_norm (bool) : whether or not to use layer normalization on the output of each layer
        bidirectional (bool) : whether or not to use bidirectional `SRUCell`s.
        highway_bias (float) : initial bias of the highway gate, typicially <= 0
        nn_rnn_compatible_return (bool) : set to True to change the layout of returned state to
            match that of pytorch nn.RNN, ie (num_layers * num_directions, batch, hidden_size)
            (this will be slower, but can make SRU a drop-in replacement for nn.RNN and nn.GRU)
        custom_m (nn.Module or List[nn.Module]) : use a custom module to compute the U matrix (and V
            matrix) given the input. The module must take as input a tensor of shape (seq_len,
            batch_size, hidden_size).
            It returns a tensor U of shape (seq_len, batch_size, hidden_size * 3), or one optional
            tensor V of shape (seq_len, batch_size, hidden_size * 2).
    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0,
        rnn_dropout=0, bidirectional=False, projection_size=0, use_tanh=
        False, layer_norm=False, highway_bias=0, has_skip_term=True,
        rescale=False, v1=False, nn_rnn_compatible_return=False, custom_m=
        None, proj_input_to_hidden_first=False):
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.projection_size = projection_size
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.use_layer_norm = layer_norm
        self.has_skip_term = has_skip_term
        self.num_directions = 2 if bidirectional else 1
        self.nn_rnn_compatible_return = nn_rnn_compatible_return
        if proj_input_to_hidden_first and input_size != self.output_size:
            first_layer_input_size = self.output_size
            self.input_to_hidden = nn.Linear(input_size, self.output_size,
                bias=False)
        else:
            first_layer_input_size = input_size
            self.input_to_hidden = None
        for i in range(num_layers):
            custom_m_i = None
            if custom_m is not None:
                custom_m_i = custom_m[i] if isinstance(custom_m, list
                    ) else copy.deepcopy(custom_m)
            l = SRUCell(first_layer_input_size if i == 0 else self.
                output_size, self.hidden_size, dropout=dropout if i + 1 !=
                num_layers else 0, rnn_dropout=rnn_dropout, bidirectional=
                bidirectional, n_proj=projection_size, use_tanh=use_tanh,
                layer_norm=layer_norm, highway_bias=highway_bias,
                has_skip_term=has_skip_term, rescale=rescale, v1=v1,
                custom_m=custom_m_i)
            self.rnn_lst.append(l)

    def forward(self, input, c0=None, mask_pad=None):
        """
        Feeds `input` forward through `num_layers` `SRUCell`s, where `num_layers`
        is a parameter on the constructor of this class.

        parameters:
        - input (FloatTensor): (sequence_length, batch_size, input_size)
        - c0 (FloatTensor): (num_layers, batch_size, hidden_size * num_directions)
        - mask_pad (ByteTensor): (sequence_length, batch_size): set to 1 to ignore the value at that position

        input can be packed, which will lead to worse execution speed, but is compatible with many usages
        of nn.RNN.

        Return:
        - prevx: output: FloatTensor, (sequence_length, batch_size, num_directions * hidden_size)
        - lstc_stack: state:
            (FloatTensor): (num_layers, batch_size, num_directions * hidden_size) if not nn_rnn_compatible_return, else
            (FloatTensor): (num_layers * num_directions, batch, hidden_size)
        """
        input_packed = isinstance(input, nn.utils.rnn.PackedSequence)
        if input_packed:
            input, lengths = nn.utils.rnn.pad_packed_sequence(input)
            max_length = lengths.max().item()
            mask_pad = torch.ByteTensor([([0] * l + [1] * (max_length - l)) for
                l in lengths.tolist()])
            mask_pad = mask_pad.to(input.device).transpose(0, 1).contiguous()
        if input.dim() != 3:
            raise ValueError(
                'There must be 3 dimensions for (length, batch_size, input_size)'
                )
        if c0 is None:
            zeros = input.data.new(input.size(1), self.output_size).zero_()
            c0 = [zeros for i in range(self.num_layers)]
        else:
            if c0.dim() != 3:
                raise ValueError(
                    'There must be 3 dimensions for (num_layers, batch_size, output_size)'
                    )
            c0 = [x.squeeze(0) for x in c0.chunk(self.num_layers, 0)]
        prevx = (input if self.input_to_hidden is None else self.
            input_to_hidden(input))
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i], mask_pad=mask_pad)
            prevx = h
            lstc.append(c)
        if input_packed:
            prevx = nn.utils.rnn.pack_padded_sequence(prevx, lengths,
                enforce_sorted=False)
        lstc_stack = torch.stack(lstc)
        if self.nn_rnn_compatible_return:
            batch_size = input.size(1)
            lstc_stack = lstc_stack.view(self.num_layers, batch_size, self.
                num_directions, self.hidden_size)
            lstc_stack = lstc_stack.transpose(1, 2).contiguous()
            lstc_stack = lstc_stack.view(self.num_layers * self.
                num_directions, batch_size, self.hidden_size)
        return prevx, lstc_stack

    def reset_parameters(self):
        for rnn in self.rnn_lst:
            rnn.reset_parameters()

    def make_backward_compatible(self):
        self.nn_rnn_compatible_return = getattr(self,
            'nn_rnn_compatible_return', False)
        if hasattr(self, 'n_in'):
            if len(self.ln_lst):
                raise Exception(
                    'Layer norm is not backward compatible for sru<=2.1.7')
            if self.use_weight_norm:
                raise Exception('Weight norm removed in sru>=2.1.9')
            self.input_size = self.n_in
            self.hidden_size = self.n_out
            self.output_size = self.out_size
            self.num_layers = self.depth
            self.projection_size = self.n_proj
            self.use_layer_norm = False
            for cell in self.rnn_lst:
                cell.input_size = cell.n_in
                cell.hidden_size = cell.n_out
                cell.output_size = (cell.n_out * 2 if cell.bidirectional else
                    cell.n_out)
                cell.num_matrices = cell.k
                cell.projection_size = cell.n_proj
                cell.layer_norm = None
                if cell.activation_type > 1:
                    raise Exception(
                        'ReLU or SeLU activation removed in sru>=2.1.9')
        if not hasattr(self, 'input_to_hidden'):
            self.input_to_hidden = None
            for cell in self.rnn_lst:
                cell.custom_m = None


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_asappresearch_sru(_paritybench_base):
    pass
    def test_000(self):
        self._check(EmbeddingLayer(*[], **{'n_d': 4, 'words': [4, 4]}), [torch.zeros([4], dtype=torch.int64)], {})

