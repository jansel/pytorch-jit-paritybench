import sys
_module = sys.modules[__name__]
del sys
conf = _module
Loss = _module
ModelConstructor = _module
Models = _module
Optim = _module
Trainer = _module
Utils = _module
onmt = _module
AudioDataset = _module
DatasetBase = _module
GCNDataset = _module
IO = _module
ImageDataset = _module
TextDataset = _module
io = _module
AudioEncoder = _module
Conv2Conv = _module
ConvMultiStepAttention = _module
CopyGenerator = _module
Embeddings = _module
Gate = _module
GlobalAttention = _module
ImageEncoder = _module
MultiHeadedAttn = _module
SRU = _module
StackedRNN = _module
StructuredAttention = _module
Transformer = _module
UtilClass = _module
WeightNorm = _module
modules = _module
GCN = _module
my_modules = _module
Beam = _module
Penalties = _module
Translation = _module
Translator = _module
translate = _module
opts = _module
preprocess = _module
setup = _module
srtask = _module
sr11_linear_input = _module
sr11_onmtgcn_input = _module
sr_onmtgcn_deanonymise = _module
srpredictions4ter = _module
test = _module
test_attention = _module
test_models = _module
test_preprocess = _module
test_simple = _module
apply_bpe = _module
average_models = _module
embeddings_to_torch = _module
extract_embeddings = _module
learn_bpe = _module
release_model = _module
test_rouge = _module
train = _module
translate = _module
CoreNLPService = _module
utils = _module
jsonrpc = _module
EntityGraph = _module
webnlg_eval_scripts = _module
benchmark_reader = _module
metrics = _module
webnlg_baseline_input = _module
webnlg_gcnonmt_input = _module
webnlg_gcnonmt_relexicalise = _module
webnlg_relexicalise = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.autograd import Variable


from torch.nn.init import xavier_uniform


import torch.nn.functional as F


from torch.nn.utils.rnn import pack_padded_sequence as pack


from torch.nn.utils.rnn import pad_packed_sequence as unpack


import torch.optim as optim


from torch.nn.utils import clip_grad_norm


import time


import math


import torchtext


from itertools import chain


from collections import Counter


from collections import defaultdict


from collections import OrderedDict


from itertools import count


import torch.autograd as autograd


import torchtext.data


import torchtext.vocab


import numpy as np


import torch.nn.init as init


import torch.cuda


import re


from torch.autograd import Function


from collections import namedtuple


from torch.nn import Parameter


from torch.nn.parameter import Parameter


import copy


import random


from torch import cuda


def filter_shard_state(state, requires_grad=True, volatile=False):
    for k, v in state.items():
        if v is not None:
            if isinstance(v, Variable) and v.requires_grad:
                v = Variable(v.data, requires_grad=requires_grad, volatile=volatile)
            yield k, v


def shards(state, shard_size, eval=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval:
        yield filter_shard_state(state, False, True)
    else:
        non_none = dict(filter_shard_state(state))
        keys, values = zip(*((k, torch.split(v, shard_size)) for k, v in non_none.items()))
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))
        variables = ((state[k], v.grad.data) for k, v in non_none.items() if isinstance(v, Variable) and v.grad is not None)
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating mutiple
    loss computations


    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, generator, tgt_vocab):
        super(LossComputeBase, self).__init__()
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.padding_idx = tgt_vocab.stoi[onmt.io.PAD_WORD]

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch or a trunc of it?
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def monolithic_compute_loss(self, batch, output, attns):
        """
        Compute the forward loss for the batch.

        Args:
          batch (batch): batch of labeled examples
          output (:obj:`FloatTensor`):
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict of :obj:`FloatTensor`) :
              dictionary of attention distributions
              `[tgt_len x batch x src_len]`
        Returns:
            :obj:`onmt.Statistics`: loss statistics
        """
        range_ = 0, batch.tgt.size(0)
        shard_state = self._make_shard_state(batch, output, range_, attns)
        _, batch_stats = self._compute_loss(batch, **shard_state)
        return batch_stats

    def sharded_compute_loss(self, batch, output, attns, cur_trunc, trunc_size, shard_size, normalization):
        """Compute the forward loss and backpropagate.  Computation is done
        with shards and optionally truncation for memory efficiency.

        Also supports truncated BPTT for long sequences by taking a
        range in the decoder output sequence to back propagate in.
        Range is from `(cur_trunc, cur_trunc + trunc_size)`.

        Note sharding is an exact efficiency trick to relieve memory
        required for the generation buffers. Truncation is an
        approximate efficiency trick to relieve the memory required
        in the RNN buffers.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          cur_trunc (int) : starting position of truncation window
          trunc_size (int) : length of truncation window
          shard_size (int) : maximum number of examples in a shard
          normalization (int) : Loss is divided by this number

        Returns:
            :obj:`onmt.Statistics`: validation loss statistics

        """
        batch_stats = onmt.Statistics()
        range_ = cur_trunc, cur_trunc + trunc_size
        shard_state = self._make_shard_state(batch, output, range_, attns)
        for shard in shards(shard_state, shard_size):
            loss, stats = self._compute_loss(batch, **shard)
            loss.div(normalization).backward()
            batch_stats.update(stats)
        return batch_stats

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum()
        return onmt.Statistics(loss[0], non_padding.sum(), num_correct)

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def _unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, normalization='sents', label_smoothing=0.0):
        super(NMTLossCompute, self).__init__(generator, tgt_vocab)
        assert label_smoothing >= 0.0 and label_smoothing <= 1.0
        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(size_average=False)
            one_hot = torch.randn(1, len(tgt_vocab))
            one_hot.fill_(label_smoothing / (len(tgt_vocab) - 2))
            one_hot[0][self.padding_idx] = 0
            self.register_buffer('one_hot', one_hot)
        else:
            weight = torch.ones(len(tgt_vocab))
            weight[self.padding_idx] = 0
            self.criterion = nn.NLLLoss(weight, size_average=False)
        self.confidence = 1.0 - label_smoothing

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {'output': output, 'target': batch.tgt[range_[0] + 1:range_[1]]}

    def _compute_loss(self, batch, output, target):
        scores = self.generator(self._bottle(output))
        gtruth = target.view(-1)
        if self.confidence < 1:
            tdata = gtruth.data
            mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
            log_likelihood = torch.gather(scores.data, 1, tdata.unsqueeze(1))
            tmp_ = self.one_hot.repeat(gtruth.size(0), 1)
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)
            if mask.dim() > 0:
                log_likelihood.index_fill_(0, mask, 0)
                tmp_.index_fill_(0, mask, 0)
            gtruth = Variable(tmp_, requires_grad=False)
        loss = self.criterion(scores, gtruth)
        if self.confidence < 1:
            loss_data = loss.data.clone()
        else:
            loss_data = loss.data.clone()
        stats = self._stats(loss_data, scores.data, target.view(-1).data)
        return loss, stats


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), 'Not all arguments have the same value: ' + str(args)


class EncoderBase(nn.Module):
    """
    Base encoder class. Specifies the interface used by different encoder types
    and required by :obj:`onmt.Models.NMTModel`.

    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
            C[Pos 1]
            D[Pos 2]
            E[Pos N]
          end
          F[Memory_Bank]
          G[Final]
          A-->C
          A-->D
          A-->E
          C-->F
          D-->F
          E-->F
          E-->G
    """

    def _check_args(self, input, lengths=None, hidden=None):
        s_len, n_batch, n_feats = input.size()
        if lengths is not None:
            n_batch_, = lengths.size()
            aeq(n_batch, n_batch_)

    def forward(self, src, lengths=None, encoder_state=None):
        """
        Args:
            src (:obj:`LongTensor`):
               padded sequences of sparse indices `[src_len x batch x nfeat]`
            lengths (:obj:`LongTensor`): length of each sequence `[batch]`
            encoder_state (rnn-class specific):
               initial encoder_state state.

        Returns:
            (tuple of :obj:`FloatTensor`, :obj:`FloatTensor`):
                * final encoder state, used to initialize decoder
                * memory bank for attention, `[src_len x batch x hidden]`
        """
        raise NotImplementedError


class MeanEncoder(EncoderBase):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    def forward(self, src, lengths=None, encoder_state=None):
        """See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths, encoder_state)
        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        mean = emb.mean(0).expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = mean, mean
        return encoder_final, memory_bank


def rnn_factory(rnn_type, **kwargs):
    no_pack_padded_seq = False
    if rnn_type == 'SRU':
        no_pack_padded_seq = True
        rnn = onmt.modules.SRU(**kwargs)
    else:
        rnn = getattr(nn, rnn_type)(**kwargs)
    return rnn, no_pack_padded_seq


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size, dropout=0.0, embeddings=None, use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None
        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.rnn, self.no_pack_padded_seq = rnn_factory(rnn_type, input_size=embeddings.embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type, hidden_size, num_layers)

    def forward(self, src, lengths=None, encoder_state=None):
        """See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths, encoder_state)
        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()
        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)
        memory_bank, encoder_final = self.rnn(packed_emb, encoder_state)
        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank

    def _initialize_bridge(self, rnn_type, hidden_size, num_layers):
        number_of_states = 2 if rnn_type == 'LSTM' else 1
        self.total_hidden_dim = hidden_size * num_layers
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim, self.total_hidden_dim, bias=True) for i in range(number_of_states)])

    def _bridge(self, hidden):
        """
        Forward hidden state through bridge
        """

        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)
        if isinstance(hidden, tuple):
            outs = tuple([bottle_hidden(layer, hidden[ix]) for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class GCNLayer(nn.Module):
    """ Graph convolutional neural network encoder.

    """

    def __init__(self, num_inputs, num_units, num_labels, in_arcs=True, out_arcs=True, batch_first=False, use_gates=True, use_glus=False):
        super(GCNLayer, self).__init__()
        self.in_arcs = in_arcs
        self.out_arcs = out_arcs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_labels = num_labels
        self.batch_first = batch_first
        self.glu = nn.GLU(3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_gates = use_gates
        self.use_glus = use_glus
        if in_arcs:
            self.V_in = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal(self.V_in)
            self.b_in = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant(self.b_in, 0)
            if self.use_gates:
                self.V_in_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal(self.V_in_gate)
                self.b_in_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant(self.b_in_gate, 1)
        if out_arcs:
            self.V_out = Parameter(torch.Tensor(self.num_inputs, self.num_units))
            nn.init.xavier_normal(self.V_out)
            self.b_out = Parameter(torch.Tensor(num_labels, self.num_units))
            nn.init.constant(self.b_out, 0)
            if self.use_gates:
                self.V_out_gate = Parameter(torch.Tensor(self.num_inputs, 1))
                nn.init.xavier_normal(self.V_out_gate)
                self.b_out_gate = Parameter(torch.Tensor(num_labels, 1))
                nn.init.constant(self.b_out_gate, 1)
        self.W_self_loop = Parameter(torch.Tensor(self.num_inputs, self.num_units))
        nn.init.xavier_normal(self.W_self_loop)
        if self.use_gates:
            self.W_self_loop_gate = Parameter(torch.Tensor(self.num_inputs, 1))
            nn.init.xavier_normal(self.W_self_loop_gate)

    def forward(self, src, lengths=None, arc_tensor_in=None, arc_tensor_out=None, label_tensor_in=None, label_tensor_out=None, mask_in=None, mask_out=None, mask_loop=None, sent_mask=None):
        if not self.batch_first:
            encoder_outputs = src.permute(1, 0, 2).contiguous()
        else:
            encoder_outputs = src.contiguous()
        batch_size = encoder_outputs.size()[0]
        seq_len = encoder_outputs.size()[1]
        max_degree = 1
        input_ = encoder_outputs.view((batch_size * seq_len, self.num_inputs))
        if self.in_arcs:
            input_in = torch.mm(input_, self.V_in)
            first_in = input_in.index_select(0, arc_tensor_in[0] * seq_len + arc_tensor_in[1])
            second_in = self.b_in.index_select(0, label_tensor_in[0])
            in_ = first_in + second_in
            degr = int(first_in.size()[0] / batch_size / seq_len)
            in_ = in_.view((batch_size, seq_len, degr, self.num_units))
            if self.use_glus:
                in_ = torch.cat((in_, in_), 3)
                in_ = self.glu(in_)
            if self.use_gates:
                input_in_gate = torch.mm(input_, self.V_in_gate)
                first_in_gate = input_in_gate.index_select(0, arc_tensor_in[0] * seq_len + arc_tensor_in[1])
                second_in_gate = self.b_in_gate.index_select(0, label_tensor_in[0])
                in_gate = (first_in_gate + second_in_gate).view((batch_size, seq_len, degr))
            max_degree += degr
        if self.out_arcs:
            input_out = torch.mm(input_, self.V_out)
            first_out = input_out.index_select(0, arc_tensor_out[0] * seq_len + arc_tensor_out[1])
            second_out = self.b_out.index_select(0, label_tensor_out[0])
            degr = int(first_out.size()[0] / batch_size / seq_len)
            max_degree += degr
            out_ = (first_out + second_out).view((batch_size, seq_len, degr, self.num_units))
            if self.use_glus:
                out_ = torch.cat((out_, out_), 3)
                out_ = self.glu(out_)
            if self.use_gates:
                input_out_gate = torch.mm(input_, self.V_out_gate)
                first_out_gate = input_out_gate.index_select(0, arc_tensor_out[0] * seq_len + arc_tensor_out[1])
                second_out_gate = self.b_out_gate.index_select(0, label_tensor_out[0])
                out_gate = (first_out_gate + second_out_gate).view((batch_size, seq_len, degr))
        same_input = torch.mm(encoder_outputs.view(-1, encoder_outputs.size(2)), self.W_self_loop).view(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        same_input = same_input.view(encoder_outputs.size(0), encoder_outputs.size(1), 1, self.W_self_loop.size(1))
        if self.use_gates:
            same_input_gate = torch.mm(encoder_outputs.view(-1, encoder_outputs.size(2)), self.W_self_loop_gate).view(encoder_outputs.size(0), encoder_outputs.size(1), -1)
        if self.in_arcs and self.out_arcs:
            potentials = torch.cat((in_, out_, same_input), dim=2)
            if self.use_gates:
                potentials_gate = torch.cat((in_gate, out_gate, same_input_gate), dim=2)
            mask_soft = torch.cat((mask_in, mask_out, mask_loop), dim=1)
        elif self.out_arcs:
            potentials = torch.cat((out_, same_input), dim=2)
            if self.use_gates:
                potentials_gate = torch.cat((out_gate, same_input_gate), dim=2)
            mask_soft = torch.cat((mask_out, mask_loop), dim=1)
        elif self.in_arcs:
            potentials = torch.cat((in_, same_input), dim=2)
            if self.use_gates:
                potentials_gate = torch.cat((in_gate, same_input_gate), dim=2)
            mask_soft = torch.cat((mask_in, mask_loop), dim=1)
        else:
            potentials = same_input
            if self.use_gates:
                potentials_gate = same_input_gate
            mask_soft = mask_loop
        potentials_resh = potentials.view((batch_size * seq_len, max_degree, self.num_units))
        if self.use_gates:
            potentials_r = potentials_gate.view((batch_size * seq_len, max_degree))
            probs_det_ = (self.sigmoid(potentials_r) * mask_soft).unsqueeze(2)
            potentials_masked = potentials_resh * probs_det_
        else:
            potentials_masked = potentials_resh * mask_soft.unsqueeze(2)
        potentials_masked_ = potentials_masked.sum(dim=1)
        potentials_masked_ = self.relu(potentials_masked_)
        result_ = potentials_masked_.view((batch_size, seq_len, self.num_units))
        result_ = result_ * sent_mask.permute(1, 0).contiguous().unsqueeze(2)
        memory_bank = result_.permute(1, 0, 2).contiguous()
        return memory_bank


class GCNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, embeddings, num_inputs, num_units, num_labels, num_layers=1, in_arcs=True, out_arcs=True, batch_first=False, residual='', use_gates=True, use_glus=False, morph_embeddings=None):
        super(GCNEncoder, self).__init__()
        self.embeddings = embeddings
        self.num_layers = num_layers
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.residual = residual
        self.use_gates = use_gates
        self.use_glus = use_glus
        if morph_embeddings is not None:
            self.morph_embeddings = morph_embeddings
            self.emb_morph_emb = nn.Linear(num_inputs + morph_embeddings.embedding_size, num_inputs)
        self.H_1 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal(self.H_1)
        self.H_2 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal(self.H_2)
        self.H_3 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal(self.H_3)
        self.H_4 = torch.nn.parameter.Parameter(torch.Tensor(self.num_units, self.num_units))
        nn.init.xavier_normal(self.H_4)
        self.gcn_layers = []
        if residual == '' or residual == 'residual':
            for i in range(self.num_layers):
                gcn = GCNLayer(num_inputs, num_units, num_labels, in_arcs=in_arcs, out_arcs=out_arcs, use_gates=self.use_gates, use_glus=self.use_glus)
                self.gcn_layers.append(gcn)
            self.gcn_seq = nn.Sequential(*self.gcn_layers)
        elif residual == 'dense':
            for i in range(self.num_layers):
                input_size = num_inputs + i * num_units
                gcn = GCNLayer(input_size, num_units, num_labels, in_arcs=in_arcs, out_arcs=out_arcs, use_gates=self.use_gates, use_glus=self.use_glus)
                self.gcn_layers.append(gcn)
            self.gcn_seq = nn.Sequential(*self.gcn_layers)

    def forward(self, src, lengths=None, arc_tensor_in=None, arc_tensor_out=None, label_tensor_in=None, label_tensor_out=None, mask_in=None, mask_out=None, mask_loop=None, sent_mask=None, morph=None, morph_mask=None):
        if morph is None:
            embeddings = self.embeddings(src)
        else:
            embeddings = self.embeddings(src)
            morph_size = morph.data.size()
            embeddings_m = self.morph_embeddings(morph.view(morph_size[0] * morph_size[1], morph_size[2], 1))
            embeddings_m = embeddings_m.view((morph_size[0], morph_size[1], morph_size[2], embeddings_m.data.size()[2]))
            embeddings_m = embeddings_m.permute(3, 0, 1, 2).contiguous()
            masked_morph = embeddings_m * morph_mask
            morph_sum = masked_morph.sum(3).permute(2, 1, 0).contiguous()
            embeddings = torch.cat([embeddings, morph_sum], dim=2)
            embeddings = torch.nn.functional.relu(self.emb_morph_emb(embeddings))
        if self.residual == '':
            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(embeddings, lengths, arc_tensor_in, arc_tensor_out, label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)
                else:
                    memory_bank = gcn(memory_bank, lengths, arc_tensor_in, arc_tensor_out, label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)
        elif self.residual == 'residual':
            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(embeddings, lengths, arc_tensor_in, arc_tensor_out, label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)
                elif g == 1:
                    prev_memory_bank = embeddings + memory_bank
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out, label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)
                else:
                    prev_memory_bank = prev_memory_bank + memory_bank
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out, label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)
        elif self.residual == 'dense':
            for g, gcn in enumerate(self.gcn_layers):
                if g == 0:
                    memory_bank = gcn(embeddings, lengths, arc_tensor_in, arc_tensor_out, label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)
                elif g == 1:
                    prev_memory_bank = torch.cat([embeddings, memory_bank], dim=2)
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out, label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)
                else:
                    prev_memory_bank = torch.cat([prev_memory_bank, memory_bank], dim=2)
                    memory_bank = gcn(prev_memory_bank, lengths, arc_tensor_in, arc_tensor_out, label_tensor_in, label_tensor_out, mask_in, mask_out, mask_loop, sent_mask)
        batch_size = memory_bank.size()[1]
        result_ = memory_bank.permute(2, 1, 0)
        res_sum = result_.sum(2)
        sent_mask = sent_mask.permute(1, 0).contiguous()
        mask_sum = sent_mask.sum(1)
        encoder_final = res_sum / mask_sum
        encoder_final = encoder_final.permute(1, 0)
        h_1 = torch.mm(encoder_final, self.H_1).view((1, batch_size, self.num_units))
        h_2 = torch.mm(encoder_final, self.H_2).view((1, batch_size, self.num_units))
        h_3 = torch.mm(encoder_final, self.H_3).view((1, batch_size, self.num_units))
        h_4 = torch.mm(encoder_final, self.H_4).view((1, batch_size, self.num_units))
        h__1 = torch.cat([h_1, h_2], dim=0)
        h__2 = torch.cat([h_3, h_4], dim=0)
        return (h__1, h__2), memory_bank


class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """

    def detach(self):
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, positions, beam_size):
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size, sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size, br // beam_size, sizes[2], sizes[3])[:, :, idx]
            sent_states.data.copy_(sent_states.data.index_select(1, positions))


class RNNDecoderState(DecoderState):

    def __init__(self, hidden_size, rnnstate):
        """
        Args:
            hidden_size (int): the size of hidden layer of the decoder.
            rnnstate: final hidden state from the encoder.
                transformed to shape: layers x batch x (directions*dim).
        """
        if not isinstance(rnnstate, tuple):
            self.hidden = rnnstate,
        else:
            self.hidden = rnnstate
        self.coverage = None
        batch_size = self.hidden[0].size(1)
        h_size = batch_size, hidden_size
        self.input_feed = Variable(self.hidden[0].data.new(*h_size).zero_(), requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnnstate, input_feed, coverage):
        if not isinstance(rnnstate, tuple):
            self.hidden = rnnstate,
        else:
            self.hidden = rnnstate
        self.input_feed = input_feed
        self.coverage = coverage

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True) for e in self._all]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]


class RNNDecoderBase(nn.Module):
    """
    Base recurrent attention-based decoder class.
    Specifies the interface used by different decoder types
    and required by :obj:`onmt.Models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[Memory_Bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :obj:`onmt.modules.GlobalAttention`
       coverage_attn (str): see :obj:`onmt.modules.GlobalAttention`
       context_gate (str): see :obj:`onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers, hidden_size, attn_type='general', coverage_attn=False, context_gate=None, copy_attn=False, dropout=0.0, embeddings=None, reuse_copy_attn=False):
        super(RNNDecoderBase, self).__init__()
        self.decoder_type = 'rnn'
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)
        self.rnn = self._build_rnn(rnn_type, input_size=self._input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = onmt.modules.context_gate_factory(context_gate, self._input_size, hidden_size, hidden_size, hidden_size)
        self._coverage = coverage_attn
        self.attn = onmt.modules.GlobalAttention(hidden_size, coverage=coverage_attn, attn_type=attn_type)
        self._copy = False
        if copy_attn and not reuse_copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(hidden_size, attn_type=attn_type)
        if copy_attn:
            self._copy = True
        self._reuse_copy_attn = reuse_copy_attn

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Args:
            tgt (`LongTensor`): sequences of padded tokens
                                `[tgt_len x batch x nfeats]`.
            memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            state (:obj:`onmt.Models.DecoderState`):
                 decoder state object to initialize the decoder
            memory_lengths (`LongTensor`): the padded source lengths
                `[batch]`.
        Returns:
            (`FloatTensor`,:obj:`onmt.Models.DecoderState`,`FloatTensor`):
                * decoder_outputs: output from the decoder (after attn)
                         `[tgt_len x batch x hidden]`.
                * decoder_state: final hidden state from the decoder
                * attns: distribution over src at each tgt
                        `[tgt_len x batch x src_len]`.
        """
        assert isinstance(state, RNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        _, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        decoder_final, decoder_outputs, attns = self._run_forward_pass(tgt, memory_bank, state, memory_lengths=memory_lengths)
        final_output = decoder_outputs[-1]
        coverage = None
        if 'coverage' in attns:
            coverage = attns['coverage'][-1].unsqueeze(0)
        state.update_state(decoder_final, final_output.unsqueeze(0), coverage)
        decoder_outputs = torch.stack(decoder_outputs)
        for k in attns:
            attns[k] = torch.stack(attns[k])
        return decoder_outputs, state, attns

    def init_decoder_state(self, src, memory_bank, encoder_final):

        def _fix_enc_hidden(h):
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h
        if isinstance(encoder_final, tuple):
            return RNNDecoderState(self.hidden_size, tuple([_fix_enc_hidden(enc_hid) for enc_hid in encoder_final]))
        else:
            return RNNDecoderState(self.hidden_size, _fix_enc_hidden(encoder_final))


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard fully batched RNN decoder with attention.
    Faster implementation, uses CuDNN for implementation.
    See :obj:`RNNDecoderBase` for options.


    Based around the approach from
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    :cite:`Bahdanau2015`


    Implemented without input_feeding and currently with no `coverage_attn`
    or `copy_attn` support.
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        Private helper for running the specific RNN forward pass.
        Must be overriden by all subclasses.
        Args:
            tgt (LongTensor): a sequence of input tokens tensors
                                 [len x batch x nfeats].
            memory_bank (FloatTensor): output(tensor sequence) from the encoder
                        RNN of size (src_len x batch x hidden_size).
            state (FloatTensor): hidden state from the encoder RNN for
                                 initializing the decoder.
            memory_lengths (LongTensor): the source memory_bank lengths.
        Returns:
            decoder_final (Variable): final hidden state from the decoder.
            decoder_outputs ([FloatTensor]): an array of output of every time
                                     step from the decoder.
            attns (dict of (str, [FloatTensor]): a dictionary of different
                            type of attention Tensor array of every time
                            step from the decoder.
        """
        assert not self._copy
        assert not self._coverage
        attns = {}
        emb = self.embeddings(tgt)
        if isinstance(self.rnn, nn.GRU):
            rnn_output, decoder_final = self.rnn(emb, state.hidden[0])
        else:
            rnn_output, decoder_final = self.rnn(emb, state.hidden)
        tgt_len, tgt_batch, _ = tgt.size()
        output_len, output_batch, _ = rnn_output.size()
        aeq(tgt_len, output_len)
        aeq(tgt_batch, output_batch)
        decoder_outputs, p_attn = self.attn(rnn_output.transpose(0, 1).contiguous(), memory_bank.transpose(0, 1), memory_lengths=memory_lengths)
        attns['std'] = p_attn
        if self.context_gate is not None:
            decoder_outputs = self.context_gate(emb.view(-1, emb.size(2)), rnn_output.view(-1, rnn_output.size(2)), decoder_outputs.view(-1, decoder_outputs.size(2)))
            decoder_outputs = decoder_outputs.view(tgt_len, tgt_batch, self.hidden_size)
        decoder_outputs = self.dropout(decoder_outputs)
        return decoder_final, decoder_outputs, attns

    def _build_rnn(self, rnn_type, **kwargs):
        rnn, _ = rnn_factory(rnn_type, **kwargs)
        return rnn

    @property
    def _input_size(self):
        """
        Private helper returning the number of expected features.
        """
        return self.embeddings.embedding_size


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Input feeding based decoder. See :obj:`RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[Memory_Bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        input_feed = state.input_feed.squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        decoder_outputs = []
        attns = {'std': []}
        if self._copy:
            attns['copy'] = []
        if self._coverage:
            attns['coverage'] = []
        emb = self.embeddings(tgt)
        assert emb.dim() == 3
        hidden = state.hidden
        coverage = state.coverage.squeeze(0) if state.coverage is not None else None
        for i, emb_t in enumerate(emb.split(1)):
            emb_t = emb_t.squeeze(0)
            decoder_input = torch.cat([emb_t, input_feed], 1)
            rnn_output, hidden = self.rnn(decoder_input, hidden)
            decoder_output, p_attn = self.attn(rnn_output, memory_bank.transpose(0, 1), memory_lengths=memory_lengths)
            if self.context_gate is not None:
                decoder_output = self.context_gate(decoder_input, rnn_output, decoder_output)
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output
            decoder_outputs += [decoder_output]
            attns['std'] += [p_attn]
            if self._coverage:
                coverage = coverage + p_attn if coverage is not None else p_attn
                attns['coverage'] += [coverage]
            if self._copy and not self._reuse_copy_attn:
                _, copy_attn = self.copy_attn(decoder_output, memory_bank.transpose(0, 1))
                attns['copy'] += [copy_attn]
            elif self._copy:
                attns['copy'] = attns['std']
        return hidden, decoder_outputs, attns

    def _build_rnn(self, rnn_type, input_size, hidden_size, num_layers, dropout):
        assert not rnn_type == 'SRU', "SRU doesn't support input feed! Please set -input_feed 0!"
        if rnn_type == 'LSTM':
            stacked_cell = onmt.modules.StackedLSTM
        else:
            stacked_cell = onmt.modules.StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """
        Using input feed by concatenating input with attention vectors.
        """
        return self.embeddings.embedding_size + self.hidden_size


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]
        enc_final, memory_bank = self.encoder(src, lengths)
        enc_state = self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_bank, enc_state if dec_state is None else dec_state, memory_lengths=lengths)
        if self.multigpu:
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class NMTModelGCN(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModelGCN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, lengths, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop, mask_sent, morph=None, mask_morph=None, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]
        enc_final, memory_bank = self.encoder(src, lengths, adj_arc_in, adj_arc_out, adj_lab_in, adj_lab_out, mask_in, mask_out, mask_loop, mask_sent, morph, mask_morph)
        enc_state = self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_bank, enc_state if dec_state is None else dec_state, memory_lengths=lengths)
        if self.multigpu:
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state


class AudioEncoder(nn.Module):
    """
    A simple encoder convolutional -> recurrent neural network for
    audio input.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec

    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout, sample_rate, window_size):
        super(AudioEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size
        self.layer1 = nn.Conv2d(1, 32, kernel_size=(41, 11), padding=(0, 10), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=(21, 11), padding=(0, 0), stride=(2, 1))
        self.batch_norm2 = nn.BatchNorm2d(32)
        input_size = int(math.floor(sample_rate * window_size / 2) + 1)
        input_size = int(math.floor(input_size - 41) / 2 + 1)
        input_size = int(math.floor(input_size - 21) / 2 + 1)
        input_size *= 32
        self.rnn = nn.LSTM(input_size, rnn_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)

    def load_pretrained_vectors(self, opt):
        pass

    def forward(self, input, lengths=None):
        """See :obj:`onmt.modules.EncoderBase.forward()`"""
        input = self.batch_norm1(self.layer1(input[:, :, :, :]))
        input = F.hardtanh(input, 0, 20, inplace=True)
        input = self.batch_norm2(self.layer2(input))
        input = F.hardtanh(input, 0, 20, inplace=True)
        batch_size = input.size(0)
        length = input.size(3)
        input = input.view(batch_size, -1, length)
        input = input.transpose(0, 2).transpose(1, 2)
        output, hidden = self.rnn(input)
        return hidden, output


def get_var_maybe_avg(namespace, var_name, training, polyak_decay):
    v = getattr(namespace, var_name)
    v_avg = getattr(namespace, var_name + '_avg')
    v_avg -= (1 - polyak_decay) * (v_avg - v.data)
    if training:
        return v
    else:
        return Variable(v_avg)


def get_vars_maybe_avg(namespace, var_names, training, polyak_decay):
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(namespace, vn, training, polyak_decay))
    return vars


class WeightNormConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, init_scale=1.0, polyak_decay=0.9995):
        super(WeightNormConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = self.bias
        self.register_buffer('V_avg', torch.zeros(self.V.size()))
        self.register_buffer('g_avg', torch.zeros(out_channels))
        self.register_buffer('b_avg', torch.zeros(out_channels))
        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05)
            v_norm = self.V.data / self.V.data.view(self.out_channels, -1).norm(2, 1).view(self.out_channels, *([1] * (len(self.kernel_size) + 1))).expand_as(self.V.data)
            x_init = F.conv2d(x, Variable(v_norm), None, self.stride, self.padding, self.dilation, self.groups).data
            t_x_init = x_init.transpose(0, 1).contiguous().view(self.out_channels, -1)
            m_init, v_init = t_x_init.mean(1).squeeze(1), t_x_init.var(1).squeeze(1)
            scale_init = self.init_scale / torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            scale_init_shape = scale_init.view(1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            m_init_shape = m_init.view(1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            x_init = scale_init_shape.expand_as(x_init) * (x_init - m_init_shape.expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
        else:
            v, g, b = get_vars_maybe_avg(self, ['V', 'g', 'b'], self.training, polyak_decay=self.polyak_decay)
            scalar = torch.norm(v.view(self.out_channels, -1), 2, 1)
            if len(scalar.size()) == 2:
                scalar = g / scalar.squeeze(1)
            else:
                scalar = g / scalar
            w = scalar.view(self.out_channels, *([1] * (len(v.size()) - 1))).expand_as(v) * v
            x = F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
            return x


class GatedConv(nn.Module):

    def __init__(self, input_size, width=3, dropout=0.2, nopad=False):
        super(GatedConv, self).__init__()
        self.conv = WeightNormConv2d(input_size, 2 * input_size, kernel_size=(width, 1), stride=(1, 1), padding=(width // 2 * (1 - nopad), 0))
        init.xavier_uniform(self.conv.weight, gain=(4 * (1 - dropout)) ** 0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_var, hidden=None):
        x_var = self.dropout(x_var)
        x_var = self.conv(x_var)
        out, gate = x_var.split(int(x_var.size(1) / 2), 1)
        out = out * F.sigmoid(gate)
        return out


SCALE_WEIGHT = 0.5 ** 0.5


class StackedCNN(nn.Module):

    def __init__(self, num_layers, input_size, cnn_kernel_width=3, dropout=0.2):
        super(StackedCNN, self).__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(GatedConv(input_size, cnn_kernel_width, dropout))

    def forward(self, x, hidden=None):
        for conv in self.layers:
            x = x + conv(x)
            x *= SCALE_WEIGHT
        return x


def shape_transform(x):
    """ Tranform the size of the tensors to fit for conv input. """
    return torch.unsqueeze(torch.transpose(x, 1, 2), 3)


class CNNEncoder(EncoderBase):
    """
    Encoder built on CNN based on
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self, num_layers, hidden_size, cnn_kernel_width, dropout, embeddings):
        super(CNNEncoder, self).__init__()
        self.embeddings = embeddings
        input_size = embeddings.embedding_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size, cnn_kernel_width, dropout)

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`onmt.modules.EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)
        emb = self.embeddings(input)
        s_len, batch, emb_dim = emb.size()
        emb = emb.transpose(0, 1).contiguous()
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)
        return emb_remap.squeeze(3).transpose(0, 1).contiguous(), out.squeeze(3).transpose(0, 1).contiguous()


class CNNDecoderState(DecoderState):

    def __init__(self, memory_bank, enc_hidden):
        self.init_src = (memory_bank + enc_hidden) * SCALE_WEIGHT
        self.previous_input = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return self.previous_input,

    def update_state(self, input):
        """ Called for every decoder forward pass. """
        self.previous_input = input

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.init_src = Variable(self.init_src.data.repeat(1, beam_size, 1), volatile=True)


class CNNDecoder(nn.Module):
    """
    Decoder built on CNN, based on :cite:`DBLP:journals/corr/GehringAGYD17`.


    Consists of residual convolutional layers, with ConvMultiStepAttention.
    """

    def __init__(self, num_layers, hidden_size, attn_type, copy_attn, cnn_kernel_width, dropout, embeddings):
        super(CNNDecoder, self).__init__()
        self.decoder_type = 'cnn'
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cnn_kernel_width = cnn_kernel_width
        self.embeddings = embeddings
        self.dropout = dropout
        input_size = self.embeddings.embedding_size
        self.linear = nn.Linear(input_size, self.hidden_size)
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv_layers.append(GatedConv(self.hidden_size, self.cnn_kernel_width, self.dropout, True))
        self.attn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.attn_layers.append(onmt.modules.ConvMultiStepAttention(self.hidden_size))
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(hidden_size, attn_type=attn_type)
            self._copy = True

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """ See :obj:`onmt.modules.RNNDecoderBase.forward()`"""
        assert isinstance(state, CNNDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        contxt_len, contxt_batch, _ = memory_bank.size()
        aeq(tgt_batch, contxt_batch)
        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
        outputs = []
        attns = {'std': []}
        assert not self._copy, 'Copy mechanism not yet tested in conv2conv'
        if self._copy:
            attns['copy'] = []
        emb = self.embeddings(tgt)
        assert emb.dim() == 3
        tgt_emb = emb.transpose(0, 1).contiguous()
        src_memory_bank_t = memory_bank.transpose(0, 1).contiguous()
        src_memory_bank_c = state.init_src.transpose(0, 1).contiguous()
        emb_reshape = tgt_emb.contiguous().view(tgt_emb.size(0) * tgt_emb.size(1), -1)
        linear_out = self.linear(emb_reshape)
        x = linear_out.view(tgt_emb.size(0), tgt_emb.size(1), -1)
        x = shape_transform(x)
        pad = Variable(torch.zeros(x.size(0), x.size(1), self.cnn_kernel_width - 1, 1))
        pad = pad.type_as(x)
        base_target_emb = x
        for conv, attention in zip(self.conv_layers, self.attn_layers):
            new_target_input = torch.cat([pad, x], 2)
            out = conv(new_target_input)
            c, attn = attention(base_target_emb, out, src_memory_bank_t, src_memory_bank_c)
            x = (x + (c + out) * SCALE_WEIGHT) * SCALE_WEIGHT
        output = x.squeeze(3).transpose(1, 2)
        outputs = output.transpose(0, 1).contiguous()
        if state.previous_input is not None:
            outputs = outputs[state.previous_input.size(0):]
            attn = attn[:, state.previous_input.size(0):].squeeze()
            attn = torch.stack([attn])
        attns['std'] = attn
        if self._copy:
            attns['copy'] = attn
        state.update_state(tgt)
        return outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        return CNNDecoderState(memory_bank, enc_hidden)


def seq_linear(linear, x):
    batch, hidden_size, length, _ = x.size()
    h = linear(torch.transpose(x, 1, 2).contiguous().view(batch * length, hidden_size))
    return torch.transpose(h.view(batch, length, hidden_size, 1), 1, 2)


class ConvMultiStepAttention(nn.Module):
    """

    Conv attention takes a key matrix, a value matrix and a query vector.
    Attention weight is calculated by key matrix with the query vector
    and sum on the value matrix. And the same operation is applied
    in each decode conv layer.

    """

    def __init__(self, input_size):
        super(ConvMultiStepAttention, self).__init__()
        self.linear_in = nn.Linear(input_size, input_size)
        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def forward(self, base_target_emb, input, encoder_out_top, encoder_out_combine):
        """
        Args:
            base_target_emb: target emb tensor
            input: output of decode conv
            encoder_out_t: the key matrix for calculation of attetion weight,
                which is the top output of encode conv
            encoder_out_combine:
                the value matrix for the attention-weighted sum,
                which is the combination of base emb and top output of encode

        """
        batch, channel, height, width = base_target_emb.size()
        batch_, channel_, height_, width_ = input.size()
        aeq(batch, batch_)
        aeq(height, height_)
        enc_batch, enc_channel, enc_height = encoder_out_top.size()
        enc_batch_, enc_channel_, enc_height_ = encoder_out_combine.size()
        aeq(enc_batch, enc_batch_)
        aeq(enc_height, enc_height_)
        preatt = seq_linear(self.linear_in, input)
        target = (base_target_emb + preatt) * SCALE_WEIGHT
        target = torch.squeeze(target, 3)
        target = torch.transpose(target, 1, 2)
        pre_attn = torch.bmm(target, encoder_out_top)
        if self.mask is not None:
            pre_attn.data.masked_fill_(self.mask, -float('inf'))
        pre_attn = pre_attn.transpose(0, 2)
        attn = F.softmax(pre_attn)
        attn = attn.transpose(0, 2).contiguous()
        context_output = torch.bmm(attn, torch.transpose(encoder_out_combine, 1, 2))
        context_output = torch.transpose(torch.unsqueeze(context_output, 3), 1, 2)
        return context_output, attn


class CopyGenerator(nn.Module):
    """Generator module that additionally considers copying
    words directly from the source.

    The main idea is that we have an extended "dynamic dictionary".
    It contains `|tgt_dict|` words plus an arbitrary number of
    additional words introduced by the source sentence.
    For each source sentence we have a `src_map` that maps
    each source word to an index in `tgt_dict` if it known, or
    else to an extra word.

    The copy generator is an extended version of the standard
    generator that computse three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of instead copying a
      word from the source, computed using a bernoulli
    * :math:`p_{copy}` the probility of copying a word instead.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       tgt_dict (Vocab): output target dictionary

    """

    def __init__(self, input_size, tgt_dict):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, len(tgt_dict))
        self.linear_copy = nn.Linear(input_size, 1)
        self.tgt_dict = tgt_dict

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by compying
        source words.

        Args:
           hidden (`FloatTensor`): hidden outputs `[batch*tlen, input_size]`
           attn (`FloatTensor`): attn for each `[batch*tlen, input_size]`
           src_map (`FloatTensor`):
             A sparse indicator matrix mapping each source word to
             its index in the "extended" vocab containing.
             `[src_len, batch, extra_words]`
        """
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        slen_, batch, cvocab = src_map.size()
        aeq(batch_by_tlen, batch_by_tlen_)
        aeq(slen, slen_)
        logits = self.linear(hidden)
        logits[:, self.tgt_dict.stoi[onmt.io.PAD_WORD]] = -float('inf')
        prob = F.softmax(logits)
        p_copy = F.sigmoid(self.linear_copy(hidden))
        out_prob = torch.mul(prob, 1 - p_copy.expand_as(prob))
        mul_attn = torch.mul(attn, p_copy.expand_as(attn))
        copy_prob = torch.bmm(mul_attn.view(-1, batch, slen).transpose(0, 1), src_map.transpose(0, 1)).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return torch.cat([out_prob, copy_prob], 1)


class CopyGeneratorCriterion(object):

    def __init__(self, vocab_size, force_copy, pad, eps=1e-20):
        self.force_copy = force_copy
        self.eps = eps
        self.offset = vocab_size
        self.pad = pad

    def __call__(self, scores, align, target):
        align_unk = align.eq(0).float()
        align_not_unk = align.ne(0).float()
        target_unk = target.eq(0).float()
        target_not_unk = target.ne(0).float()
        out = scores.gather(1, align.view(-1, 1) + self.offset).view(-1)
        out = out.mul(align_not_unk) + self.eps
        tmp = scores.gather(1, target.view(-1, 1)).view(-1)
        if not self.force_copy:
            out = out + tmp.mul(target_not_unk)
            out = out + tmp.mul(align_unk).mul(target_unk)
        else:
            out = out + tmp.mul(align_unk)
        loss = -out.log().mul(target.ne(self.pad).float())
        return loss


class CopyGeneratorLossCompute(onmt.Loss.LossComputeBase):
    """
    Copy Generator Loss Computation.
    """

    def __init__(self, generator, tgt_vocab, force_copy, normalize_by_length, eps=1e-20):
        super(CopyGeneratorLossCompute, self).__init__(generator, tgt_vocab)
        self.cur_dataset = None
        self.force_copy = force_copy
        self.normalize_by_length = normalize_by_length
        self.criterion = CopyGeneratorCriterion(len(tgt_vocab), force_copy, self.padding_idx)

    def _make_shard_state(self, batch, output, range_, attns):
        """ See base class for args description. """
        if getattr(batch, 'alignment', None) is None:
            raise AssertionError('using -copy_attn you need to pass in -dynamic_dict during preprocess stage.')
        return {'output': output, 'target': batch.tgt[range_[0] + 1:range_[1]], 'copy_attn': attns.get('copy'), 'align': batch.alignment[range_[0] + 1:range_[1]]}

    def _compute_loss(self, batch, output, target, copy_attn, align):
        """
        Compute the loss. The args must match self._make_shard_state().
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(self._bottle(output), self._bottle(copy_attn), batch.src_map)
        loss = self.criterion(scores, align, target)
        scores_data = scores.data.clone()
        scores_data = onmt.io.TextDataset.collapse_copy_scores(self._unbottle(scores_data, batch.batch_size), batch, self.tgt_vocab, self.cur_dataset.src_vocabs)
        scores_data = self._bottle(scores_data)
        target_data = target.data.clone()
        correct_mask = target_data.eq(0) * align.data.ne(0)
        correct_copy = (align.data + len(self.tgt_vocab)) * correct_mask.long()
        target_data = target_data + correct_copy
        loss_data = loss.sum().data.clone()
        stats = self._stats(loss_data, scores_data, target_data)
        if self.normalize_by_length:
            pad_ix = batch.dataset.fields['tgt'].vocab.stoi[onmt.io.PAD_WORD]
            tgt_lens = batch.tgt.ne(pad_ix).sum(0).float()
            loss = loss.view(-1, batch.batch_size).sum(0)
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()
        return loss, stats


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding for
    non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        emb = emb * math.sqrt(self.dim)
        emb = emb + Variable(self.pe[:emb.size(0)], requires_grad=False)
        emb = self.dropout(emb)
        return emb


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Variable whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Variable.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, input):
        inputs = [feat.squeeze(2) for feat in input.split(1, dim=2)]
        assert len(self) == len(inputs)
        outputs = [f(x) for f, x in zip(self, inputs)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class Embeddings(nn.Module):
    """
    Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feats_padding_idx (list of int): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes ([int], optional): list of size of dictionary
                                    of embeddings for each feature.

        position_encoding (bool): see :obj:`onmt.modules.PositionalEncoding`

        feat_merge (string): merge action for the features embeddings:
                    concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
                    embedding size is N^feat_dim_exponent, where N is the
                    number of values of feature takes.
        feat_vec_size (int): embedding dimension for features when using
                    `-feat_merge mlp`
        dropout (float): dropout probability.
    """

    def __init__(self, word_vec_size, word_vocab_size, word_padding_idx, position_encoding=False, feat_merge='concat', feat_vec_exponent=0.7, feat_vec_size=-1, feat_padding_idx=[], feat_vocab_sizes=[], dropout=0, sparse=False):
        self.word_padding_idx = word_padding_idx
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent) for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse) for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)
        self.embedding_size = sum(emb_dims) if feat_merge == 'concat' else word_vec_size
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)
        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            out_dim = word_vec_size
            mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)
        if position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

    @property
    def word_lut(self):
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file, fixed):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
          fixed (bool) : if true, embeddings are not updated
        """
        if emb_file:
            pretrained = torch.load(emb_file)
            self.word_lut.weight.data.copy_(pretrained)
            if fixed:
                self.word_lut.weight.requires_grad = False

    def forward(self, input):
        """
        Computes the embeddings for words and features.

        Args:
            input (`LongTensor`): index tensor `[len x batch x nfeat]`
        Return:
            `FloatTensor`: word embeddings `[len x batch x embedding_size]`
        """
        in_length, in_batch, nfeat = input.size()
        aeq(nfeat, len(self.emb_luts))
        emb = self.make_embedding(input)
        out_length, out_batch, emb_size = emb.size()
        aeq(in_length, out_length)
        aeq(in_batch, out_batch)
        aeq(emb_size, self.embedding_size)
        return emb


class ContextGate(nn.Module):
    """
    Context gate is a decoder module that takes as input the previous word
    embedding, the current decoder state and the attention state, and
    produces a gate.
    The gate can be used to select the input from the target side context
    (decoder state), from the source context (attention state) or both.
    """

    def __init__(self, embeddings_size, decoder_size, attention_size, output_size):
        super(ContextGate, self).__init__()
        input_size = embeddings_size + decoder_size + attention_size
        self.gate = nn.Linear(input_size, output_size, bias=True)
        self.sig = nn.Sigmoid()
        self.source_proj = nn.Linear(attention_size, output_size)
        self.target_proj = nn.Linear(embeddings_size + decoder_size, output_size)

    def forward(self, prev_emb, dec_state, attn_state):
        input_tensor = torch.cat((prev_emb, dec_state, attn_state), dim=1)
        z = self.sig(self.gate(input_tensor))
        proj_source = self.source_proj(attn_state)
        proj_target = self.target_proj(torch.cat((prev_emb, dec_state), dim=1))
        return z, proj_source, proj_target


class SourceContextGate(nn.Module):
    """Apply the context gate only to the source context"""

    def __init__(self, embeddings_size, decoder_size, attention_size, output_size):
        super(SourceContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size, attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh(target + z * source)


class TargetContextGate(nn.Module):
    """Apply the context gate only to the target context"""

    def __init__(self, embeddings_size, decoder_size, attention_size, output_size):
        super(TargetContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size, attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh(z * target + source)


class BothContextGate(nn.Module):
    """Apply the context gate to both contexts"""

    def __init__(self, embeddings_size, decoder_size, attention_size, output_size):
        super(BothContextGate, self).__init__()
        self.context_gate = ContextGate(embeddings_size, decoder_size, attention_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, prev_emb, dec_state, attn_state):
        z, source, target = self.context_gate(prev_emb, dec_state, attn_state)
        return self.tanh((1.0 - z) * target + z * source)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len).type_as(lengths).repeat(batch_size, 1).lt(lengths.unsqueeze(1))


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \\sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """

    def __init__(self, dim, coverage=False, attn_type='dot'):
        super(GlobalAttention, self).__init__()
        self.dim = dim
        self.attn_type = attn_type
        assert self.attn_type in ['dot', 'general', 'mlp'], 'Please select a valid attention type.'
        if self.attn_type == 'general':
            self.linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == 'mlp':
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
        out_bias = self.attn_type == 'mlp'
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)
        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()
        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)
        if self.attn_type in ['general', 'dot']:
            if self.attn_type == 'general':
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            wq = self.linear_query(h_t.view(-1, dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)
            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)
            wquh = self.tanh(wq + uh)
            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input, memory_bank, memory_lengths=None, coverage=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False
        batch, sourceL, dim = memory_bank.size()
        batch_, targetL, dim_ = input.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = self.tanh(memory_bank)
        align = self.score(input, memory_bank)
        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths)
            mask = mask.unsqueeze(1)
            align.data.masked_fill_(1 - mask, -float('inf'))
        align_vectors = self.sm(align.view(batch * targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)
        c = torch.bmm(align_vectors, memory_bank)
        concat_c = torch.cat([c, input], 2).view(batch * targetL, dim * 2)
        attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ['general', 'dot']:
            attn_h = self.tanh(attn_h)
        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        return attn_h, align_vectors


class ImageEncoder(nn.Module):
    """
    A simple encoder convolutional -> recurrent neural network for
    image input.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout):
        super(ImageEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size
        self.layer1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)
        input_size = 512
        self.rnn = nn.LSTM(input_size, rnn_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional)
        self.pos_lut = nn.Embedding(1000, input_size)

    def load_pretrained_vectors(self, opt):
        pass

    def forward(self, input, lengths=None):
        """See :obj:`onmt.modules.EncoderBase.forward()`"""
        batch_size = input.size(0)
        input = F.relu(self.layer1(input[:, :, :, :] - 0.5), True)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = F.relu(self.layer2(input), True)
        input = F.max_pool2d(input, kernel_size=(2, 2), stride=(2, 2))
        input = F.relu(self.batch_norm1(self.layer3(input)), True)
        input = F.relu(self.layer4(input), True)
        input = F.max_pool2d(input, kernel_size=(1, 2), stride=(1, 2))
        input = F.relu(self.batch_norm2(self.layer5(input)), True)
        input = F.max_pool2d(input, kernel_size=(2, 1), stride=(2, 1))
        input = F.relu(self.batch_norm3(self.layer6(input)), True)
        all_outputs = []
        for row in range(input.size(2)):
            inp = input[:, :, row, :].transpose(0, 2).transpose(1, 2)
            row_vec = torch.Tensor(batch_size).type_as(inp.data).long().fill_(row)
            pos_emb = self.pos_lut(Variable(row_vec))
            with_pos = torch.cat((pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)
        return hidden_t, out


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.

    .. mermaid::

       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O

    Also includes several additional tricks.

    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim
        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count
        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors.

        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :

           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """
        batch, k_len, d = key.size()
        batch_, k_len_, d_ = value.size()
        aeq(batch, batch_)
        aeq(k_len, k_len_)
        aeq(d, d_)
        batch_, q_len, d_ = query.size()
        aeq(batch, batch_)
        aeq(d, d_)
        aeq(self.model_dim % 8, 0)
        if mask is not None:
            batch_, q_len_, k_len_ = mask.size()
            aeq(batch_, batch)
            aeq(k_len_, k_len)
            aeq(q_len_ == q_len)
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch_size, -1, head_count * dim_per_head)
        key_up = shape(self.linear_keys(key))
        value_up = shape(self.linear_values(value))
        query_up = shape(self.linear_query(query))
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(Variable(mask), -1e+18)
        attn = self.sm(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value_up))
        output = self.final_linear(context)
        batch_, q_len_, d_ = output.size()
        aeq(q_len, q_len_)
        aeq(batch, batch_)
        aeq(d, d_)
        top_attn = attn.view(batch_size, head_count, query_len, key_len)[:, 0, :, :].contiguous()
        return output, top_attn


class SRU_Compute(Function):

    def __init__(self, activation_type, d_out, bidirectional=False):
        super(SRU_Compute, self).__init__()
        self.activation_type = activation_type
        self.d_out = d_out
        self.bidirectional = bidirectional

    def forward(self, u, x, bias, init=None, mask_h=None):
        bidir = 2 if self.bidirectional else 1
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1
        init_ = x.new(ncols).zero_() if init is None else init
        size = (length, batch, d * bidir) if x.dim() == 3 else (batch, d * bidir)
        c = x.new(*size)
        h = x.new(*size)
        FUNC = SRU_FWD_FUNC if not self.bidirectional else SRU_BiFWD_FUNC
        FUNC(args=[u.contiguous().data_ptr(), x.contiguous().data_ptr() if k_ == 3 else 0, bias.data_ptr(), init_.contiguous().data_ptr(), mask_h.data_ptr() if mask_h is not None else 0, length, batch, d, k_, h.data_ptr(), c.data_ptr(), self.activation_type], block=(thread_per_block, 1, 1), grid=(num_block, 1, 1), stream=SRU_STREAM)
        self.save_for_backward(u, x, bias, init, mask_h)
        self.intermediate = c
        if x.dim() == 2:
            last_hidden = c
        elif self.bidirectional:
            last_hidden = torch.stack((c[-1, :, :d], c[0, :, d:]))
        else:
            last_hidden = c[-1]
        return h, last_hidden

    def backward(self, grad_h, grad_last):
        if self.bidirectional:
            grad_last = torch.cat((grad_last[0], grad_last[1]), 1)
        bidir = 2 if self.bidirectional else 1
        u, x, bias, init, mask_h = self.saved_tensors
        c = self.intermediate
        length = x.size(0) if x.dim() == 3 else 1
        batch = x.size(-2)
        d = self.d_out
        k = u.size(-1) // d
        k_ = k // 2 if self.bidirectional else k
        ncols = batch * d * bidir
        thread_per_block = min(512, ncols)
        num_block = (ncols - 1) // thread_per_block + 1
        init_ = x.new(ncols).zero_() if init is None else init
        grad_u = u.new(*u.size())
        grad_bias = x.new(2, batch, d * bidir)
        grad_init = x.new(batch, d * bidir)
        grad_x = x.new(*x.size()) if k_ == 3 else None
        FUNC = SRU_BWD_FUNC if not self.bidirectional else SRU_BiBWD_FUNC
        FUNC(args=[u.contiguous().data_ptr(), x.contiguous().data_ptr() if k_ == 3 else 0, bias.data_ptr(), init_.contiguous().data_ptr(), mask_h.data_ptr() if mask_h is not None else 0, c.data_ptr(), grad_h.contiguous().data_ptr(), grad_last.contiguous().data_ptr(), length, batch, d, k_, grad_u.data_ptr(), grad_x.data_ptr() if k_ == 3 else 0, grad_bias.data_ptr(), grad_init.data_ptr(), self.activation_type], block=(thread_per_block, 1, 1), grid=(num_block, 1, 1), stream=SRU_STREAM)
        return grad_u, grad_x, grad_bias.sum(1).view(-1), grad_init, None


class SRUCell(nn.Module):

    def __init__(self, n_in, n_out, dropout=0, rnn_dropout=0, bidirectional=False, use_tanh=1, use_relu=0):
        super(SRUCell, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.rnn_dropout = rnn_dropout
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation_type = 2 if use_relu else 1 if use_tanh else 0
        out_size = n_out * 2 if bidirectional else n_out
        k = 4 if n_in != out_size else 3
        self.size_per_dir = n_out * k
        self.weight = nn.Parameter(torch.Tensor(n_in, self.size_per_dir * 2 if bidirectional else self.size_per_dir))
        self.bias = nn.Parameter(torch.Tensor(n_out * 4 if bidirectional else n_out * 2))
        self.init_weight()

    def init_weight(self):
        val_range = (3.0 / self.n_in) ** 0.5
        self.weight.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()

    def set_bias(self, bias_val=0):
        n_out = self.n_out
        if self.bidirectional:
            self.bias.data[n_out * 2:].zero_().add_(bias_val)
        else:
            self.bias.data[n_out:].zero_().add_(bias_val)

    def forward(self, input, c0=None):
        assert input.dim() == 2 or input.dim() == 3
        n_in, n_out = self.n_in, self.n_out
        batch = input.size(-2)
        if c0 is None:
            c0 = Variable(input.data.new(batch, n_out if not self.bidirectional else n_out * 2).zero_())
        if self.training and self.rnn_dropout > 0:
            mask = self.get_dropout_mask_((batch, n_in), self.rnn_dropout)
            x = input * mask.expand_as(input)
        else:
            x = input
        x_2d = x if x.dim() == 2 else x.contiguous().view(-1, n_in)
        u = x_2d.mm(self.weight)
        if self.training and self.dropout > 0:
            bidir = 2 if self.bidirectional else 1
            mask_h = self.get_dropout_mask_((batch, n_out * bidir), self.dropout)
            h, c = SRU_Compute(self.activation_type, n_out, self.bidirectional)(u, input, self.bias, c0, mask_h)
        else:
            h, c = SRU_Compute(self.activation_type, n_out, self.bidirectional)(u, input, self.bias, c0)
        return h, c

    def get_dropout_mask_(self, size, p):
        w = self.weight.data
        return Variable(w.new(*size).bernoulli_(1 - p).div_(1 - p))


def check_sru_requirement(abort=False):
    """
    Return True if check pass; if check fails and abort is True,
    raise an Exception, othereise return False.
    """
    try:
        if platform.system() == 'Windows':
            subprocess.check_output('pip freeze | findstr cupy', shell=True)
            subprocess.check_output('pip freeze | findstr pynvrtc', shell=True)
        else:
            subprocess.check_output('pip freeze | grep -w cupy', shell=True)
            subprocess.check_output('pip freeze | grep -w pynvrtc', shell=True)
    except subprocess.CalledProcessError:
        if not abort:
            return False
        raise AssertionError("Using SRU requires 'cupy' and 'pynvrtc' python packages installed.")
    if torch.cuda.is_available() is False:
        if not abort:
            return False
        raise AssertionError('Using SRU requires pytorch built with cuda.')
    pattern = re.compile('.*cuda/lib.*')
    ld_path = os.getenv('LD_LIBRARY_PATH', '')
    if re.match(pattern, ld_path) is None:
        if not abort:
            return False
        raise AssertionError('Using SRU requires setting cuda lib path, e.g. export LD_LIBRARY_PATH=/usr/local/cuda/lib64.')
    return True


class SRU(nn.Module):
    """
    Implementation of "Training RNNs as Fast as CNNs"
    :cite:`DBLP:journals/corr/abs-1709-02755`

    TODO: turn to pytorch's implementation when it is available.

    This implementation is adpoted from the author of the paper:
    https://github.com/taolei87/sru/blob/master/cuda_functional.py.

    Args:
      input_size (int): input to model
      hidden_size (int): hidden dimension
      num_layers (int): number of layers
      dropout (float): dropout to use (stacked)
      rnn_dropout (float): dropout to use (recurrent)
      bidirectional (bool): bidirectional
      use_tanh (bool): activation
      use_relu (bool): activation

    """

    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0, rnn_dropout=0, bidirectional=False, use_tanh=1, use_relu=0):
        check_sru_requirement(abort=True)
        super(SRU, self).__init__()
        self.n_in = input_size
        self.n_out = hidden_size
        self.depth = num_layers
        self.dropout = dropout
        self.rnn_dropout = rnn_dropout
        self.rnn_lst = nn.ModuleList()
        self.bidirectional = bidirectional
        self.out_size = hidden_size * 2 if bidirectional else hidden_size
        for i in range(num_layers):
            sru_cell = SRUCell(n_in=self.n_in if i == 0 else self.out_size, n_out=self.n_out, dropout=dropout if i + 1 != num_layers else 0, rnn_dropout=rnn_dropout, bidirectional=bidirectional, use_tanh=use_tanh, use_relu=use_relu)
            self.rnn_lst.append(sru_cell)

    def set_bias(self, bias_val=0):
        for l in self.rnn_lst:
            l.set_bias(bias_val)

    def forward(self, input, c0=None, return_hidden=True):
        assert input.dim() == 3
        dir_ = 2 if self.bidirectional else 1
        if c0 is None:
            zeros = Variable(input.data.new(input.size(1), self.n_out * dir_).zero_())
            c0 = [zeros for i in range(self.depth)]
        else:
            if isinstance(c0, tuple):
                c0 = c0[0]
            assert c0.dim() == 3
            c0 = [h.squeeze(0) for h in c0.chunk(self.depth, 0)]
        prevx = input
        lstc = []
        for i, rnn in enumerate(self.rnn_lst):
            h, c = rnn(prevx, c0[i])
            prevx = h
            lstc.append(c)
        if self.bidirectional:
            fh = torch.cat(lstc)
        else:
            fh = torch.stack(lstc)
        if return_hidden:
            return prevx, fh
        else:
            return prevx


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
        h_1 = torch.stack(h_1)
        return input, (h_1,)


class MatrixTree(nn.Module):
    """Implementation of the matrix-tree theorem for computing marginals
    of non-projective dependency parsing. This attention layer is used
    in the paper "Learning Structured Text Representations."


    :cite:`DBLP:journals/corr/LiuL17d`
    """

    def __init__(self, eps=1e-05):
        self.eps = eps
        super(MatrixTree, self).__init__()

    def forward(self, input):
        laplacian = input.exp() + self.eps
        output = input.clone()
        for b in range(input.size(0)):
            lap = laplacian[b].masked_fill(Variable(torch.eye(input.size(1)).ne(0)), 0)
            lap = -lap + torch.diag(lap.sum(0))
            lap[0] = input[b].diag().exp()
            inv_laplacian = lap.inverse()
            factor = inv_laplacian.diag().unsqueeze(1).expand_as(input[b]).transpose(0, 1)
            term1 = input[b].exp().mul(factor).clone()
            term2 = input[b].exp().mul(inv_laplacian.transpose(0, 1)).clone()
            term1[:, 0] = 0
            term2[0] = 0
            output[b] = term1 - term2
            roots_output = input[b].diag().exp().mul(inv_laplacian.transpose(0, 1)[0])
            output[b] = output[b] + torch.diag(roots_output)
        return output


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

        Args:
            size (int): the size of input for the first-layer of the FFN.
            hidden_size (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, size)
        self.layer_norm = onmt.modules.LayerNorm(size)
        self.dropout_1 = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
            size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
            droput(float): dropout probability(0-1.0).
            head_count(int): the number of head for MultiHeadedAttention.
            hidden_size(int): the second-layer of the PositionwiseFeedForward.
    """

    def __init__(self, size, dropout, head_count=8, hidden_size=2048):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, hidden_size, dropout)
        self.layer_norm = onmt.modules.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class TransformerEncoder(EncoderBase):
    """
    The Transformer encoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O



    Args:
       num_layers (int): number of encoder layers
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings
    """

    def __init__(self, num_layers, hidden_size, dropout, embeddings):
        super(TransformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList([TransformerEncoderLayer(hidden_size, dropout) for i in range(num_layers)])
        self.layer_norm = onmt.modules.LayerNorm(hidden_size)

    def forward(self, input, lengths=None, hidden=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)
        emb = self.embeddings(input)
        s_len, n_batch, emb_dim = emb.size()
        out = emb.transpose(0, 1).contiguous()
        words = input[:, :, 0].transpose(0, 1)
        out_batch, out_len, _ = out.size()
        w_batch, w_len = words.size()
        aeq(out_batch, w_batch)
        aeq(out_len, w_len)
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1).expand(w_batch, w_len, w_len)
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)
        return Variable(emb.data), out.transpose(0, 1).contiguous()


MAX_SIZE = 5000


class TransformerDecoderLayer(nn.Module):
    """
    Args:
      size(int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      droput(float): dropout probability(0-1.0).
      head_count(int): the number of heads for MultiHeadedAttention.
      hidden_size(int): the second-layer of the PositionwiseFeedForward.
    """

    def __init__(self, size, dropout, head_count=8, hidden_size=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = onmt.modules.MultiHeadedAttention(head_count, size, dropout=dropout)
        self.context_attn = onmt.modules.MultiHeadedAttention(head_count, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, hidden_size, dropout)
        self.layer_norm_1 = onmt.modules.LayerNorm(size)
        self.layer_norm_2 = onmt.modules.LayerNorm(size)
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        self.register_buffer('mask', mask)

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, previous_input=None):
        input_batch, input_len, _ = inputs.size()
        if previous_input is not None:
            pi_batch, _, _ = previous_input.size()
            aeq(pi_batch, input_batch)
        contxt_batch, contxt_len, _ = memory_bank.size()
        aeq(input_batch, contxt_batch)
        src_batch, t_len, s_len = src_pad_mask.size()
        tgt_batch, t_len_, t_len__ = tgt_pad_mask.size()
        aeq(input_batch, contxt_batch, src_batch, tgt_batch)
        aeq(s_len, contxt_len)
        dec_mask = torch.gt(tgt_pad_mask + self.mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm
        if previous_input is not None:
            all_input = torch.cat((previous_input, input_norm), dim=1)
            dec_mask = None
        query, attn = self.self_attn(all_input, all_input, input_norm, mask=dec_mask)
        query = self.drop(query) + inputs
        query_norm = self.layer_norm_2(query)
        mid, attn = self.context_attn(memory_bank, memory_bank, query_norm, mask=src_pad_mask)
        output = self.feed_forward(self.drop(mid) + query)
        output_batch, output_len, _ = output.size()
        aeq(input_len, output_len)
        aeq(contxt_batch, output_batch)
        n_batch_, t_len_, s_len_ = attn.size()
        aeq(input_batch, n_batch_)
        aeq(contxt_len, s_len_)
        aeq(input_len, t_len_)
        return output, attn, all_input

    def _get_attn_subsequent_mask(self, size):
        """ Get an attention mask to avoid using the subsequent info."""
        attn_shape = 1, size, size
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoderState(DecoderState):

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        return self.previous_input, self.previous_layer_inputs, self.src

    def update_state(self, input, previous_layer_inputs):
        """ Called for every decoder forward pass. """
        state = TransformerDecoderState(self.src)
        state.previous_input = input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = Variable(self.src.data.repeat(1, beam_size, 1), volatile=True)


class TransformerDecoder(nn.Module):
    """
    The Transformer decoder from "Attention is All You Need".


    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
       num_layers (int): number of encoder layers.
       hidden_size (int): number of hidden units
       dropout (float): dropout parameters
       embeddings (:obj:`onmt.modules.Embeddings`):
          embeddings to use, should have positional encodings

       attn_type (str): if using a seperate copy attention
    """

    def __init__(self, num_layers, hidden_size, attn_type, copy_attn, dropout, embeddings):
        super(TransformerDecoder, self).__init__()
        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer_layers = nn.ModuleList([TransformerDecoderLayer(hidden_size, dropout) for _ in range(num_layers)])
        self._copy = False
        if copy_attn:
            self.copy_attn = onmt.modules.GlobalAttention(hidden_size, attn_type=attn_type)
            self._copy = True
        self.layer_norm = onmt.modules.LayerNorm(hidden_size)

    def forward(self, tgt, memory_bank, state, memory_lengths=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        assert isinstance(state, TransformerDecoderState)
        tgt_len, tgt_batch, _ = tgt.size()
        memory_len, memory_batch, _ = memory_bank.size()
        aeq(tgt_batch, memory_batch)
        src = state.src
        src_words = src[:, :, 0].transpose(0, 1)
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        src_batch, src_len = src_words.size()
        tgt_batch, tgt_len = tgt_words.size()
        aeq(tgt_batch, memory_batch, src_batch, tgt_batch)
        aeq(memory_len, src_len)
        if state.previous_input is not None:
            tgt = torch.cat([state.previous_input, tgt], 0)
        outputs = []
        attns = {'std': []}
        if self._copy:
            attns['copy'] = []
        emb = self.embeddings(tgt)
        if state.previous_input is not None:
            emb = emb[state.previous_input.size(0):,]
        assert emb.dim() == 3
        output = emb.transpose(0, 1).contiguous()
        src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        padding_idx = self.embeddings.word_padding_idx
        src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1).expand(src_batch, tgt_len, src_len)
        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1).expand(tgt_batch, tgt_len, tgt_len)
        saved_inputs = []
        for i in range(self.num_layers):
            prev_layer_input = None
            if state.previous_input is not None:
                prev_layer_input = state.previous_layer_inputs[i]
            output, attn, all_input = self.transformer_layers[i](output, src_memory_bank, src_pad_mask, tgt_pad_mask, previous_input=prev_layer_input)
            saved_inputs.append(all_input)
        saved_inputs = torch.stack(saved_inputs)
        output = self.layer_norm(output)
        outputs = output.transpose(0, 1).contiguous()
        attn = attn.transpose(0, 1).contiguous()
        attns['std'] = attn
        if self._copy:
            attns['copy'] = attn
        state = state.update_state(tgt, saved_inputs)
        return outputs, state, attns

    def init_decoder_state(self, src, memory_bank, enc_hidden):
        return TransformerDecoderState(src)


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class WeightNormLinear(nn.Linear):
    """
    Implementation of "Weight Normalization: A Simple Reparameterization
    to Accelerate Training of Deep Neural Networks"
    :cite:`DBLP:journals/corr/SalimansK16`

    As a reparameterization method, weight normalization is same
    as BatchNormalization, but it doesn't depend on minibatch.
    """

    def __init__(self, in_features, out_features, init_scale=1.0, polyak_decay=0.9995):
        super(WeightNormLinear, self).__init__(in_features, out_features, bias=True)
        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_features))
        self.b = self.bias
        self.register_buffer('V_avg', torch.zeros(out_features, in_features))
        self.register_buffer('g_avg', torch.zeros(out_features))
        self.register_buffer('b_avg', torch.zeros(out_features))
        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05)
            v_norm = self.V.data / self.V.data.norm(2, 1).expand_as(self.V.data)
            x_init = F.linear(x, Variable(v_norm)).data
            m_init, v_init = x_init.mean(0).squeeze(0), x_init.var(0).squeeze(0)
            scale_init = self.init_scale / torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            x_init = scale_init.view(1, -1).expand_as(x_init) * (x_init - m_init.view(1, -1).expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
        else:
            V, g, b = get_vars_maybe_avg(self, ['V', 'g', 'b'], self.training, polyak_decay=self.polyak_decay)
            x = F.linear(x, V)
            scalar = g / torch.norm(V, 2, 1).squeeze(1)
            x = scalar.view(1, -1).expand_as(x) * x + b.view(1, -1).expand_as(x)
            return x


class WeightNormConvTranspose2d(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, init_scale=1.0, polyak_decay=0.9995):
        super(WeightNormConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups)
        self.V = self.weight
        self.g = Parameter(torch.Tensor(out_channels))
        self.b = self.bias
        self.register_buffer('V_avg', torch.zeros(self.V.size()))
        self.register_buffer('g_avg', torch.zeros(out_channels))
        self.register_buffer('b_avg', torch.zeros(out_channels))
        self.init_scale = init_scale
        self.polyak_decay = polyak_decay
        self.reset_parameters()

    def reset_parameters(self):
        return

    def forward(self, x, init=False):
        if init is True:
            self.V.data.copy_(torch.randn(self.V.data.size()).type_as(self.V.data) * 0.05)
            v_norm = self.V.data / self.V.data.transpose(0, 1).contiguous().view(self.out_channels, -1).norm(2, 1).view(self.in_channels, self.out_channels, *([1] * len(self.kernel_size))).expand_as(self.V.data)
            x_init = F.conv_transpose2d(x, Variable(v_norm), None, self.stride, self.padding, self.output_padding, self.groups).data
            t_x_init = x_init.tranpose(0, 1).contiguous().view(self.out_channels, -1)
            m_init, v_init = t_x_init.mean(1).squeeze(1), t_x_init.var(1).squeeze(1)
            scale_init = self.init_scale / torch.sqrt(v_init + 1e-10)
            self.g.data.copy_(scale_init)
            self.b.data.copy_(-m_init * scale_init)
            scale_init_shape = scale_init.view(1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            m_init_shape = m_init.view(1, self.out_channels, *([1] * (len(x_init.size()) - 2)))
            x_init = scale_init_shape.expand_as(x_init) * (x_init - m_init_shape.expand_as(x_init))
            self.V_avg.copy_(self.V.data)
            self.g_avg.copy_(self.g.data)
            self.b_avg.copy_(self.b.data)
            return Variable(x_init)
        else:
            V, g, b = get_vars_maybe_avg(self, ['V', 'g', 'b'], self.training, polyak_decay=self.polyak_decay)
            scalar = g / torch.norm(V.transpose(0, 1).contiguous().view(self.out_channels, -1), 2, 1).squeeze(1)
            w = scalar.view(self.in_channels, self.out_channels, *([1] * (len(V.size()) - 2))).expand_as(V) * V
            x = F.conv_transpose2d(x, w, b, self.stride, self.padding, self.output_padding, self.groups)
            return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BothContextGate,
     lambda: ([], {'embeddings_size': 4, 'decoder_size': 4, 'attention_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (ContextGate,
     lambda: ([], {'embeddings_size': 4, 'decoder_size': 4, 'attention_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (GatedConv,
     lambda: ([], {'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GlobalAttention,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MatrixTree,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MeanEncoder,
     lambda: ([], {'num_layers': 1, 'embeddings': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (PositionalEncoding,
     lambda: ([], {'dropout': 0.5, 'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (PositionwiseFeedForward,
     lambda: ([], {'size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SourceContextGate,
     lambda: ([], {'embeddings_size': 4, 'decoder_size': 4, 'attention_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (StackedCNN,
     lambda: ([], {'num_layers': 1, 'input_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (TargetContextGate,
     lambda: ([], {'embeddings_size': 4, 'decoder_size': 4, 'attention_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])], {}),
     True),
    (WeightNormConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_diegma_graph_2_text(_paritybench_base):
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

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

