import sys
_module = sys.modules[__name__]
del sys
FB = _module
config = _module
data = _module
domain = _module
engine = _module
gen_config = _module
models = _module
dialog_model = _module
modules = _module
train = _module
utils = _module
vis = _module
gen_selfplay_eval = _module
experiments_deal = _module
reinforce_cat = _module
reinforce_gauss = _module
reinforce_word = _module
sl_cat = _module
sl_gauss = _module
sl_word = _module
experiments_woz = _module
dialog_utils = _module
reinforce_cat = _module
reinforce_gauss = _module
reinforce_word = _module
sl_cat = _module
sl_gauss = _module
sl_word = _module
latent_dialog = _module
agent_deal = _module
agent_task = _module
base_data_loaders = _module
base_models = _module
corpora = _module
criterions = _module
data_loaders = _module
dialog_deal = _module
dialog_task = _module
enc2dec = _module
base_modules = _module
classifier = _module
decoders = _module
encoders = _module
evaluators = _module
judgment = _module
main = _module
metric = _module
models_deal = _module
models_task = _module
nn_lib = _module
normalizer = _module
delexicalize = _module
record = _module
utils = _module

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


import random


import copy


import re


from collections import OrderedDict


import logging


import torch


import numpy as np


import time


import itertools


from torch import optim


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


import torch.nn.init


import torch.optim as optim


import torch as th


from torch.nn.modules.loss import _Loss


from torch.nn.modules.module import _addindent


from torch import nn


from collections import defaultdict


class CudaModule(nn.Module):
    """A helper to run a module on a particular device using CUDA."""

    def __init__(self, device_id):
        super(CudaModule, self).__init__()
        self.device_id = device_id

    def to_device(self, m):
        if self.device_id is not None:
            return m
        return m


def init_rnn(rnn, init_range, weights=None, biases=None):
    """Initializes RNN uniformly."""
    weights = weights or ['weight_ih_l0', 'weight_hh_l0']
    biases = biases or ['bias_ih_l0', 'bias_hh_l0']
    for w in weights:
        rnn._parameters[w].data.uniform_(-init_range, init_range)
    for b in biases:
        rnn._parameters[b].data.fill_(0)


class RnnContextEncoder(CudaModule):
    """A module that encodes dialogues context using an RNN."""

    def __init__(self, n, k, nembed, nhid, init_range, device_id):
        super(RnnContextEncoder, self).__init__(device_id)
        self.nhid = nhid
        self.embeder = nn.Embedding(n, nembed)
        self.encoder = nn.GRU(input_size=nembed, hidden_size=nhid, bias=True)
        self.embeder.weight.data.uniform_(-init_range, init_range)
        init_rnn(self.encoder, init_range)

    def forward(self, ctx):
        ctx_h = self.to_device(torch.zeros(1, ctx.size(1), self.nhid))
        ctx_emb = self.embeder(ctx)
        _, ctx_h = self.encoder(ctx_emb, Variable(ctx_h))
        return ctx_h


def init_cont(cont, init_range):
    """Initializes a container uniformly."""
    for m in cont:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-init_range, init_range)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0)


class MlpContextEncoder(CudaModule):
    """A module that encodes dialogues context using an MLP."""

    def __init__(self, n, k, nembed, nhid, init_range, device_id):
        super(MlpContextEncoder, self).__init__(device_id)
        self.cnt_enc = nn.Embedding(n, nembed)
        self.val_enc = nn.Embedding(n, nembed)
        self.encoder = nn.Sequential(nn.Tanh(), nn.Linear(k * nembed, nhid))
        self.cnt_enc.weight.data.uniform_(-init_range, init_range)
        self.val_enc.weight.data.uniform_(-init_range, init_range)
        init_cont(self.encoder, init_range)

    def forward(self, ctx):
        idx = np.arange(ctx.size(0) // 2)
        cnt_idx = Variable(self.to_device(torch.from_numpy(2 * idx + 0)))
        val_idx = Variable(self.to_device(torch.from_numpy(2 * idx + 1)))
        cnt = ctx.index_select(0, cnt_idx)
        val = ctx.index_select(0, val_idx)
        cnt_emb = self.cnt_enc(cnt)
        val_emb = self.val_enc(val)
        h = torch.mul(cnt_emb, val_emb)
        h = h.transpose(0, 1).contiguous().view(ctx.size(1), -1)
        ctx_h = self.encoder(h).unsqueeze(0)
        return ctx_h


FLOAT = 2


INT = 0


LONG = 1


def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(th.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(th.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(th.cuda.FloatTensor)
        else:
            raise ValueError('Unknown dtype')
    elif dtype == INT:
        var = var.type(th.IntTensor)
    elif dtype == LONG:
        var = var.type(th.LongTensor)
    elif dtype == FLOAT:
        var = var.type(th.FloatTensor)
    else:
        raise ValueError('Unknown dtype')
    return var


class BaseModel(nn.Module):

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.use_gpu = config.use_gpu
        self.config = config
        self.kl_w = 0.0

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(th.from_numpy(inputs)), dtype, self.use_gpu)

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, loss, batch_cnt):
        total_loss = self.valid_loss(loss, batch_cnt)
        total_loss.backward()

    def valid_loss(self, loss, batch_cnt=None):
        total_loss = 0.0
        for k, l in loss.items():
            if l is not None:
                total_loss += l
        return total_loss

    def get_optimizer(self, config, verbose=True):
        if config.op == 'adam':
            if verbose:
                None
            return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=config.init_lr, weight_decay=config.l2_norm)
        elif config.op == 'sgd':
            None
            return optim.SGD(self.parameters(), lr=config.init_lr, momentum=config.momentum)
        elif config.op == 'rmsprop':
            None
            return optim.RMSprop(self.parameters(), lr=config.init_lr, momentum=config.momentum)

    def get_clf_optimizer(self, config):
        params = []
        params.extend(self.gru_attn_encoder.parameters())
        params.extend(self.feat_projecter.parameters())
        params.extend(self.sel_classifier.parameters())
        if config.fine_tune_op == 'adam':
            None
            return optim.Adam(params, lr=config.fine_tune_lr)
        elif config.fine_tune_op == 'sgd':
            None
            return optim.SGD(params, lr=config.fine_tune_lr, momentum=config.fine_tune_momentum)
        elif config.fine_tune_op == 'rmsprop':
            None
            return optim.RMSprop(params, lr=config.fine_tune_lr, momentum=config.fine_tune_momentum)

    def model_sel_loss(self, loss, batch_cnt):
        return self.valid_loss(loss, batch_cnt)

    def extract_short_ctx(self, context, context_lens, backward_size=1):
        utts = []
        for b_id in range(context.shape[0]):
            utts.append(context[b_id, context_lens[b_id] - 1])
        return np.array(utts)

    def flatten_context(self, context, context_lens, align_right=False):
        utts = []
        temp_lens = []
        for b_id in range(context.shape[0]):
            temp = []
            for t_id in range(context_lens[b_id]):
                for token in context[b_id, t_id]:
                    if token != 0:
                        temp.append(token)
            temp_lens.append(len(temp))
            utts.append(temp)
        max_temp_len = np.max(temp_lens)
        results = np.zeros((context.shape[0], max_temp_len))
        for b_id in range(context.shape[0]):
            if align_right:
                results[(b_id), -temp_lens[b_id]:] = utts[b_id]
            else:
                results[(b_id), 0:temp_lens[b_id]] = utts[b_id]
        return results


class NLLEntropy(_Loss):

    def __init__(self, padding_idx, avg_type):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.avg_type = avg_type

    def forward(self, net_output, labels):
        batch_size = net_output.size(0)
        pred = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)
        if self.avg_type is None:
            loss = F.nll_loss(pred, target, size_average=False, ignore_index=self.padding_idx)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(pred, target, size_average=False, ignore_index=self.padding_idx)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(pred, target, ignore_index=self.padding_idx, reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = th.sum(loss, dim=1)
            word_cnt = th.sum(th.sign(labels), dim=1).float()
            loss = loss / word_cnt
            loss = th.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(pred, target, size_average=True, ignore_index=self.padding_idx)
        else:
            raise ValueError('Unknown average type')
        return loss


class NLLEntropy4CLF(_Loss):

    def __init__(self, dictionary, bad_tokens=['<disconnect>', '<disagree>'], reduction='elementwise_mean'):
        super(NLLEntropy4CLF, self).__init__()
        w = th.Tensor(len(dictionary)).fill_(1)
        for token in bad_tokens:
            w[dictionary[token]] = 0.0
        self.crit = nn.CrossEntropyLoss(w, reduction=reduction)

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        labels = labels.view(-1)
        return self.crit(preds, labels)


domain = 'object_division'


class CombinedNLLEntropy4CLF(_Loss):

    def __init__(self, dictionary, corpus, np2var, bad_tokens=['<disconnect>', '<disagree>']):
        super(CombinedNLLEntropy4CLF, self).__init__()
        self.dictionary = dictionary
        self.domain = domain.get_domain('object_division')
        self.corpus = corpus
        self.np2var = np2var
        self.bad_tokens = bad_tokens

    def forward(self, preds, goals_id, outcomes_id):
        batch_size = len(goals_id)
        losses = []
        for bth in range(batch_size):
            pred = preds[bth]
            goal = goals_id[bth]
            goal_str = self.corpus.id2goal(goal)
            outcome = outcomes_id[bth]
            outcome_str = self.corpus.id2outcome(outcome)
            if outcome_str[0] in self.bad_tokens:
                continue
            choices = self.domain.generate_choices(goal_str)
            sel_outs = [pred[i] for i in range(pred.size(0))]
            choices_logits = []
            for i in range(self.domain.selection_length()):
                idxs = np.array([self.dictionary[c[i]] for c in choices])
                idxs_var = self.np2var(idxs, LONG)
                choices_logits.append(th.gather(sel_outs[i], 0, idxs_var).unsqueeze(1))
            choice_logit = th.sum(th.cat(choices_logits, 1), 1, keepdim=False)
            choice_logit = choice_logit.sub(choice_logit.max().item())
            prob = F.softmax(choice_logit, dim=0)
            label = choices.index(outcome_str)
            target_prob = prob[label]
            losses.append(-th.log(target_prob))
        return sum(losses) / float(len(losses))


class CatKLLoss(_Loss):

    def __init__(self):
        super(CatKLLoss, self).__init__()

    def forward(self, log_qy, log_py, batch_size=None, unit_average=False):
        """
        qy * log(q(y)/p(y))
        """
        qy = th.exp(log_qy)
        y_kl = th.sum(qy * (log_qy - log_py), dim=1)
        if unit_average:
            return th.mean(y_kl)
        else:
            return th.sum(y_kl) / batch_size


class Entropy(_Loss):

    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, log_qy, batch_size=None, unit_average=False):
        """
        -qy log(qy)
        """
        if log_qy.dim() > 2:
            log_qy = log_qy.squeeze()
        qy = th.exp(log_qy)
        h_q = th.sum(-1 * log_qy * qy, dim=1)
        if unit_average:
            return th.mean(h_q)
        else:
            return th.sum(h_q) / batch_size


class BinaryNLLEntropy(_Loss):

    def __init__(self, size_average=True):
        super(BinaryNLLEntropy, self).__init__()
        self.size_average = size_average

    def forward(self, net_output, label_output):
        """
        :param net_output: batch_size x
        :param labels:
        :return:
        """
        batch_size = net_output.size(0)
        loss = F.binary_cross_entropy_with_logits(net_output, label_output, size_average=self.size_average)
        if self.size_average is False:
            loss /= batch_size
        return loss


class NormKLLoss(_Loss):

    def __init__(self, unit_average=False):
        super(NormKLLoss, self).__init__()
        self.unit_average = unit_average

    def forward(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        loss = 1.0 + (recog_logvar - prior_logvar)
        loss -= th.div(th.pow(prior_mu - recog_mu, 2), th.exp(prior_logvar))
        loss -= th.div(th.exp(recog_logvar), th.exp(prior_logvar))
        if self.unit_average:
            kl_loss = -0.5 * th.mean(loss, dim=1)
        else:
            kl_loss = -0.5 * th.sum(loss, dim=1)
        avg_kl_loss = th.mean(kl_loss)
        return avg_kl_loss


class BaseRNN(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional):
        super(BaseRNN, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell Type: {0}'.format(rnn_cell))
        self.rnn = self.rnn_cell(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=output_dropout_p, bidirectional=bidirectional)
        if rnn_cell.lower() == 'lstm':
            for names in self.rnn._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)


class EncoderGRUATTN(BaseRNN):

    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderGRUATTN, self).__init__(input_dropout_p=input_dropout_p, rnn_cell=rnn_cell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_dropout_p=output_dropout_p, bidirectional=bidirectional)
        self.variable_lengths = variable_lengths
        self.nhid_attn = hidden_size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attn = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))

    def forward(self, residual_var, input_var, turn_feat, mask=None, init_state=None, input_lengths=None):
        require_embed = True
        if require_embed:
            input_cat = th.cat([input_var, residual_var, turn_feat], 2)
        else:
            input_cat = th.cat([input_var, turn_feat], 2)
        if mask is not None:
            input_mask = mask.view(input_cat.size(0), input_cat.size(1), 1)
            input_cat = th.mul(input_cat, input_mask)
        embedded = self.input_dropout(input_cat)
        require_rnn = True
        if require_rnn:
            if init_state is not None:
                h, _ = self.rnn(embedded, init_state)
            else:
                h, _ = self.rnn(embedded)
            logit = self.attn(h.contiguous().view(-1, 2 * self.nhid_attn)).view(h.size(0), h.size(1))
            prob = F.softmax(logit, dim=1).unsqueeze(2).expand_as(h)
            attn = th.sum(th.mul(h, prob), 1)
            return attn
        else:
            logit = self.attn(embedded.contiguous().view(input_cat.size(0) * input_cat.size(1), -1)).view(input_cat.size(0), input_cat.size(1))
            if mask is not None:
                logit_mask = mask.view(input_cat.size(0), input_cat.size(1))
                logit_mask = -999.0 * logit_mask
                logit = logit_mask + logit
            prob = F.softmax(logit, dim=1).unsqueeze(2).expand_as(embedded)
            attn = th.sum(th.mul(embedded, prob), 1)
            return attn


class FeatureProjecter(nn.Module):

    def __init__(self, input_dropout_p, input_size, output_size):
        super(FeatureProjecter, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.sel_encoder = nn.Sequential(nn.Linear(input_size, output_size), nn.Tanh())

    def forward(self, goals_h, attn_outs):
        h = th.cat([attn_outs, goals_h], 1)
        h = self.input_dropout(h)
        h = self.sel_encoder.forward(h)
        return h


class SelectionClassifier(nn.Module):

    def __init__(self, selection_length, input_size, output_size):
        super(SelectionClassifier, self).__init__()
        self.sel_decoders = nn.ModuleList()
        for _ in range(selection_length):
            self.sel_decoders.append(nn.Linear(input_size, output_size))

    def forward(self, proj_outs):
        outs = [decoder.forward(proj_outs).unsqueeze(1) for decoder in self.sel_decoders]
        outs = th.cat(outs, 1)
        return outs


class Attention(nn.Module):

    def __init__(self, dec_cell_size, ctx_cell_size, attn_mode, project):
        super(Attention, self).__init__()
        self.dec_cell_size = dec_cell_size
        self.ctx_cell_size = ctx_cell_size
        self.attn_mode = attn_mode
        if project:
            self.linear_out = nn.Linear(dec_cell_size + ctx_cell_size, dec_cell_size)
        else:
            self.linear_out = None
        if attn_mode == 'general':
            self.dec_w = nn.Linear(dec_cell_size, ctx_cell_size)
        elif attn_mode == 'cat':
            self.dec_w = nn.Linear(dec_cell_size, dec_cell_size)
            self.attn_w = nn.Linear(ctx_cell_size, dec_cell_size)
            self.query_w = nn.Linear(dec_cell_size, 1)

    def forward(self, output, context):
        batch_size = output.size(0)
        max_ctx_len = context.size(1)
        if self.attn_mode == 'dot':
            attn = th.bmm(output, context.transpose(1, 2))
        elif self.attn_mode == 'general':
            mapped_output = self.dec_w(output)
            attn = th.bmm(mapped_output, context.transpose(1, 2))
        elif self.attn_mode == 'cat':
            mapped_output = self.dec_w(output)
            mapped_attn = self.attn_w(context)
            tiled_output = mapped_output.unsqueeze(2).repeat(1, 1, max_ctx_len, 1)
            tiled_attn = mapped_attn.unsqueeze(1)
            fc1 = F.tanh(tiled_output + tiled_attn)
            attn = self.query_w(fc1).squeeze(-1)
        else:
            raise ValueError('Unknown attention mode')
        attn = F.softmax(attn.view(-1, max_ctx_len), dim=1).view(batch_size, -1, max_ctx_len)
        mix = th.bmm(attn, context)
        combined = th.cat((mix, output), dim=2)
        if self.linear_out is None:
            return combined, attn
        else:
            output = F.tanh(self.linear_out(combined.view(-1, self.dec_cell_size + self.ctx_cell_size))).view(batch_size, -1, self.dec_cell_size)
            return output, attn


BOD = '<d>'


PAD = '<pad>'


SYS = 'THEM:'


UNK = '<unk>'


USR = 'YOU:'


DECODING_MASKED_TOKENS = [PAD, UNK, USR, SYS, BOD]


EOS = '<eos>'


GEN = 'gen'


TEACH_FORCE = 'teacher_forcing'


class DecoderRNN(BaseRNN):

    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, vocab_size, use_attn, ctx_cell_size, attn_mode, sys_id, eos_id, use_gpu, max_dec_len, embedding=None):
        super(DecoderRNN, self).__init__(input_dropout_p=input_dropout_p, rnn_cell=rnn_cell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_dropout_p=output_dropout_p, bidirectional=bidirectional)
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.embedding = embedding
        self.use_attn = use_attn
        if self.use_attn:
            self.attention = Attention(dec_cell_size=hidden_size, ctx_cell_size=ctx_cell_size, attn_mode=attn_mode, project=True)
        self.dec_cell_size = hidden_size
        self.output_size = vocab_size
        self.project = nn.Linear(self.dec_cell_size, self.output_size)
        self.log_softmax = F.log_softmax
        self.sys_id = sys_id
        self.eos_id = eos_id
        self.use_gpu = use_gpu
        self.max_dec_len = max_dec_len

    def forward(self, batch_size, dec_inputs, dec_init_state, attn_context, mode, gen_type, beam_size, goal_hid=None):
        ret_dict = dict()
        if self.use_attn:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()
        if mode == GEN:
            dec_inputs = None
        if gen_type != 'beam':
            beam_size = 1
        if dec_inputs is not None:
            decoder_input = dec_inputs
        else:
            with th.no_grad():
                bos_var = Variable(th.LongTensor([self.sys_id]))
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size * beam_size, 1)
        if mode == GEN and gen_type == 'beam':
            pass
        else:
            decoder_hidden_state = dec_init_state
        prob_outputs = []
        symbol_outputs = []

        def decode(step, cum_sum, step_output, step_attn):
            prob_outputs.append(step_output)
            step_output_slice = step_output.squeeze(1)
            if self.use_attn:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            if gen_type == 'greedy':
                _, symbols = step_output_slice.topk(1)
            elif gen_type == 'sample':
                pass
            elif gen_type == 'beam':
                pass
            else:
                raise ValueError('Unsupported decoding mode')
            symbol_outputs.append(symbols)
            return cum_sum, symbols
        if mode == TEACH_FORCE:
            prob_outputs, decoder_hidden_state, attn = self.forward_step(input_var=decoder_input, hidden_state=decoder_hidden_state, encoder_outputs=attn_context, goal_hid=goal_hid)
        else:
            cum_sum = None
            for step in range(self.max_dec_len):
                decoder_output, decoder_hidden_state, step_attn = self.forward_step(decoder_input, decoder_hidden_state, attn_context, goal_hid=goal_hid)
                cum_sum, symbols = decode(step, cum_sum, decoder_output, step_attn)
                decoder_input = symbols
            prob_outputs = th.cat(prob_outputs, dim=1)
        ret_dict[DecoderRNN.KEY_SEQUENCE] = symbol_outputs
        return prob_outputs, decoder_hidden_state, ret_dict

    def forward_step(self, input_var, hidden_state, encoder_outputs, goal_hid):
        batch_size, output_seq_len = input_var.size()
        embedded = self.embedding(input_var)
        if goal_hid is not None:
            goal_hid = goal_hid.view(goal_hid.size(0), 1, goal_hid.size(1))
            goal_rep = goal_hid.repeat(1, output_seq_len, 1)
            embedded = th.cat([embedded, goal_rep], dim=2)
        embedded = self.input_dropout(embedded)
        output, hidden_s = self.rnn(embedded, hidden_state)
        attn = None
        if self.use_attn:
            output, attn = self.attention(output, encoder_outputs)
        logits = self.project(output.contiguous().view(-1, self.dec_cell_size))
        prediction = self.log_softmax(logits, dim=logits.dim() - 1).view(batch_size, output_seq_len, -1)
        return prediction, hidden_s, attn

    def _step(self, input_var, hidden_state, encoder_outputs, goal_hid):
        batch_size, output_seq_len = input_var.size()
        embedded = self.embedding(input_var)
        if goal_hid is not None:
            goal_hid = goal_hid.view(goal_hid.size(0), 1, goal_hid.size(1))
            goal_rep = goal_hid.repeat(1, output_seq_len, 1)
            embedded = th.cat([embedded, goal_rep], dim=2)
        embedded = self.input_dropout(embedded)
        output, hidden_s = self.rnn(embedded, hidden_state)
        attn = None
        if self.use_attn:
            output, attn = self.attention(output, encoder_outputs)
        logits = self.project(output.view(-1, self.dec_cell_size))
        prediction = logits.view(batch_size, output_seq_len, -1)
        return prediction, hidden_s

    def write(self, input_var, hidden_state, encoder_outputs, max_words, vocab, stop_tokens, goal_hid=None, mask=True, decoding_masked_tokens=DECODING_MASKED_TOKENS):
        logprob_outputs = []
        symbol_outputs = []
        decoder_input = input_var
        decoder_hidden_state = hidden_state
        if type(encoder_outputs) is list:
            encoder_outputs = th.cat(encoder_outputs, 1)
        if mask:
            special_token_mask = Variable(th.FloatTensor([(-999.0 if token in decoding_masked_tokens else 0.0) for token in vocab]))
            special_token_mask = cast_type(special_token_mask, FLOAT, self.use_gpu)

        def _sample(dec_output, num_i):
            dec_output = dec_output.view(-1)
            prob = F.softmax(dec_output / 0.6, dim=0)
            logprob = F.log_softmax(dec_output, dim=0)
            symbol = prob.multinomial(num_samples=1).detach()
            _, tmp_symbol = prob.topk(1)
            logprob = logprob.gather(0, symbol)
            return logprob, symbol
        for i in range(max_words):
            decoder_output, decoder_hidden_state = self._step(decoder_input, decoder_hidden_state, encoder_outputs, goal_hid)
            if mask:
                decoder_output += special_token_mask.expand(1, 1, -1)
            logprob, symbol = _sample(decoder_output, i)
            logprob_outputs.append(logprob)
            symbol_outputs.append(symbol)
            decoder_input = symbol.view(1, -1)
            if vocab[symbol.item()] in stop_tokens:
                break
        assert len(logprob_outputs) == len(symbol_outputs)
        logprob_list = logprob_outputs
        symbol_list = [t.item() for t in symbol_outputs]
        return logprob_list, symbol_list

    def forward_rl(self, batch_size, dec_init_state, attn_context, vocab, max_words, goal_hid=None, mask=True, temp=0.1):
        with th.no_grad():
            bos_var = Variable(th.LongTensor([self.sys_id]))
        bos_var = cast_type(bos_var, LONG, self.use_gpu)
        decoder_input = bos_var.expand(batch_size, 1)
        decoder_hidden_state = dec_init_state
        encoder_outputs = attn_context
        logprob_outputs = []
        symbol_outputs = []
        if mask:
            special_token_mask = Variable(th.FloatTensor([(-999.0 if token in DECODING_MASKED_TOKENS else 0.0) for token in vocab]))
            special_token_mask = cast_type(special_token_mask, FLOAT, self.use_gpu)

        def _sample(dec_output, num_i):
            dec_output = dec_output.view(batch_size, -1)
            prob = F.softmax(dec_output / temp, dim=1)
            logprob = F.log_softmax(dec_output, dim=1)
            symbol = prob.multinomial(num_samples=1).detach()
            _, tmp_symbol = prob.topk(1)
            logprob = logprob.gather(1, symbol)
            return logprob, symbol
        stopped_samples = set()
        for i in range(max_words):
            decoder_output, decoder_hidden_state = self._step(decoder_input, decoder_hidden_state, encoder_outputs, goal_hid)
            if mask:
                decoder_output += special_token_mask.expand(1, 1, -1)
            logprob, symbol = _sample(decoder_output, i)
            logprob_outputs.append(logprob)
            symbol_outputs.append(symbol)
            decoder_input = symbol.view(batch_size, -1)
            for b_id in range(batch_size):
                if vocab[symbol[b_id].item()] == EOS:
                    stopped_samples.add(b_id)
            if len(stopped_samples) == batch_size:
                break
        assert len(logprob_outputs) == len(symbol_outputs)
        symbol_outputs = th.cat(symbol_outputs, dim=1).cpu().data.numpy().tolist()
        logprob_outputs = th.cat(logprob_outputs, dim=1)
        logprob_list = []
        symbol_list = []
        for b_id in range(batch_size):
            b_logprob = []
            b_symbol = []
            for t_id in range(logprob_outputs.shape[1]):
                symbol = symbol_outputs[b_id][t_id]
                if vocab[symbol] == EOS and t_id != 0:
                    break
                b_symbol.append(symbol_outputs[b_id][t_id])
                b_logprob.append(logprob_outputs[b_id][t_id])
            logprob_list.append(b_logprob)
            symbol_list.append(b_symbol)
        if batch_size == 1:
            logprob_list = logprob_list[0]
            symbol_list = symbol_list[0]
        return logprob_list, symbol_list


class EncoderRNN(BaseRNN):

    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size, num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderRNN, self).__init__(input_dropout_p=input_dropout_p, rnn_cell=rnn_cell, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_dropout_p=output_dropout_p, bidirectional=bidirectional)
        self.variable_lengths = variable_lengths
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, input_var, init_state=None, input_lengths=None, goals=None):
        if goals is not None:
            batch_size, max_ctx_len, ctx_nhid = input_var.size()
            goals = goals.view(goals.size(0), 1, goals.size(1))
            goals_rep = goals.repeat(1, max_ctx_len, 1).view(batch_size, max_ctx_len, -1)
            input_var = th.cat([input_var, goals_rep], dim=2)
        embedded = self.input_dropout(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


class RnnUttEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, feat_size, goal_nhid, rnn_cell, utt_cell_size, num_layers, input_dropout_p, output_dropout_p, bidirectional, variable_lengths, use_attn, embedding=None):
        super(RnnUttEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = embedding
        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p, rnn_cell=rnn_cell, input_size=embedding_dim + feat_size + goal_nhid, hidden_size=utt_cell_size, num_layers=num_layers, output_dropout_p=output_dropout_p, bidirectional=bidirectional, variable_lengths=variable_lengths)
        self.utt_cell_size = utt_cell_size
        self.multiplier = 2 if bidirectional else 1
        self.output_size = self.multiplier * self.utt_cell_size
        self.use_attn = use_attn
        if self.use_attn:
            self.key_w = nn.Linear(self.output_size, self.utt_cell_size)
            self.query = nn.Linear(self.utt_cell_size, 1)

    def forward(self, utterances, feats=None, init_state=None, goals=None):
        batch_size, max_ctx_len, max_utt_len = utterances.size()
        flat_words = utterances.view(-1, max_utt_len)
        word_embeddings = self.embedding(flat_words)
        flat_mask = th.sign(flat_words).float()
        if feats is not None:
            flat_feats = feats.view(-1, 1)
            flat_feats = flat_feats.unsqueeze(1).repeat(1, max_utt_len, 1)
            word_embeddings = th.cat([word_embeddings, flat_feats], dim=2)
        if goals is not None:
            goals = goals.view(goals.size(0), 1, 1, goals.size(1))
            goals_rep = goals.repeat(1, max_ctx_len, max_utt_len, 1).view(batch_size * max_ctx_len, max_utt_len, -1)
            word_embeddings = th.cat([word_embeddings, goals_rep], dim=2)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)
        if self.use_attn:
            fc1 = th.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            attn = F.softmax(attn, attn.dim() - 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True) + 1e-10)).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = th.sum(utt_embedded, dim=1)
        else:
            attn = None
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)
        utt_embedded = utt_embedded.view(batch_size, max_ctx_len, self.output_size)
        return utt_embedded, word_embeddings.contiguous().view(batch_size, max_ctx_len * max_utt_len, -1), enc_outs.contiguous().view(batch_size, max_ctx_len * max_utt_len, -1)


class MlpGoalEncoder(nn.Module):

    def __init__(self, goal_vocab_size, k, nembed, nhid, init_range):
        super(MlpGoalEncoder, self).__init__()
        self.cnt_enc = nn.Embedding(goal_vocab_size, nembed)
        self.val_enc = nn.Embedding(goal_vocab_size, nembed)
        self.encoder = nn.Sequential(nn.Tanh(), nn.Linear(k * nembed, nhid))
        self.cnt_enc.weight.data.uniform_(-init_range, init_range)
        self.val_enc.weight.data.uniform_(-init_range, init_range)
        self._init_cont(self.encoder, init_range)

    def _init_cont(self, cont, init_range):
        """initializes a container uniformly."""
        for m in cont:
            if hasattr(m, 'weight'):
                m.weight.data.uniform_(-init_range, init_range)
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0)

    def forward(self, goal):
        goal = goal.transpose(0, 1).contiguous()
        idx = np.arange(goal.size(0) // 2)
        cnt_idx = Variable(th.from_numpy(2 * idx + 0))
        val_idx = Variable(th.from_numpy(2 * idx + 1))
        if goal.is_cuda:
            cnt_idx = cnt_idx.type(th.cuda.LongTensor)
            val_idx = val_idx.type(th.cuda.LongTensor)
        else:
            cnt_idx = cnt_idx.type(th.LongTensor)
            val_idx = val_idx.type(th.LongTensor)
        cnt = goal.index_select(0, cnt_idx)
        val = goal.index_select(0, val_idx)
        cnt_emb = self.cnt_enc(cnt)
        val_emb = self.val_enc(val)
        h = th.mul(cnt_emb, val_emb)
        h = h.transpose(0, 1).contiguous().view(goal.size(1), -1)
        goal_h = self.encoder(h)
        return goal_h


class TaskMlpGoalEncoder(nn.Module):

    def __init__(self, goal_vocab_sizes, nhid, init_range):
        super(TaskMlpGoalEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        for v_size in goal_vocab_sizes:
            domain_encoder = nn.Sequential(nn.Linear(v_size, nhid), nn.Tanh())
            self._init_cont(domain_encoder, init_range)
            self.encoder.append(domain_encoder)

    def _init_cont(self, cont, init_range):
        """initializes a container uniformly."""
        for m in cont:
            if hasattr(m, 'weight'):
                m.weight.data.uniform_(-init_range, init_range)
            if hasattr(m, 'bias'):
                m.bias.data.fill_(0)

    def forward(self, goals_list):
        outs = [encoder.forward(goal) for goal, encoder in zip(goals_list, self.encoder)]
        outs = th.sum(th.stack(outs), dim=0)
        return outs


class SelfAttn(nn.Module):

    def __init__(self, hidden_size):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(hidden_size, 1)

    def forward(self, keys, values, attn_mask=None):
        """
        :param attn_inputs: batch_size x time_len x hidden_size
        :param attn_mask: batch_size x time_len
        :return: summary state
        """
        alpha = F.softmax(self.query(keys), dim=1)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
            alpha = alpha / th.sum(alpha, dim=1, keepdim=True)
        summary = th.sum(values * alpha, dim=1)
        return summary


class Bi2UniConnector(nn.Module):

    def __init__(self, rnn_cell, num_layer, hidden_size, output_size):
        super(Bi2UniConnector, self).__init__()
        if rnn_cell == 'lstm':
            self.fch = nn.Linear(hidden_size * 2 * num_layer, output_size)
            self.fcc = nn.Linear(hidden_size * 2 * num_layer, output_size)
        else:
            self.fc = nn.Linear(hidden_size * 2 * num_layer, output_size)
        self.rnn_cell = rnn_cell
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, hidden_state):
        """
        :param hidden_state: [num_layer, batch_size, feat_size]
        :param inputs: [batch_size, feat_size]
        :return: 
        """
        if self.rnn_cell == 'lstm':
            h, c = hidden_state
            num_layer = h.size()[0]
            flat_h = h.transpose(0, 1).contiguous()
            flat_c = c.transpose(0, 1).contiguous()
            new_h = self.fch(flat_h.view(-1, self.hidden_size * num_layer))
            new_c = self.fch(flat_c.view(-1, self.hidden_size * num_layer))
            return new_h.view(1, -1, self.output_size), new_c.view(1, -1, self.output_size)
        else:
            num_layer = hidden_state.size()[0]
            new_s = self.fc(hidden_state.view(-1, self.hidden_size * num_layer))
            new_s = new_s.view(1, -1, self.output_size)
            return new_s


class IdentityConnector(nn.Module):

    def __init(self):
        super(IdentityConnector, self).__init__()

    def forward(self, hidden_state):
        return hidden_state


class Pack(dict):

    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


class HRED(BaseModel):

    def __init__(self, corpus, config):
        super(HRED, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.goal_vocab = corpus.goal_vocab
        self.goal_vocab_dict = corpus.goal_vocab_dict
        self.goal_vocab_size = len(self.goal_vocab)
        self.outcome_vocab = corpus.outcome_vocab
        self.outcome_vocab_dict = corpus.outcome_vocab_dict
        self.outcome_vocab_size = len(self.outcome_vocab)
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.goal_encoder = MlpGoalEncoder(goal_vocab_size=self.goal_vocab_size, k=config.k, nembed=config.goal_embed_size, nhid=config.goal_nhid, init_range=config.init_range)
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size, embedding_dim=config.embed_size, feat_size=1, goal_nhid=config.goal_nhid, rnn_cell=config.utt_rnn_cell, utt_cell_size=config.utt_cell_size, num_layers=config.num_layers, input_dropout_p=config.dropout, output_dropout_p=config.dropout, bidirectional=config.bi_utt_cell, variable_lengths=False, use_attn=config.enc_use_attn, embedding=self.embedding)
        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0, rnn_cell=config.ctx_rnn_cell, input_size=self.utt_encoder.output_size, hidden_size=config.ctx_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=config.bi_ctx_cell, variable_lengths=False)
        if config.bi_ctx_cell:
            self.connector = Bi2UniConnector(rnn_cell=config.ctx_rnn_cell, num_layer=1, hidden_size=config.ctx_cell_size, output_size=config.dec_cell_size)
        else:
            self.connector = IdentityConnector()
        self.decoder = DecoderRNN(input_dropout_p=config.dropout, rnn_cell=config.dec_rnn_cell, input_size=config.embed_size + config.goal_nhid, hidden_size=config.dec_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=False, vocab_size=self.vocab_size, use_attn=config.dec_use_attn, ctx_cell_size=self.ctx_encoder.output_size, attn_mode=config.dec_attn_mode, sys_id=self.sys_id, eos_id=self.eos_id, use_gpu=config.use_gpu, max_dec_len=config.max_dec_len, embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        clf = False
        if not clf:
            ctx_lens = data_feed['context_lens']
            ctx_utts = self.np2var(data_feed['contexts'], LONG)
            ctx_confs = self.np2var(data_feed['context_confs'], FLOAT)
            out_utts = self.np2var(data_feed['outputs'], LONG)
            goals = self.np2var(data_feed['goals'], LONG)
            batch_size = len(ctx_lens)
            goals_h = self.goal_encoder(goals)
            enc_inputs, _, _ = self.utt_encoder(ctx_utts, feats=ctx_confs, goals=goals_h)
            enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)
            dec_inputs = out_utts[:, :-1]
            labels = out_utts[:, 1:].contiguous()
            if self.config.dec_use_attn:
                attn_context = enc_outs
            else:
                attn_context = None
            dec_init_state = self.connector(enc_last)
            dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size, dec_inputs=dec_inputs, dec_init_state=dec_init_state, attn_context=attn_context, mode=mode, gen_type=gen_type, beam_size=self.config.beam_size, goal_hid=goals_h)
            if mode == GEN:
                return ret_dict, labels
            if return_latent:
                return Pack(nll=self.nll(dec_outputs, labels), latent_action=dec_init_state)
            else:
                return Pack(nll=self.nll(dec_outputs, labels))


class GaussHRED(BaseModel):

    def __init__(self, corpus, config):
        super(GaussHRED, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.goal_vocab = corpus.goal_vocab
        self.goal_vocab_dict = corpus.goal_vocab_dict
        self.goal_vocab_size = len(self.goal_vocab)
        self.outcome_vocab = corpus.outcome_vocab
        self.outcome_vocab_dict = corpus.outcome_vocab_dict
        self.outcome_vocab_size = len(self.outcome_vocab)
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.simple_posterior = config.simple_posterior
        self.goal_encoder = MlpGoalEncoder(goal_vocab_size=self.goal_vocab_size, k=config.k, nembed=config.goal_embed_size, nhid=config.goal_nhid, init_range=config.init_range)
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size, embedding_dim=config.embed_size, feat_size=0, goal_nhid=config.goal_nhid, rnn_cell=config.utt_rnn_cell, utt_cell_size=config.utt_cell_size, num_layers=config.num_layers, input_dropout_p=config.dropout, output_dropout_p=config.dropout, bidirectional=config.bi_utt_cell, variable_lengths=False, use_attn=config.enc_use_attn, embedding=self.embedding)
        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0, rnn_cell=config.ctx_rnn_cell, input_size=self.utt_encoder.output_size, hidden_size=config.ctx_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=config.bi_ctx_cell, variable_lengths=False)
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size, config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(config.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.ctx_encoder.output_size, config.y_size, is_lstm=False)
        self.decoder = DecoderRNN(input_dropout_p=config.dropout, rnn_cell=config.dec_rnn_cell, input_size=config.embed_size + config.goal_nhid, hidden_size=config.dec_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=False, vocab_size=self.vocab_size, use_attn=config.dec_use_attn, ctx_cell_size=self.ctx_encoder.output_size, attn_mode=config.dec_attn_mode, sys_id=self.sys_id, eos_id=self.eos_id, use_gpu=config.use_gpu, max_dec_len=config.max_dec_len, embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = criterions.NormKLLoss(unit_average=True)
        self.zero = utils.cast_type(th.zeros(1), FLOAT, self.use_gpu)

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl
        return total_loss

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2 * np.pi))
        logprob = constant - 0.5 * logvar - th.pow(mu - sample_z, 2) / (2.0 * var)
        return logprob

    def z2dec(self, last_h, requires_grad):
        p_mu, p_logvar = self.c2z(last_h)
        if requires_grad:
            sample_z = self.gauss_connector(p_mu, p_logvar)
            joint_logpz = None
        else:
            sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
            logprob_sample_z = self.gaussian_logprob(p_mu, p_logvar, sample_z)
            joint_logpz = th.sum(logprob_sample_z.squeeze(0), dim=1)
        dec_init_state = self.z_embedding(sample_z)
        attn_context = None
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        return dec_init_state, attn_context, joint_logpz

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        goals = self.np2var(data_feed['goals'], LONG)
        batch_size = len(ctx_lens)
        goals_h = self.goal_encoder(goals)
        enc_inputs, _, _ = self.utt_encoder(ctx_utts, goals=goals_h)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1), goals=goals_h)
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1).unsqueeze(0)], dim=2))
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)
        dec_init_state = self.z_embedding(sample_z)
        attn_context = None
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size, dec_inputs=dec_inputs, dec_init_state=dec_init_state, attn_context=attn_context, mode=mode, gen_type=gen_type, beam_size=self.config.beam_size, goal_hid=goals_h)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            return ret_dict, labels
        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result


class CatHRED(BaseModel):

    def __init__(self, corpus, config):
        super(CatHRED, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.goal_vocab = corpus.goal_vocab
        self.goal_vocab_dict = corpus.goal_vocab_dict
        self.goal_vocab_size = len(self.goal_vocab)
        self.outcome_vocab = corpus.outcome_vocab
        self.outcome_vocab_dict = corpus.outcome_vocab_dict
        self.outcome_vocab_size = len(self.outcome_vocab)
        self.sys_id = self.vocab_dict[SYS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.simple_posterior = config.simple_posterior
        self.goal_encoder = MlpGoalEncoder(goal_vocab_size=self.goal_vocab_size, k=config.k, nembed=config.goal_embed_size, nhid=config.goal_nhid, init_range=config.init_range)
        self.embedding = nn.Embedding(self.vocab_size, config.embed_size, padding_idx=self.pad_id)
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size, embedding_dim=config.embed_size, feat_size=0, goal_nhid=config.goal_nhid, rnn_cell=config.utt_rnn_cell, utt_cell_size=config.utt_cell_size, num_layers=config.num_layers, input_dropout_p=config.dropout, output_dropout_p=config.dropout, bidirectional=config.bi_utt_cell, variable_lengths=False, use_attn=config.enc_use_attn, embedding=self.embedding)
        self.ctx_encoder = EncoderRNN(input_dropout_p=0.0, rnn_cell=config.ctx_rnn_cell, input_size=self.utt_encoder.output_size, hidden_size=config.ctx_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=config.bi_ctx_cell, variable_lengths=False)
        self.c2z = nn_lib.Hidden2Discrete(self.ctx_encoder.output_size, config.y_size, config.k_size, is_lstm=config.ctx_rnn_cell == 'lstm')
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Discrete(self.ctx_encoder.output_size + self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        self.z_embedding = nn.Linear(config.y_size * config.k_size, config.dec_cell_size, bias=False)
        self.decoder = DecoderRNN(input_dropout_p=config.dropout, rnn_cell=config.dec_rnn_cell, input_size=config.embed_size + config.goal_nhid, hidden_size=config.dec_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=False, vocab_size=self.vocab_size, use_attn=config.dec_use_attn, ctx_cell_size=self.ctx_encoder.output_size, attn_mode=config.dec_attn_mode, sys_id=self.sys_id, eos_id=self.eos_id, use_gpu=config.use_gpu, max_dec_len=config.max_dec_len, embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = criterions.CatKLLoss()
        self.entropy_loss = criterions.Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss -= self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl
        return total_loss

    def z2dec(self, last_h, requires_grad):
        logits, log_qy = self.c2z(last_h)
        if requires_grad:
            sample_y = self.gumbel_connector(logits)
            logprob_z = None
        else:
            idx = th.multinomial(th.exp(log_qy), 1).detach()
            logprob_z = th.sum(log_qy.gather(1, idx))
            sample_y = utils.cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
            sample_y.scatter_(1, idx, 1.0)
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.config.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.config.y_size):
                attn_context.append(th.mm(temp_sample_y[:, (z_id)], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            attn_context = None
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
        return dec_init_state, attn_context, logprob_z

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']
        ctx_utts = self.np2var(data_feed['contexts'], LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        goals = self.np2var(data_feed['goals'], LONG)
        batch_size = len(ctx_lens)
        goals_h = self.goal_encoder(goals)
        enc_inputs, _, _ = self.utt_encoder(ctx_utts, goals=goals_h)
        enc_outs, enc_last = self.ctx_encoder(enc_inputs, input_lengths=ctx_lens, goals=None)
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1), goals=goals_h)
            logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1).unsqueeze(0)], dim=2))
            if mode == GEN or use_py:
                sample_y = self.gumbel_connector(logits_py)
            else:
                sample_y = self.gumbel_connector(logits_qy)
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.config.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.config.y_size):
                attn_context.append(th.mm(temp_sample_y[:, (z_id)], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            attn_context = None
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size, dec_inputs=dec_inputs, dec_init_state=dec_init_state, attn_context=attn_context, mode=mode, gen_type=gen_type, beam_size=self.config.beam_size, goal_hid=goals_h)
        if mode == GEN:
            return ret_dict, labels
        else:
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            pi_h = self.entropy_loss(log_qy, unit_average=True)
            results = Pack(nll=self.nll(dec_outputs, labels), mi=mi, pi_kl=pi_kl, pi_h=pi_h)
            if return_latent:
                results['latent_action'] = dec_init_state
            return results


BOS = '<s>'


class SysPerfectBD2Word(BaseModel):

    def __init__(self, corpus, config):
        super(SysPerfectBD2Word, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size, embedding_dim=config.embed_size, feat_size=0, goal_nhid=0, rnn_cell=config.utt_rnn_cell, utt_cell_size=config.utt_cell_size, num_layers=config.num_layers, input_dropout_p=config.dropout, output_dropout_p=config.dropout, bidirectional=config.bi_utt_cell, variable_lengths=False, use_attn=config.enc_use_attn, embedding=self.embedding)
        self.policy = nn.Sequential(nn.Linear(self.utt_encoder.output_size + self.db_size + self.bs_size, config.dec_cell_size), nn.Tanh(), nn.Dropout(config.dropout))
        self.decoder = DecoderRNN(input_dropout_p=config.dropout, rnn_cell=config.dec_rnn_cell, input_size=config.embed_size, hidden_size=config.dec_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=False, vocab_size=self.vocab_size, use_attn=config.dec_use_attn, ctx_cell_size=self.utt_encoder.output_size, attn_mode=config.dec_attn_mode, sys_id=self.bos_id, eos_id=self.eos_id, use_gpu=config.use_gpu, max_dec_len=config.max_dec_len, embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', return_latent=False):
        ctx_lens = data_feed['context_lens']
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)
        db_label = self.np2var(data_feed['db'], FLOAT)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None
        dec_init_state = self.policy(th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)).unsqueeze(0)
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size, dec_inputs=dec_inputs, dec_init_state=dec_init_state, attn_context=attn_context, mode=mode, gen_type=gen_type, beam_size=self.config.beam_size)
        if mode == GEN:
            return ret_dict, labels
        if return_latent:
            return Pack(nll=self.nll(dec_outputs, labels), latent_action=dec_init_state)
        else:
            return Pack(nll=self.nll(dec_outputs, labels))

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)
        db_label = self.np2var(data_feed['db'], FLOAT)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        if self.config.dec_use_attn:
            attn_context = enc_outs
        else:
            attn_context = None
        dec_init_state = self.policy(th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)).unsqueeze(0)
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size, dec_init_state=dec_init_state, attn_context=attn_context, vocab=self.vocab, max_words=max_words, temp=temp)
        return logprobs, outs


class SysPerfectBD2Cat(BaseModel):

    def __init__(self, corpus, config):
        super(SysPerfectBD2Cat, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.k_size = config.k_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.contextual_posterior = config.contextual_posterior
        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size, embedding_dim=config.embed_size, feat_size=0, goal_nhid=0, rnn_cell=config.utt_rnn_cell, utt_cell_size=config.utt_cell_size, num_layers=config.num_layers, input_dropout_p=config.dropout, output_dropout_p=config.dropout, bidirectional=config.bi_utt_cell, variable_lengths=False, use_attn=config.enc_use_attn, embedding=self.embedding)
        self.c2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size + self.db_size + self.bs_size, config.y_size, config.k_size, is_lstm=False)
        self.z_embedding = nn.Linear(self.y_size * self.k_size, config.dec_cell_size, bias=False)
        self.gumbel_connector = nn_lib.GumbelConnector(config.use_gpu)
        if not self.simple_posterior:
            if self.contextual_posterior:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size, config.y_size, config.k_size, is_lstm=False)
            else:
                self.xc2z = nn_lib.Hidden2Discrete(self.utt_encoder.output_size, config.y_size, config.k_size, is_lstm=False)
        self.decoder = DecoderRNN(input_dropout_p=config.dropout, rnn_cell=config.dec_rnn_cell, input_size=config.embed_size, hidden_size=config.dec_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=False, vocab_size=self.vocab_size, use_attn=config.dec_use_attn, ctx_cell_size=config.dec_cell_size, attn_mode=config.dec_attn_mode, sys_id=self.bos_id, eos_id=self.eos_id, use_gpu=config.use_gpu, max_dec_len=config.max_dec_len, embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.cat_kl_loss = CatKLLoss()
        self.entropy_loss = Entropy()
        self.log_uniform_y = Variable(th.log(th.ones(1) / config.k_size))
        self.eye = Variable(th.eye(self.config.y_size).unsqueeze(0))
        self.beta = self.config.beta if hasattr(self.config, 'beta') else 0.0
        if self.use_gpu:
            self.log_uniform_y = self.log_uniform_y
            self.eye = self.eye

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl
        if self.config.use_mi:
            total_loss += loss.b_pr * self.beta
        if self.config.use_diversity:
            total_loss += loss.diversity
        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)
        db_label = self.np2var(data_feed['db'], FLOAT)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        if self.simple_posterior:
            logits_qy, log_qy = self.c2z(enc_last)
            sample_y = self.gumbel_connector(logits_qy, hard=mode == GEN)
            log_py = self.log_uniform_y
        else:
            logits_py, log_py = self.c2z(enc_last)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            if self.contextual_posterior:
                logits_qy, log_qy = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            else:
                logits_qy, log_qy = self.xc2z(x_h.squeeze(1))
            if mode == GEN or use_py is not None and use_py is True:
                sample_y = self.gumbel_connector(logits_py, hard=False)
            else:
                sample_y = self.gumbel_connector(logits_qy, hard=True)
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, (z_id)], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size, dec_inputs=dec_inputs, dec_init_state=dec_init_state, attn_context=attn_context, mode=mode, gen_type=gen_type, beam_size=self.config.beam_size)
        if mode == GEN:
            ret_dict['sample_z'] = sample_y
            ret_dict['log_qy'] = log_qy
            return ret_dict, labels
        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            avg_log_qy = th.exp(log_qy.view(-1, self.config.y_size, self.config.k_size))
            avg_log_qy = th.log(th.mean(avg_log_qy, dim=0) + 1e-15)
            b_pr = self.cat_kl_loss(avg_log_qy, self.log_uniform_y, batch_size, unit_average=True)
            mi = self.entropy_loss(avg_log_qy, unit_average=True) - self.entropy_loss(log_qy, unit_average=True)
            pi_kl = self.cat_kl_loss(log_qy, log_py, batch_size, unit_average=True)
            q_y = th.exp(log_qy).view(-1, self.config.y_size, self.config.k_size)
            p = th.pow(th.bmm(q_y, th.transpose(q_y, 1, 2)) - self.eye, 2)
            result['pi_kl'] = pi_kl
            result['diversity'] = th.mean(p)
            result['nll'] = self.nll(dec_outputs, labels)
            result['b_pr'] = b_pr
            result['mi'] = mi
            return result

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)
        db_label = self.np2var(data_feed['db'], FLOAT)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        if self.simple_posterior:
            logits_py, log_qy = self.c2z(enc_last)
        else:
            logits_py, log_qy = self.c2z(enc_last)
        qy = F.softmax(logits_py / temp, dim=1)
        log_qy = F.log_softmax(logits_py, dim=1)
        idx = th.multinomial(qy, 1).detach()
        logprob_sample_z = log_qy.gather(1, idx).view(-1, self.y_size)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        sample_y = cast_type(Variable(th.zeros(log_qy.size())), FLOAT, self.use_gpu)
        sample_y.scatter_(1, idx, 1.0)
        if self.config.dec_use_attn:
            z_embeddings = th.t(self.z_embedding.weight).split(self.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.config.y_size, self.config.k_size)
            for z_id in range(self.y_size):
                attn_context.append(th.mm(temp_sample_y[:, (z_id)], z_embeddings[z_id]).unsqueeze(1))
            attn_context = th.cat(attn_context, dim=1)
            dec_init_state = th.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.z_embedding(sample_y.view(1, -1, self.config.y_size * self.config.k_size))
            attn_context = None
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size, dec_init_state=dec_init_state, attn_context=attn_context, vocab=self.vocab, max_words=max_words, temp=0.1)
        return logprobs, outs, joint_logpz, sample_y


class SysPerfectBD2Gauss(BaseModel):

    def __init__(self, corpus, config):
        super(SysPerfectBD2Gauss, self).__init__(config)
        self.vocab = corpus.vocab
        self.vocab_dict = corpus.vocab_dict
        self.vocab_size = len(self.vocab)
        self.bos_id = self.vocab_dict[BOS]
        self.eos_id = self.vocab_dict[EOS]
        self.pad_id = self.vocab_dict[PAD]
        self.bs_size = corpus.bs_size
        self.db_size = corpus.db_size
        self.y_size = config.y_size
        self.simple_posterior = config.simple_posterior
        self.embedding = None
        self.utt_encoder = RnnUttEncoder(vocab_size=self.vocab_size, embedding_dim=config.embed_size, feat_size=0, goal_nhid=0, rnn_cell=config.utt_rnn_cell, utt_cell_size=config.utt_cell_size, num_layers=config.num_layers, input_dropout_p=config.dropout, output_dropout_p=config.dropout, bidirectional=config.bi_utt_cell, variable_lengths=False, use_attn=config.enc_use_attn, embedding=self.embedding)
        self.c2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size + self.db_size + self.bs_size, config.y_size, is_lstm=False)
        self.gauss_connector = nn_lib.GaussianConnector(self.use_gpu)
        self.z_embedding = nn.Linear(self.y_size, config.dec_cell_size)
        if not self.simple_posterior:
            self.xc2z = nn_lib.Hidden2Gaussian(self.utt_encoder.output_size * 2 + self.db_size + self.bs_size, config.y_size, is_lstm=False)
        self.decoder = DecoderRNN(input_dropout_p=config.dropout, rnn_cell=config.dec_rnn_cell, input_size=config.embed_size, hidden_size=config.dec_cell_size, num_layers=config.num_layers, output_dropout_p=config.dropout, bidirectional=False, vocab_size=self.vocab_size, use_attn=config.dec_use_attn, ctx_cell_size=config.dec_cell_size, attn_mode=config.dec_attn_mode, sys_id=self.bos_id, eos_id=self.eos_id, use_gpu=config.use_gpu, max_dec_len=config.max_dec_len, embedding=self.embedding)
        self.nll = NLLEntropy(self.pad_id, config.avg_type)
        self.gauss_kl = NormKLLoss(unit_average=True)
        self.zero = cast_type(th.zeros(1), FLOAT, self.use_gpu)

    def valid_loss(self, loss, batch_cnt=None):
        if self.simple_posterior:
            total_loss = loss.nll
            if self.config.use_pr > 0.0:
                total_loss += self.config.beta * loss.pi_kl
        else:
            total_loss = loss.nll + loss.pi_kl
        return total_loss

    def forward(self, data_feed, mode, clf=False, gen_type='greedy', use_py=None, return_latent=False):
        ctx_lens = data_feed['context_lens']
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        out_utts = self.np2var(data_feed['outputs'], LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)
        db_label = self.np2var(data_feed['db'], FLOAT)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        dec_inputs = out_utts[:, :-1]
        labels = out_utts[:, 1:].contiguous()
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        if self.simple_posterior:
            q_mu, q_logvar = self.c2z(enc_last)
            sample_z = self.gauss_connector(q_mu, q_logvar)
            p_mu, p_logvar = self.zero, self.zero
        else:
            p_mu, p_logvar = self.c2z(enc_last)
            x_h, _, _ = self.utt_encoder(out_utts.unsqueeze(1))
            q_mu, q_logvar = self.xc2z(th.cat([enc_last, x_h.squeeze(1)], dim=1))
            if mode == GEN or use_py:
                sample_z = self.gauss_connector(p_mu, p_logvar)
            else:
                sample_z = self.gauss_connector(q_mu, q_logvar)
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        dec_outputs, dec_hidden_state, ret_dict = self.decoder(batch_size=batch_size, dec_inputs=dec_inputs, dec_init_state=dec_init_state, attn_context=attn_context, mode=mode, gen_type=gen_type, beam_size=self.config.beam_size)
        if mode == GEN:
            ret_dict['sample_z'] = sample_z
            return ret_dict, labels
        else:
            result = Pack(nll=self.nll(dec_outputs, labels))
            pi_kl = self.gauss_kl(q_mu, q_logvar, p_mu, p_logvar)
            result['pi_kl'] = pi_kl
            result['nll'] = self.nll(dec_outputs, labels)
            return result

    def gaussian_logprob(self, mu, logvar, sample_z):
        var = th.exp(logvar)
        constant = float(-0.5 * np.log(2 * np.pi))
        logprob = constant - 0.5 * logvar - th.pow(mu - sample_z, 2) / (2.0 * var)
        return logprob

    def forward_rl(self, data_feed, max_words, temp=0.1):
        ctx_lens = data_feed['context_lens']
        short_ctx_utts = self.np2var(self.extract_short_ctx(data_feed['contexts'], ctx_lens), LONG)
        bs_label = self.np2var(data_feed['bs'], FLOAT)
        db_label = self.np2var(data_feed['db'], FLOAT)
        batch_size = len(ctx_lens)
        utt_summary, _, enc_outs = self.utt_encoder(short_ctx_utts.unsqueeze(1))
        enc_last = th.cat([bs_label, db_label, utt_summary.squeeze(1)], dim=1)
        p_mu, p_logvar = self.c2z(enc_last)
        sample_z = th.normal(p_mu, th.sqrt(th.exp(p_logvar))).detach()
        logprob_sample_z = self.gaussian_logprob(p_mu, self.zero, sample_z)
        joint_logpz = th.sum(logprob_sample_z, dim=1)
        dec_init_state = self.z_embedding(sample_z.unsqueeze(0))
        attn_context = None
        if self.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])
        logprobs, outs = self.decoder.forward_rl(batch_size=batch_size, dec_init_state=dec_init_state, attn_context=attn_context, vocab=self.vocab, max_words=max_words, temp=0.1)
        return logprobs, outs, joint_logpz, sample_z


class Hidden2Gaussian(nn.Module):

    def __init__(self, input_size, output_size, is_lstm=False, has_bias=True):
        super(Hidden2Gaussian, self).__init__()
        if is_lstm:
            self.mu_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar_h = nn.Linear(input_size, output_size, bias=has_bias)
            self.mu_c = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar_c = nn.Linear(input_size, output_size, bias=has_bias)
        else:
            self.mu = nn.Linear(input_size, output_size, bias=has_bias)
            self.logvar = nn.Linear(input_size, output_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h, c = inputs
            if h.dim() == 3:
                h = h.squeeze(0)
                c = c.squeeze(0)
            mu_h, mu_c = self.mu_h(h), self.mu_c(c)
            logvar_h, logvar_c = self.logvar_h(h), self.logvar_c(c)
            return mu_h + mu_c, logvar_h + logvar_c
        else:
            mu = self.mu(inputs)
            logvar = self.logvar(inputs)
            return mu, logvar


class Hidden2Discrete(nn.Module):

    def __init__(self, input_size, y_size, k_size, is_lstm=False, has_bias=True):
        super(Hidden2Discrete, self).__init__()
        self.y_size = y_size
        self.k_size = k_size
        latent_size = self.k_size * self.y_size
        if is_lstm:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)
            self.p_c = nn.Linear(input_size, latent_size, bias=has_bias)
        else:
            self.p_h = nn.Linear(input_size, latent_size, bias=has_bias)
        self.is_lstm = is_lstm

    def forward(self, inputs):
        """
        :param inputs: batch_size x input_size
        :return:
        """
        if self.is_lstm:
            h, c = inputs
            if h.dim() == 3:
                h = h.squeeze(0)
                c = c.squeeze(0)
            logits = self.p_h(h) + self.p_c(c)
        else:
            logits = self.p_h(inputs)
        logits = logits.view(-1, self.k_size)
        log_qy = F.log_softmax(logits, dim=1)
        return logits, log_qy


class GaussianConnector(nn.Module):

    def __init__(self, use_gpu):
        super(GaussianConnector, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, mu, logvar):
        """
        Sample a sample from a multivariate Gaussian distribution with a diagonal covariance matrix using the
        reparametrization trick.
        TODO: this should be better be a instance method in a Gaussian class.
        :param mu: a tensor of size [batch_size, variable_dim]. Batch_size can be None to support dynamic batching
        :param logvar: a tensor of size [batch_size, variable_dim]. Batch_size can be None.
        :return:
        """
        epsilon = th.randn(logvar.size())
        epsilon = cast_type(Variable(epsilon), FLOAT, self.use_gpu)
        std = th.exp(0.5 * logvar)
        z = mu + std * epsilon
        return z


class GumbelConnector(nn.Module):

    def __init__(self, use_gpu):
        super(GumbelConnector, self).__init__()
        self.use_gpu = use_gpu

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = th.rand(logits.size())
        sample = Variable(-th.log(-th.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps
        return F.softmax(y / temperature, dim=y.dim() - 1)

    def forward(self, logits, temperature=1.0, hard=False, return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :param return_max_id
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        _, y_hard = th.max(y, dim=1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(th.zeros(y.size())), FLOAT, self.use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BinaryNLLEntropy,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (FeatureProjecter,
     lambda: ([], {'input_dropout_p': 0.5, 'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (GaussianConnector,
     lambda: ([], {'use_gpu': False}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (GumbelConnector,
     lambda: ([], {'use_gpu': False}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Hidden2Discrete,
     lambda: ([], {'input_size': 4, 'y_size': 4, 'k_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Hidden2Gaussian,
     lambda: ([], {'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (IdentityConnector,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (NormKLLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (RnnContextEncoder,
     lambda: ([], {'n': 4, 'k': 4, 'nembed': 4, 'nhid': 4, 'init_range': 4, 'device_id': 0}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64)], {}),
     False),
    (SelectionClassifier,
     lambda: ([], {'selection_length': 4, 'input_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SelfAttn,
     lambda: ([], {'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (TaskMlpGoalEncoder,
     lambda: ([], {'goal_vocab_sizes': [4, 4], 'nhid': 4, 'init_range': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_snakeztc_NeuralDialog_LaRL(_paritybench_base):
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

