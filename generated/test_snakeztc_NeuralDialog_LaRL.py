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


import random


import itertools


import copy


import re


import logging


import torch


from torch import optim


import torch.nn as nn


from torch.autograd import Variable


import numpy as np


import torch.nn.functional as F


import torch.nn.init


import torch.optim as optim


import torch as th


from torch.nn.modules.loss import _Loss


from torch.nn.modules.module import _addindent


class CudaModule(nn.Module):
    """A helper to run a module on a particular device using CUDA."""

    def __init__(self, device_id):
        super(CudaModule, self).__init__()
        self.device_id = device_id

    def to_device(self, m):
        if self.device_id is not None:
            return m
        return m


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
            return optim.Adam(filter(lambda p: p.requires_grad, self.
                parameters()), lr=config.init_lr, weight_decay=config.l2_norm)
        elif config.op == 'sgd':
            None
            return optim.SGD(self.parameters(), lr=config.init_lr, momentum
                =config.momentum)
        elif config.op == 'rmsprop':
            None
            return optim.RMSprop(self.parameters(), lr=config.init_lr,
                momentum=config.momentum)

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
            return optim.SGD(params, lr=config.fine_tune_lr, momentum=
                config.fine_tune_momentum)
        elif config.fine_tune_op == 'rmsprop':
            None
            return optim.RMSprop(params, lr=config.fine_tune_lr, momentum=
                config.fine_tune_momentum)

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
            loss = F.nll_loss(pred, target, size_average=False,
                ignore_index=self.padding_idx)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(pred, target, size_average=False,
                ignore_index=self.padding_idx)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(pred, target, ignore_index=self.padding_idx,
                reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = th.sum(loss, dim=1)
            word_cnt = th.sum(th.sign(labels), dim=1).float()
            loss = loss / word_cnt
            loss = th.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(pred, target, size_average=True, ignore_index
                =self.padding_idx)
        else:
            raise ValueError('Unknown average type')
        return loss


class NLLEntropy4CLF(_Loss):

    def __init__(self, dictionary, bad_tokens=['<disconnect>', '<disagree>'
        ], reduction='elementwise_mean'):
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

    def __init__(self, dictionary, corpus, np2var, bad_tokens=[
        '<disconnect>', '<disagree>']):
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
                choices_logits.append(th.gather(sel_outs[i], 0, idxs_var).
                    unsqueeze(1))
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
        loss = F.binary_cross_entropy_with_logits(net_output, label_output,
            size_average=self.size_average)
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

    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size,
        num_layers, output_dropout_p, bidirectional):
        super(BaseRNN, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError('Unsupported RNN Cell Type: {0}'.format(rnn_cell))
        self.rnn = self.rnn_cell(input_size=input_size, hidden_size=
            hidden_size, num_layers=num_layers, batch_first=True, dropout=
            output_dropout_p, bidirectional=bidirectional)
        if rnn_cell.lower() == 'lstm':
            for names in self.rnn._all_weights:
                for name in filter(lambda n: 'bias' in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.0)


class FeatureProjecter(nn.Module):

    def __init__(self, input_dropout_p, input_size, output_size):
        super(FeatureProjecter, self).__init__()
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.sel_encoder = nn.Sequential(nn.Linear(input_size, output_size),
            nn.Tanh())

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
        outs = [decoder.forward(proj_outs).unsqueeze(1) for decoder in self
            .sel_decoders]
        outs = th.cat(outs, 1)
        return outs


class Attention(nn.Module):

    def __init__(self, dec_cell_size, ctx_cell_size, attn_mode, project):
        super(Attention, self).__init__()
        self.dec_cell_size = dec_cell_size
        self.ctx_cell_size = ctx_cell_size
        self.attn_mode = attn_mode
        if project:
            self.linear_out = nn.Linear(dec_cell_size + ctx_cell_size,
                dec_cell_size)
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
            tiled_output = mapped_output.unsqueeze(2).repeat(1, 1,
                max_ctx_len, 1)
            tiled_attn = mapped_attn.unsqueeze(1)
            fc1 = F.tanh(tiled_output + tiled_attn)
            attn = self.query_w(fc1).squeeze(-1)
        else:
            raise ValueError('Unknown attention mode')
        attn = F.softmax(attn.view(-1, max_ctx_len), dim=1).view(batch_size,
            -1, max_ctx_len)
        mix = th.bmm(attn, context)
        combined = th.cat((mix, output), dim=2)
        if self.linear_out is None:
            return combined, attn
        else:
            output = F.tanh(self.linear_out(combined.view(-1, self.
                dec_cell_size + self.ctx_cell_size))).view(batch_size, -1,
                self.dec_cell_size)
            return output, attn


class EncoderRNN(BaseRNN):

    def __init__(self, input_dropout_p, rnn_cell, input_size, hidden_size,
        num_layers, output_dropout_p, bidirectional, variable_lengths):
        super(EncoderRNN, self).__init__(input_dropout_p=input_dropout_p,
            rnn_cell=rnn_cell, input_size=input_size, hidden_size=
            hidden_size, num_layers=num_layers, output_dropout_p=
            output_dropout_p, bidirectional=bidirectional)
        self.variable_lengths = variable_lengths
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(self, input_var, init_state=None, input_lengths=None, goals
        =None):
        if goals is not None:
            batch_size, max_ctx_len, ctx_nhid = input_var.size()
            goals = goals.view(goals.size(0), 1, goals.size(1))
            goals_rep = goals.repeat(1, max_ctx_len, 1).view(batch_size,
                max_ctx_len, -1)
            input_var = th.cat([input_var, goals_rep], dim=2)
        embedded = self.input_dropout(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded,
                input_lengths, batch_first=True)
        if init_state is not None:
            output, hidden = self.rnn(embedded, init_state)
        else:
            output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output,
                batch_first=True)
        return output, hidden


class RnnUttEncoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, feat_size, goal_nhid,
        rnn_cell, utt_cell_size, num_layers, input_dropout_p,
        output_dropout_p, bidirectional, variable_lengths, use_attn,
        embedding=None):
        super(RnnUttEncoder, self).__init__()
        if embedding is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size,
                embedding_dim=embedding_dim)
        else:
            self.embedding = embedding
        self.rnn = EncoderRNN(input_dropout_p=input_dropout_p, rnn_cell=
            rnn_cell, input_size=embedding_dim + feat_size + goal_nhid,
            hidden_size=utt_cell_size, num_layers=num_layers,
            output_dropout_p=output_dropout_p, bidirectional=bidirectional,
            variable_lengths=variable_lengths)
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
            goals_rep = goals.repeat(1, max_ctx_len, max_utt_len, 1).view(
                batch_size * max_ctx_len, max_utt_len, -1)
            word_embeddings = th.cat([word_embeddings, goals_rep], dim=2)
        enc_outs, enc_last = self.rnn(word_embeddings, init_state=init_state)
        if self.use_attn:
            fc1 = th.tanh(self.key_w(enc_outs))
            attn = self.query(fc1).squeeze(2)
            attn = F.softmax(attn, attn.dim() - 1)
            attn = attn * flat_mask
            attn = (attn / (th.sum(attn, dim=1, keepdim=True) + 1e-10)
                ).unsqueeze(2)
            utt_embedded = attn * enc_outs
            utt_embedded = th.sum(utt_embedded, dim=1)
        else:
            attn = None
            utt_embedded = enc_last.transpose(0, 1).contiguous()
            utt_embedded = utt_embedded.view(-1, self.output_size)
        utt_embedded = utt_embedded.view(batch_size, max_ctx_len, self.
            output_size)
        return utt_embedded, word_embeddings.contiguous().view(batch_size, 
            max_ctx_len * max_utt_len, -1), enc_outs.contiguous().view(
            batch_size, max_ctx_len * max_utt_len, -1)


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
        outs = [encoder.forward(goal) for goal, encoder in zip(goals_list,
            self.encoder)]
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


class IdentityConnector(nn.Module):

    def __init(self):
        super(IdentityConnector, self).__init__()

    def forward(self, hidden_state):
        return hidden_state


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
            return new_h.view(1, -1, self.output_size), new_c.view(1, -1,
                self.output_size)
        else:
            num_layer = hidden_state.size()[0]
            new_s = self.fc(hidden_state.view(-1, self.hidden_size * num_layer)
                )
            new_s = new_s.view(1, -1, self.output_size)
            return new_s


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

    def __init__(self, input_size, y_size, k_size, is_lstm=False, has_bias=True
        ):
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

    def forward(self, logits, temperature=1.0, hard=False, return_max_id=False
        ):
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
            y_onehot = cast_type(Variable(th.zeros(y.size())), FLOAT, self.
                use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_snakeztc_NeuralDialog_LaRL(_paritybench_base):
    pass
    def test_000(self):
        self._check(BinaryNLLEntropy(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(FeatureProjecter(*[], **{'input_dropout_p': 0.5, 'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(GaussianConnector(*[], **{'use_gpu': False}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(GumbelConnector(*[], **{'use_gpu': False}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(Hidden2Discrete(*[], **{'input_size': 4, 'y_size': 4, 'k_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(Hidden2Gaussian(*[], **{'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(IdentityConnector(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(NormKLLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(SelectionClassifier(*[], **{'selection_length': 4, 'input_size': 4, 'output_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(SelfAttn(*[], **{'hidden_size': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_010(self):
        self._check(TaskMlpGoalEncoder(*[], **{'goal_vocab_sizes': [4, 4], 'nhid': 4, 'init_range': 4}), [torch.rand([4, 4, 4, 4])], {})

