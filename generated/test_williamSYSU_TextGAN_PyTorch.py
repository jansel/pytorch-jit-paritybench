import sys
_module = sys.modules[__name__]
del sys
config = _module
instructor = _module
jsdgan_instructor = _module
leakgan_instructor = _module
maligan_instructor = _module
relgan_instructor = _module
sentigan_instructor = _module
seqgan_instructor = _module
instructor = _module
relgan_instructor = _module
main = _module
basic = _module
bleu = _module
clas_acc = _module
nll = _module
ppl = _module
JSDGAN_G = _module
LeakGAN_D = _module
LeakGAN_G = _module
MaliGAN_D = _module
MaliGAN_G = _module
Oracle = _module
RelGAN_D = _module
RelGAN_G = _module
SentiGAN_D = _module
SentiGAN_G = _module
SeqGAN_D = _module
SeqGAN_G = _module
discriminator = _module
generator = _module
relational_rnn_general = _module
run_jsdgan = _module
run_leakgan = _module
run_maligan = _module
run_relgan = _module
run_sentigan = _module
run_seqgan = _module
cat_data_loader = _module
data_loader = _module
data_utils = _module
helpers = _module
rollout = _module
text_process = _module
visualization = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import math


import time


import torch.autograd as autograd


from torch import nn


from time import strftime


from time import localtime


import logging


from time import gmtime


import copy


dis_num_filters = [200, 200, 200, 200]


goal_out_size = sum(dis_num_filters)


def truncated_normal_(tensor, mean=0, std=1):
    """
    Implemented by @ruotianluo
    See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


_global_config['gen_init'] = 4


_global_config['batch_size'] = 4


_global_config['start_letter'] = 4


class LeakGAN_G(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len,
        padding_idx, goal_size, step_size, gpu=False):
        super(LeakGAN_G, self).__init__()
        self.name = 'leakgan'
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.goal_size = goal_size
        self.goal_out_size = goal_out_size
        self.step_size = step_size
        self.gpu = gpu
        self.temperature = 1.5
        self.embeddings = nn.Embedding(vocab_size, embedding_dim,
            padding_idx=padding_idx)
        self.worker = nn.LSTM(embedding_dim, hidden_dim)
        self.manager = nn.LSTM(goal_out_size, hidden_dim)
        self.work2goal = nn.Linear(hidden_dim, vocab_size * goal_size)
        self.mana2goal = nn.Linear(hidden_dim, goal_out_size)
        self.goal2goal = nn.Linear(goal_out_size, goal_size, bias=False)
        self.goal_init = nn.Parameter(torch.rand((cfg.batch_size,
            goal_out_size)))
        self.init_params()

    def forward(self, idx, inp, work_hidden, mana_hidden, feature,
        real_goal, no_log=False, train=False):
        """
        Embeds input and sample on token at a time (seq_len = 1)

        :param idx: index of current token in sentence
        :param inp: [batch_size]
        :param work_hidden: 1 * batch_size * hidden_dim
        :param mana_hidden: 1 * batch_size * hidden_dim
        :param feature: 1 * batch_size * total_num_filters, feature of current sentence
        :param real_goal: batch_size * goal_out_size, real_goal in LeakGAN source code
        :param no_log: no log operation
        :param train: if train

        :return: out, cur_goal, work_hidden, mana_hidden
            - out: batch_size * vocab_size
            - cur_goal: batch_size * 1 * goal_out_size
        """
        emb = self.embeddings(inp).unsqueeze(0)
        mana_out, mana_hidden = self.manager(feature, mana_hidden)
        mana_out = self.mana2goal(mana_out.permute([1, 0, 2]))
        cur_goal = F.normalize(mana_out, dim=-1)
        _real_goal = self.goal2goal(real_goal)
        _real_goal = F.normalize(_real_goal, p=2, dim=-1).unsqueeze(-1)
        work_out, work_hidden = self.worker(emb, work_hidden)
        work_out = self.work2goal(work_out).view(-1, self.vocab_size, self.
            goal_size)
        out = torch.matmul(work_out, _real_goal).squeeze(-1)
        if idx > 1:
            if train:
                temperature = 1.0
            else:
                temperature = self.temperature
        else:
            temperature = self.temperature
        out = temperature * out
        if no_log:
            out = F.softmax(out, dim=-1)
        else:
            out = F.log_softmax(out, dim=-1)
        return out, cur_goal, work_hidden, mana_hidden

    def sample(self, num_samples, batch_size, dis, start_letter=cfg.
        start_letter, train=False):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return: samples: batch_size * max_seq_len
        """
        num_batch = (num_samples // batch_size + 1 if num_samples !=
            batch_size else 1)
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        fake_sentences = torch.zeros((batch_size, self.max_seq_len))
        for b in range(num_batch):
            leak_sample, _, _, _ = self.forward_leakgan(fake_sentences, dis,
                if_sample=True, no_log=False, start_letter=start_letter,
                train=False)
            assert leak_sample.shape == (batch_size, self.max_seq_len)
            samples[b * batch_size:(b + 1) * batch_size, :] = leak_sample
        samples = samples[:num_samples, :]
        return samples

    def pretrain_loss(self, target, dis, start_letter=cfg.start_letter):
        """
        Returns the pretrain_generator Loss for predicting target sequence.

        Inputs: target, dis, start_letter
            - target: batch_size * seq_len

        """
        batch_size, seq_len = target.size()
        _, feature_array, goal_array, leak_out_array = self.forward_leakgan(
            target, dis, if_sample=False, no_log=False, start_letter=
            start_letter)
        mana_cos_loss = self.manager_cos_loss(batch_size, feature_array,
            goal_array)
        manager_loss = -torch.sum(mana_cos_loss) / (batch_size * (seq_len //
            self.step_size))
        work_nll_loss = self.worker_nll_loss(target, leak_out_array)
        work_loss = torch.sum(work_nll_loss) / (batch_size * seq_len)
        return manager_loss, work_loss

    def adversarial_loss(self, target, rewards, dis, start_letter=cfg.
        start_letter):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/

        Inputs: target, rewards, dis, start_letter
            - target: batch_size * seq_len
            - rewards: batch_size * seq_len (discriminator rewards for each token)
        """
        batch_size, seq_len = target.size()
        _, feature_array, goal_array, leak_out_array = self.forward_leakgan(
            target, dis, if_sample=False, no_log=False, start_letter=
            start_letter, train=True)
        t0 = time.time()
        mana_cos_loss = self.manager_cos_loss(batch_size, feature_array,
            goal_array)
        mana_loss = -torch.sum(rewards * mana_cos_loss) / (batch_size * (
            seq_len // self.step_size))
        work_nll_loss = self.worker_nll_loss(target, leak_out_array)
        work_cos_reward = self.worker_cos_reward(feature_array, goal_array)
        work_loss = -torch.sum(work_nll_loss * work_cos_reward) / (batch_size *
            seq_len)
        return mana_loss, work_loss

    def manager_cos_loss(self, batch_size, feature_array, goal_array):
        """
        Get manager cosine distance loss

        :return cos_loss: batch_size * (seq_len / step_size)
        """
        sub_feature = torch.zeros(batch_size, self.max_seq_len // self.
            step_size, self.goal_out_size)
        real_goal = torch.zeros(batch_size, self.max_seq_len // self.
            step_size, self.goal_out_size)
        for i in range(self.max_seq_len // self.step_size):
            idx = i * self.step_size
            sub_feature[:, (i), :] = feature_array[:, (idx + self.step_size), :
                ] - feature_array[:, (idx), :]
            if i == 0:
                real_goal[:, (i), :] = self.goal_init[:batch_size, :]
            else:
                idx = (i - 1) * self.step_size + 1
                real_goal[:, (i), :] = torch.sum(goal_array[:, idx:idx + 4,
                    :], dim=1)
        sub_feature = F.normalize(sub_feature, p=2, dim=-1)
        real_goal = F.normalize(real_goal, p=2, dim=-1)
        cos_loss = F.cosine_similarity(sub_feature, real_goal, dim=-1)
        return cos_loss

    def worker_nll_loss(self, target, leak_out_array):
        """
        Get NLL loss for worker

        :return loss: batch_size * seq_len
        """
        loss_fn = nn.NLLLoss(reduction='none')
        loss = loss_fn(leak_out_array.permute([0, 2, 1]), target)
        return loss

    def worker_cos_reward(self, feature_array, goal_array):
        """
        Get reward for worker (cosine distance)

        :return: cos_loss: batch_size * seq_len
        """
        for i in range(int(self.max_seq_len / self.step_size)):
            real_feature = feature_array[:, (i * self.step_size), :].unsqueeze(
                1).expand((-1, self.step_size, -1))
            feature_array[:, i * self.step_size:(i + 1) * self.step_size, :
                ] = real_feature
            if i > 0:
                sum_goal = torch.sum(goal_array[:, (i - 1) * self.step_size
                    :i * self.step_size, :], dim=1, keepdim=True)
            else:
                sum_goal = goal_array[:, (0), :].unsqueeze(1)
            goal_array[:, i * self.step_size:(i + 1) * self.step_size, :
                ] = sum_goal.expand((-1, self.step_size, -1))
        offset_feature = feature_array[:, 1:, :]
        goal_array = goal_array[:, :self.max_seq_len, :]
        sub_feature = offset_feature - goal_array
        sub_feature = F.normalize(sub_feature, p=2, dim=-1)
        all_goal = F.normalize(goal_array, p=2, dim=-1)
        cos_loss = F.cosine_similarity(sub_feature, all_goal, dim=-1)
        return cos_loss

    def forward_leakgan(self, sentences, dis, if_sample, no_log=False,
        start_letter=cfg.start_letter, train=False):
        """
        Get all feature and goals according to given sentences
        :param sentences: batch_size * max_seq_len, not include start token
        :param dis: discriminator model
        :param if_sample: if use to sample token
        :param no_log: if use log operation
        :param start_letter:
        :param train: if use temperature parameter
        :return samples, feature_array, goal_array, leak_out_array:
            - samples: batch_size * max_seq_len
            - feature_array: batch_size * (max_seq_len + 1) * total_num_filter
            - goal_array: batch_size * (max_seq_len + 1) * goal_out_size
            - leak_out_array: batch_size * max_seq_len * vocab_size
        """
        batch_size, seq_len = sentences.size()
        feature_array = torch.zeros((batch_size, seq_len + 1, self.
            goal_out_size))
        goal_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))
        leak_out_array = torch.zeros((batch_size, seq_len + 1, self.vocab_size)
            )
        samples = torch.zeros(batch_size, seq_len + 1).long()
        work_hidden = self.init_hidden(batch_size)
        mana_hidden = self.init_hidden(batch_size)
        leak_inp = torch.LongTensor([start_letter] * batch_size)
        real_goal = self.goal_init[:batch_size, :]
        if self.gpu:
            feature_array = feature_array
            goal_array = goal_array
            leak_out_array = leak_out_array
        goal_array[:, (0), :] = real_goal
        for i in range(seq_len + 1):
            if if_sample:
                dis_inp = samples[:, :seq_len]
            else:
                dis_inp = torch.zeros(batch_size, seq_len).long()
                if i > 0:
                    dis_inp[:, :i] = sentences[:, :i]
                    leak_inp = sentences[:, (i - 1)]
            if self.gpu:
                dis_inp = dis_inp
                leak_inp = leak_inp
            feature = dis.get_feature(dis_inp).unsqueeze(0)
            feature_array[:, (i), :] = feature.squeeze(0)
            out, cur_goal, work_hidden, mana_hidden = self.forward(i,
                leak_inp, work_hidden, mana_hidden, feature, real_goal,
                no_log=no_log, train=train)
            leak_out_array[:, (i), :] = out
            goal_array[:, (i), :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.goal_init[:batch_size, :]
            if not no_log:
                out = torch.exp(out)
            out = torch.multinomial(out, 1).view(-1)
            samples[:, (i)] = out.data
            leak_inp = out
        samples = samples[:, :seq_len]
        leak_out_array = leak_out_array[:, :seq_len, :]
        return samples, feature_array, goal_array, leak_out_array

    def batchNLLLoss(self, target, dis, start_letter=cfg.start_letter):
        _, _, _, leak_out_array = self.forward_leakgan(target, dis,
            if_sample=False, no_log=False, start_letter=start_letter)
        nll_loss = torch.mean(self.worker_nll_loss(target, leak_out_array))
        return nll_loss

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)
        if self.gpu:
            return h, c
        else:
            return h, c

    def init_goal(self, batch_size):
        goal = torch.rand((batch_size, self.goal_out_size)).normal_(std=0.1)
        goal = nn.Parameter(goal)
        if self.gpu:
            return goal
        else:
            return goal

    def split_params(self):
        mana_params = list()
        work_params = list()
        mana_params += list(self.manager.parameters())
        mana_params += list(self.mana2goal.parameters())
        mana_params.append(self.goal_init)
        work_params += list(self.embeddings.parameters())
        work_params += list(self.worker.parameters())
        work_params += list(self.work2goal.parameters())
        work_params += list(self.goal2goal.parameters())
        return mana_params, work_params

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.gen_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)


_global_config['dis_init'] = 4


class CNNDiscriminator(nn.Module):

    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters,
        padding_idx, gpu=False, dropout=0.2):
        super(CNNDiscriminator, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        self.gpu = gpu
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=
            padding_idx)
        self.convs = nn.ModuleList([nn.Conv2d(1, n, (f, embed_dim)) for n,
            f in zip(num_filters, filter_sizes)])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)
        self.init_params()

    def forward(self, inp):
        """
        Get final predictions of discriminator
        :param inp: batch_size * seq_len
        :return: pred: batch_size * 2
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))
        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        pred = torch.cat(pools, 1)
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1.0 - torch.
            sigmoid(highway)) * pred
        return pred

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.dis_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.dis_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.dis_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)


class GRUDiscriminator(nn.Module):

    def __init__(self, embedding_dim, vocab_size, hidden_dim, feature_dim,
        max_seq_len, padding_idx, gpu=False, dropout=0.2):
        super(GRUDiscriminator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.gpu = gpu
        self.embeddings = nn.Embedding(vocab_size, embedding_dim,
            padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2,
            bidirectional=True, dropout=dropout)
        self.gru2hidden = nn.Linear(2 * 2 * hidden_dim, feature_dim)
        self.feature2out = nn.Linear(feature_dim, 2)
        self.dropout = nn.Dropout(dropout)
        self.init_params()

    def init_hidden(self, batch_size):
        h = autograd.Variable(torch.zeros(2 * 2 * 1, batch_size, self.
            hidden_dim))
        if self.gpu:
            return h
        else:
            return h

    def forward(self, inp):
        """
        Get final feature of discriminator
        :param inp: batch_size * seq_len
        :return pred: batch_size * 2
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))
        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        hidden = self.init_hidden(inp.size(0))
        emb = self.embeddings(input)
        emb = emb.permute(1, 0, 2)
        _, hidden = self.gru(emb, hidden)
        hidden = hidden.permute(1, 0, 2).contiguous()
        out = self.gru2hidden(hidden.view(-1, 4 * self.hidden_dim))
        feature = torch.tanh(out)
        return feature

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.dis_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.dis_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.dis_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)


class LSTMGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len,
        padding_idx, gpu=False):
        super(LSTMGenerator, self).__init__()
        self.name = 'vanilla'
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu
        self.temperature = 1.0
        self.embeddings = nn.Embedding(vocab_size, embedding_dim,
            padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.init_params()

    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        emb = self.embeddings(inp)
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.lstm2out(out)
        pred = self.softmax(out)
        if need_hidden:
            return pred, hidden
        else:
            return pred

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = (num_samples // batch_size + 1 if num_samples !=
            batch_size else 1)
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp
            for i in range(self.max_seq_len):
                out, hidden = self.forward(inp, hidden, need_hidden=True)
                next_token = torch.multinomial(torch.exp(out), 1)
                samples[b * batch_size:(b + 1) * batch_size, (i)
                    ] = next_token.view(-1)
                inp = next_token.view(-1)
        samples = samples[:num_samples]
        return samples

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.gen_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)

    def init_oracle(self):
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_hidden(self, batch_size=cfg.batch_size):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)
        if self.gpu:
            return h, c
        else:
            return h, c


class RelationalMemory(nn.Module):
    """
    Constructs a `RelationalMemory` object.
    This class is same as the RMC from relational_rnn_models.py, but without language modeling-specific variables.
    Args:
      mem_slots: The total number of memory slots to use.
      head_size: The size of an attention head.
      input_size: The size of input per step. i.e. the dimension of each input vector
      num_heads: The number of attention heads to use. Defaults to 1.
      num_blocks: Number of times to compute attention per time step. Defaults
        to 1.
      forget_bias: Bias to use for the forget gate, assuming we are using
        some form of gating. Defaults to 1.
      input_bias: Bias to use for the input gate, assuming we are using
        some form of gating. Defaults to 0.
      gate_style: Whether to use per-element gating ('unit'),
        per-memory slot gating ('memory'), or no gating at all (None).
        Defaults to `unit`.
      attention_mlp_layers: Number of layers to use in the post-attention
        MLP. Defaults to 2.
      key_size: Size of vector to use for key & query vectors in the attention
        computation. Defaults to None, in which case we use `head_size`.

      # NEW flag for this class
      return_all_outputs: Whether the model returns outputs for each step (like seq2seq) or only the final output.
    Raises:
      ValueError: gate_style not one of [None, 'memory', 'unit'].
      ValueError: num_blocks is < 1.
      ValueError: attention_mlp_layers is < 1.
    """

    def __init__(self, mem_slots, head_size, input_size, num_heads=1,
        num_blocks=1, forget_bias=1.0, input_bias=0.0, gate_style='unit',
        attention_mlp_layers=2, key_size=None, return_all_outputs=False):
        super(RelationalMemory, self).__init__()
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.mem_size = self.head_size * self.num_heads
        self.mem_slots_plus_input = self.mem_slots + 1
        if num_blocks < 1:
            raise ValueError('num_blocks must be >=1. Got: {}.'.format(
                num_blocks))
        self.num_blocks = num_blocks
        if gate_style not in ['unit', 'memory', None]:
            raise ValueError(
                "gate_style must be one of ['unit', 'memory', None]. got: {}."
                .format(gate_style))
        self.gate_style = gate_style
        if attention_mlp_layers < 1:
            raise ValueError('attention_mlp_layers must be >= 1. Got: {}.'.
                format(attention_mlp_layers))
        self.attention_mlp_layers = attention_mlp_layers
        self.key_size = key_size if key_size else self.head_size
        self.value_size = self.head_size
        self.qkv_size = 2 * self.key_size + self.value_size
        self.total_qkv_size = self.qkv_size * self.num_heads
        self.qkv_projector = nn.Linear(self.mem_size, self.total_qkv_size)
        self.qkv_layernorm = nn.LayerNorm([self.mem_slots_plus_input, self.
            total_qkv_size])
        self.attention_mlp = nn.ModuleList([nn.Linear(self.mem_size, self.
            mem_size)] * self.attention_mlp_layers)
        self.attended_memory_layernorm = nn.LayerNorm([self.
            mem_slots_plus_input, self.mem_size])
        self.attended_memory_layernorm2 = nn.LayerNorm([self.
            mem_slots_plus_input, self.mem_size])
        self.input_size = input_size
        self.input_projector = nn.Linear(self.input_size, self.mem_size)
        self.num_gates = 2 * self.calculate_gate_size()
        self.input_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        self.memory_gate_projector = nn.Linear(self.mem_size, self.num_gates)
        self.forget_bias = nn.Parameter(torch.tensor(forget_bias, dtype=
            torch.float32))
        self.input_bias = nn.Parameter(torch.tensor(input_bias, dtype=torch
            .float32))
        self.return_all_outputs = return_all_outputs

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def initial_state(self, batch_size, trainable=False):
        """
        Creates the initial memory.
        We should ensure each row of the memory is initialized to be unique,
        so initialize the matrix to be the identity. We then pad or truncate
        as necessary so that init_state is of size
        (batch_size, self.mem_slots, self.mem_size).
        Args:
          batch_size: The size of the batch.
          trainable: Whether the initial state is trainable. This is always True.
        Returns:
          init_state: A truncated or padded matrix of size
            (batch_size, self.mem_slots, self.mem_size).
        """
        init_state = torch.stack([torch.eye(self.mem_slots) for _ in range(
            batch_size)])
        if self.mem_size > self.mem_slots:
            difference = self.mem_size - self.mem_slots
            pad = torch.zeros((batch_size, self.mem_slots, difference))
            init_state = torch.cat([init_state, pad], -1)
        elif self.mem_size < self.mem_slots:
            init_state = init_state[:, :, :self.mem_size]
        return init_state

    def multihead_attention(self, memory):
        """
        Perform multi-head attention from 'Attention is All You Need'.
        Implementation of the attention mechanism from
        https://arxiv.org/abs/1706.03762.
        Args:
          memory: Memory tensor to perform attention on.
        Returns:
          new_memory: New memory tensor.
        """
        qkv = self.qkv_projector(memory)
        qkv = self.qkv_layernorm(qkv)
        mem_slots = memory.shape[1]
        qkv_reshape = qkv.view(qkv.shape[0], mem_slots, self.num_heads,
            self.qkv_size)
        qkv_transpose = qkv_reshape.permute(0, 2, 1, 3)
        q, k, v = torch.split(qkv_transpose, [self.key_size, self.key_size,
            self.value_size], -1)
        q *= self.key_size ** -0.5
        dot_product = torch.matmul(q, k.permute(0, 1, 3, 2))
        weights = F.softmax(dot_product, dim=-1)
        output = torch.matmul(weights, v)
        output_transpose = output.permute(0, 2, 1, 3).contiguous()
        new_memory = output_transpose.view((output_transpose.shape[0],
            output_transpose.shape[1], -1))
        return new_memory

    @property
    def state_size(self):
        return [self.mem_slots, self.mem_size]

    @property
    def output_size(self):
        return self.mem_slots * self.mem_size

    def calculate_gate_size(self):
        """
        Calculate the gate size from the gate_style.
        Returns:
          The per sample, per head parameter size of each gate.
        """
        if self.gate_style == 'unit':
            return self.mem_size
        elif self.gate_style == 'memory':
            return 1
        else:
            return 0

    def create_gates(self, inputs, memory):
        """
        Create input and forget gates for this step using `inputs` and `memory`.
        Args:
          inputs: Tensor input.
          memory: The current state of memory.
        Returns:
          input_gate: A LSTM-like insert gate.
          forget_gate: A LSTM-like forget gate.
        """
        memory = torch.tanh(memory)
        if len(inputs.shape) == 3:
            if inputs.shape[1] > 1:
                raise ValueError(
                    'input seq length is larger than 1. create_gate function is meant to be called for each step, with input seq length of 1'
                    )
            inputs = inputs.view(inputs.shape[0], -1)
            gate_inputs = self.input_gate_projector(inputs)
            gate_inputs = gate_inputs.unsqueeze(dim=1)
            gate_memory = self.memory_gate_projector(memory)
        else:
            raise ValueError(
                'input shape of create_gate function is 2, expects 3')
        gates = gate_memory + gate_inputs
        gates = torch.split(gates, split_size_or_sections=int(gates.shape[2
            ] / 2), dim=2)
        input_gate, forget_gate = gates
        assert input_gate.shape[2] == forget_gate.shape[2]
        input_gate = torch.sigmoid(input_gate + self.input_bias)
        forget_gate = torch.sigmoid(forget_gate + self.forget_bias)
        return input_gate, forget_gate

    def attend_over_memory(self, memory):
        """
        Perform multiheaded attention over `memory`.
            Args:
              memory: Current relational memory.
            Returns:
              The attended-over memory.
        """
        for _ in range(self.num_blocks):
            attended_memory = self.multihead_attention(memory)
            memory = self.attended_memory_layernorm(memory + attended_memory)
            attention_mlp = memory
            for i, l in enumerate(self.attention_mlp):
                attention_mlp = self.attention_mlp[i](attention_mlp)
                attention_mlp = F.relu(attention_mlp)
            memory = self.attended_memory_layernorm2(memory + attention_mlp)
        return memory

    def forward_step(self, inputs, memory, treat_input_as_matrix=False):
        """
        Forward step of the relational memory core.
        Args:
          inputs: Tensor input.
          memory: Memory output from the previous time step.
          treat_input_as_matrix: Optional, whether to treat `input` as a sequence
            of matrices. Default to False, in which case the input is flattened
            into a vector.
        Returns:
          output: This time step's output.
          next_memory: The next version of memory to use.
        """
        if treat_input_as_matrix:
            inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
            inputs_reshape = self.input_projector(inputs)
        else:
            inputs = inputs.view(inputs.shape[0], -1)
            inputs = self.input_projector(inputs)
            inputs_reshape = inputs.unsqueeze(dim=1)
        memory_plus_input = torch.cat([memory, inputs_reshape], dim=1)
        next_memory = self.attend_over_memory(memory_plus_input)
        n = inputs_reshape.shape[1]
        next_memory = next_memory[:, :-n, :]
        if self.gate_style == 'unit' or self.gate_style == 'memory':
            input_gate, forget_gate = self.create_gates(inputs_reshape, memory)
            next_memory = input_gate * torch.tanh(next_memory)
            next_memory += forget_gate * memory
        output = next_memory.view(next_memory.shape[0], -1)
        return output, next_memory

    def forward(self, inputs, memory, treat_input_as_matrix=False):
        logit = 0
        logits = []
        for idx_step in range(inputs.shape[1]):
            logit, memory = self.forward_step(inputs[:, (idx_step)], memory)
            logits.append(logit.unsqueeze(1))
        logits = torch.cat(logits, dim=1)
        if self.return_all_outputs:
            return logits, memory
        else:
            return logit.unsqueeze(1), memory


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_williamSYSU_TextGAN_PyTorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(RelationalMemory(*[], **{'mem_slots': 4, 'head_size': 4, 'input_size': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

