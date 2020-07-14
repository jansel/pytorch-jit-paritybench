import sys
_module = sys.modules[__name__]
del sys
eval_D = _module
eval_G = _module
eval_G_DIS = _module
eval_all = _module
sample_dialog = _module
misc = _module
dataLoader = _module
encoder_Q = _module
encoder_QI = _module
encoder_QIH = _module
model = _module
netG = _module
share_Linear = _module
utils = _module
download = _module
prepro = _module
train_D = _module
train_G = _module
train_all = _module

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


import time


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import torchvision.transforms as transforms


import torchvision.utils as vutils


from torch.autograd import Variable


import torch.utils.data as data


import math


from torch.nn.parameter import Parameter


from torch.nn import Module


class _netE(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ninp, nhid, nlayers, dropout, img_feat_size):
        super(_netE, self).__init__()
        self.d = dropout
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.nhid = nhid
        self.ninp = ninp
        self.img_embed = nn.Linear(img_feat_size, nhid)
        self.ques_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.his_rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.Wq_1 = nn.Linear(self.nhid, self.nhid)
        self.Wh_1 = nn.Linear(self.nhid, self.nhid)
        self.Wa_1 = nn.Linear(self.nhid, 1)
        self.Wq_2 = nn.Linear(self.nhid, self.nhid)
        self.Wh_2 = nn.Linear(self.nhid, self.nhid)
        self.Wi_2 = nn.Linear(self.nhid, self.nhid)
        self.Wa_2 = nn.Linear(self.nhid, 1)
        self.fc1 = nn.Linear(self.nhid * 3, self.ninp)

    def forward(self, ques_emb, his_emb, img_raw, ques_hidden, his_hidden, rnd):
        img_emb = F.tanh(self.img_embed(img_raw))
        ques_feat, ques_hidden = self.ques_rnn(ques_emb, ques_hidden)
        ques_feat = ques_feat[-1]
        his_feat, his_hidden = self.his_rnn(his_emb, his_hidden)
        his_feat = his_feat[-1]
        ques_emb_1 = self.Wq_1(ques_feat).view(-1, 1, self.nhid)
        his_emb_1 = self.Wh_1(his_feat).view(-1, rnd, self.nhid)
        atten_emb_1 = F.tanh(his_emb_1 + ques_emb_1.expand_as(his_emb_1))
        his_atten_weight = F.softmax(self.Wa_1(F.dropout(atten_emb_1, self.d, training=self.training).view(-1, self.nhid)).view(-1, rnd))
        his_attn_feat = torch.bmm(his_atten_weight.view(-1, 1, rnd), his_feat.view(-1, rnd, self.nhid))
        his_attn_feat = his_attn_feat.view(-1, self.nhid)
        ques_emb_2 = self.Wq_2(ques_feat).view(-1, 1, self.nhid)
        his_emb_2 = self.Wh_2(his_attn_feat).view(-1, 1, self.nhid)
        img_emb_2 = self.Wi_2(img_emb).view(-1, 49, self.nhid)
        atten_emb_2 = F.tanh(img_emb_2 + ques_emb_2.expand_as(img_emb_2) + his_emb_2.expand_as(img_emb_2))
        img_atten_weight = F.softmax(self.Wa_2(F.dropout(atten_emb_2, self.d, training=self.training).view(-1, self.nhid)).view(-1, 49))
        img_attn_feat = torch.bmm(img_atten_weight.view(-1, 1, 49), img_emb.view(-1, 49, self.nhid))
        concat_feat = torch.cat((ques_feat, his_attn_feat.view(-1, self.nhid), img_attn_feat.view(-1, self.nhid)), 1)
        encoder_feat = F.tanh(self.fc1(F.dropout(concat_feat, self.d, training=self.training)))
        return encoder_feat, ques_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class share_Linear(Module):
    """Applies a linear transformation to the incoming data: :math:`y = Ax + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
    Shape:
        - Input: :math:`(N, in\\_features)`
        - Output: :math:`(N, out\\_features)`
    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, weight):
        super(share_Linear, self).__init__()
        self.in_features = weight.size(0)
        self.out_features = weight.size(1)
        self.weight = weight.t()
        self.register_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class _netW(nn.Module):

    def __init__(self, ntoken, ninp, dropout):
        super(_netW, self).__init__()
        self.word_embed = nn.Embedding(ntoken + 1, ninp)
        self.Linear = share_Linear(self.word_embed.weight)
        self.init_weights()
        self.d = dropout

    def init_weights(self):
        initrange = 0.1
        self.word_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, format='index'):
        if format == 'onehot':
            out = F.dropout(self.Linear(input), self.d, training=self.training)
        elif format == 'index':
            out = F.dropout(self.word_embed(input), self.d, training=self.training)
        return out

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class _netD(nn.Module):
    """
    Given the real/wrong/fake answer, use a RNN (LSTM) to embed the answer.
    """

    def __init__(self, rnn_type, ninp, nhid, nlayers, ntoken, dropout):
        super(_netD, self).__init__()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntoken = ntoken
        self.ninp = ninp
        self.d = dropout
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        self.W1 = nn.Linear(self.nhid, self.nhid)
        self.W2 = nn.Linear(self.nhid, 1)
        self.fc = nn.Linear(nhid, ninp)

    def forward(self, input_feat, idx, hidden, vocab_size):
        output, _ = self.rnn(input_feat, hidden)
        mask = idx.data.eq(0)
        mask[idx.data == vocab_size] = 1
        if isinstance(input_feat, Variable):
            mask = Variable(mask, volatile=input_feat.volatile)
        atten = self.W2(F.dropout(F.tanh(self.W1(output.view(-1, self.nhid))), self.d, training=self.training)).view(idx.size())
        atten.masked_fill_(mask, -99999)
        weight = F.softmax(atten.t()).view(-1, 1, idx.size(0))
        feat = torch.bmm(weight, output.transpose(0, 1)).view(-1, self.nhid)
        feat = F.dropout(feat, self.d, training=self.training)
        transform_output = F.tanh(self.fc(feat))
        return transform_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class LMCriterion(nn.Module):

    def __init__(self):
        super(LMCriterion, self).__init__()

    def forward(self, input, target):
        logprob_select = torch.gather(input, 1, target)
        mask = target.data.gt(0)
        if isinstance(input, Variable):
            mask = Variable(mask, volatile=input.volatile)
        out = torch.masked_select(logprob_select, mask)
        loss = -torch.sum(out)
        return loss


class mixture_of_softmaxes(torch.nn.Module):
    """
    Breaking the Softmax Bottleneck: A High-Rank RNN Language Model (ICLR 2018)    
    """

    def __init__(self, nhid, n_experts, ntoken):
        super(mixture_of_softmaxes, self).__init__()
        self.nhid = nhid
        self.ntoken = ntoken
        self.n_experts = n_experts
        self.prior = nn.Linear(nhid, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(nhid, n_experts * nhid), nn.Tanh())
        self.decoder = nn.Linear(nhid, ntoken)

    def forward(self, x):
        latent = self.latent(x)
        logit = self.decoder(latent.view(-1, self.nhid))
        prior_logit = self.prior(x).view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit)
        prob = nn.functional.softmax(logit.view(-1, self.ntoken)).view(-1, self.n_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)
        return prob


class nPairLoss(nn.Module):
    """
    Given the right, fake, wrong, wrong_sampled embedding, use the N Pair Loss
    objective (which is an extension to the triplet loss)

    Loss = log(1+exp(feat*wrong - feat*right + feat*fake - feat*right)) + L2 norm.

    Improved Deep Metric Learning with Multi-class N-pair Loss Objective (NIPS)
    """

    def __init__(self, ninp, margin):
        super(nPairLoss, self).__init__()
        self.ninp = ninp
        self.margin = np.log(margin)

    def forward(self, feat, right, wrong, batch_wrong, fake=None, fake_diff_mask=None):
        num_wrong = wrong.size(1)
        batch_size = feat.size(0)
        feat = feat.view(-1, self.ninp, 1)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        wrong_dis = torch.bmm(wrong, feat)
        batch_wrong_dis = torch.bmm(batch_wrong, feat)
        wrong_score = torch.sum(torch.exp(wrong_dis - right_dis.expand_as(wrong_dis)), 1) + torch.sum(torch.exp(batch_wrong_dis - right_dis.expand_as(batch_wrong_dis)), 1)
        loss_dis = torch.sum(torch.log(wrong_score + 1))
        loss_norm = right.norm() + feat.norm() + wrong.norm() + batch_wrong.norm()
        if fake:
            fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), feat)
            fake_score = torch.masked_select(torch.exp(fake_dis - right_dis), fake_diff_mask)
            margin_score = F.relu(torch.log(fake_score + 1) - self.margin)
            loss_fake = torch.sum(margin_score)
            loss_dis += loss_fake
            loss_norm += fake.norm()
        loss = (loss_dis + 0.1 * loss_norm) / batch_size
        if fake:
            return loss, loss_fake.data[0] / batch_size
        else:
            return loss


class G_loss(nn.Module):
    """
    Generator loss:
    minimize right feature and fake feature L2 norm.
    maximinze the fake feature and wrong feature.
    """

    def __init__(self, ninp):
        super(G_loss, self).__init__()
        self.ninp = ninp

    def forward(self, feat, right, fake):
        batch_size = feat.size(0)
        feat = feat.view(-1, self.ninp, 1)
        fake_dis = torch.bmm(fake.view(-1, 1, self.ninp), feat)
        right_dis = torch.bmm(right.view(-1, 1, self.ninp), feat)
        fake_score = torch.exp(right_dis - fake_dis)
        loss_fake = torch.sum(torch.log(fake_score + 1))
        loss_norm = feat.norm() + fake.norm() + right.norm()
        loss = (loss_fake + 0.1 * loss_norm) / batch_size
        return loss, loss_fake.data[0] / batch_size


class gumbel_sampler(nn.Module):

    def __init__(self):
        super(gumbel_sampler, self).__init__()

    def forward(self, input, noise, temperature=0.5):
        eps = 1e-20
        noise.data.add_(eps).log_().neg_()
        noise.data.add_(eps).log_().neg_()
        y = (input + noise) / temperature
        y = F.softmax(y)
        max_val, max_idx = torch.max(y, y.dim() - 1)
        y_hard = y == max_val.view(-1, 1).expand_as(y)
        y = (y_hard.float() - y).detach() + y
        return y, max_idx.view(1, -1)


class AxB(nn.Module):

    def __init__(self, nhid):
        super(AxB, self).__init__()
        self.nhid = nhid

    def forward(self, nhA, nhB):
        mat = torch.bmm(nhB.view(-1, 100, self.nhid), nhA.view(-1, self.nhid, 1))
        return mat.view(-1, 100)


class _netG(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout, mos):
        super(_netG, self).__init__()
        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.mos_flag = mos
        self.mos_layer = mixture_of_softmaxes(nhid, 5, ntoken + 1)
        self.decoder = nn.Linear(nhid, ntoken + 1)
        self.d = dropout
        self.beta = 3
        self.vocab_size = ntoken

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, emb, hidden):
        output, hidden = self.rnn(emb, hidden)
        output = F.dropout(output, self.d, training=self.training)
        if self.mos_flag:
            decoded = self.mos_layer(output.view(output.size(0) * output.size(1), output.size(2)))
            logprob = torch.log(self.beta * decoded)
        else:
            decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
            logprob = F.log_softmax(self.beta * decoded)
        return logprob, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def sample_beam(self, netW, input, hidden_state, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = input.size(1)
        seq_all = torch.LongTensor(self.seq_length, batch_size, beam_size).zero_()
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = []
            for state_tmp in hidden_state:
                state.append(state_tmp[:, (k), :].view(1, 1, -1).expand(1, beam_size, self.nhid).clone())
            state = tuple(state)
            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size)
            for t in range(self.seq_length + 1):
                if t == 0:
                    it = input.data.resize_(1, beam_size).fill_(self.vocab_size)
                    xt = netW(Variable(it, requires_grad=False))
                else:
                    """perform a beam merge. that is,
                    for every previous beam we now many new possibilities to branch out
                    we need to resort our beams to maintain the loop invariant of keeping
                    the top beam_size most likely sequences."""
                    logprobsf = logprobs.float()
                    ys, ix = torch.sort(logprobsf, 1, True)
                    candidates = []
                    cols = min(beam_size, ys.size(1))
                    rows = beam_size
                    if t == 1:
                        rows = 1
                    for cc in range(cols):
                        for qq in range(rows):
                            local_logprob = ys[qq, cc]
                            if beam_seq[t - 2, qq] == self.vocab_size:
                                local_logprob.data.fill_(-9999)
                            candidate_logprob = beam_logprobs_sum[qq] + local_logprob
                            candidates.append({'c': ix.data[qq, cc], 'q': qq, 'p': candidate_logprob.data[0], 'r': local_logprob.data[0]})
                    candidates = sorted(candidates, key=lambda x: -x['p'])
                    new_state = [_.clone() for _ in state]
                    if t > 1:
                        beam_seq_prev = beam_seq[:t - 1].clone()
                        beam_seq_logprobs_prev = beam_seq_logprobs[:t - 1].clone()
                    for vix in range(beam_size):
                        v = candidates[vix]
                        if t > 1:
                            beam_seq[:t - 1, (vix)] = beam_seq_prev[:, (v['q'])]
                            beam_seq_logprobs[:t - 1, (vix)] = beam_seq_logprobs_prev[:, (v['q'])]
                        for state_ix in range(len(new_state)):
                            new_state[state_ix][0, vix] = state[state_ix][0, v['q']]
                        beam_seq[t - 1, vix] = v['c']
                        beam_seq_logprobs[t - 1, vix] = v['r']
                        beam_logprobs_sum[vix] = v['p']
                        if v['c'] == self.vocab_size or t == self.seq_length:
                            self.done_beams[k].append({'seq': beam_seq[:, (vix)].clone(), 'logps': beam_seq_logprobs[:, (vix)].clone(), 'p': beam_logprobs_sum[vix]})
                    it = beam_seq[t - 1].view(1, -1)
                    xt = netW(Variable(it))
                if t >= 1:
                    state = new_state
                output, state = self.rnn(xt, state)
                output = F.dropout(output, self.d, training=self.training)
                if self.mos_flag:
                    decoded = self.mos_layer(output.view(output.size(0) * output.size(1), output.size(2)))
                    logprob = torch.log(self.beta * decoded)
                else:
                    decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
                    logprob = F.log_softmax(self.beta * decoded)
            self.done_beams[k] = sorted(self.done_beams[k], key=lambda x: -x['p'])
            seq[:, (k)] = self.done_beams[k][0]['seq']
            seqLogprobs[:, (k)] = self.done_beams[k][0]['logps']
            for ii in range(beam_size):
                seq_all[:, (k), (ii)] = self.done_beams[k][ii]['seq']
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, netW, input, state, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        seq_length = opt.get('seq_length', 9)
        self.seq_length = seq_length
        if beam_size > 1:
            return self.sample_beam(netW, input, state, opt)
        batch_size = input.size(1)
        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0:
                it = input.data
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu()
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False))
                it = it.view(-1).long()
            xt = netW(Variable(it.view(1, -1), requires_grad=False))
            if t >= 1:
                seq.append(it)
                seqLogprobs.append(sampleLogprobs.view(-1))
            output, state = self.rnn(xt, state)
            output = F.dropout(output, self.d, training=self.training)
            if self.mos_flag:
                decoded = self.mos_layer(output.view(output.size(0) * output.size(1), output.size(2)))
                logprob = torch.log(self.beta * decoded)
            else:
                decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
                logprob = F.log_softmax(self.beta * decoded)
        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AxB,
     lambda: ([], {'nhid': 4}),
     lambda: ([torch.rand([400, 100, 4]), torch.rand([40000, 100, 4])], {}),
     True),
    (LMCriterion,
     lambda: ([], {}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64), torch.ones([4, 4], dtype=torch.int64)], {}),
     True),
    (_netW,
     lambda: ([], {'ntoken': 4, 'ninp': 4, 'dropout': 0.5}),
     lambda: ([torch.ones([4], dtype=torch.int64)], {}),
     False),
    (gumbel_sampler,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (mixture_of_softmaxes,
     lambda: ([], {'nhid': 4, 'n_experts': 4, 'ntoken': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (nPairLoss,
     lambda: ([], {'ninp': 4, 'margin': 4}),
     lambda: ([torch.rand([16, 4, 4]), torch.rand([64, 4]), torch.rand([64, 4, 4]), torch.rand([64, 4, 4])], {}),
     False),
    (share_Linear,
     lambda: ([], {'weight': torch.rand([4, 4])}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_jiasenlu_visDial_pytorch(_paritybench_base):
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

