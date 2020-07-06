import sys
_module = sys.modules[__name__]
del sys
eval = _module
download = _module
preprocess = _module
score = _module
speech = _module
loader = _module
models = _module
ctc_decoder = _module
ctc_model = _module
model = _module
seq2seq = _module
transducer_model = _module
utils = _module
convert = _module
data_helpers = _module
io = _module
wave = _module
ctc_test = _module
io_test = _module
loader_test = _module
model_test = _module
seq2seq_test = _module
shared = _module
wave_test = _module
train = _module

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


import numpy as np


import random


import scipy.signal


import torch.autograd as autograd


import torch.utils.data as tud


import math


import torch.nn as nn


import time


import torch.optim


class Model(nn.Module):

    def __init__(self, input_dim, config):
        super().__init__()
        self.input_dim = input_dim
        encoder_cfg = config['encoder']
        conv_cfg = encoder_cfg['conv']
        convs = []
        in_c = 1
        for out_c, h, w, s in conv_cfg:
            conv = nn.Conv2d(in_c, out_c, (h, w), stride=(s, s), padding=0)
            convs.extend([conv, nn.ReLU()])
            if config['dropout'] != 0:
                convs.append(nn.Dropout(p=config['dropout']))
            in_c = out_c
        self.conv = nn.Sequential(*convs)
        conv_out = out_c * self.conv_out_size(input_dim, 1)
        assert conv_out > 0, 'Convolutional ouptut frequency dimension is negative.'
        rnn_cfg = encoder_cfg['rnn']
        self.rnn = nn.GRU(input_size=conv_out, hidden_size=rnn_cfg['dim'], num_layers=rnn_cfg['layers'], batch_first=True, dropout=config['dropout'], bidirectional=rnn_cfg['bidirectional'])
        self._encoder_dim = rnn_cfg['dim']
        self.volatile = False

    def conv_out_size(self, n, dim):
        for c in self.conv.children():
            if type(c) == nn.Conv2d:
                k = c.kernel_size[dim]
                s = c.stride[dim]
                n = (n - k + 1) / s
                n = int(math.ceil(n))
        return n

    def forward(self, batch):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def encode(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = torch.transpose(x, 1, 2).contiguous()
        b, t, f, c = x.size()
        x = x.view((b, t, f * c))
        x, h = self.rnn(x)
        if self.rnn.bidirectional:
            half = x.size()[-1] // 2
            x = x[:, :, :half] + x[:, :, half:]
        return x

    def loss(self, x, y):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def set_eval(self):
        """
        Set the model to evaluation mode.
        """
        self.eval()
        self.volatile = True

    def set_train(self):
        """
        Set the model to training mode.
        """
        self.train()
        self.volatile = False

    def infer(self, x):
        """
        Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def is_cuda(self):
        return list(self.parameters())[0].is_cuda

    @property
    def encoder_dim(self):
        return self._encoder_dim


class LinearND(nn.Module):

    def __init__(self, *args):
        """
        A torch.nn.Linear layer modified to accept ND arrays.
        The function treats the last dimension of the input
        as the hidden dimension.
        """
        super(LinearND, self).__init__()
        self.fc = nn.Linear(*args)

    def forward(self, x):
        size = x.size()
        n = int(np.prod(size[:-1]))
        out = x.contiguous().view(n, size[-1])
        out = self.fc(out)
        size = list(size)
        size[-1] = out.size()[-1]
        return out.view(size)


class NNAttention(nn.Module):

    def __init__(self, n_channels, kernel_size=15, log_t=False):
        super(NNAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, n_channels, kernel_size, padding=padding)
        self.nn = nn.Sequential(nn.ReLU(), model.LinearND(n_channels, 1))
        self.log_t = log_t

    def forward(self, eh, dhx, ax=None):
        pax = eh + dhx
        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).transpose(1, 2)
            pax = pax + ax
        pax = self.nn(pax)
        pax = pax.squeeze(dim=2)
        if self.log_t:
            log_t = math.log(pax.size()[1])
            pax = log_t * pax
        ax = nn.functional.softmax(pax, dim=1)
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax


def end_pad_concat(labels):
    batch_size = len(labels)
    end_tok = labels[0][-1]
    max_len = max(len(l) for l in labels)
    cat_labels = np.full((batch_size, max_len), fill_value=end_tok, dtype=np.int64)
    for e, l in enumerate(labels):
        cat_labels[(e), :len(l)] = l
    return cat_labels


class Seq2Seq(model.Model):

    def __init__(self, freq_dim, vocab_size, config):
        super().__init__(freq_dim, config)
        decoder_cfg = config['decoder']
        rnn_dim = self.encoder_dim
        embed_dim = decoder_cfg['embedding_dim']
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_rnn = nn.GRUCell(input_size=embed_dim, hidden_size=rnn_dim)
        self.attend = NNAttention(rnn_dim, log_t=decoder_cfg.get('log_t', False))
        self.sample_prob = decoder_cfg.get('sample_prob', 0)
        self.scheduled_sampling = self.sample_prob != 0
        self.fc = model.LinearND(rnn_dim, vocab_size - 1)

    def set_eval(self):
        """
        Set the model to evaluation mode.
        """
        self.eval()
        self.volatile = True
        self.scheduled_sampling = False

    def set_train(self):
        """
        Set the model to training mode.
        """
        self.train()
        self.volatile = False
        self.scheduled_sampling = self.sample_prob != 0

    def loss(self, batch):
        x, y = self.collate(*batch)
        if self.is_cuda:
            x = x
            y = y
        out, alis = self.forward_impl(x, y)
        batch_size, _, out_dim = out.size()
        out = out.view((-1, out_dim))
        y = y[:, 1:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(out, y, size_average=False)
        loss = loss / batch_size
        return loss

    def forward_impl(self, x, y):
        x = self.encode(x)
        out, alis = self.decode(x, y)
        return out, alis

    def forward(self, batch):
        x, y = self.collate(*batch)
        if self.is_cuda:
            x = x
            y = y
        return self.forward_impl(x, y)[0]

    def decode(self, x, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        inputs = self.embedding(y[:, :-1])
        out = []
        aligns = []
        hx = torch.zeros((x.shape[0], x.shape[2]), requires_grad=False)
        if self.is_cuda:
            hx
        ax = None
        sx = None
        for t in range(y.size()[1] - 1):
            sample = out and self.scheduled_sampling
            if sample and random.random() < self.sample_prob:
                ix = torch.max(out[-1], dim=2)[1]
                ix = self.embedding(ix)
            else:
                ix = inputs[:, t:t + 1, :]
            if sx is not None:
                ix = ix + sx
            hx = self.dec_rnn(ix.squeeze(dim=1), hx)
            ox = hx.unsqueeze(dim=1)
            sx, ax = self.attend(x, ox, ax)
            aligns.append(ax)
            out.append(self.fc(ox + sx))
        out = torch.cat(out, dim=1)
        aligns = torch.stack(aligns, dim=1)
        return out, aligns

    def decode_step(self, x, y, state=None, softmax=False):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        if state is None:
            hx = torch.zeros((x.shape[0], x.shape[2]), requires_grad=False)
            if self.is_cuda:
                hx
            ax = None
            sx = None
        else:
            hx, ax, sx = state
        ix = self.embedding(y)
        if sx is not None:
            ix = ix + sx
        hx = self.dec_rnn(ix.squeeze(dim=1), hx=hx)
        ox = hx.unsqueeze(dim=1)
        sx, ax = self.attend(x, ox, ax=ax)
        out = ox + sx
        out = self.fc(out.squeeze(dim=1))
        if softmax:
            out = nn.functional.log_softmax(out, dim=1)
        return out, (hx, ax, sx)

    def predict(self, batch):
        probs = self(batch)
        argmaxs = torch.max(probs, dim=2)[1]
        argmaxs = argmaxs.cpu().data.numpy()
        return [seq.tolist() for seq in argmaxs]

    def infer_decode(self, x, y, end_tok, max_len):
        probs = []
        argmaxs = [y]
        state = None
        for e in range(max_len):
            out, state = self.decode_step(x, y, state=state)
            probs.append(out)
            y = torch.max(out, dim=1)[1]
            y = y.unsqueeze(dim=1)
            argmaxs.append(y)
            if torch.sum(y.data == end_tok) == y.numel():
                break
        probs = torch.cat(probs)
        argmaxs = torch.cat(argmaxs, dim=1)
        return probs, argmaxs

    def infer(self, batch, max_len=200):
        """
        Infer a likely output. No beam search yet.
        """
        x, y = self.collate(*batch)
        end_tok = y.data[0, -1]
        t = y
        if self.is_cuda:
            x = x
            t = y
        x = self.encode(x)
        y = t[:, 0:1]
        _, argmaxs = self.infer_decode(x, y, end_tok, max_len)
        argmaxs = argmaxs.cpu().data.numpy()
        return [seq.tolist() for seq in argmaxs]

    def beam_search(self, batch, beam_size=10, max_len=200):
        x, y = self.collate(*batch)
        start_tok = y.data[0, 0]
        end_tok = y.data[0, -1]
        if self.is_cuda:
            x = x
            y = y
        x = self.encode(x)
        y = y[:, 0:1].clone()
        beam = [((start_tok,), 0, None)]
        complete = []
        for _ in range(max_len):
            new_beam = []
            for hyp, score, state in beam:
                y[0] = hyp[-1]
                out, state = self.decode_step(x, y, state=state, softmax=True)
                out = out.cpu().data.numpy().squeeze(axis=0).tolist()
                for i, p in enumerate(out):
                    new_score = score + p
                    new_hyp = hyp + (i,)
                    new_beam.append((new_hyp, new_score, state))
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)
            for cand in new_beam[:beam_size]:
                if cand[0][-1] == end_tok:
                    complete.append(cand)
            beam = filter(lambda x: x[0][-1] != end_tok, new_beam)
            beam = beam[:beam_size]
            if len(beam) == 0:
                break
            if sum(c[1] > beam[0][1] for c in complete) >= beam_size:
                break
        complete = sorted(complete, key=lambda x: x[1], reverse=True)
        if len(complete) == 0:
            complete = beam
        hyp, score, _ = complete[0]
        return [hyp]

    def collate(self, inputs, labels):
        inputs = model.zero_pad_concat(inputs)
        labels = end_pad_concat(labels)
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        if self.volatile:
            inputs.volatile = True
            labels.volatile = True
        return inputs, labels


class Attention(nn.Module):

    def __init__(self, kernel_size=11, log_t=False):
        """
        Module which Performs a single attention step along the
        second axis of a given encoded input. The module uses
        both 'content' and 'location' based attention.

        The 'content' based attention is an inner product of the
        decoder hidden state with each time-step of the encoder
        state.

        The 'location' based attention performs a 1D convollution
        on the previous attention vector and adds this into the
        next attention vector prior to normalization.

        *NB* Should compute attention differently if using cuda or cpu
        based on performance. See
        https://gist.github.com/awni/9989dd31642d42405903dec8ab91d1f0
        """
        super(Attention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding)
        self.log_t = log_t

    def forward(self, eh, dhx, ax=None):
        """
        Arguments:
            eh (FloatTensor): the encoder hidden state with
                shape (batch size, time, hidden dimension).
            dhx (FloatTensor): one time step of the decoder hidden
                state with shape (batch size, hidden dimension).
                The hidden dimension must match that of the
                encoder state.
            ax (FloatTensor): one time step of the attention
                vector.

        Returns the summary of the encoded hidden state
        and the corresponding alignment.
        """
        pax = eh * dhx
        pax = torch.sum(pax, dim=2)
        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).squeeze(dim=1)
            pax = pax + ax
        if self.log_t:
            log_t = math.log(pax.size()[1])
            pax = log_t * pax
        ax = nn.functional.softmax(pax, dim=1)
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax


class ProdAttention(nn.Module):

    def __init__(self):
        super(ProdAttention, self).__init__()

    def forward(self, eh, dhx, ax=None):
        pax = eh * dhx
        pax = torch.sum(pax, dim=2)
        ax = nn.functional.softmax(pax, dim=1)
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * sx, dim=1, keepdim=True)
        return sx, ax


class Transducer(model.Model):

    def __init__(self, freq_dim, vocab_size, config):
        super().__init__(freq_dim, config)
        decoder_cfg = config['decoder']
        rnn_dim = self.encoder_dim
        embed_dim = decoder_cfg['embedding_dim']
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dec_rnn = nn.GRU(input_size=embed_dim, hidden_size=rnn_dim, num_layers=decoder_cfg['layers'], batch_first=True, dropout=config['dropout'])
        self.blank = vocab_size
        self.fc1 = model.LinearND(rnn_dim, rnn_dim)
        self.fc2 = model.LinearND(rnn_dim, vocab_size + 1)

    def forward(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        y_mat = self.label_collate(batch[1])
        return self.forward_impl(x, y_mat)

    def forward_impl(self, x, y):
        if self.is_cuda:
            x = x
            y = y
        x = self.encode(x)
        out = self.decode(x, y)
        return out

    def loss(self, batch):
        x, y, x_lens, y_lens = self.collate(*batch)
        y_mat = self.label_collate(batch[1])
        out = self.forward_impl(x, y_mat)
        loss_fn = transducer.TransducerLoss()
        loss = loss_fn(out, y, x_lens, y_lens)
        return loss

    def decode(self, x, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        y = self.embedding(y)
        b, t, h = y.shape
        start = torch.zeros((b, 1, h))
        if self.is_cuda:
            start = start
        y = torch.cat([start, y], dim=1)
        y, _ = self.dec_rnn(y)
        x = x.unsqueeze(dim=2)
        y = y.unsqueeze(dim=1)
        out = self.fc1(x) + self.fc1(y)
        out = nn.functional.relu(out)
        out = self.fc2(out)
        out = nn.functional.log_softmax(out, dim=3)
        return out

    def collate(self, inputs, labels):
        max_t = max(i.shape[0] for i in inputs)
        max_t = self.conv_out_size(max_t, 0)
        x_lens = torch.IntTensor([max_t] * len(inputs))
        x = torch.FloatTensor(model.zero_pad_concat(inputs))
        y_lens = torch.IntTensor([len(l) for l in labels])
        y = torch.IntTensor([l for label in labels for l in label])
        batch = [x, y, x_lens, y_lens]
        if self.volatile:
            for v in batch:
                v.volatile = True
        return batch

    def infer(self, batch, beam_size=4):
        out = self(batch)
        out = out.cpu().data.numpy()
        preds = []
        for e, (i, l) in enumerate(zip(*batch)):
            T = i.shape[0]
            U = len(l) + 1
            lp = out[(e), :T, :U, :]
            preds.append(td.decode_static(lp, beam_size, blank=self.blank)[0])
        return preds

    def label_collate(self, labels):
        batch_size = len(labels)
        end_tok = labels[0][-1]
        max_len = max(len(l) for l in labels)
        cat_labels = np.full((batch_size, max_len), fill_value=end_tok, dtype=np.int64)
        for e, l in enumerate(labels):
            cat_labels[(e), :len(l)] = l
        labels = torch.LongTensor(cat_labels)
        if self.volatile:
            labels.volatile = True
        return labels


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NNAttention,
     lambda: ([], {'n_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ProdAttention,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_awni_speech(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

