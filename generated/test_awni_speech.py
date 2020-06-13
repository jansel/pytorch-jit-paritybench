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


import numpy as np


import torch


import torch.autograd as autograd


import math


import torch.nn as nn


import random


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
        self.rnn = nn.GRU(input_size=conv_out, hidden_size=rnn_cfg['dim'],
            num_layers=rnn_cfg['layers'], batch_first=True, dropout=config[
            'dropout'], bidirectional=rnn_cfg['bidirectional'])
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_awni_speech(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(Attention(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(NNAttention(*[], **{'n_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ProdAttention(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

