import sys
_module = sys.modules[__name__]
del sys
conf = _module
freezing = _module
history = _module
mnist = _module
train = _module
dataset = _module
model = _module
prepare_dataset = _module
utils = _module
model = _module
data = _module
generate = _module
model = _module
net = _module
setup = _module
skorch = _module
callbacks = _module
base = _module
logging = _module
lr_scheduler = _module
regularization = _module
scoring = _module
training = _module
classifier = _module
cli = _module
exceptions = _module
helper = _module
net = _module
regressor = _module
setter = _module
tests = _module
test_all = _module
test_logging = _module
test_lr_scheduler = _module
test_regularization = _module
test_scoring = _module
test_training = _module
conftest = _module
test_classifier = _module
test_cli = _module
test_dataset = _module
test_helper = _module
test_history = _module
test_net = _module
test_regressor = _module
test_setter = _module
test_toy = _module
test_utils = _module
toy = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import time


import numpy as np


import torch


from torch import nn


import torch.nn as nn


from torchvision import models


from torch.nn.functional import binary_cross_entropy_with_logits


from torchvision.transforms.functional import to_pil_image


from torch.autograd import Variable


from torch.nn.utils import clip_grad_norm_


import warnings


from functools import partial


from itertools import product


import re


from torch.utils.data import DataLoader


from itertools import chain


from collections import Sequence


from collections import namedtuple


from collections import OrderedDict


from math import cos


from torch.nn import RReLU


import torch.utils.data


import torch.nn.functional as F


from scipy import sparse


import copy


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pack_padded_sequence


from collections.abc import Sequence


from enum import Enum


from itertools import tee


from torch.utils.data.dataset import Subset


class ClassifierModule(nn.Module):

    def __init__(self):
        super(ClassifierModule, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, (3, 3)), nn.ReLU(), nn.Conv2d(32, 64, (3, 3)), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout(0.25))
        self.out = nn.Sequential(nn.Linear(64 * 12 * 12, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10), nn.Softmax(dim=-1))

    def forward(self, X, **kwargs):
        X = self.cnn(X)
        X = X.reshape(-1, 64 * 12 * 12)
        X = self.out(X)
        return X


N_CLASSES = 2


N_FEATURES = 20


class MLPClassifier(nn.Module):
    """A simple multi-layer perceptron module.

    This can be adapted for usage in different contexts, e.g. binary
    and multi-class classification, regression, etc.

    Note: This docstring is used to create the help for the CLI.

    Parameters
    ----------
    hidden_units : int (default=10)
      Number of units in hidden layers.

    num_hidden : int (default=1)
      Number of hidden layers.

    nonlin : torch.nn.Module instance (default=torch.nn.ReLU())
      Non-linearity to apply after hidden layers.

    dropout : float (default=0)
      Dropout rate. Dropout is applied between layers.

    """

    def __init__(self, hidden_units=10, num_hidden=1, nonlin=nn.ReLU(), dropout=0):
        super().__init__()
        self.hidden_units = hidden_units
        self.num_hidden = num_hidden
        self.nonlin = nonlin
        self.dropout = dropout
        self.reset_params()

    def reset_params(self):
        """(Re)set all parameters."""
        units = [N_FEATURES]
        units += [self.hidden_units] * self.num_hidden
        units += [N_CLASSES]
        sequence = []
        for u0, u1 in zip(units, units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        return nn.Softmax(dim=-1)(self.sequential(X))


def make_decoder_block(in_channels, middle_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, middle_channels, 3, padding=1), nn.ReLU(inplace=True), nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """UNet Model inspired by the the original UNet paper

    Parameters
    ----------
    pretrained: bool (default=True)
        Option to use pretrained vgg16_bn based on ImageNet

    References
    ----------

    .. [1] Olaf Ronneberger, Philipp Fischer, Thomas Brox, 2015,
        "U-Net: Convolutional Networks for Biomedical Image Segmentation,".
        "MICCAI" `<https://arxiv.org/abs/1505.04597>`_
    """

    def __init__(self, pretrained=False):
        super().__init__()
        encoder = models.vgg16_bn(pretrained=pretrained).features
        self.conv1 = encoder[:6]
        self.conv2 = encoder[6:13]
        self.conv3 = encoder[13:23]
        self.conv4 = encoder[23:33]
        self.conv5 = encoder[33:43]
        self.center = nn.Sequential(encoder[43], make_decoder_block(512, 512, 256))
        self.dec5 = make_decoder_block(256 + 512, 512, 256)
        self.dec4 = make_decoder_block(256 + 512, 512, 256)
        self.dec3 = make_decoder_block(256 + 256, 256, 64)
        self.dec2 = make_decoder_block(64 + 128, 128, 32)
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 64, 32, 3, padding=1), nn.ReLU(inplace=True))
        self.final = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        center = self.center(conv5)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)


class RNNClassifier(nn.Module):

    def __init__(self, embedding_dim=128, rec_layer_type='lstm', num_units=128, num_layers=2, dropout=0, vocab_size=1000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.rec_layer_type = rec_layer_type.lower()
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.emb = nn.Embedding(vocab_size + 1, embedding_dim=self.embedding_dim)
        rec_layer = {'lstm': nn.LSTM, 'gru': nn.GRU}[self.rec_layer_type]
        self.rec = rec_layer(self.embedding_dim, self.num_units, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(self.num_units, 2)

    def forward(self, X):
        embeddings = self.emb(X)
        if self.rec_layer_type == 'gru':
            _, rec_out = self.rec(embeddings)
        else:
            _, (rec_out, _) = self.rec(embeddings)
        rec_out = rec_out[-1]
        drop = F.dropout(rec_out, p=self.dropout)
        out = F.softmax(self.output(drop), dim=-1)
        return out


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class MLPModule(nn.Module):
    """A simple multi-layer perceptron module.

    This can be adapted for usage in different contexts, e.g. binary
    and multi-class classification, regression, etc.

    Parameters
    ----------
    input_units : int (default=20)
      Number of input units.

    output_units : int (default=2)
      Number of output units.

    hidden_units : int (default=10)
      Number of units in hidden layers.

    num_hidden : int (default=1)
      Number of hidden layers.

    nonlin : torch.nn.Module instance (default=torch.nn.ReLU())
      Non-linearity to apply after hidden layers.

    output_nonlin : torch.nn.Module instance or None (default=None)
      Non-linearity to apply after last layer, if any.

    dropout : float (default=0)
      Dropout rate. Dropout is applied between layers.

    squeeze_output : bool (default=False)
      Whether to squeeze output. Squeezing can be helpful if you wish
      your output to be 1-dimensional (e.g. for
      NeuralNetBinaryClassifier).

    """

    def __init__(self, input_units=20, output_units=2, hidden_units=10, num_hidden=1, nonlin=nn.ReLU(), output_nonlin=None, dropout=0, squeeze_output=False):
        super().__init__()
        self.input_units = input_units
        self.output_units = output_units
        self.hidden_units = hidden_units
        self.num_hidden = num_hidden
        self.nonlin = nonlin
        self.output_nonlin = output_nonlin
        self.dropout = dropout
        self.squeeze_output = squeeze_output
        self.reset_params()

    def reset_params(self):
        """(Re)set all parameters."""
        units = [self.input_units]
        units += [self.hidden_units] * self.num_hidden
        units += [self.output_units]
        sequence = []
        for u0, u1 in zip(units, units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]
        if self.output_nonlin:
            sequence.append(self.output_nonlin)
        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        if self.squeeze_output:
            X = X.squeeze(-1)
        return X


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ClassifierModule,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (MLPClassifier,
     lambda: ([], {}),
     lambda: ([torch.rand([20, 20])], {}),
     False),
    (MLPModule,
     lambda: ([], {}),
     lambda: ([torch.rand([20, 20])], {}),
     False),
    (UNet,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_skorch_dev_skorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

