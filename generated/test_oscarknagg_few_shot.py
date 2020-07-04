import sys
_module = sys.modules[__name__]
del sys
config = _module
maml = _module
matching_nets = _module
proto_nets = _module
few_shot = _module
callbacks = _module
core = _module
datasets = _module
eval = _module
maml = _module
matching = _module
metrics = _module
models = _module
proto = _module
train = _module
utils = _module
scratch = _module
scripts = _module
prepare_mini_imagenet = _module
prepare_omniglot = _module
tests = _module
test_few_shot = _module
test_maml = _module
test_matching = _module
test_proto = _module
test_utils = _module

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


from torch.utils.data import DataLoader


from torch.optim import Adam


import torch


from torch.nn import Module


from typing import Callable


from typing import List


from typing import Union


from collections import OrderedDict


from torch.optim import Optimizer


from typing import Dict


from torch.nn.utils import clip_grad_norm_


from torch.nn.modules.loss import _Loss as Loss


from torch import nn


import numpy as np


import torch.nn.functional as F


from typing import Tuple


from torch.nn.modules.distance import CosineSimilarity


from torch.nn.modules.distance import PairwiseDistance


class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


class GlobalMaxPool1d(nn.Module):
    """Performs global max pooling over the entire length of a batched 1D tensor

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return nn.functional.max_pool1d(input, kernel_size=input.size()[2:]
            ).view(-1, input.size(1))


class GlobalAvgPool2d(nn.Module):
    """Performs global average pooling over the entire height and width of a batched 2D tensor

    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return nn.functional.avg_pool2d(input, kernel_size=input.size()[2:]
            ).view(-1, input.size(1))


def conv_block(in_channels: int, out_channels: int) ->nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.

    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
        stride=2))


def functional_conv_block(x: torch.Tensor, weights: torch.Tensor, biases:
    torch.Tensor, bn_weights, bn_biases) ->torch.Tensor:
    """Performs 3x3 convolution, ReLu activation, 2x2 max pooling in a functional fashion.

    # Arguments:
        x: Input Tensor for the conv block
        weights: Weights for the convolutional block
        biases: Biases for the convolutional block
        bn_weights:
        bn_biases:
    """
    x = F.conv2d(x, weights, biases, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None, weight=
        bn_weights, bias=bn_biases, training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2)
    return x


class FewShotClassifier(nn.Module):

    def __init__(self, num_input_channels: int, k_way: int,
        final_layer_size: int=64):
        """Creates a few shot classifier as used in MAML.

        This network should be identical to the one created by `get_few_shot_encoder` but with a
        classification layer on top.

        # Arguments:
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            k_way: Number of classes the model will discriminate between
            final_layer_size: 64 for Omniglot, 1600 for miniImageNet
        """
        super(FewShotClassifier, self).__init__()
        self.conv1 = conv_block(num_input_channels, 64)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.logits = nn.Linear(final_layer_size, k_way)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.logits(x)

    def functional_forward(self, x, weights):
        """Applies the same forward pass using PyTorch functional operators using a specified set of weights."""
        for block in [1, 2, 3, 4]:
            x = functional_conv_block(x, weights[f'conv{block}.0.weight'],
                weights[f'conv{block}.0.bias'], weights.get(
                f'conv{block}.1.weight'), weights.get(f'conv{block}.1.bias'))
        x = x.view(x.size(0), -1)
        x = F.linear(x, weights['logits.weight'], weights['logits.bias'])
        return x


def get_few_shot_encoder(num_input_channels=1) ->nn.Module:
    """Creates a few shot encoder as used in Matching and Prototypical Networks

    # Arguments:
        num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
            miniImageNet = 3
    """
    return nn.Sequential(conv_block(num_input_channels, 64), conv_block(64,
        64), conv_block(64, 64), conv_block(64, 64), Flatten())


class MatchingNetwork(nn.Module):

    def __init__(self, n: int, k: int, q: int, fce: bool,
        num_input_channels: int, lstm_layers: int, lstm_input_size: int,
        unrolling_steps: int, device: torch.device):
        """Creates a Matching Network as described in Vinyals et al.

        # Arguments:
            n: Number of examples per class in the support set
            k: Number of classes in the few shot classification task
            q: Number of examples per class in the query set
            fce: Whether or not to us fully conditional embeddings
            num_input_channels: Number of color channels the model expects input data to contain. Omniglot = 1,
                miniImageNet = 3
            lstm_layers: Number of LSTM layers in the bidrectional LSTM g that embeds the support set (fce = True)
            lstm_input_size: Input size for the bidirectional and Attention LSTM. This is determined by the embedding
                dimension of the few shot encoder which is in turn determined by the size of the input data. Hence we
                have Omniglot -> 64, miniImageNet -> 1600.
            unrolling_steps: Number of unrolling steps to run the Attention LSTM
            device: Device on which to run computation
        """
        super(MatchingNetwork, self).__init__()
        self.n = n
        self.k = k
        self.q = q
        self.fce = fce
        self.num_input_channels = num_input_channels
        self.encoder = get_few_shot_encoder(self.num_input_channels)
        if self.fce:
            self.g = BidrectionalLSTM(lstm_input_size, lstm_layers)
            self.f = AttentionLSTM(lstm_input_size, unrolling_steps=
                unrolling_steps)

    def forward(self, inputs):
        pass


class BidrectionalLSTM(nn.Module):

    def __init__(self, size: int, layers: int):
        """Bidirectional LSTM used to generate fully conditional embeddings (FCE) of the support set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            layers: Number of LSTM layers
        """
        super(BidrectionalLSTM, self).__init__()
        self.num_layers = layers
        self.batch_size = 1
        self.lstm = nn.LSTM(input_size=size, num_layers=layers, hidden_size
            =size, bidirectional=True)

    def forward(self, inputs):
        output, (hn, cn) = self.lstm(inputs, None)
        forward_output = output[:, :, :self.lstm.hidden_size]
        backward_output = output[:, :, self.lstm.hidden_size:]
        output = forward_output + backward_output + inputs
        return output, hn, cn


class AttentionLSTM(nn.Module):

    def __init__(self, size: int, unrolling_steps: int):
        """Attentional LSTM used to generate fully conditional embeddings (FCE) of the query set as described
        in the Matching Networks paper.

        # Arguments
            size: Size of input and hidden layers. These are constrained to be the same in order to implement the skip
                connection described in Appendix A.2
            unrolling_steps: Number of steps of attention over the support set to compute. Analogous to number of
                layers in a regular LSTM
        """
        super(AttentionLSTM, self).__init__()
        self.unrolling_steps = unrolling_steps
        self.lstm_cell = nn.LSTMCell(input_size=size, hidden_size=size)

    def forward(self, support, queries):
        if support.shape[-1] != queries.shape[-1]:
            raise ValueError(
                'Support and query set have different embedding dimension!')
        batch_size = queries.shape[0]
        embedding_dim = queries.shape[1]
        h_hat = torch.zeros_like(queries).double()
        c = torch.zeros(batch_size, embedding_dim).double()
        for k in range(self.unrolling_steps):
            h = h_hat + queries
            attentions = torch.mm(h, support.t())
            attentions = attentions.softmax(dim=1)
            readout = torch.mm(attentions, support)
            h_hat, c = self.lstm_cell(queries, (h + readout, c))
        h = h_hat + queries
        return h


class DummyModel(torch.nn.Module):
    """Dummy 1 layer (0 hidden layer) model for testing purposes"""

    def __init__(self, k: int):
        super(DummyModel, self).__init__()
        self.out = torch.nn.Linear(2, k, bias=False)

    def forward(self, x):
        x = self.out(x)
        return x

    def functional_forward(self, x, weights):
        x = F.linear(x, weights['out.weight'])
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_oscarknagg_few_shot(_paritybench_base):
    pass
    def test_000(self):
        self._check(BidrectionalLSTM(*[], **{'size': 4, 'layers': 1}), [torch.rand([4, 4, 4])], {})

    def test_001(self):
        self._check(DummyModel(*[], **{'k': 4}), [torch.rand([2, 2])], {})

    def test_002(self):
        self._check(Flatten(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(GlobalMaxPool1d(*[], **{}), [torch.rand([4, 4, 4])], {})

    def test_005(self):
        self._check(MatchingNetwork(*[], **{'n': 4, 'k': 4, 'q': 4, 'fce': 4, 'num_input_channels': 4, 'lstm_layers': 1, 'lstm_input_size': 4, 'unrolling_steps': 4, 'device': 0}), [torch.rand([4, 4, 4, 4])], {})

