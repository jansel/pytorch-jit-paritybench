import sys
_module = sys.modules[__name__]
del sys
NAO_V1 = _module
controller = _module
decoder = _module
encoder = _module
model = _module
model_search = _module
operations = _module
test_cifar = _module
train_cifar = _module
train_controller = _module
train_imagenet = _module
train_search = _module
utils = _module
NAO_V2 = _module
autoaugment = _module
controller = _module
decoder = _module
encoder = _module
model = _module
model_search = _module
operations = _module
test_cifar = _module
test_controller = _module
train_cifar = _module
train_controller = _module
train_imagenet = _module
train_search = _module
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


import logging


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import time


from torch import Tensor


import copy


import random


import torch.utils


import torchvision.datasets as dset


import torch.backends.cudnn as cudnn


import torchvision.transforms as transforms


import torch.utils.data as data


class Attention(nn.Module):

    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, input, source_hids):
        batch_size = input.size(0)
        source_len = source_hids.size(1)
        x = self.input_proj(input)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)
        mix = torch.bmm(attn, source_hids)
        combined = torch.cat((mix, input), dim=2)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1, self.output_dim)
        return output, attn


EOS_ID = 0


SOS_ID = 0


class Decoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, layers, vocab_size, hidden_size, dropout, length, encoder_length):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length
        self.encoder_length = encoder_length
        self.vocab_size = vocab_size
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.init_input = None
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

    def forward_step(self, x, hidden, encoder_outputs):
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output, attn = self.attention(output, encoder_outputs)
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, x, encoder_hidden=None, encoder_outputs=None):
        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if x is None:
            inference = True
        else:
            inference = False
        x, batch_size, length = self._validate_args(x, encoder_hidden, encoder_outputs)
        assert length == self.length
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn)
            if step % 2 == 0:
                index = step // 2 % 10 // 2 + 3
                symbols = decoder_outputs[-1][:, 1:index].topk(1)[1] + 1
            else:
                symbols = decoder_outputs[-1][:, 7:].topk(1)[1] + 7
            sequence_symbols.append(symbols)
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = (lengths > step) & eos_batches != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols
        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols
        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()
        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden

    def _validate_args(self, x, encoder_hidden, encoder_outputs):
        if encoder_outputs is None:
            raise ValueError('Argument encoder_outputs cannot be None when attention is used.')
        if x is None and encoder_hidden is None:
            batch_size = 1
        elif x is not None:
            batch_size = x.size(0)
        else:
            batch_size = encoder_hidden[0].size(1)
        if x is None:
            x = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.length
        else:
            max_length = x.size(1)
        return x, batch_size, max_length

    def eval(self):
        return

    def infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden


class Encoder(nn.Module):

    def __init__(self, layers, vocab_size, hidden_size, dropout, length, source_length, emb_size, mlp_layers, mlp_hidden_size, mlp_dropout):
        super(Encoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.length = length
        self.source_length = source_length
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(nn.Linear(self.hidden_size, self.mlp_hidden_size), nn.ReLU(inplace=False), nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size), nn.ReLU(inplace=False), nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        if self.source_length != self.length:
            assert self.source_length % self.length == 0
            ratio = self.source_length // self.length
            embedded = embedded.view(-1, self.source_length // ratio, ratio * self.emb_size)
        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out
        encoder_hidden = hidden
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        arch_emb = out
        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value

    def infer(self, x, predict_lambda, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb = torch.mean(new_encoder_outputs, dim=1)
        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb


class NAO(nn.Module):

    def __init__(self, encoder_layers, encoder_vocab_size, encoder_hidden_size, encoder_dropout, encoder_length, source_length, encoder_emb_size, mlp_layers, mlp_hidden_size, mlp_dropout, decoder_layers, decoder_vocab_size, decoder_hidden_size, decoder_dropout, decoder_length):
        super(NAO, self).__init__()
        self.encoder = Encoder(encoder_layers, encoder_vocab_size, encoder_hidden_size, encoder_dropout, encoder_length, source_length, encoder_emb_size, mlp_layers, mlp_hidden_size, mlp_dropout)
        self.decoder = Decoder(decoder_layers, decoder_vocab_size, decoder_hidden_size, decoder_dropout, decoder_length, encoder_length)
        self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
        decoder_hidden = arch_emb.unsqueeze(0), arch_emb.unsqueeze(0)
        decoder_outputs, decoder_hidden, ret = self.decoder(target_variable, decoder_hidden, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, arch

    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0)
        decoder_outputs, decoder_hidden, ret = self.decoder(None, new_encoder_hidden, new_encoder_outputs)
        new_arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_arch


class WSAvgPool2d(nn.Module):

    def __init__(self, kernel_size, padding):
        super(WSAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x, x_id, stride=1, bn_train=False):
        return F.avg_pool2d(x, self.kernel_size, stride, self.padding, count_include_pad=False)


INPLACE = False


class WSBN(nn.Module):

    def __init__(self, num_possible_inputs, num_features, eps=1e-05, momentum=0.1, affine=True):
        super(WSBN, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.weight = nn.ParameterList([nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
            self.bias = nn.ParameterList([nn.Parameter(torch.Tensor(num_features)) for _ in range(num_possible_inputs)])
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            for i in range(self.num_possible_inputs):
                self.weight[i].data.fill_(1)
                self.bias[i].data.zero_()

    def forward(self, x, x_id, bn_train=False):
        training = self.training
        if bn_train:
            training = True
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight[x_id], self.bias[x_id], training, self.momentum, self.eps)


class WSDilSepConv(nn.Module):

    def __init__(self, num_possible_inputs, C_in, C_out, kernel_size, padding, dilation=2, affine=True):
        super(WSDilSepConv, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.C_out = C_out
        self.C_in = C_in
        self.padding = padding
        self.dilation = dilation
        self.relu = nn.ReLU(inplace=INPLACE)
        self.W_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn = WSBN(num_possible_inputs, C_in, affine=affine)

    def forward(self, x, x_id, stride=1, bn_train=False):
        x = self.relu(x)
        x = F.conv2d(x, self.W_depthwise[x_id], stride=stride, padding=self.padding, dilation=self.dilation, groups=self.C_in)
        x = F.conv2d(x, self.W_pointwise[x_id], padding=0)
        x = self.bn(x, x_id, bn_train=bn_train)
        return x


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, shape, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False), nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False), nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn.train()
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), 'constant', 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out


class WSIdentity(nn.Module):

    def __init__(self, c_in, c_out, stride, affine=True):
        super(WSIdentity, self).__init__()
        if stride == 2:
            self.reduce = nn.ModuleList()
            self.reduce.append(FactorizedReduce(c_in, c_out, [0, 0, 0], affine=affine))
            self.reduce.append(FactorizedReduce(c_in, c_out, [0, 0, 0], affine=affine))

    def forward(self, x, x_id, stride=1, bn_train=False):
        if stride == 2:
            return self.reduce[x_id](x, bn_train=bn_train)
        return x


class WSMaxPool2d(nn.Module):

    def __init__(self, kernel_size, padding):
        super(WSMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x, x_id, stride=1, bn_train=False):
        return F.max_pool2d(x, self.kernel_size, stride, self.padding)


class WSSepConv(nn.Module):

    def __init__(self, num_possible_inputs, C_in, C_out, kernel_size, padding, affine=True):
        super(WSSepConv, self).__init__()
        self.num_possible_inputs = num_possible_inputs
        self.C_out = C_out
        self.C_in = C_in
        self.padding = padding
        self.relu1 = nn.ReLU(inplace=INPLACE)
        self.W1_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W1_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn1 = WSBN(num_possible_inputs, C_in, affine=affine)
        self.relu2 = nn.ReLU(inplace=INPLACE)
        self.W2_depthwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_in, 1, kernel_size, kernel_size)) for i in range(num_possible_inputs)])
        self.W2_pointwise = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, 1, 1)) for i in range(num_possible_inputs)])
        self.bn2 = WSBN(num_possible_inputs, C_in, affine=affine)

    def forward(self, x, x_id, stride=1, bn_train=False):
        x = self.relu1(x)
        x = F.conv2d(x, self.W1_depthwise[x_id], stride=stride, padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W1_pointwise[x_id], padding=0)
        x = self.bn1(x, x_id, bn_train=bn_train)
        x = self.relu2(x)
        x = F.conv2d(x, self.W2_depthwise[x_id], padding=self.padding, groups=self.C_in)
        x = F.conv2d(x, self.W2_pointwise[x_id], padding=0)
        x = self.bn2(x, x_id, bn_train=bn_train)
        return x


class WSZero(nn.Module):

    def __init__(self):
        super(WSZero, self).__init__()

    def forward(self, x, x_id, stride=1, bn_train=False):
        if stride == 1:
            return x.mul(0.0)
        return x[:, :, ::stride, ::stride].mul(0.0)


OPERATIONS_search_middle = {(0): lambda n, c_in, c_out, stride, affine: WSZero(), (1): lambda n, c_in, c_out, stride, affine: WSIdentity(c_in, c_out, stride, affine=affine), (2): lambda n, c_in, c_out, stride, affine: WSSepConv(n, c_in, c_out, 3, 1, affine=affine), (3): lambda n, c_in, c_out, stride, affine: WSSepConv(n, c_in, c_out, 5, 2, affine=affine), (4): lambda n, c_in, c_out, stride, affine: WSSepConv(n, c_in, c_out, 7, 3, affine=affine), (5): lambda n, c_in, c_out, stride, affine: WSDilSepConv(n, c_in, c_out, 3, 2, 2, affine=affine), (6): lambda n, c_in, c_out, stride, affine: WSDilSepConv(n, c_in, c_out, 5, 4, 2, affine=affine), (7): lambda n, c_in, c_out, stride, affine: WSDilSepConv(n, c_in, c_out, 7, 6, 2, affine=affine), (8): lambda n, c_in, c_out, stride, affine: WSAvgPool2d(3, padding=1), (9): lambda n, c_in, c_out, stride, affine: WSMaxPool2d(3, padding=1)}


OPERATIONS_search_small = {(0): lambda n, c_in, c_out, stride, affine: WSSepConv(n, c_in, c_out, 3, 1, affine=affine), (1): lambda n, c_in, c_out, stride, affine: WSSepConv(n, c_in, c_out, 5, 2, affine=affine), (2): lambda n, c_in, c_out, stride, affine: WSAvgPool2d(3, padding=1), (3): lambda n, c_in, c_out, stride, affine: WSMaxPool2d(3, padding=1), (4): lambda n, c_in, c_out, stride, affine: WSIdentity(c_in, c_out, stride, affine=affine)}


def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id + 1) / layers
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.0:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob)
        x = x / drop_path_keep_prob * mask
    return x


class Node(nn.Module):

    def __init__(self, search_space, prev_layers, channels, stride, drop_path_keep_prob=None, node_id=0, layer_id=0, layers=0, steps=0):
        super(Node, self).__init__()
        self.search_space = search_space
        self.channels = channels
        self.stride = stride
        self.drop_path_keep_prob = drop_path_keep_prob
        self.node_id = node_id
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.x_op = nn.ModuleList()
        self.y_op = nn.ModuleList()
        num_possible_inputs = node_id + 2
        if search_space == 'small':
            OPERATIONS = OPERATIONS_search_small
        elif search_space == 'middle':
            OPERATIONS = OPERATIONS_search_middle
        else:
            OPERATIONS = OPERATIONS_search_small
        for k, v in OPERATIONS.items():
            self.x_op.append(v(num_possible_inputs, channels, channels, stride, True))
            self.y_op.append(v(num_possible_inputs, channels, channels, stride, True))
        self.out_shape = [prev_layers[0][0] // stride, prev_layers[0][1] // stride, channels]

    def forward(self, x, x_id, x_op, y, y_id, y_op, step, bn_train=False):
        stride = self.stride if x_id in [0, 1] else 1
        x = self.x_op[x_op](x, x_id, stride, bn_train)
        stride = self.stride if y_id in [0, 1] else 1
        y = self.y_op[y_op](y, y_id, stride, bn_train)
        X_DROP = False
        Y_DROP = False
        if self.search_space == 'small':
            if self.drop_path_keep_prob is not None and self.training:
                X_DROP = True
            if self.drop_path_keep_prob is not None and self.training:
                Y_DROP = True
        elif self.search_space == 'middle':
            if self.drop_path_keep_prob is not None and self.training:
                X_DROP = True
            if self.drop_path_keep_prob is not None and self.training:
                Y_DROP = True
        if X_DROP:
            x = apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        if Y_DROP:
            y = apply_drop_path(y, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        return x + y


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU(inplace=INPLACE)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, bn_train=False):
        x = self.relu(x)
        x = self.conv(x)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        return x


class MaybeCalibrateSize(nn.Module):

    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        hw = [layer[0] for layer in layers]
        c = [layer[-1] for layer in layers]
        x_out_shape = [hw[0], hw[0], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            self.relu = nn.ReLU(inplace=INPLACE)
            self.preprocess_x = FactorizedReduce(c[0], channels, [hw[0], hw[0], c[0]], affine)
            x_out_shape = [hw[1], hw[1], channels]
        elif c[0] != channels:
            self.preprocess_x = ReLUConvBN(c[0], channels, 1, 1, 0, [hw[0], hw[0]], affine)
            x_out_shape = [hw[0], hw[0], channels]
        if c[1] != channels:
            self.preprocess_y = ReLUConvBN(c[1], channels, 1, 1, 0, [hw[1], hw[1]], affine)
            y_out_shape = [hw[1], hw[1], channels]
        self.out_shape = [x_out_shape, y_out_shape]

    def forward(self, s0, s1, bn_train=False):
        if s0.size(2) != s1.size(2):
            s0 = self.relu(s0)
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        elif s0.size(1) != self.channels:
            s0 = self.preprocess_x(s0, bn_train=bn_train)
        if s1.size(1) != self.channels:
            s1 = self.preprocess_y(s1, bn_train=bn_train)
        return [s0, s1]


class WSReLUConvBN(nn.Module):

    def __init__(self, num_possible_inputs, C_out, C_in, kernel_size, stride=1, padding=0):
        super(WSReLUConvBN, self).__init__()
        self.stride = stride
        self.padding = padding
        self.relu = nn.ReLU(inplace=INPLACE)
        self.w = nn.ParameterList([nn.Parameter(torch.Tensor(C_out, C_in, kernel_size, kernel_size)) for _ in range(num_possible_inputs)])
        self.bn = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x, x_id, bn_train=False):
        x = self.relu(x)
        w = torch.cat([self.w[i] for i in x_id], dim=1)
        x = F.conv2d(x, w, stride=self.stride, padding=self.padding)
        if bn_train:
            self.bn.train()
        x = self.bn(x)
        return x


class Cell(nn.Module):

    def __init__(self, search_space, prev_layers, nodes, channels, reduction, layer_id, layers, steps, drop_path_keep_prob=None):
        super(Cell, self).__init__()
        self.search_space = search_space
        assert len(prev_layers) == 2
        None
        self.reduction = reduction
        self.layer_id = layer_id
        self.layers = layers
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleList()
        self.nodes = nodes
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        prev_layers = self.maybe_calibrate_size.out_shape
        stride = 2 if self.reduction else 1
        for i in range(self.nodes):
            node = Node(search_space, prev_layers, channels, stride, drop_path_keep_prob, i, layer_id, layers, steps)
            self.ops.append(node)
            prev_layers.append(node.out_shape)
        out_hw = min([shape[0] for i, shape in enumerate(prev_layers)])
        if reduction:
            self.fac_1 = FactorizedReduce(prev_layers[0][-1], channels, prev_layers[0])
            self.fac_2 = FactorizedReduce(prev_layers[1][-1], channels, prev_layers[1])
        self.final_combine_conv = WSReLUConvBN(self.nodes + 2, channels, channels, 1)
        self.out_shape = [out_hw, out_hw, channels]

    def forward(self, s0, s1, arch, step, bn_train=False):
        s0, s1 = self.maybe_calibrate_size(s0, s1, bn_train=bn_train)
        states = [s0, s1]
        used = [0] * (self.nodes + 2)
        for i in range(self.nodes):
            x_id, x_op, y_id, y_op = arch[4 * i], arch[4 * i + 1], arch[4 * i + 2], arch[4 * i + 3]
            used[x_id] += 1
            used[y_id] += 1
            out = self.ops[i](states[x_id], x_id, x_op, states[y_id], y_id, y_op, step, bn_train=bn_train)
            states.append(out)
        concat = []
        for i, c in enumerate(used):
            if used[i] == 0:
                concat.append(i)
        if self.reduction:
            if 0 in concat:
                states[0] = self.fac_1(states[0])
            if 1 in concat:
                states[1] = self.fac_2(states[1])
        out = torch.cat([states[i] for i in concat], dim=1)
        out = self.final_combine_conv(out, concat, bn_train=bn_train)
        return out


class AuxHeadCIFAR(nn.Module):

    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NASNetworkCIFAR(nn.Module):

    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, arch):
        super(NASNetworkCIFAR, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        if isinstance(arch, str):
            arch = list(map(int, arch.strip().split()))
        elif isinstance(arch, list) and len(arch) == 2:
            arch = arch[0] + arch[1]
        self.conv_arch = arch[:4 * self.nodes]
        self.reduc_arch = arch[4 * self.nodes:]
        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
        outs = [[32, 32, channels], [32, 32, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers + 2):
            if i not in self.pool_layers:
                cell = Cell(self.search_space, self.conv_arch, outs, channels, False, i, self.layers + 2, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.search_space, self.reduc_arch, outs, channels, True, i, self.layers + 2, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input, step=None):
        aux_logits = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits


class AuxHeadImageNet(nn.Module):

    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, classes)

    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NASNetworkImageNet(nn.Module):

    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps, arch):
        super(NASNetworkImageNet, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        arch = list(map(int, arch.strip().split()))
        self.conv_arch = arch[:4 * self.nodes]
        self.reduc_arch = arch[4 * self.nodes:]
        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.layers = self.layers * 3
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]
        channels = self.channels
        self.stem0 = nn.Sequential(nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=True), nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels))
        outs = [[56, 56, channels], [28, 28, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers + 2):
            if i not in self.pool_layers:
                cell = Cell(self.search_space, self.conv_arch, outs, channels, False, i, self.layers + 2, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.search_space, self.reduc_arch, outs, channels, True, i, self.layers + 2, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input, step=None):
        aux_logits = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits


class NASWSNetworkCIFAR(nn.Module):

    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps):
        super(NASWSNetworkCIFAR, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.total_layers = self.layers * 3 + 2
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))
        outs = [[32, 32, channels], [32, 32, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.total_layers):
            if i not in self.pool_layers:
                cell = Cell(self.search_space, outs, self.nodes, channels, False, i, self.total_layers, self.steps, self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = Cell(self.search_space, outs, self.nodes, channels, True, i, self.total_layers, self.steps, self.drop_path_keep_prob)
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def new(self):
        model_new = NASWSNetworkCIFAR(self.search_space, self.classes, self.layers, self.nodes, self.channels, self.keep_prob, self.drop_path_keep_prob, self.use_aux_head, self.steps)
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, arch, step=None, bn_train=False):
        aux_logits = None
        conv_arch, reduc_arch = arch
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                assert i in self.pool_layers
                s0, s1 = s1, cell(s0, s1, reduc_arch, step, bn_train=bn_train)
            else:
                s0, s1 = s1, cell(s0, s1, conv_arch, step, bn_train=bn_train)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1, bn_train=bn_train)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits


class NASWSNetworkImageNet(nn.Module):

    def __init__(self, args, classes, layers, nodes, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps):
        super(NASWSNetworkImageNet, self).__init__()
        self.args = args
        self.search_space = args.search_space
        self.classes = classes
        self.layers = layers
        self.nodes = nodes
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        self.pool_layers = [self.layers, 2 * self.layers + 1]
        self.total_layers = self.layers * 3 + 2
        if self.use_aux_head:
            self.aux_head_index = self.pool_layers[-1]
        channels = self.channels
        self.stem0 = nn.Sequential(nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels // 2), nn.ReLU(inplace=False), nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels // 2))
        self.stem1 = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(channels))
        outs = [[112, 112, channels // 2], [56, 56, channels]]
        self.cells = nn.ModuleList()
        for i in range(self.total_layers):
            if i not in self.pool_layers:
                cell = Cell(self.search_space, outs, self.nodes, channels, False, i, self.total_layers, self.steps, self.drop_path_keep_prob)
                outs = [outs[-1], cell.out_shape]
            else:
                channels *= 2
                cell = Cell(self.search_space, outs, self.nodes, channels, True, i, self.total_layers, self.steps, self.drop_path_keep_prob)
                outs = [outs[-1], cell.out_shape]
            self.cells.append(cell)
            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        self.init_parameters()

    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def new(self):
        model_new = NASWSNetworkImageNet(self.search_space, self.classes, self.layers, self.nodes, self.channels, self.keep_prob, self.drop_path_keep_prob, self.use_aux_head, self.steps)
        for x, y in zip(model_new.parameters(), self.parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input, arch, step=None, bn_train=False):
        aux_logits = None
        conv_arch, reduc_arch = arch
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                assert i in self.pool_layers
                s0, s1 = s1, cell(s0, s1, reduc_arch, step, bn_train=bn_train)
            else:
                s0, s1 = s1, cell(s0, s1, conv_arch, step, bn_train=bn_train)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1, bn_train=bn_train)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits


class Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(Conv, self).__init__()
        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(nn.ReLU(inplace=INPLACE), nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(C_out, affine=affine))
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(nn.ReLU(inplace=INPLACE), nn.Conv2d(C_out, C_out, (k1, k2), stride=(1, stride), padding=padding[0], bias=False), nn.BatchNorm2d(C_out, affine=affine), nn.ReLU(inplace=INPLACE), nn.Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        x = self.ops(x)
        return x


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, shape, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=INPLACE), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_in, affine=affine), nn.ReLU(inplace=INPLACE), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class FinalCombine(nn.Module):

    def __init__(self, layers, out_hw, channels, concat, affine=True):
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat = concat
        self.ops = nn.ModuleList()
        self.concat_fac_op_dict = {}
        for i in concat:
            hw = layers[i][0]
            if hw > out_hw:
                assert hw == 2 * out_hw and i in [0, 1]
                self.concat_fac_op_dict[i] = len(self.ops)
                op = FactorizedReduce(layers[i][-1], channels, [hw, hw], affine)
                self.ops.append(op)

    def forward(self, states, bn_train=False):
        for i in self.concat:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](states[i], bn_train)
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class DilSepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, shape, affine=True):
        super(DilSepConv, self).__init__()
        self.op = nn.Sequential(nn.ReLU(inplace=False), nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False), nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False), nn.BatchNorm2d(C_out, affine=affine))

    def forward(self, x):
        return self.op(x)


class AvgPool(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=0, count_include_pad=False):
        super(AvgPool, self).__init__()
        self.op = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, count_include_pad=count_include_pad)

    def forward(self, x):
        return self.op(x)


class MaxPool(nn.Module):

    def __init__(self, kernel_size, stride=1, padding=0):
        super(MaxPool, self).__init__()
        self.op = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.0)
        return x[:, :, ::self.stride, ::self.stride].mul(0.0)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (AvgPool,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Conv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DilSepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'dilation': 1, 'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FactorizedReduce,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaxPool,
     lambda: ([], {'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ReLUConvBN,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SepConv,
     lambda: ([], {'C_in': 4, 'C_out': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4, 'shape': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (WSBN,
     lambda: ([], {'num_possible_inputs': 4, 'num_features': 4}),
     lambda: ([torch.rand([4, 4]), 0], {}),
     False),
    (WSIdentity,
     lambda: ([], {'c_in': 4, 'c_out': 4, 'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (WSZero,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Zero,
     lambda: ([], {'stride': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_renqianluo_NAO_pytorch(_paritybench_base):
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

