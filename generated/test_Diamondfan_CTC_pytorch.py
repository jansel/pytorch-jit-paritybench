import sys
_module = sys.modules[__name__]
del sys
cnn_lstm_ctc = _module
ctcDecoder = _module
data_loader = _module
lstm_ctc = _module
model = _module
test = _module
utils = _module
normalize_phone = _module
model_ctc = _module
get_model_units = _module
test_ctc = _module
train_ctc = _module
visualize = _module
BeamSearch = _module
NgramLM = _module
ctcDecoder = _module
data_loader = _module
tools = _module

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
xrange = range
wraps = functools.wraps


import torch


import torch.nn as nn


from torch.autograd import Variable


import time


import numpy as np


import copy


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


import scipy.signal


import math


import torch.nn.functional as F


from collections import OrderedDict


import torchaudio


def position_encoding_init(n_position, d_pos_vec):
    position_enc = np.array([([(pos / np.power(10000, 2 * i / d_pos_vec)) for i in range(d_pos_vec)] if pos != 0 else np.zeros(d_pos_vec)) for pos in range(n_position)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class Encoder(nn.Module):

    def __init__(self, n_position, d_word_vec=512):
        super(Encoder, self).__init__()
        self.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=0)
        self.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

    def forward(self, src_pos):
        enc_input = self.position_enc(src_pos)
        return enc_input


class SequenceWise(nn.Module):

    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        try:
            x, batch_size_len = x.data, x.batch_sizes
            x = self.module(x)
            x = nn.utils.rnn.PackedSequence(x, batch_size_len)
        except:
            t, n = x.size(0), x.size(1)
            x = x.view(t * n, -1)
            x = self.module(x)
            x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class InferenceBatchLogSoftmax(nn.Module):

    def forward(self, x):
        if not self.training:
            seq_len = x.size()[0]
            return torch.stack([F.log_softmax(x[i]) for i in range(seq_len)], 0)
        else:
            return x


class BatchRNN(nn.Module):
    """
    Add BatchNorm before rnn to generate a batchrnn layer
    """

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True, dropout=0.1):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size, bidirectional=bidirectional, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if self.batch_norm is not None:
            x = x.transpose(-1, -2)
            x = self.batch_norm(x)
            x = x.transpose(-1, -2)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        return x


class CTC_RNN(nn.Module):

    def __init__(self, rnn_input_size=40, rnn_hidden_size=768, rnn_layers=5, rnn_type=nn.LSTM, bidirectional=True, batch_norm=True, num_class=28, drop_out=0.1):
        super(CTC_RNN, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.num_class = num_class
        self.num_directions = 2 if bidirectional else 1
        self.name = 'CTC_RNN'
        self._drop_out = drop_out
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for i in range(rnn_layers - 1):
            rnn = BatchRNN(input_size=self.num_directions * rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, dropout=drop_out, batch_norm=batch_norm)
            rnns.append(('%d' % (i + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        if batch_norm:
            fc = nn.Sequential(nn.BatchNorm1d(self.num_directions * rnn_hidden_size), nn.Linear(self.num_directions * rnn_hidden_size, num_class + 1, bias=False))
        else:
            fc = nn.Linear(self.num_directions * rnn_hidden_size, num_class + 1, bias=False)
        self.fc = SequenceWise(fc)
        self.inference_log_softmax = InferenceBatchLogSoftmax()

    def forward(self, x):
        x = self.rnns(x)
        x = self.fc(x)
        x, batch_seq = nn.utils.rnn.pad_packed_sequence(x, batch_first=False)
        x = self.inference_log_softmax(x)
        return x

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, training_cer_results=None, dev_cer_results=None):
        package = {'input_size': model.rnn_input_size, 'hidden_size': model.rnn_hidden_size, 'rnn_layers': model.rnn_layers, 'rnn_type': model.rnn_type, 'num_class': model.num_class, 'bidirectional': model.num_directions, '_drop_out': model._drop_out, 'name': model.name, 'state_dict': model.state_dict()}
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['training_cer_results'] = training_cer_results
            package['dev_cer_results'] = dev_cer_results
        return package


class CNN_LSTM_CTC(nn.Module):

    def __init__(self, rnn_input_size=201, rnn_hidden_size=256, rnn_layers=4, rnn_type=nn.LSTM, bidirectional=True, batch_norm=True, num_class=48, drop_out=0.1):
        super(CNN_LSTM_CTC, self).__init__()
        self.rnn_input_size = rnn_input_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.num_class = num_class
        self.num_directions = 2 if bidirectional else 1
        self._drop_out = drop_out
        self.name = 'CNN_LSTM_CTC'
        self.conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(11, 5), stride=(2, 2)), nn.BatchNorm2d(16), nn.Hardtanh(0, 20, inplace=True))
        rnn_input_size = int(math.floor(rnn_input_size - 5) / 2 + 1)
        rnn_input_size *= 16
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for i in range(rnn_layers - 1):
            rnn = BatchRNN(input_size=self.num_directions * rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, dropout=drop_out, batch_norm=batch_norm)
            rnns.append(('%d' % (i + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        if batch_norm:
            fc = nn.Sequential(nn.BatchNorm1d(self.num_directions * rnn_hidden_size), nn.Linear(self.num_directions * rnn_hidden_size, num_class + 1, bias=False))
        else:
            fc = nn.Linear(self.num_directions * rnn_hidden_size, num_class + 1, bias=False)
        self.fc = SequenceWise(fc)
        self.inference_log_softmax = InferenceBatchLogSoftmax()

    def forward(self, x):
        x = self.conv(x)
        x = x.transpose(2, 3).contiguous()
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        x = self.rnns(x)
        x = self.fc(x)
        x = self.inference_log_softmax(x)
        return x

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, training_cer_results=None, dev_cer_results=None):
        package = {'input_size': model.rnn_input_size, 'hidden_size': model.rnn_hidden_size, 'rnn_layers': model.rnn_layers, 'rnn_type': model.rnn_type, 'num_class': model.num_class, 'bidirectional': model.num_directions, '_drop_out': model._drop_out, 'name': model.name, 'state_dict': model.state_dict()}
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['training_cer_results'] = training_cer_results
            package['dev_cer_results'] = dev_cer_results
        return package


class LayerCNN(nn.Module):
    """
    One CNN layer include conv2d, batchnorm, activation and maxpooling
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, pooling_size=None, activation_function=nn.ReLU, batch_norm=True, dropout=0.1):
        super(LayerCNN, self).__init__()
        if len(kernel_size) == 2:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
            self.batch_norm = nn.BatchNorm2d(out_channel) if batch_norm else None
        else:
            self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
            self.batch_norm = nn.BatchNorm1d(out_channel) if batch_norm else None
        self.activation = activation_function(inplace=True)
        if pooling_size is not None and len(kernel_size) == 2:
            self.pooling = nn.MaxPool2d(pooling_size)
        elif len(kernel_size) == 1:
            self.pooling = nn.MaxPool1d(pooling_size)
        else:
            self.pooling = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.dropout(x)
        return x


class CTC_Model(nn.Module):

    def __init__(self, add_cnn=False, cnn_param=None, rnn_param=None, num_class=39, drop_out=0.1):
        """
        add_cnn   [bool]:  whether add cnn in the model
        cnn_param [dict]:  cnn parameters, only support Conv2d i.e.
            cnn_param = {"layer":[[(in_channel, out_channel), (kernel_size), (stride), (padding), (pooling_size)],...], 
                            "batch_norm":True, "activate_function":nn.ReLU}
        rnn_param [dict]:  rnn parameters i.e.
            rnn_param = {"rnn_input_size":201, "rnn_hidden_size":256, ....}
        num_class  [int]:  the number of modelling units, add blank to be the number of classes
        drop_out [float]:  drop_out rate for all
        """
        super(CTC_Model, self).__init__()
        self.add_cnn = add_cnn
        self.cnn_param = cnn_param
        if rnn_param is None or type(rnn_param) != dict:
            raise ValueError('rnn_param need to be a dict to contain all params of rnn!')
        self.rnn_param = rnn_param
        self.num_class = num_class
        self.num_directions = 2 if rnn_param['bidirectional'] else 1
        self.drop_out = drop_out
        if add_cnn:
            cnns = []
            activation = cnn_param['activate_function']
            batch_norm = cnn_param['batch_norm']
            rnn_input_size = rnn_param['rnn_input_size']
            cnn_layers = cnn_param['layer']
            for n in range(len(cnn_layers)):
                in_channel = cnn_layers[n][0][0]
                out_channel = cnn_layers[n][0][1]
                kernel_size = cnn_layers[n][1]
                stride = cnn_layers[n][2]
                padding = cnn_layers[n][3]
                pooling_size = cnn_layers[n][4]
                cnn = LayerCNN(in_channel, out_channel, kernel_size, stride, padding, pooling_size, activation_function=activation, batch_norm=batch_norm, dropout=drop_out)
                cnns.append(('%d' % n, cnn))
                try:
                    rnn_input_size = int(math.floor((rnn_input_size + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1)
                except:
                    rnn_input_size = rnn_input_size
            self.conv = nn.Sequential(OrderedDict(cnns))
            rnn_input_size *= out_channel
        else:
            rnn_input_size = rnn_param['rnn_input_size']
        rnns = []
        rnn_hidden_size = rnn_param['rnn_hidden_size']
        rnn_type = rnn_param['rnn_type']
        rnn_layers = rnn_param['rnn_layers']
        bidirectional = rnn_param['bidirectional']
        batch_norm = rnn_param['batch_norm']
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, dropout=drop_out, batch_norm=False)
        rnns.append(('0', rnn))
        for i in range(rnn_layers - 1):
            rnn = BatchRNN(input_size=self.num_directions * rnn_hidden_size, hidden_size=rnn_hidden_size, rnn_type=rnn_type, bidirectional=bidirectional, dropout=drop_out, batch_norm=batch_norm)
            rnns.append(('%d' % (i + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        if batch_norm:
            self.fc = nn.Sequential(nn.BatchNorm1d(self.num_directions * rnn_hidden_size), nn.Linear(self.num_directions * rnn_hidden_size, num_class, bias=False))
        else:
            self.fc = nn.Linear(self.num_directions * rnn_hidden_size, num_class, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, visualize=False):
        if visualize:
            visual = [x]
        if self.add_cnn:
            x = self.conv(x.unsqueeze(1))
            if visualize:
                visual.append(x)
            x = x.transpose(1, 2).contiguous()
            sizes = x.size()
            if len(sizes) > 3:
                x = x.view(sizes[0], sizes[1], sizes[2] * sizes[3])
            x = x.transpose(0, 1).contiguous()
            if visualize:
                visual.append(x)
            x = self.rnns(x)
            seq_len, batch, _ = x.size()
            x = x.view(seq_len * batch, -1)
            x = self.fc(x)
            x = x.view(seq_len, batch, -1)
            out = self.log_softmax(x)
            if visualize:
                visual.append(out)
                return out, visual
            return out
        else:
            x = x.transpose(0, 1)
            x = self.rnns(x)
            seq_len, batch, _ = x.size()
            x = x.view(seq_len * batch, -1)
            x = self.fc(x)
            x = x.view(seq_len, batch, -1)
            out = self.log_softmax(x)
            if visualize:
                visual.append(out)
                return out, visual
            return out

    def compute_wer(self, index, input_sizes, targets, target_sizes):
        batch_errs = 0
        batch_tokens = 0
        for i in range(len(index)):
            label = targets[i][:target_sizes[i]]
            pred = []
            for j in range(len(index[i][:input_sizes[i]])):
                if index[i][j] == 0:
                    continue
                if j == 0:
                    pred.append(index[i][j])
                if j > 0 and index[i][j] != index[i][j - 1]:
                    pred.append(index[i][j])
            batch_errs += ed.eval(label, pred)
            batch_tokens += len(label)
        return batch_errs, batch_tokens

    def add_weights_noise(self):
        for param in self.parameters():
            weight_noise = param.data.new(param.size()).normal_(0, 0.075).type_as(param.type())
            param = torch.nn.parameter.Parameter(param.data + weight_noise)

    @staticmethod
    def save_package(model, optimizer=None, decoder=None, epoch=None, loss_results=None, dev_loss_results=None, dev_cer_results=None):
        package = {'rnn_param': model.rnn_param, 'add_cnn': model.add_cnn, 'cnn_param': model.cnn_param, 'num_class': model.num_class, '_drop_out': model.drop_out, 'state_dict': model.state_dict()}
        if optimizer is not None:
            package['optim_dict'] = optimizer.state_dict()
        if decoder is not None:
            package['decoder'] = decoder
        if epoch is not None:
            package['epoch'] = epoch
        if loss_results is not None:
            package['loss_results'] = loss_results
            package['dev_loss_results'] = dev_loss_results
            package['dev_cer_results'] = dev_cer_results
        return package


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BatchRNN,
     lambda: ([], {'input_size': 4, 'hidden_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'n_position': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (InferenceBatchLogSoftmax,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (LayerCNN,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': [4, 4], 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SequenceWise,
     lambda: ([], {'module': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_Diamondfan_CTC_pytorch(_paritybench_base):
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

