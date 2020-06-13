import sys
_module = sys.modules[__name__]
del sys
configs = _module
data_loader = _module
feature_extraction = _module
layers = _module
discriminator = _module
lstmcell = _module
summarizer = _module
weight_norm = _module
solver = _module
train = _module
utils = _module

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


import torch.nn as nn


import torch


from torch.autograd import Variable


from torch.nn.utils import weight_norm


import torch.optim as optim


import numpy as np


class ResNetFeature(nn.Module):

    def __init__(self, feature='resnet101'):
        """
        Args:
            feature (string): resnet101 or resnet152
        """
        super(ResNetFeature, self).__init__()
        if feature == 'resnet101':
            resnet = models.resnet101(pretrained=True)
        else:
            resnet = models.resnet152(pretrained=True)
        resnet.float()
        resnet
        resnet.eval()
        module_list = list(resnet.children())
        self.conv5 = nn.Sequential(*module_list[:-2])
        self.pool5 = module_list[-2]

    def forward(self, x):
        res5c = self.conv5(x)
        pool5 = self.pool5(res5c)
        pool5 = pool5.view(pool5.size(0), -1)
        return res5c, pool5


class cLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        """Discriminator LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, input_size]
        Return:
            last_h: [1, hidden_size]
        """
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(features, init_hidden)
        last_h = h_n[-1]
        return last_h


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        """Discriminator: cLSTM + output projection to probability"""
        super().__init__()
        self.cLSTM = cLSTM(input_size, hidden_size, num_layers)
        self.out = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h : [1, hidden_size]
                Last h from top layer of discriminator
            prob: [1=batch_size, 1]
                Probability to be original feature from CNN
        """
        h = self.cLSTM(features)
        prob = self.out(h).squeeze()
        return h, prob


class StackedLSTMCell(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout=0.0):
        super(StackedLSTMCell, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, x, h_c):
        """
        Args:
            x: [batch_size, input_size]
            h_c: [2, num_layers, batch_size, hidden_size]
        Return:
            last_h_c: [2, batch_size, hidden_size] (h from last layer)
            h_c_list: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
        """
        h_0, c_0 = h_c
        h_list, c_list = [], []
        for i, layer in enumerate(self.layers):
            h_i, c_i = layer(x, (h_0[i], c_0[i]))
            x = h_i
            if i + 1 != self.num_layers:
                x = self.dropout(x)
            h_list += [h_i]
            c_list += [c_i]
        last_h_c = h_list[-1], c_list[-1]
        h_list = torch.stack(h_list)
        c_list = torch.stack(c_list)
        h_c_list = h_list, c_list
        return last_h_c, h_c_list


class sLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
            bidirectional=True)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, 1), nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 100] (compressed pool5 features)
        Return:
            scores [seq_len, 1]
        """
        self.lstm.flatten_parameters()
        features, (h_n, c_n) = self.lstm(features)
        scores = self.out(features.squeeze(1))
        return scores


class eLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear_mu = nn.Linear(hidden_size, hidden_size)
        self.linear_var = nn.Linear(hidden_size, hidden_size)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, 1, hidden_size]
        Return:
            last hidden
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """
        self.lstm.flatten_parameters()
        _, (h_last, c_last) = self.lstm(frame_features)
        return h_last, c_last


class dLSTM(nn.Module):

    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        """Decoder LSTM"""
        super().__init__()
        self.lstm_cell = StackedLSTMCell(num_layers, input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, init_hidden):
        """
        Args:
            seq_len (int)
            init_hidden
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """
        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)
        x = Variable(torch.zeros(batch_size, hidden_size))
        h, c = init_hidden
        out_features = []
        for i in range(seq_len):
            (last_h, last_c), (h, c) = self.lstm_cell(x, (h, c))
            x = self.out(last_h)
            out_features.append(last_h)
        return out_features


class VAE(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)
        self.softplus = nn.Softplus()

    def reparameterize(self, mu, log_variance):
        """Sample z via reparameterization trick
        Args:
            mu: [num_layers, hidden_size]
            log_var: [num_layers, hidden_size]
        Return:
            h: [num_layers, 1, hidden_size]
        """
        std = torch.exp(0.5 * log_variance)
        epsilon = Variable(torch.randn(std.size()))
        return (mu + epsilon * std).unsqueeze(1)

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            h: [2=num_layers, 1, hidden_size]
            decoded_features: [seq_len, 1, 2048]
        """
        seq_len = features.size(0)
        h, c = self.e_lstm(features)
        h = h.squeeze(1)
        h_mu = self.e_lstm.linear_mu(h)
        h_log_variance = torch.log(self.softplus(self.e_lstm.linear_var(h)))
        h = self.reparameterize(h_mu, h_log_variance)
        decoded_features = self.d_lstm(seq_len, init_hidden=(h, c))
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return h_mu, h_log_variance, decoded_features


class Summarizer(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.s_lstm = sLSTM(input_size, hidden_size, num_layers)
        self.vae = VAE(input_size, hidden_size, num_layers)

    def forward(self, image_features, uniform=False):
        if not uniform:
            scores = self.s_lstm(image_features)
            weighted_features = image_features * scores.view(-1, 1, 1)
        else:
            scores = None
            weighted_features = image_features
        h_mu, h_log_variance, decoded_features = self.vae(weighted_features)
        return scores, h_mu, h_log_variance, decoded_features


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_j_min_Adversarial_Video_Summary(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(cLSTM(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Discriminator(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(sLSTM(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(eLSTM(*[], **{'input_size': 4, 'hidden_size': 4}), [torch.rand([4, 4, 4])], {})

