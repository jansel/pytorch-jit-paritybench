import sys
_module = sys.modules[__name__]
del sys
Linear_Regression = _module
Logistic_Regression = _module
neural_network = _module
convolution_network = _module
logger = _module
recurrent_network = _module
data_utils = _module
Variational_autoencoder = _module
conv_autoencoder = _module
simple_autoencoder = _module
conv_gan = _module
simple_Gan = _module
backward = _module
custom_data_io = _module

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


from torch import nn


from torch.autograd import Variable


import time


import torch.nn.functional as F


from torch.utils.data import DataLoader


from torch import optim


import torch.nn as nn


import math


import random


from collections import namedtuple


from itertools import count


from copy import deepcopy


import torch.optim as optim


class linearRegression(nn.Module):

    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        out = self.linear(x)
        return out


class Logistic_Regression(nn.Module):

    def __init__(self, in_dim, n_class):
        super(Logistic_Regression, self).__init__()
        self.logistic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logistic(x)
        return out


class neuralNetwork(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(neuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(
            True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.
            ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim), nn.ReLU
            (True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Cnn(nn.Module):

    def __init__(self, in_dim, n_class):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, 6, 3, stride=1, padding
            =1), nn.ReLU(True), nn.MaxPool2d(2, 2), nn.Conv2d(6, 16, 5,
            stride=1, padding=0), nn.ReLU(True), nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(400, 120), nn.Linear(120, 84), nn
            .Linear(84, n_class))

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class Rnn(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, (-1), :]
        out = self.classifier(out)
        return out


class NgramModel(nn.Module):

    def __init__(self, vocb_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocb_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob


class CBOW(nn.Module):

    def __init__(self, n_word, n_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(n_word, n_dim)
        self.project = nn.Linear(n_dim, n_dim, bias=False)
        self.linear1 = nn.Linear(n_dim, 128)
        self.linear2 = nn.Linear(128, n_word)

    def forward(self, x):
        x = self.embedding(x)
        x = self.project(x)
        x = torch.sum(x, 0, keepdim=True)
        x = self.linear1(x)
        x = F.relu(x, inplace=True)
        x = self.linear2(x)
        x = F.log_softmax(x)
        return x


class CharLSTM(nn.Module):

    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(n_char, char_dim)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True)

    def forward(self, x):
        x = self.char_embedding(x)
        _, h = self.char_lstm(x)
        return h[0]


character_to_idx = {}


class LSTMTagger(nn.Module):

    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden,
        n_hidden, n_tag):
        super(LSTMTagger, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim)
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        self.lstm = nn.LSTM(n_dim + char_hidden, n_hidden, batch_first=True)
        self.linear1 = nn.Linear(n_hidden, n_tag)

    def forward(self, x, word):
        char = torch.FloatTensor()
        for each in word:
            char_list = []
            for letter in each:
                char_list.append(character_to_idx[letter.lower()])
            char_list = torch.LongTensor(char_list)
            char_list = char_list.unsqueeze(0)
            if torch.cuda.is_available():
                tempchar = self.char_lstm(Variable(char_list))
            else:
                tempchar = self.char_lstm(Variable(char_list))
            tempchar = tempchar.squeeze(0)
            char = torch.cat((char, tempchar.cpu().data), 0)
        if torch.cuda.is_available():
            char = char
        char = Variable(char)
        x = self.word_embedding(x)
        x = torch.cat((x, char), 1)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        x = self.linear1(x)
        y = F.log_softmax(x)
        return y


class languagemodel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super(languagemodel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first
            =True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        x, hi = self.lstm(x, h)
        b, s, h = x.size()
        x = x.contiguous().view(b * s, h)
        x = self.linear(x)
        return x, hi


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, stride=3, padding=
            1), nn.ReLU(True), nn.MaxPool2d(2, stride=2), nn.Conv2d(16, 8, 
            3, stride=2, padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=1))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(True), nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1
            ), nn.ReLU(True), nn.ConvTranspose2d(8, 1, 2, stride=2, padding
            =1), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(True),
            nn.Linear(128, 64), nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(
            True), nn.Linear(12, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 12), nn.ReLU(True), nn.
            Linear(12, 64), nn.ReLU(True), nn.Linear(64, 128), nn.ReLU(True
            ), nn.Linear(128, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, padding=2), nn.
            LeakyReLU(0.2, True), nn.AvgPool2d(2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, padding=2), nn.
            LeakyReLU(0.2, True), nn.AvgPool2d(2, stride=2))
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 1024), nn.LeakyReLU(
            0.2, True), nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x):
        """
        x: batch, width, height, channel=1
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class generator(nn.Module):

    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU(True))
        self.downsample1 = nn.Sequential(nn.Conv2d(1, 50, 3, stride=1,
            padding=1), nn.BatchNorm2d(50), nn.ReLU(True))
        self.downsample2 = nn.Sequential(nn.Conv2d(50, 25, 3, stride=1,
            padding=1), nn.BatchNorm2d(25), nn.ReLU(True))
        self.downsample3 = nn.Sequential(nn.Conv2d(25, 1, 2, stride=2), nn.
            Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn
            .Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.
            Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(100, 256), nn.ReLU(True), nn.
            Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv_bn1 = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2), nn.
            BatchNorm2d(16), nn.ReLU(True))
        self.conv_bn2 = nn.Sequential(nn.Conv2d(16, 32, 5, stride=2), nn.
            BatchNorm2d(32), nn.ReLU(True))
        self.conv_bn3 = nn.Sequential(nn.Conv2d(32, 32, 5, stride=2), nn.
            BatchNorm2d(32), nn.ReLU(True))
        self.move = nn.Linear(448, 2)

    def forward(self, x):
        x = self.conv_bn1(x)
        x = self.conv_bn2(x)
        x = self.conv_bn3(x)
        x = x.view(x.size(0), -1)
        x = self.move(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_L1aoXingyu_pytorch_beginner(_paritybench_base):
    pass
    def test_000(self):
        self._check(CBOW(*[], **{'n_word': 4, 'n_dim': 4, 'context_size': 4}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_001(self):
        self._check(Logistic_Regression(*[], **{'in_dim': 4, 'n_class': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(NgramModel(*[], **{'vocb_size': 4, 'context_size': 4, 'n_dim': 4}), [torch.zeros([4], dtype=torch.int64)], {})

    def test_003(self):
        self._check(Rnn(*[], **{'in_dim': 4, 'hidden_dim': 4, 'n_layer': 1, 'n_class': 4}), [torch.rand([4, 4, 4])], {})

    def test_004(self):
        self._check(autoencoder(*[], **{}), [torch.rand([784, 784])], {})

    def test_005(self):
        self._check(discriminator(*[], **{}), [torch.rand([784, 784])], {})

    def test_006(self):
        self._check(generator(*[], **{}), [torch.rand([100, 100])], {})

    def test_007(self):
        self._check(linearRegression(*[], **{}), [torch.rand([1, 1])], {})

    def test_008(self):
        self._check(neuralNetwork(*[], **{'in_dim': 4, 'n_hidden_1': 4, 'n_hidden_2': 4, 'out_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

