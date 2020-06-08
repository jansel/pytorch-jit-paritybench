import sys
_module = sys.modules[__name__]
del sys
build_kaggle_dataset = _module
build_vocab = _module
evaluate = _module
model = _module
data_loader = _module
net = _module
search_hyperparams = _module
synthesize_results = _module
train = _module
utils = _module
build_dataset = _module
evaluate = _module
net = _module
train = _module
utils = _module
evaluation = _module
input_fn = _module
model_fn = _module
training = _module

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


import logging


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.autograd import Variable


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)
        self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim,
            batch_first=True)
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        s = self.embedding(s)
        s, _ = self.lstm(s)
        s = s.contiguous()
        s = s.view(-1, s.shape[2])
        s = self.fc(s)
        return F.log_softmax(s, dim=1)


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions

    such as F.relu, F.sigmoid, F.softmax, F.max_pool2d. Be careful to ensure your dimensions are correct after each
    step. You are encouraged to have a look at the network in pytorch/nlp/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available o you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an convolutional network that predicts the sign from an image. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.num_channels = params.num_channels
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3,
            stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.num_channels * 2)
        self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 4,
            3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(self.num_channels * 4)
        self.fc1 = nn.Linear(8 * 8 * self.num_channels * 4, self.
            num_channels * 4)
        self.fcbn1 = nn.BatchNorm1d(self.num_channels * 4)
        self.fc2 = nn.Linear(self.num_channels * 4, 6)
        self.dropout_rate = params.dropout_rate

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .

        Returns:
            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.

        Note: the dimensions after each step are provided
        """
        s = self.bn1(self.conv1(s))
        s = F.relu(F.max_pool2d(s, 2))
        s = self.bn2(self.conv2(s))
        s = F.relu(F.max_pool2d(s, 2))
        s = self.bn3(self.conv3(s))
        s = F.relu(F.max_pool2d(s, 2))
        s = s.view(-1, 8 * 8 * self.num_channels * 4)
        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))), p=self.dropout_rate,
            training=self.training)
        s = self.fc2(s)
        return F.log_softmax(s, dim=1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cs230_stanford_cs230_code_examples(_paritybench_base):
    pass
