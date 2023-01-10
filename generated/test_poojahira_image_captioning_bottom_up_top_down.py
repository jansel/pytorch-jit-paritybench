import sys
_module = sys.modules[__name__]
del sys
tsv = _module
utils = _module
create_input_files = _module
datasets = _module
eval = _module
models = _module
nlgeval = _module
pycocoevalcap = _module
bleu = _module
bleu_scorer = _module
cider = _module
cider_scorer = _module
meteor = _module
test_meteor = _module
rouge = _module
skipthoughts = _module
tests = _module
test_nlgeval = _module
word2vec = _module
evaluate = _module
generate_w2v_files = _module
setup = _module
test = _module
api = _module
train = _module
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


import numpy as np


import torch


import torch.nn as nn


from torch.utils.data import Dataset


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


import torchvision.transforms as transforms


import torch.nn.functional as F


from torch import nn


import torchvision


from torch.nn.utils.weight_norm import weight_norm


import time


from torch.nn.utils.rnn import pack_padded_sequence


from collections import Counter


from random import seed


from random import choice


from random import sample


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, features_dim, decoder_dim, attention_dim, dropout=0.5):
        """
        :param features_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.features_att = weight_norm(nn.Linear(features_dim, attention_dim))
        self.decoder_att = weight_norm(nn.Linear(decoder_dim, attention_dim))
        self.full_att = weight_norm(nn.Linear(attention_dim, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_features, decoder_hidden):
        """
        Forward propagation.
        :param image_features: encoded images, a tensor of dimension (batch_size, 36, features_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.features_att(image_features)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.dropout(self.relu(att1 + att2.unsqueeze(1)))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (image_features * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, features_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param features_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        self.features_dim = features_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.attention = Attention(features_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.top_down_attention = nn.LSTMCell(embed_dim + features_dim + decoder_dim, decoder_dim, bias=True)
        self.language_model = nn.LSTMCell(features_dim + decoder_dim, decoder_dim, bias=True)
        self.fc1 = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.fc = weight_norm(nn.Linear(decoder_dim, vocab_size))
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size, self.decoder_dim)
        c = torch.zeros(batch_size, self.decoder_dim)
        return h, c

    def forward(self, image_features, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = image_features.size(0)
        vocab_size = self.vocab_size
        image_features_mean = image_features.mean(1)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        image_features = image_features[sort_ind]
        image_features_mean = image_features_mean[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        embeddings = self.embedding(encoded_captions)
        h1, c1 = self.init_hidden_state(batch_size)
        h2, c2 = self.init_hidden_state(batch_size)
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        predictions1 = torch.zeros(batch_size, max(decode_lengths), vocab_size)
        for t in range(max(decode_lengths)):
            batch_size_t = sum([(l > t) for l in decode_lengths])
            h1, c1 = self.top_down_attention(torch.cat([h2[:batch_size_t], image_features_mean[:batch_size_t], embeddings[:batch_size_t, t, :]], dim=1), (h1[:batch_size_t], c1[:batch_size_t]))
            attention_weighted_encoding = self.attention(image_features[:batch_size_t], h1[:batch_size_t])
            preds1 = self.fc1(self.dropout(h1))
            h2, c2 = self.language_model(torch.cat([attention_weighted_encoding[:batch_size_t], h1[:batch_size_t]], dim=1), (h2[:batch_size_t], c2[:batch_size_t]))
            preds = self.fc(self.dropout(h2))
            predictions[:batch_size_t, t, :] = preds
            predictions1[:batch_size_t, t, :] = preds1
        return predictions, predictions1, encoded_captions, decode_lengths, sort_ind


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Attention,
     lambda: ([], {'features_dim': 4, 'decoder_dim': 4, 'attention_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_poojahira_image_captioning_bottom_up_top_down(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

