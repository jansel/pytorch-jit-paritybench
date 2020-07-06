import sys
_module = sys.modules[__name__]
del sys
master = _module
dataset = _module
collate_fn = _module
data_transform = _module
test_data = _module
text_data = _module
lr_policy = _module
models = _module
crnn = _module
model_loader = _module
test = _module
train = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


from torch.utils.data import Dataset


import string


import random


import torch.nn as nn


from torch.autograd import Variable


import torchvision.models as models


from collections import OrderedDict


from torch import nn


from torch.utils.data import DataLoader


from torchvision.transforms import Compose


from torch import optim


from torch import Tensor


class CRNN(nn.Module):

    def __init__(self, abc=string.digits, backend='resnet18', rnn_hidden_size=128, rnn_num_layers=2, rnn_dropout=0, seq_proj=[0, 0]):
        super(CRNN, self).__init__()
        self.abc = abc
        self.num_classes = len(self.abc)
        self.feature_extractor = getattr(models, backend)(pretrained=True)
        self.cnn = nn.Sequential(self.feature_extractor.conv1, self.feature_extractor.bn1, self.feature_extractor.relu, self.feature_extractor.maxpool, self.feature_extractor.layer1, self.feature_extractor.layer2, self.feature_extractor.layer3, self.feature_extractor.layer4)
        self.fully_conv = seq_proj[0] == 0
        if not self.fully_conv:
            self.proj = nn.Conv2d(seq_proj[0], seq_proj[1], kernel_size=1)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.rnn = nn.GRU(self.get_block_size(self.cnn), rnn_hidden_size, rnn_num_layers, batch_first=False, dropout=rnn_dropout, bidirectional=True)
        self.linear = nn.Linear(rnn_hidden_size * 2, self.num_classes + 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, decode=False):
        hidden = self.init_hidden(x.size(0), next(self.parameters()).is_cuda)
        features = self.cnn(x)
        features = self.features_to_sequence(features)
        seq, hidden = self.rnn(features, hidden)
        seq = self.linear(seq)
        if not self.training:
            seq = self.softmax(seq)
            if decode:
                seq = self.decode(seq)
        return seq

    def init_hidden(self, batch_size, gpu=False):
        h0 = Variable(torch.zeros(self.rnn_num_layers * 2, batch_size, self.rnn_hidden_size))
        if gpu:
            h0 = h0
        return h0

    def features_to_sequence(self, features):
        b, c, h, w = features.size()
        assert h == 1, 'the height of out must be 1'
        if not self.fully_conv:
            features = features.permute(0, 3, 2, 1)
            features = self.proj(features)
            features = features.permute(1, 0, 2, 3)
        else:
            features = features.permute(3, 0, 2, 1)
        features = features.squeeze(2)
        return features

    def get_block_size(self, layer):
        return layer[-1][-1].bn2.weight.size()[0]

    def pred_to_string(self, pred):
        seq = []
        for i in range(pred.shape[0]):
            label = np.argmax(pred[i])
            seq.append(label - 1)
        out = []
        for i in range(len(seq)):
            if len(out) == 0:
                if seq[i] != -1:
                    out.append(seq[i])
            elif seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
        out = ''.join(self.abc[i] for i in out)
        return out

    def decode(self, pred):
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        seq = []
        for i in range(pred.shape[0]):
            seq.append(self.pred_to_string(pred[i]))
        return seq

