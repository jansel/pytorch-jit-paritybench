import sys
_module = sys.modules[__name__]
del sys
Dataset = _module
MLP = _module
evaluate = _module
itempop = _module
train_test_split = _module
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


import scipy.sparse as sp


import numpy as np


import pandas as pd


import torch


from torch.utils.data import Dataset


import torch.nn.functional as F


from torch.autograd import Variable


from torch import nn


from torch.utils.data import DataLoader


from time import time


import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()


device = torch.device('cuda:0' if use_cuda else 'cpu')


class MLP(nn.Module):

    def __init__(self, n_users, n_items, layers=[16, 8], dropout=False):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert layers[0] % 2 == 0, 'layers[0] must be an even number'
        self.__alias__ = 'MLP {}'.format(layers)
        self.__dropout__ = dropout
        embedding_dim = int(layers[0] / 2)
        self.user_embedding = torch.nn.Embedding(n_users, embedding_dim)
        self.item_embedding = torch.nn.Embedding(n_items, embedding_dim)
        self.fc_layers = torch.nn.ModuleList()
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.output_layer = torch.nn.Linear(layers[-1], 1)

    def forward(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.__dropout__, training=self.training)
        logit = self.output_layer(x)
        rating = torch.sigmoid(logit)
        return rating

    def predict(self, feed_dict):
        for key in feed_dict:
            if type(feed_dict[key]) != type(None):
                feed_dict[key] = torch.from_numpy(feed_dict[key])
        output_scores = self.forward(feed_dict)
        return output_scores.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__

