import sys
_module = sys.modules[__name__]
del sys
data = _module
dataset = _module
main = _module
DeepFM = _module
model = _module
utils = _module
dataPreprocess = _module

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


import torch


from torch.utils.data import Dataset


import pandas as pd


import numpy as np


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.data import sampler


import torch.nn as nn


import torch.nn.functional as F


from time import time


class DeepFM(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this 
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, embedding_size=4, hidden_dims=[32, 32], num_classes=1, dropout=[0.5, 0.5], use_cuda=True, verbose=False):
        """
        Initialize a new network

        Inputs: 
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.long
        self.bias = torch.nn.Parameter(torch.randn(1))
        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """
        self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        """
            init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i), nn.Linear(all_dims[i - 1], all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i), nn.Dropout(dropout[i - 1]))

    def forward(self, Xi, Xv):
        """
        Forward process of network. 

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """
        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, (i), :]), 1).t() * Xv[:, (i)]).t() for i, emb in enumerate(self.fm_first_order_embeddings)]
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, (i), :]), 1).t() * Xv[:, (i)]).t() for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb
        fm_second_order_emb_square = [(item * item) for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)
        fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        """
            deep part
        """
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        """
            sum
        """
        total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(deep_out, 1) + self.bias
        return total_sum

    def fit(self, loader_train, loader_val, optimizer, epochs=100, verbose=False, print_every=100):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations. 
        """
        """
            load input data
        """
        model = self.train()
        criterion = F.binary_cross_entropy_with_logits
        for _ in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi
                xv = xv
                y = y
                total = model(xi, xv)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose and t % print_every == 0:
                    None
                    self.check_accuracy(loader_val, model)
                    None

    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            None
        else:
            None
        num_correct = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi
                xv = xv
                y = y
                total = model(xi, xv)
                preds = F.sigmoid(total) > 0.5
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            None


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DeepFM,
     lambda: ([], {'feature_sizes': [4, 4]}),
     lambda: ([torch.ones([4, 4, 4], dtype=torch.int64), torch.rand([4, 4, 4])], {}),
     False),
]

class Test_chenxijun1029_DeepFM_with_PyTorch(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

