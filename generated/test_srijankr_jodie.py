import sys
_module = sys.modules[__name__]
del sys
evaluate_interaction_prediction = _module
evaluate_state_change_prediction = _module
get_final_performance_numbers = _module
jodie = _module
library_data = _module
library_models = _module
tbatch = _module

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


import torch


import torch.nn as nn


from torch.nn import functional as F


from torch.autograd import Variable


from torch import optim


import numpy as np


import math


import random


from collections import defaultdict


from itertools import chain


class NormalLinear(nn.Linear):

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)


class JODIE(nn.Module):

    def __init__(self, args, num_features, num_users, num_items):
        super(JODIE, self).__init__()
        None
        self.modelname = args.model
        self.embedding_dim = args.embedding_dim
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embedding_size = num_users
        self.item_static_embedding_size = num_items
        None
        self.initial_user_embedding = nn.Parameter(torch.Tensor(args.
            embedding_dim))
        self.initial_item_embedding = nn.Parameter(torch.Tensor(args.
            embedding_dim))
        rnn_input_size_items = rnn_input_size_users = (self.embedding_dim +
            1 + num_features)
        None
        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embedding_dim)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embedding_dim)
        None
        self.linear_layer1 = nn.Linear(self.embedding_dim, 50)
        self.linear_layer2 = nn.Linear(50, 2)
        self.prediction_layer = nn.Linear(self.user_static_embedding_size +
            self.item_static_embedding_size + self.embedding_dim * 2, self.
            item_static_embedding_size + self.embedding_dim)
        self.embedding_layer = NormalLinear(1, self.embedding_dim)
        None

    def forward(self, user_embeddings, item_embeddings, timediffs=None,
        features=None, select=None):
        if select == 'item_update':
            input1 = torch.cat([user_embeddings, timediffs, features], dim=1)
            item_embedding_output = self.item_rnn(input1, item_embeddings)
            return F.normalize(item_embedding_output)
        elif select == 'user_update':
            input2 = torch.cat([item_embeddings, timediffs, features], dim=1)
            user_embedding_output = self.user_rnn(input2, user_embeddings)
            return F.normalize(user_embedding_output)
        elif select == 'project':
            user_projected_embedding = self.context_convert(user_embeddings,
                timediffs, features)
            return user_projected_embedding

    def context_convert(self, embeddings, timediffs, features):
        new_embeddings = embeddings * (1 + self.embedding_layer(timediffs))
        return new_embeddings

    def predict_label(self, user_embeddings):
        X_out = nn.ReLU()(self.linear_layer1(user_embeddings))
        X_out = self.linear_layer2(X_out)
        return X_out

    def predict_item_embedding(self, user_embeddings):
        X_out = self.prediction_layer(user_embeddings)
        return X_out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_srijankr_jodie(_paritybench_base):
    pass

    def test_000(self):
        self._check(NormalLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(JODIE(*[], **{'args': _mock_config(model=4, embedding_dim=4), 'num_features': 4, 'num_users': 4, 'num_items': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})
