import sys
_module = sys.modules[__name__]
del sys
active_learning = _module
active_learning_basics = _module
advanced_active_learning = _module
diversity_sampling = _module
pytorch_clusters = _module
uncertainty_sampling = _module

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


import torch.nn.functional as F


import torch.optim as optim


import random


import math


import re


from random import shuffle


from collections import defaultdict


import copy


class SimpleTextClassifier(nn.Module):
    """Text Classifier with 1 hidden layer 

    """

    def __init__(self, num_labels, vocab_size):
        super(SimpleTextClassifier, self).__init__()
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec, return_all_layers=False):
        hidden1 = self.linear1(feature_vec).clamp(min=0)
        output = self.linear2(hidden1)
        log_softmax = F.log_softmax(output, dim=1)
        if return_all_layers:
            return [hidden1, output, log_softmax]
        else:
            return log_softmax


class SimpleTextClassifier(nn.Module):
    """Text Classifier with 1 hidden layer 

    """

    def __init__(self, num_labels, vocab_size):
        super(SimpleTextClassifier, self).__init__()
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec):
        hidden1 = self.linear1(feature_vec).clamp(min=0)
        output = self.linear2(hidden1)
        return F.log_softmax(output, dim=1)


class SimpleUncertaintyPredictor(nn.Module):
    """Simple model to predict whether an item will be classified correctly    

    """

    def __init__(self, vocab_size):
        super(SimpleUncertaintyPredictor, self).__init__()
        self.linear = nn.Linear(vocab_size, 2)

    def forward(self, feature_vec, return_all_layers=False):
        output = self.linear(feature_vec).clamp(min=-1)
        log_softmax = F.log_softmax(output, dim=1)
        if return_all_layers:
            return [output, log_softmax]
        else:
            return log_softmax


class AdvancedUncertaintyPredictor(nn.Module):
    """Simple model to predict whether an item will be classified correctly    

    """

    def __init__(self, vocab_size):
        super(AdvancedUncertaintyPredictor, self).__init__()
        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec, return_all_layers=False):
        hidden1 = self.linear1(feature_vec).clamp(min=0)
        output = self.linear2(hidden1)
        log_softmax = F.log_softmax(output, dim=1)
        if return_all_layers:
            return [hidden1, output, log_softmax]
        else:
            return log_softmax


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_rmunro_pytorch_active_learning(_paritybench_base):
    pass
    def test_000(self):
        self._check(SimpleTextClassifier(*[], **{'num_labels': 4, 'vocab_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(SimpleUncertaintyPredictor(*[], **{'vocab_size': 4}), [torch.rand([4, 4, 4, 4])], {})

