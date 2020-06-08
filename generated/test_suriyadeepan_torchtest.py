import sys
_module = sys.modules[__name__]
del sys
examples = _module
setup = _module
tc = _module
test_torchtest = _module
torchtest = _module
torchtest = _module

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


from torch.autograd import Variable


class LstmClassifier(nn.Module):

    def __init__(self, hparams, weights=None):
        """
    LSTM RNN Classifier

    Args:
      hparams : dictionary of hyperparameters
      
    """
        super(LstmClassifier, self).__init__()
        self.hparams = hparams
        self.weights = weights
        self.embedding = nn.Embedding(hparams['vocab_size'], hparams['emb_dim']
            )
        if weights:
            self.embedding.weight = nn.Parameter(weights['glove'],
                requires_grad=False)
        self.lstm = nn.LSTM(hparams['emb_dim'], hparams['hidden_dim'])
        self.linear = nn.Linear(hparams['hidden_dim'], hparams['output_size'])

    def forward(self, sequence, batch_size=None, get_hidden=False):
        """
    Forward Operation.

    Args:
      sequence : list of indices based off a sentence

    """
        input = self.embedding(sequence)
        input = input.permute(1, 0, 2)
        batch_size = batch_size if batch_size else self.hparams['batch_size']
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(1, batch_size, self.hparams[
                'hidden_dim']))
            c0 = Variable(torch.zeros(1, batch_size, self.hparams[
                'hidden_dim']))
        else:
            h0 = Variable(torch.zeros(1, batch_size, self.hparams[
                'hidden_dim']))
            c0 = Variable(torch.zeros(1, batch_size, self.hparams[
                'hidden_dim']))
        self.lstm.flatten_parameters()
        lstm_out, (h, c) = self.lstm(input, (h0, c0))
        self.h = h[-1]
        linear_out = self.linear(h[-1])
        if get_hidden:
            return linear_out, self.h
        return linear_out


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_suriyadeepan_torchtest(_paritybench_base):
    pass
