import sys
_module = sys.modules[__name__]
del sys
model = _module
plot = _module
preprocess = _module
train = _module

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


import torch as t


import torch.nn as nn


from torch import LongTensor as LT


from torch import FloatTensor as FT


class Bundler(nn.Module):

    def forward(self, data):
        raise NotImplementedError

    def forward_i(self, data):
        raise NotImplementedError

    def forward_o(self, data):
        raise NotImplementedError


class SGNS(nn.Module):

    def __init__(self, embedding, vocab_size=20000, n_negs=20, weights=None):
        super(SGNS, self).__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size
        self.n_negs = n_negs
        self.weights = None
        if weights is not None:
            wf = np.power(weights, 0.75)
            wf = wf / wf.sum()
            self.weights = FT(wf)

    def forward(self, iword, owords):
        batch_size = iword.size()[0]
        context_size = owords.size()[1]
        if self.weights is not None:
            nwords = t.multinomial(self.weights, batch_size * context_size *
                self.n_negs, replacement=True).view(batch_size, -1)
        else:
            nwords = FT(batch_size, context_size * self.n_negs).uniform_(0,
                self.vocab_size - 1).long()
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        oloss = t.bmm(ovectors, ivectors).squeeze().sigmoid().log().mean(1)
        nloss = t.bmm(nvectors, ivectors).squeeze().sigmoid().log().view(-1,
            context_size, self.n_negs).sum(2).mean(1)
        return -(oloss + nloss).mean()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_theeluwin_pytorch_sgns(_paritybench_base):
    pass
