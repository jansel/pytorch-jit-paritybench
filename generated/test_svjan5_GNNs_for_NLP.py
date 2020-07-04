import sys
_module = sys.modules[__name__]
del sys
pytorch_gcn = _module
tf_gcn = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


import torch.nn.functional as F


class KipfGCN(torch.nn.Module):

    def __init__(self, data, num_class, params):
        super(KipfGCN, self).__init__()
        self.p = params
        self.data = data
        self.conv1 = GCNConv(self.data.num_features, self.p.gcn_dim, cached
            =True)
        self.conv2 = GCNConv(self.p.gcn_dim, num_class, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.p.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_svjan5_GNNs_for_NLP(_paritybench_base):
    pass
