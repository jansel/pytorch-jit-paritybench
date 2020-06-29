import sys
_module = sys.modules[__name__]
del sys
conv = _module
data = _module
model = _module
preprocess_OAG = _module
train_author_disambiguation = _module
train_paper_field = _module
train_paper_venue = _module
utils = _module

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


import math


from warnings import filterwarnings


class RelTemporalEncoding(nn.Module):
    """
        Implement the Temporal Encoding (Sinusoid) function.
    """

    def __init__(self, n_hid, max_len=240, dropout=0.2):
        super(RelTemporalEncoding, self).__init__()
        self.drop = nn.Dropout(dropout)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = 1 / (10000 ** torch.arange(0.0, n_hid * 2, 2.0) / n_hid / 2)
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = torch.sin(position * div_term
            ) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = torch.cos(position * div_term
            ) / math.sqrt(n_hid)
        self.emb.requires_grad = False
        self.lin = nn.Linear(n_hid * 2, n_hid)

    def forward(self, x, t):
        return x + self.lin(self.drop(self.emb(t)))


class Classifier(nn.Module):

    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(self.__class__.__name__,
            self.n_hid, self.n_out)


class Matcher(nn.Module):
    """
        Matching between a pair of nodes to conduct link prediction.
        Use multi-head attention as matching model.
    """

    def __init__(self, n_hid):
        super(Matcher, self).__init__()
        self.left_linear = nn.Linear(n_hid, n_hid)
        self.right_linear = nn.Linear(n_hid, n_hid)
        self.sqrt_hd = math.sqrt(n_hid)
        self.cache = None

    def forward(self, x, y, infer=False, pair=False):
        ty = self.right_linear(y)
        if infer:
            """
                During testing, we will consider millions or even billions of nodes as candidates (x).
                It's not possible to calculate them again for different query (y)
                Since the model is fixed, we propose to cache them, and dirrectly use the results.
            """
            if self.cache != None:
                tx = self.cache
            else:
                tx = self.left_linear(x)
                self.cache = tx
        else:
            tx = self.left_linear(x)
        if pair:
            res = (tx * ty).sum(dim=-1)
        else:
            res = torch.matmul(tx, ty.transpose(0, 1))
        return res / self.sqrt_hd

    def __repr__(self):
        return '{}(n_hid={})'.format(self.__class__.__name__, self.n_hid)


class GNN(nn.Module):

    def __init__(self, in_dim, n_hid, num_types, num_relations, n_heads,
        n_layers, dropout=0.2, conv_name='hgt'):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.num_types = num_types
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.adapt_ws = nn.ModuleList()
        self.drop = nn.Dropout(dropout)
        for t in range(num_types):
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))
        for l in range(n_layers):
            self.gcs.append(GeneralConv(conv_name, n_hid, n_hid, num_types,
                num_relations, n_heads, dropout))

    def forward(self, node_feature, node_type, edge_time, edge_index, edge_type
        ):
        res = torch.zeros(node_feature.size(0), self.n_hid)
        for t_id in range(self.num_types):
            idx = node_type == int(t_id)
            if idx.sum() == 0:
                continue
            res[idx] = torch.tanh(self.adapt_ws[t_id](node_feature[idx]))
        meta_xs = self.drop(res)
        del res
        for gc in self.gcs:
            meta_xs = gc(meta_xs, node_type, edge_index, edge_type, edge_time)
        return meta_xs


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_acbull_pyHGT(_paritybench_base):
    pass
    def test_000(self):
        self._check(Classifier(*[], **{'n_hid': 4, 'n_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Matcher(*[], **{'n_hid': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(RelTemporalEncoding(*[], **{'n_hid': 4}), [torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {})

