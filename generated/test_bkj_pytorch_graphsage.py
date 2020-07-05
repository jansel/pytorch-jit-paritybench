import sys
_module = sys.modules[__name__]
del sys
helpers = _module
lr = _module
models = _module
nn_modules = _module
problem = _module
train = _module
convert = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from functools import partial


import torch


from torch import nn


from torch.nn import functional as F


from torch.autograd import Variable


import numpy as np


from scipy import sparse


from scipy.sparse import csr_matrix


from time import time


class LRSchedule(object):

    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @staticmethod
    def constant(x, lr_init=0.1, epochs=1):
        return lr_init

    @staticmethod
    def step(x, breaks=(150, 250)):
        if x < breaks[0]:
            return 0.1
        elif x < breaks[1]:
            return 0.01
        else:
            return 0.001

    @staticmethod
    def linear(x, lr_init=0.1, epochs=1):
        return lr_init * float(epochs - x) / epochs

    @staticmethod
    def cyclical(x, lr_init=0.1, epochs=1):
        """ Cyclical learning rate w/ annealing """
        if x < 1:
            return 0.05
        else:
            return lr_init * (1 - x % 1) * (epochs - np.floor(x)) / epochs


class GSSupervised(nn.Module):

    def __init__(self, input_dim, n_nodes, n_classes, layer_specs, aggregator_class, prep_class, sampler_class, adj, train_adj, lr_init=0.01, weight_decay=0.0, lr_schedule='constant', epochs=10):
        super(GSSupervised, self).__init__()
        self.train_sampler = sampler_class(adj=train_adj)
        self.val_sampler = sampler_class(adj=adj)
        self.train_sample_fns = [partial(self.train_sampler, n_samples=s['n_train_samples']) for s in layer_specs]
        self.val_sample_fns = [partial(self.val_sampler, n_samples=s['n_val_samples']) for s in layer_specs]
        self.prep = prep_class(input_dim=input_dim, n_nodes=n_nodes)
        input_dim = self.prep.output_dim
        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_class(input_dim=input_dim, output_dim=spec['output_dim'], activation=spec['activation'])
            agg_layers.append(agg)
            input_dim = agg.output_dim
        self.agg_layers = nn.Sequential(*agg_layers)
        self.fc = nn.Linear(input_dim, n_classes, bias=True)
        self.lr_scheduler = partial(getattr(LRSchedule, lr_schedule), lr_init=lr_init)
        self.lr = self.lr_scheduler(0.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

    def forward(self, ids, feats, train=True):
        sample_fns = self.train_sample_fns if train else self.val_sample_fns
        has_feats = feats is not None
        tmp_feats = feats[ids] if has_feats else None
        all_feats = [self.prep(ids, tmp_feats, layer_idx=0)]
        for layer_idx, sampler_fn in enumerate(sample_fns):
            ids = sampler_fn(ids=ids).contiguous().view(-1)
            tmp_feats = feats[ids] if has_feats else None
            all_feats.append(self.prep(ids, tmp_feats, layer_idx=layer_idx + 1))
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
        assert len(all_feats) == 1, 'len(all_feats) != 1'
        out = F.normalize(all_feats[0], dim=1)
        return self.fc(out)

    def set_progress(self, progress):
        self.lr = self.lr_scheduler(progress)
        LRSchedule.set_lr(self.optimizer, self.lr)

    def train_step(self, ids, feats, targets, loss_fn):
        self.optimizer.zero_grad()
        preds = self(ids, feats, train=True)
        loss = loss_fn(preds, targets.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return preds


class IdentityPrep(nn.Module):

    def __init__(self, input_dim, n_nodes=None):
        """ Example of preprocessor -- doesn't do anything """
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

    def forward(self, ids, feats, layer_idx=0):
        return feats


class NodeEmbeddingPrep(nn.Module):

    def __init__(self, input_dim, n_nodes, embedding_dim=64):
        """ adds node embedding """
        super(NodeEmbeddingPrep, self).__init__()
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes + 1, embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)

    @property
    def output_dim(self):
        if self.input_dim:
            return self.input_dim + self.embedding_dim
        else:
            return self.embedding_dim

    def forward(self, ids, feats, layer_idx=0):
        if layer_idx > 0:
            embs = self.embedding(ids)
        else:
            embs = self.embedding(Variable(ids.clone().data.zero_() + self.n_nodes))
        embs = self.fc(embs)
        if self.input_dim:
            return torch.cat([feats, embs], dim=1)
        else:
            return embs


class LinearPrep(nn.Module):

    def __init__(self, input_dim, n_nodes, output_dim=32):
        """ adds node embedding """
        super(LinearPrep, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim = output_dim

    def forward(self, ids, feats, layer_idx=0):
        return self.fc(feats)


class AggregatorMixin(object):

    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)


class MeanAggregator(nn.Module, AggregatorMixin):

    def __init__(self, input_dim, output_dim, activation, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanAggregator, self).__init__()
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = agg_neib.mean(dim=1)
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        return out


class PoolAggregator(nn.Module, AggregatorMixin):

    def __init__(self, input_dim, output_dim, pool_fn, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(PoolAggregator, self).__init__()
        self.mlp = nn.Sequential(*[nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU()])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)
        self.output_dim_ = output_dim
        self.activation = activation
        self.pool_fn = pool_fn
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        h_neibs = self.mlp(neibs)
        agg_neib = h_neibs.view(x.size(0), -1, h_neibs.size(1))
        agg_neib = self.pool_fn(agg_neib)
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        return out


class LSTMAggregator(nn.Module, AggregatorMixin):

    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, bidirectional=False, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(LSTMAggregator, self).__init__()
        assert not hidden_dim % 2, 'LSTMAggregator: hiddem_dim % 2 != 0'
        self.lstm = nn.LSTM(input_dim, hidden_dim // (1 + bidirectional), bidirectional=bidirectional, batch_first=True)
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        x_emb = self.fc_x(x)
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib, _ = self.lstm(agg_neib)
        agg_neib = agg_neib[:, (-1), :]
        neib_emb = self.fc_neib(agg_neib)
        out = self.combine_fn([x_emb, neib_emb])
        if self.activation:
            out = self.activation(out)
        return out


class AttentionAggregator(nn.Module, AggregatorMixin):

    def __init__(self, input_dim, output_dim, activation, hidden_dim=32, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(AttentionAggregator, self).__init__()
        self.att = nn.Sequential(*[nn.Linear(input_dim, hidden_dim, bias=False), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim, bias=False)])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        neib_att = self.att(neibs)
        x_att = self.att(x)
        neib_att = neib_att.view(x.size(0), -1, neib_att.size(1))
        x_att = x_att.view(x_att.size(0), x_att.size(1), 1)
        ws = F.softmax(torch.bmm(neib_att, x_att).squeeze())
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.sum(agg_neib * ws.unsqueeze(-1), dim=1)
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttentionAggregator,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (IdentityPrep,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LSTMAggregator,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (LinearPrep,
     lambda: ([], {'input_dim': 4, 'n_nodes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (MeanAggregator,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (NodeEmbeddingPrep,
     lambda: ([], {'input_dim': 4, 'n_nodes': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64), torch.rand([4, 4])], {}),
     False),
    (PoolAggregator,
     lambda: ([], {'input_dim': 4, 'output_dim': 4, 'pool_fn': _mock_layer(), 'activation': _mock_layer()}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_bkj_pytorch_graphsage(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

    def test_003(self):
        self._check(*TESTCASES[3])

    def test_004(self):
        self._check(*TESTCASES[4])

    def test_005(self):
        self._check(*TESTCASES[5])

    def test_006(self):
        self._check(*TESTCASES[6])

