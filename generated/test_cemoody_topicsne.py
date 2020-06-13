import sys
_module = sys.modules[__name__]
del sys
run = _module
topic_sne = _module
tsne = _module
vtsne = _module
wrapper = _module

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


import torch.autograd


import torch.nn.functional as F


from torch.autograd import Variable


from torch import nn


import numpy as np


def pairwise(data):
    n_obs, dim = data.size()
    xk = data.unsqueeze(0).expand(n_obs, n_obs, dim)
    xl = data.unsqueeze(1).expand(n_obs, n_obs, dim)
    dkl2 = ((xk - xl) ** 2.0).sum(2).squeeze()
    return dkl2


class TopicSNE(nn.Module):

    def __init__(self, n_points, n_topics, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TopicSNE, self).__init__()
        self.logits = nn.Embedding(n_points, n_topics)
        self.topic = nn.Linear(n_topics, n_dim)

    def positions(self):
        x = self.logits.weight
        return x

    def dirichlet_ll(self):
        pass

    def forward(self, pij, i, j):
        with torch.cuda.device(pij.get_device()):
            alli = torch.from_numpy(np.arange(self.n_points))
            alli = Variable(alli)
        x = self.logits(alli)
        dkl2 = pairwise(x)
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        xi = self.logits(i)
        xj = self.logits(j)
        num = (1.0 + (xi - xj) ** 2.0).sum(1).pow(-1.0).squeeze()
        qij = num / part.expand_as(num)
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)


class TSNE(nn.Module):

    def __init__(self, n_points, n_topics, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(TSNE, self).__init__()
        self.logits = nn.Embedding(n_points, n_topics)

    def forward(self, pij, i, j):
        x = self.logits.weight
        dkl2 = pairwise(x)
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        xi = self.logits(i)
        xj = self.logits(j)
        num = (1.0 + (xi - xj) ** 2.0).sum(1).pow(-1.0).squeeze()
        qij = num / part.expand_as(num)
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        return loss_kld.sum()

    def __call__(self, *args):
        return self.forward(*args)


class VTSNE(nn.Module):

    def __init__(self, n_points, n_topics, n_dim):
        self.n_points = n_points
        self.n_dim = n_dim
        super(VTSNE, self).__init__()
        self.logits_mu = nn.Embedding(n_points, n_topics)
        self.logits_lv = nn.Embedding(n_points, n_topics)

    @property
    def logits(self):
        return self.logits_mu

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        kld = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld = torch.sum(kld).mul_(-0.5)
        return z, kld

    def sample_logits(self, i=None):
        if i is None:
            return self.reparametrize(self.logits_mu.weight, self.logits_lv
                .weight)
        else:
            return self.reparametrize(self.logits_mu(i), self.logits_lv(i))

    def forward(self, pij, i, j):
        x, loss_kldrp = self.sample_logits()
        dkl2 = pairwise(x)
        n_diagonal = dkl2.size()[0]
        part = (1 + dkl2).pow(-1.0).sum() - n_diagonal
        xi, _ = self.sample_logits(i)
        xj, _ = self.sample_logits(j)
        num = (1.0 + (xi - xj) ** 2.0).sum(1).pow(-1.0).squeeze()
        qij = num / part.expand_as(num)
        loss_kld = pij * (torch.log(pij) - torch.log(qij))
        return loss_kld.sum() + loss_kldrp.sum() * 1e-07

    def __call__(self, *args):
        return self.forward(*args)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cemoody_topicsne(_paritybench_base):
    pass
    def test_000(self):
        self._check(TSNE(*[], **{'n_points': 4, 'n_topics': 4, 'n_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {})

    @_fails_compile()
    def test_001(self):
        self._check(TopicSNE(*[], **{'n_points': 4, 'n_topics': 4, 'n_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64), torch.zeros([4], dtype=torch.int64)], {})

