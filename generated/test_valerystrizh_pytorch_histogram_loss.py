import sys
_module = sys.modules[__name__]
del sys
datasets = _module
evaluation = _module
evaluation_old = _module
layers = _module
losses = _module
main = _module
samplers = _module
visualizer = _module

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


import torch


import torch.nn as nn


from torch.autograd import Variable


from scipy.spatial.distance import cdist


from numpy.testing import assert_almost_equal


import random


import time


import torch.optim as optim


from torch.optim import lr_scheduler


from torch.utils.data import DataLoader


class DropoutShared(nn.Module):

    def __init__(self, p=0.5, use_gpu=True):
        super(DropoutShared, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                'dropout probability has to be between 0 and 1, but got {}'
                .format(p))
        self.p = p
        self.use_gpu = use_gpu

    def forward(self, input):
        if self.training:
            index = torch.arange(0, input.size()[1])[torch.Tensor(input.
                size()[1]).uniform_(0, 1).le(self.p)].long()
            input_cloned = input.clone()
            if self.use_gpu:
                input_cloned[:, (index)] = 0
            else:
                input_cloned[:, (index)] = 0
            return input_cloned / (1 - self.p)
        else:
            return input

    def __repr__(self):
        return self.__class__.__name__ + ' (' + 'p=' + str(self.p) + ')'


class L2Normalization(nn.Module):

    def __init__(self):
        super(L2Normalization, self).__init__()

    def forward(self, input):
        input = input.squeeze()
        return input.div(torch.norm(input, dim=1).view(-1, 1))

    def __repr__(self):
        return self.__class__.__name__


class HistogramLoss(torch.nn.Module):

    def __init__(self, num_steps, cuda=True):
        super(HistogramLoss, self).__init__()
        self.step = 2 / (num_steps - 1)
        self.eps = 1 / num_steps
        self.cuda = cuda
        self.t = torch.arange(-1, 1 + self.step, self.step).view(-1, 1)
        self.tsize = self.t.size()[0]
        if self.cuda:
            self.t = self.t

    def forward(self, features, classes):

        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (s_repeat_floor - (self.t - self.step) > -self.eps) & (
                s_repeat_floor - (self.t - self.step) < self.eps) & inds
            assert indsa.nonzero().size()[0
                ] == size, 'Another number of bins should be used'
            zeros = torch.zeros((1, indsa.size()[1])).byte()
            if self.cuda:
                zeros = zeros
            indsb = torch.cat((indsa, zeros))[1:, :]
            s_repeat_[~(indsb | indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - self.t + self.step)[indsa
                ] / self.step
            s_repeat_[indsb] = (-s_repeat_ + self.t + self.step)[indsb
                ] / self.step
            return s_repeat_.sum(1) / size
        classes_size = classes.size()[0]
        classes_eq = (classes.repeat(classes_size, 1) == classes.view(-1, 1
            ).repeat(1, classes_size)).data
        dists = torch.mm(features, features.transpose(0, 1))
        assert (dists > 1 + self.eps).sum().item() + (dists < -1 - self.eps
            ).sum().item() == 0, 'L2 normalization should be used'
        s_inds = torch.triu(torch.ones(classes_eq.size()), 1).byte()
        if self.cuda:
            s_inds = s_inds
        pos_inds = classes_eq[s_inds].repeat(self.tsize, 1)
        neg_inds = ~classes_eq[s_inds].repeat(self.tsize, 1)
        pos_size = classes_eq[s_inds].sum().item()
        neg_size = (~classes_eq[s_inds]).sum().item()
        s = dists[s_inds].view(1, -1)
        s_repeat = s.repeat(self.tsize, 1)
        s_repeat_floor = (torch.floor(s_repeat.data / self.step) * self.step
            ).float()
        histogram_pos = histogram(pos_inds, pos_size)
        assert_almost_equal(histogram_pos.sum().item(), 1, decimal=1,
            err_msg='Not good positive histogram', verbose=True)
        histogram_neg = histogram(neg_inds, neg_size)
        assert_almost_equal(histogram_neg.sum().item(), 1, decimal=1,
            err_msg='Not good negative histogram', verbose=True)
        histogram_pos_repeat = histogram_pos.view(-1, 1).repeat(1,
            histogram_pos.size()[0])
        histogram_pos_inds = torch.tril(torch.ones(histogram_pos_repeat.
            size()), -1).byte()
        if self.cuda:
            histogram_pos_inds = histogram_pos_inds
        histogram_pos_repeat[histogram_pos_inds] = 0
        histogram_pos_cdf = histogram_pos_repeat.sum(0)
        loss = torch.sum(histogram_neg * histogram_pos_cdf)
        return loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_valerystrizh_pytorch_histogram_loss(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(DropoutShared(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

