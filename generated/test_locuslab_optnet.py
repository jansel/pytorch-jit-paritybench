import sys
_module = sys.modules[__name__]
del sys
densenet = _module
models = _module
plot = _module
train = _module
create = _module
main = _module
models = _module
models = _module
train = _module

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


import torch.nn as nn


import torch.optim as optim


import torch.nn.functional as F


from torch.autograd import Variable


import torchvision.datasets as dset


import torchvision.transforms as transforms


from torch.utils.data import DataLoader


import torchvision.models as models


import math


from torch.autograd import Function


from torch.nn.parameter import Parameter


from torchvision.utils import save_image


import numpy as np


import numpy.random as npr


from itertools import product


import scipy.sparse as spa


from torch.nn import Module


import time


class Bottleneck(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):

    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
            padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):

    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias
            =False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()
        nDenseBlocks = (depth - 4) // 3
        if bottleneck:
            nDenseBlocks //= 2
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias
            =False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
            bottleneck)
        nChannels += nDenseBlocks * growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc = nn.Linear(nChannels, nClasses)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = F.log_softmax(self.fc(out))
        return out


class Lenet(nn.Module):

    def __init__(self, nHidden, nCls=10, proj='softmax'):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.fc1 = nn.Linear(50 * 4 * 4, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)
        self.proj = proj
        self.nCls = nCls
        if proj == 'simproj':
            self.Q = Variable(0.5 * torch.eye(nCls).double())
            self.G = Variable(-torch.eye(nCls).double())
            self.h = Variable(-1e-05 * torch.ones(nCls).double())
            self.A = Variable(torch.ones(1, nCls).double())
            self.b = Variable(torch.Tensor([1.0]).double())

            def projF(x):
                nBatch = x.size(0)
                Q = self.Q.unsqueeze(0).expand(nBatch, nCls, nCls)
                G = self.G.unsqueeze(0).expand(nBatch, nCls, nCls)
                h = self.h.unsqueeze(0).expand(nBatch, nCls)
                A = self.A.unsqueeze(0).expand(nBatch, 1, nCls)
                b = self.b.unsqueeze(0).expand(nBatch, 1)
                x = QPFunction()(Q, -x.double(), G, h, A, b).float()
                x = x.log()
                return x
            self.projF = projF
        else:
            self.projF = F.log_softmax

    def forward(self, x):
        nBatch = x.size(0)
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.projF(x)


class LenetOptNet(nn.Module):

    def __init__(self, nHidden=50, nineq=200, neq=0, eps=0.0001):
        super(LenetOptNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.qp_o = nn.Linear(50 * 4 * 4, nHidden)
        self.qp_z0 = nn.Linear(50 * 4 * 4, nHidden)
        self.qp_s0 = nn.Linear(50 * 4 * 4, nineq)
        assert neq == 0
        self.M = Variable(torch.tril(torch.ones(nHidden, nHidden)))
        self.L = Parameter(torch.tril(torch.rand(nHidden, nHidden)))
        self.G = Parameter(torch.Tensor(nineq, nHidden).uniform_(-1, 1))
        self.nHidden = nHidden
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

    def forward(self, x):
        nBatch = x.size(0)
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(nBatch, -1)
        L = self.M * self.L
        Q = L.mm(L.t()) + self.eps * Variable(torch.eye(self.nHidden))
        Q = Q.unsqueeze(0).expand(nBatch, self.nHidden, self.nHidden)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nHidden)
        z0 = self.qp_z0(x)
        s0 = self.qp_s0(x)
        h = z0.mm(self.G.t()) + s0
        e = Variable(torch.Tensor())
        inputs = self.qp_o(x)
        x = QPFunction()(Q, inputs, G, h, e, e)
        x = x[:, :10]
        return F.log_softmax(x)


class FC(nn.Module):

    def __init__(self, nHidden, bn):
        super().__init__()
        self.bn = bn
        self.fc1 = nn.Linear(784, nHidden)
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(nHidden, 10)
        self.fc3 = nn.Linear(10, 10)

    def forward(self, x):
        nBatch = x.size(0)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)
        x = self.fc3(x)
        return F.log_softmax(x)


class OptNet(nn.Module):

    def __init__(self, nFeatures, nHidden, nCls, bn, nineq=200, neq=0, eps=
        0.0001):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.bn = bn
        self.nCls = nCls
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)
            self.bn2 = nn.BatchNorm1d(nCls)
        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)
        assert neq == 0
        self.M = Variable(torch.tril(torch.ones(nCls, nCls)))
        self.L = Parameter(torch.tril(torch.rand(nCls, nCls)))
        self.G = Parameter(torch.Tensor(nineq, nCls).uniform_(-1, 1))
        self.z0 = Parameter(torch.zeros(nCls))
        self.s0 = Parameter(torch.ones(nineq))
        self.nineq = nineq
        self.neq = neq
        self.eps = eps

    def forward(self, x):
        nBatch = x.size(0)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = F.relu(self.fc2(x))
        if self.bn:
            x = self.bn2(x)
        L = self.M * self.L
        Q = L.mm(L.t()) + self.eps * Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nCls)
        z0 = self.z0.unsqueeze(0).expand(nBatch, self.nCls)
        s0 = self.s0.unsqueeze(0).expand(nBatch, self.nineq)
        h = z0.mm(self.G.t()) + s0
        e = Variable(torch.Tensor())
        inputs = x
        x = QPFunction(verbose=-1)(Q.double(), inputs.double(), G.double(),
            h.double(), e, e)
        x = x.float()
        return F.log_softmax(x)


class OptNetEq(nn.Module):

    def __init__(self, nFeatures, nHidden, nCls, neq, Qpenalty=0.1, eps=0.0001
        ):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.nCls = nCls
        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nCls)
        self.Q = Variable(Qpenalty * torch.eye(nHidden).double())
        self.G = Variable(-torch.eye(nHidden).double())
        self.h = Variable(torch.zeros(nHidden).double())
        self.A = Parameter(torch.rand(neq, nHidden).double())
        self.b = Variable(torch.ones(self.A.size(0)).double())
        self.neq = neq

    def forward(self, x):
        nBatch = x.size(0)
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        Q = self.Q.unsqueeze(0).expand(nBatch, self.Q.size(0), self.Q.size(1))
        p = -x.view(nBatch, -1)
        G = self.G.unsqueeze(0).expand(nBatch, self.G.size(0), self.G.size(1))
        h = self.h.unsqueeze(0).expand(nBatch, self.h.size(0))
        A = self.A.unsqueeze(0).expand(nBatch, self.A.size(0), self.A.size(1))
        b = self.b.unsqueeze(0).expand(nBatch, self.b.size(0))
        x = QPFunction(verbose=False)(Q, p.double(), G, h, A, b).float()
        x = self.fc2(x)
        return F.log_softmax(x)


class ReluNet(nn.Module):

    def __init__(self, nFeatures, nHidden, bn=False):
        super().__init__()
        self.bn = bn
        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.fc2 = nn.Linear(nHidden, nFeatures)
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden)

    def __call__(self, x):
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        x = self.fc2(x)
        return x


class OptNet(nn.Module):

    def __init__(self, nFeatures, args):
        super(OptNet, self).__init__()
        nHidden, neq, nineq = 2 * nFeatures - 1, 0, 2 * nFeatures - 2
        assert neq == 0
        self.fc1 = nn.Linear(nFeatures, nHidden)
        self.M = Variable(torch.tril(torch.ones(nHidden, nHidden)))
        if args.tvInit:
            Q = 1e-08 * torch.eye(nHidden)
            Q[:nFeatures, :nFeatures] = torch.eye(nFeatures)
            self.L = Parameter(torch.potrf(Q))
            D = torch.zeros(nFeatures - 1, nFeatures)
            D[:nFeatures - 1, :nFeatures - 1] = torch.eye(nFeatures - 1)
            D[:nFeatures - 1, 1:nFeatures] -= torch.eye(nFeatures - 1)
            G_ = block(((D, -torch.eye(nFeatures - 1)), (-D, -torch.eye(
                nFeatures - 1))))
            self.G = Parameter(G_)
            self.s0 = Parameter(torch.ones(2 * nFeatures - 2) + 1e-06 *
                torch.randn(2 * nFeatures - 2))
            G_pinv = (G_.t().mm(G_) + 1e-05 * torch.eye(nHidden)).inverse().mm(
                G_.t())
            self.z0 = Parameter(-G_pinv.mv(self.s0.data) + 1e-06 * torch.
                randn(nHidden))
            lam = 21.21
            W_fc1, b_fc1 = self.fc1.weight, self.fc1.bias
            W_fc1.data[:, :] = 0.001 * torch.randn((2 * nFeatures - 1,
                nFeatures))
            W_fc1.data[:nFeatures, :nFeatures] += -torch.eye(nFeatures)
            b_fc1.data[:] = 0.0
            b_fc1.data[nFeatures:2 * nFeatures - 1] = lam
        else:
            self.L = Parameter(torch.tril(torch.rand(nHidden, nHidden)))
            self.G = Parameter(torch.Tensor(nineq, nHidden).uniform_(-1, 1))
            self.z0 = Parameter(torch.zeros(nHidden))
            self.s0 = Parameter(torch.ones(nineq))
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.neq = neq
        self.nineq = nineq
        self.args = args

    def cuda(self):
        for x in [self.L, self.G, self.z0, self.s0]:
            x.data = x.data
        return super()

    def forward(self, x):
        nBatch = x.size(0)
        x = self.fc1(x)
        L = self.M * self.L
        Q = L.mm(L.t()) + self.args.eps * Variable(torch.eye(self.nHidden))
        Q = Q.unsqueeze(0).expand(nBatch, self.nHidden, self.nHidden)
        G = self.G.unsqueeze(0).expand(nBatch, self.nineq, self.nHidden)
        h = self.G.mv(self.z0) + self.s0
        h = h.unsqueeze(0).expand(nBatch, self.nineq)
        e = Variable(torch.Tensor())
        x = QPFunction()(Q, x, G, h, e, e)
        x = x[:, :self.nFeatures]
        return x


class OptNet_LearnD(nn.Module):

    def __init__(self, nFeatures, args):
        super().__init__()
        nHidden, neq, nineq = 2 * nFeatures - 1, 0, 2 * nFeatures - 2
        assert neq == 0
        self.M = Variable(torch.tril(torch.ones(nHidden, nHidden)))
        Q = 1e-08 * torch.eye(nHidden)
        Q[:nFeatures, :nFeatures] = torch.eye(nFeatures)
        self.L = Variable(torch.potrf(Q))
        self.D = Parameter(0.3 * torch.randn(nFeatures - 1, nFeatures))
        self.h = Variable(torch.zeros(nineq))
        self.nFeatures = nFeatures
        self.nHidden = nHidden
        self.neq = neq
        self.nineq = nineq
        self.args = args

    def cuda(self):
        for x in [self.L, self.D, self.h]:
            x.data = x.data
        return super()

    def forward(self, x):
        nBatch = x.size(0)
        L = self.M * self.L
        Q = L.mm(L.t()) + self.args.eps * Variable(torch.eye(self.nHidden))
        Q = Q.unsqueeze(0).expand(nBatch, self.nHidden, self.nHidden)
        nI = Variable(-torch.eye(self.nFeatures - 1).type_as(Q.data))
        G = torch.cat((torch.cat((self.D, nI), 1), torch.cat((-self.D, nI), 1))
            )
        G = G.unsqueeze(0).expand(nBatch, self.nineq, self.nHidden)
        h = self.h.unsqueeze(0).expand(nBatch, self.nineq)
        e = Variable(torch.Tensor())
        p = torch.cat((-x, Parameter(13.0 * torch.ones(nBatch, self.
            nFeatures - 1))), 1)
        x = QPFunction()(Q.double(), p.double(), G.double(), h.double(), e, e
            ).float()
        x = x[:, :self.nFeatures]
        return x


class FC(nn.Module):

    def __init__(self, nFeatures, nHidden, bn=False):
        super().__init__()
        self.bn = bn
        fcs = []
        prevSz = nFeatures
        for sz in nHidden:
            fc = nn.Linear(prevSz, sz)
            prevSz = sz
            fcs.append(fc)
        for sz in (list(reversed(nHidden)) + [nFeatures]):
            fc = nn.Linear(prevSz, sz)
            prevSz = sz
            fcs.append(fc)
        self.fcs = nn.ModuleList(fcs)

    def __call__(self, x):
        nBatch = x.size(0)
        Nsq = x.size(1)
        in_x = x
        x = x.view(nBatch, -1)
        for fc in self.fcs:
            x = F.relu(fc(x))
        x = x.view_as(in_x)
        ex = x.exp()
        exs = ex.sum(3).expand(nBatch, Nsq, Nsq, Nsq)
        x = ex / exs
        return x


class Conv(nn.Module):

    def __init__(self, boardSz):
        super().__init__()
        self.boardSz = boardSz
        convs = []
        Nsq = boardSz ** 2
        prevSz = Nsq
        szs = [512] * 10 + [Nsq]
        for sz in szs:
            conv = nn.Conv2d(prevSz, sz, kernel_size=3, padding=1)
            convs.append(conv)
            prevSz = sz
        self.convs = nn.ModuleList(convs)

    def __call__(self, x):
        nBatch = x.size(0)
        Nsq = x.size(1)
        for i in range(len(self.convs) - 1):
            x = F.relu(self.convs[i](x))
        x = self.convs[-1](x)
        ex = x.exp()
        exs = ex.sum(3).expand(nBatch, Nsq, Nsq, Nsq)
        x = ex / exs
        return x


def get_sudoku_matrix(n):
    X = np.array([[cp.Variable(n ** 2) for i in range(n ** 2)] for j in
        range(n ** 2)])
    cons = [(x >= 0) for row in X for x in row] + [(cp.sum(x) == 1) for row in
        X for x in row] + [(sum(row) == np.ones(n ** 2)) for row in X] + [(
        sum([row[i] for row in X]) == np.ones(n ** 2)) for i in range(n ** 2)
        ] + [(sum([sum(row[i:i + n]) for row in X[j:j + n]]) == np.ones(n **
        2)) for i in range(0, n ** 2, n) for j in range(0, n ** 2, n)]
    f = sum([cp.sum(x) for row in X for x in row])
    prob = cp.Problem(cp.Minimize(f), cons)
    A = np.asarray(prob.get_problem_data(cp.ECOS)[0]['A'].todense())
    A0 = [A[0]]
    rank = 1
    for i in range(1, A.shape[0]):
        if np.linalg.matrix_rank(A0 + [A[i]], tol=1e-12) > rank:
            A0.append(A[i])
            rank += 1
    return np.array(A0)


class OptNetEq(nn.Module):

    def __init__(self, n, Qpenalty, qp_solver, trueInit=False):
        super().__init__()
        self.qp_solver = qp_solver
        nx = (n ** 2) ** 3
        self.Q = Variable(Qpenalty * torch.eye(nx).double())
        self.Q_idx = spa.csc_matrix(self.Q.detach().cpu().numpy()).nonzero()
        self.G = Variable(-torch.eye(nx).double())
        self.h = Variable(torch.zeros(nx).double())
        t = get_sudoku_matrix(n)
        if trueInit:
            self.A = Parameter(torch.DoubleTensor(t))
        else:
            self.A = Parameter(torch.rand(t.shape).double())
        self.log_z0 = Parameter(torch.zeros(nx).double())
        if self.qp_solver == 'osqpth':
            t = torch.cat((self.A, self.G), dim=0)
            self.AG_idx = spa.csc_matrix(t.detach().cpu().numpy()).nonzero()

    def forward(self, puzzles):
        nBatch = puzzles.size(0)
        p = -puzzles.view(nBatch, -1)
        b = self.A.mv(self.log_z0.exp())
        if self.qp_solver == 'qpth':
            y = QPFunction(verbose=-1)(self.Q, p.double(), self.G, self.h,
                self.A, b).float().view_as(puzzles)
        elif self.qp_solver == 'osqpth':
            _l = torch.cat((b, torch.full(self.h.shape, float('-inf'),
                device=self.h.device, dtype=self.h.dtype)), dim=0)
            _u = torch.cat((b, self.h), dim=0)
            Q_data = self.Q[self.Q_idx[0], self.Q_idx[1]]
            AG = torch.cat((self.A, self.G), dim=0)
            AG_data = AG[self.AG_idx[0], self.AG_idx[1]]
            y = OSQP(self.Q_idx, self.Q.shape, self.AG_idx, AG.shape,
                diff_mode=DiffModes.FULL)(Q_data, p.double(), AG_data, _l, _u
                ).float().view_as(puzzles)
        else:
            assert False
        return y


class SpOptNetEq(nn.Module):

    def __init__(self, n, Qpenalty, trueInit=False):
        super().__init__()
        nx = (n ** 2) ** 3
        self.nx = nx
        spTensor = torch.sparse.DoubleTensor
        iTensor = torch.LongTensor
        dTensor = torch.DoubleTensor
        self.Qi = iTensor([range(nx), range(nx)])
        self.Qv = Variable(dTensor(nx).fill_(Qpenalty))
        self.Qsz = torch.Size([nx, nx])
        self.Gi = iTensor([range(nx), range(nx)])
        self.Gv = Variable(dTensor(nx).fill_(-1.0))
        self.Gsz = torch.Size([nx, nx])
        self.h = Variable(torch.zeros(nx).double())
        t = get_sudoku_matrix(n)
        neq = t.shape[0]
        if trueInit:
            I = t != 0
            self.Av = Parameter(dTensor(t[I]))
            Ai_np = np.nonzero(t)
            self.Ai = torch.stack((torch.LongTensor(Ai_np[0]), torch.
                LongTensor(Ai_np[1])))
            self.Asz = torch.Size([neq, nx])
        else:
            self.Ai = torch.stack((iTensor(list(range(neq))).unsqueeze(1).
                repeat(1, nx).view(-1), iTensor(list(range(nx))).repeat(neq)))
            self.Av = Parameter(dTensor(neq * nx).uniform_())
            self.Asz = torch.Size([neq, nx])
        self.b = Variable(torch.ones(neq).double())

    def forward(self, puzzles):
        nBatch = puzzles.size(0)
        p = -puzzles.view(nBatch, -1).double()
        return SpQPFunction(self.Qi, self.Qsz, self.Gi, self.Gsz, self.Ai,
            self.Asz, verbose=-1)(self.Qv.expand(nBatch, self.Qv.size(0)),
            p, self.Gv.expand(nBatch, self.Gv.size(0)), self.h.expand(
            nBatch, self.h.size(0)), self.Av.expand(nBatch, self.Av.size(0)
            ), self.b.expand(nBatch, self.b.size(0))).float().view_as(puzzles)


class OptNetIneq(nn.Module):

    def __init__(self, n, Qpenalty, nineq):
        super().__init__()
        nx = (n ** 2) ** 3
        self.Q = Variable(Qpenalty * torch.eye(nx).double())
        self.G1 = Variable(-torch.eye(nx).double())
        self.h1 = Variable(torch.zeros(nx).double())
        self.A = Parameter(torch.rand(50, nx).double())
        self.G2 = Parameter(torch.Tensor(128, nx).uniform_(-1, 1).double())
        self.z2 = Parameter(torch.zeros(nx).double())
        self.s2 = Parameter(torch.ones(128).double())

    def forward(self, puzzles):
        nBatch = puzzles.size(0)
        p = -puzzles.view(nBatch, -1)
        h2 = self.G2.mv(self.z2) + self.s2
        G = torch.cat((self.G1, self.G2), 0)
        h = torch.cat((self.h1, h2), 0)
        e = Variable(torch.Tensor())
        return QPFunction(verbose=False)(self.Q, p.double(), G, h, e, e).float(
            ).view_as(puzzles)


class OptNetLatent(nn.Module):

    def __init__(self, n, Qpenalty, nLatent, nineq, trueInit=False):
        super().__init__()
        nx = (n ** 2) ** 3
        self.fc_in = nn.Linear(nx, nLatent)
        self.Q = Variable(Qpenalty * torch.eye(nLatent))
        self.G = Parameter(torch.Tensor(nineq, nLatent).uniform_(-1, 1))
        self.z = Parameter(torch.zeros(nLatent))
        self.s = Parameter(torch.ones(nineq))
        self.fc_out = nn.Linear(nLatent, nx)

    def forward(self, puzzles):
        nBatch = puzzles.size(0)
        x = puzzles.view(nBatch, -1)
        x = self.fc_in(x)
        e = Variable(torch.Tensor())
        h = self.G.mv(self.z) + self.s
        x = QPFunction(verbose=False)(self.Q, x, self.G, h, e, e)
        x = self.fc_out(x)
        x = x.view_as(puzzles)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_locuslab_optnet(_paritybench_base):
    pass
    def test_000(self):
        self._check(Bottleneck(*[], **{'nChannels': 4, 'growthRate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(SingleLayer(*[], **{'nChannels': 4, 'growthRate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Transition(*[], **{'nChannels': 4, 'nOutChannels': 4}), [torch.rand([4, 4, 4, 4])], {})

