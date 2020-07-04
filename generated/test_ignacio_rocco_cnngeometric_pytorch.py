import sys
_module = sys.modules[__name__]
del sys
caltech_dataset = _module
download_datasets = _module
pf_dataset = _module
synth_dataset = _module
tss_dataset = _module
demo = _module
eval = _module
flow = _module
grid_gen = _module
point_tnf = _module
transformation = _module
normalization = _module
cnn_geometric_model = _module
loss = _module
options = _module
train = _module
dataloader = _module
eval_util = _module
py_util = _module
torch_util = _module
train_test_fn = _module

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


from collections import OrderedDict


import torch


import torch.nn as nn


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


import numpy as np


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn.modules.module import Module


import torchvision.models as models


import numpy.matlib


import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter


class AffineGridGen(Module):

    def __init__(self, out_h=240, out_w=240, out_ch=3, use_cuda=True):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        b = theta.size()[0]
        if not theta.size() == (b, 2, 3):
            theta = theta.view(-1, 2, 3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w)
            )
        return F.affine_grid(theta, out_size)


def expand_dim(tensor, dim, desired_dim_len):
    sz = list(tensor.size())
    sz[dim] = desired_dim_len
    return tensor.expand(tuple(sz))


class AffineGridGenV2(Module):

    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(AffineGridGenV2, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
            np.linspace(-1, 1, out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X
            self.grid_Y = self.grid_Y

    def forward(self, theta):
        b = theta.size(0)
        if not theta.size() == (b, 6):
            theta = theta.view(b, 6)
            theta = theta.contiguous()
        t0 = theta[:, (0)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1 = theta[:, (1)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t2 = theta[:, (2)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3 = theta[:, (3)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4 = theta[:, (4)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5 = theta[:, (5)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)
        grid_Xp = grid_X * t0 + grid_Y * t1 + t2
        grid_Yp = grid_X * t3 + grid_Y * t4 + t5
        return torch.cat((grid_Xp, grid_Yp), 3)


def homography_mat_from_4_pts(theta):
    b = theta.size(0)
    if not theta.size() == (b, 8):
        theta = theta.view(b, 8)
        theta = theta.contiguous()
    xp = theta[:, :4].unsqueeze(2)
    yp = theta[:, 4:].unsqueeze(2)
    x = Variable(torch.FloatTensor([-1, -1, 1, 1])).unsqueeze(1).unsqueeze(0
        ).expand(b, 4, 1)
    y = Variable(torch.FloatTensor([-1, 1, -1, 1])).unsqueeze(1).unsqueeze(0
        ).expand(b, 4, 1)
    z = Variable(torch.zeros(4)).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    o = Variable(torch.ones(4)).unsqueeze(1).unsqueeze(0).expand(b, 4, 1)
    single_o = Variable(torch.ones(1)).unsqueeze(1).unsqueeze(0).expand(b, 1, 1
        )
    if theta.is_cuda:
        x = x.cuda()
        y = y.cuda()
        z = z.cuda()
        o = o.cuda()
        single_o = single_o.cuda()
    A = torch.cat([torch.cat([-x, -y, -o, z, z, z, x * xp, y * xp, xp], 2),
        torch.cat([z, z, z, -x, -y, -o, x * yp, y * yp, yp], 2)], 1)
    h = torch.bmm(torch.inverse(A[:, :, :8]), -A[:, :, (8)].unsqueeze(2))
    h = torch.cat([h, single_o], 1)
    H = h.squeeze(2)
    return H


class HomographyGridGen(Module):

    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(HomographyGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
            np.linspace(-1, 1, out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X
            self.grid_Y = self.grid_Y

    def forward(self, theta):
        b = theta.size(0)
        if theta.size(1) == 9:
            H = theta
        else:
            H = homography_mat_from_4_pts(theta)
        h0 = H[:, (0)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1 = H[:, (1)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2 = H[:, (2)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3 = H[:, (3)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4 = H[:, (4)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5 = H[:, (5)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6 = H[:, (6)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7 = H[:, (7)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8 = H[:, (8)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)
        grid_Xp = grid_X * h0 + grid_Y * h1 + h2
        grid_Yp = grid_X * h3 + grid_Y * h4 + h5
        k = grid_X * h6 + grid_Y * h7 + h8
        grid_Xp /= k
        grid_Yp /= k
        return torch.cat((grid_Xp, grid_Yp), 3)


class TpsGridGen(Module):

    def __init__(self, out_h=240, out_w=240, use_regular_grid=True,
        grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
            np.linspace(-1, 1, out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X
            self.grid_Y = self.grid_Y
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))
            P_Y = np.reshape(P_Y, (-1, 1))
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = Variable(self.compute_L_inverse(P_X, P_Y).unsqueeze(0
                ), requires_grad=False)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(
                0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(
                0, 4)
            self.P_X = Variable(self.P_X, requires_grad=False)
            self.P_Y = Variable(self.P_Y, requires_grad=False)
            if use_cuda:
                self.P_X = self.P_X
                self.P_Y = self.P_Y

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.
            grid_X, self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(
            Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        if self.reg_factor != 0:
            K += torch.eye(K.size(0), K.size(1)) * self.reg_factor
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1),
            Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        batch_size = theta.size()[0]
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
            self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
            self.N, self.N)), Q_Y)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
            points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
            points_h, points_w, 1, 1)
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3,
            self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3,
            self.N)), Q_Y)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
            points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
            points_h, points_w, 1, 1)
        points_X_for_summation = points[:, :, :, (0)].unsqueeze(3).unsqueeze(4
            ).expand(points[:, :, :, (0)].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, (1)].unsqueeze(3).unsqueeze(4
            ).expand(points[:, :, :, (1)].size() + (1, self.N))
        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            delta_X = points_X_for_summation - P_X.expand_as(
                points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(
                points_Y_for_summation)
        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))
        points_X_batch = points[:, :, :, (0)].unsqueeze(3)
        points_Y_batch = points[:, :, :, (1)].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                points_Y_batch.size()[1:])
        points_X_prime = A_X[:, :, :, :, (0)] + torch.mul(A_X[:, :, :, :, (
            1)], points_X_batch) + torch.mul(A_X[:, :, :, :, (2)],
            points_Y_batch) + torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)
        points_Y_prime = A_Y[:, :, :, :, (0)] + torch.mul(A_Y[:, :, :, :, (
            1)], points_X_batch) + torch.mul(A_Y[:, :, :, :, (2)],
            points_Y_batch) + torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)


class AffineGridGen(Module):

    def __init__(self, out_h=240, out_w=240, out_ch=3, use_cuda=True):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        b = theta.size()[0]
        if not theta.size() == (b, 2, 3):
            theta = theta.view(-1, 2, 3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w)
            )
        return F.affine_grid(theta, out_size)


class AffineGridGenV2(Module):

    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(AffineGridGenV2, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
            np.linspace(-1, 1, out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X
            self.grid_Y = self.grid_Y

    def forward(self, theta):
        b = theta.size(0)
        if not theta.size() == (b, 6):
            theta = theta.view(b, 6)
            theta = theta.contiguous()
        t0 = theta[:, (0)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t1 = theta[:, (1)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t2 = theta[:, (2)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t3 = theta[:, (3)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t4 = theta[:, (4)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        t5 = theta[:, (5)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)
        grid_Xp = grid_X * t0 + grid_Y * t1 + t2
        grid_Yp = grid_X * t3 + grid_Y * t4 + t5
        return torch.cat((grid_Xp, grid_Yp), 3)


class HomographyGridGen(Module):

    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        super(HomographyGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.use_cuda = use_cuda
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
            np.linspace(-1, 1, out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X
            self.grid_Y = self.grid_Y

    def forward(self, theta):
        b = theta.size(0)
        if theta.size(1) == 9:
            H = theta
        else:
            H = homography_mat_from_4_pts(theta)
        h0 = H[:, (0)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h1 = H[:, (1)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h2 = H[:, (2)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h3 = H[:, (3)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h4 = H[:, (4)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h5 = H[:, (5)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h6 = H[:, (6)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h7 = H[:, (7)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        h8 = H[:, (8)].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        grid_X = expand_dim(self.grid_X, 0, b)
        grid_Y = expand_dim(self.grid_Y, 0, b)
        grid_Xp = grid_X * h0 + grid_Y * h1 + h2
        grid_Yp = grid_X * h3 + grid_Y * h4 + h5
        k = grid_X * h6 + grid_Y * h7 + h8
        grid_Xp /= k
        grid_Yp /= k
        return torch.cat((grid_Xp, grid_Yp), 3)


class TpsGridGen(Module):

    def __init__(self, out_h=240, out_w=240, use_regular_grid=True,
        grid_size=3, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda
        self.grid_X, self.grid_Y = np.meshgrid(np.linspace(-1, 1, out_w),
            np.linspace(-1, 1, out_h))
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        self.grid_X = Variable(self.grid_X, requires_grad=False)
        self.grid_Y = Variable(self.grid_Y, requires_grad=False)
        if use_cuda:
            self.grid_X = self.grid_X
            self.grid_Y = self.grid_Y
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))
            P_Y = np.reshape(P_Y, (-1, 1))
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.Li = Variable(self.compute_L_inverse(P_X, P_Y).unsqueeze(0
                ), requires_grad=False)
            self.P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(
                0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(
                0, 4)
            self.P_X = Variable(self.P_X, requires_grad=False)
            self.P_Y = Variable(self.P_Y, requires_grad=False)
            if use_cuda:
                self.P_X = self.P_X
                self.P_Y = self.P_Y

    def forward(self, theta):
        warped_grid = self.apply_transformation(theta, torch.cat((self.
            grid_X, self.grid_Y), 3))
        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(Xmat - Xmat.transpose(0, 1), 2) + torch.pow(
            Ymat - Ymat.transpose(0, 1), 2)
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        if self.reg_factor != 0:
            K += torch.eye(K.size(0), K.size(1)) * self.reg_factor
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat((P.transpose(0, 1),
            Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        batch_size = theta.size()[0]
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
            self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand((batch_size,
            self.N, self.N)), Q_Y)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
            points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
            points_h, points_w, 1, 1)
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3,
            self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand((batch_size, 3,
            self.N)), Q_Y)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
            points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1,
            points_h, points_w, 1, 1)
        points_X_for_summation = points[:, :, :, (0)].unsqueeze(3).unsqueeze(4
            ).expand(points[:, :, :, (0)].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, (1)].unsqueeze(3).unsqueeze(4
            ).expand(points[:, :, :, (1)].size() + (1, self.N))
        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            delta_X = points_X_for_summation - P_X.expand_as(
                points_X_for_summation)
            delta_Y = points_Y_for_summation - P_Y.expand_as(
                points_Y_for_summation)
        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        dist_squared[dist_squared == 0] = 1
        U = torch.mul(dist_squared, torch.log(dist_squared))
        points_X_batch = points[:, :, :, (0)].unsqueeze(3)
        points_Y_batch = points[:, :, :, (1)].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand((batch_size,) +
                points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand((batch_size,) +
                points_Y_batch.size()[1:])
        points_X_prime = A_X[:, :, :, :, (0)] + torch.mul(A_X[:, :, :, :, (
            1)], points_X_batch) + torch.mul(A_X[:, :, :, :, (2)],
            points_Y_batch) + torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)
        points_Y_prime = A_Y[:, :, :, :, (0)] + torch.mul(A_Y[:, :, :, :, (
            1)], points_X_batch) + torch.mul(A_Y[:, :, :, :, (2)],
            points_Y_batch) + torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)
        return torch.cat((points_X_prime, points_Y_prime), 3)


def featureL2Norm(feature):
    epsilon = 1e-06
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5
        ).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class FeatureExtraction(torch.nn.Module):

    def __init__(self, train_fe=False, feature_extraction_cnn='vgg',
        normalization=True, last_layer='', use_cuda=True):
        super(FeatureExtraction, self).__init__()
        self.normalization = normalization
        if feature_extraction_cnn == 'vgg':
            self.model = models.vgg16(pretrained=True)
            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2',
                'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2',
                'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2',
                'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
                'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3',
                'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
                'conv5_3', 'relu5_3', 'pool5']
            if last_layer == '':
                last_layer = 'pool4'
            last_layer_idx = vgg_feature_layers.index(last_layer)
            self.model = nn.Sequential(*list(self.model.features.children()
                )[:last_layer_idx + 1])
        if feature_extraction_cnn == 'resnet101':
            self.model = models.resnet101(pretrained=True)
            resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool',
                'layer1', 'layer2', 'layer3', 'layer4']
            if last_layer == '':
                last_layer = 'layer3'
            last_layer_idx = resnet_feature_layers.index(last_layer)
            resnet_module_list = [self.model.conv1, self.model.bn1, self.
                model.relu, self.model.maxpool, self.model.layer1, self.
                model.layer2, self.model.layer3, self.model.layer4]
            self.model = nn.Sequential(*resnet_module_list[:last_layer_idx + 1]
                )
        if feature_extraction_cnn == 'resnet101_v2':
            self.model = models.resnet101(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        if feature_extraction_cnn == 'densenet201':
            self.model = models.densenet201(pretrained=True)
            self.model = nn.Sequential(*list(self.model.features.children()
                )[:-4])
        if not train_fe:
            for param in self.model.parameters():
                param.requires_grad = False
        if use_cuda:
            self.model = self.model

    def forward(self, image_batch):
        features = self.model(image_batch)
        if self.normalization:
            features = featureL2Norm(features)
        return features


class FeatureCorrelation(torch.nn.Module):

    def __init__(self, shape='3D', normalization=True, matching_type=
        'correlation'):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.matching_type = matching_type
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        if self.matching_type == 'correlation':
            if self.shape == '3D':
                feature_A = feature_A.transpose(2, 3).contiguous().view(b,
                    c, h * w)
                feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
                feature_mul = torch.bmm(feature_B, feature_A)
                correlation_tensor = feature_mul.view(b, h, w, h * w
                    ).transpose(2, 3).transpose(1, 2)
            elif self.shape == '4D':
                feature_A = feature_A.view(b, c, h * w).transpose(1, 2)
                feature_B = feature_B.view(b, c, h * w)
                feature_mul = torch.bmm(feature_A, feature_B)
                correlation_tensor = feature_mul.view(b, h, w, h, w).unsqueeze(
                    1)
            if self.normalization:
                correlation_tensor = featureL2Norm(self.ReLU(
                    correlation_tensor))
            return correlation_tensor
        if self.matching_type == 'subtraction':
            return feature_A.sub(feature_B)
        if self.matching_type == 'concatenation':
            return torch.cat((feature_A, feature_B), 1)


class FeatureRegression(nn.Module):

    def __init__(self, output_dim=6, use_cuda=True, batch_normalization=
        True, kernel_sizes=[7, 5, 5], channels=[225, 128, 64]):
        super(FeatureRegression, self).__init__()
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers - 1):
            k_size = kernel_sizes[i]
            ch_in = channels[i]
            ch_out = channels[i + 1]
            nn_modules.append(nn.Conv2d(ch_in, ch_out, kernel_size=k_size,
                padding=0))
            if batch_normalization:
                nn_modules.append(nn.BatchNorm2d(ch_out))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        self.linear = nn.Linear(ch_out * kernel_sizes[-1] * kernel_sizes[-1
            ], output_dim)
        if use_cuda:
            self.conv
            self.linear

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class CNNGeometric(nn.Module):

    def __init__(self, output_dim=6, feature_extraction_cnn='vgg',
        feature_extraction_last_layer='', return_correlation=False,
        fr_kernel_sizes=[7, 5, 5], fr_channels=[225, 128, 64],
        feature_self_matching=False, normalize_features=True,
        normalize_matches=True, batch_normalization=True, train_fe=False,
        use_cuda=True, matching_type='correlation'):
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.feature_self_matching = feature_self_matching
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.return_correlation = return_correlation
        self.FeatureExtraction = FeatureExtraction(train_fe=train_fe,
            feature_extraction_cnn=feature_extraction_cnn, last_layer=
            feature_extraction_last_layer, normalization=normalize_features,
            use_cuda=self.use_cuda)
        self.FeatureCorrelation = FeatureCorrelation(shape='3D',
            normalization=normalize_matches, matching_type=matching_type)
        self.FeatureRegression = FeatureRegression(output_dim, use_cuda=
            self.use_cuda, kernel_sizes=fr_kernel_sizes, channels=
            fr_channels, batch_normalization=batch_normalization)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, tnf_batch):
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])
        correlation = self.FeatureCorrelation(feature_A, feature_B)
        theta = self.FeatureRegression(correlation)
        if self.return_correlation:
            return theta, correlation
        else:
            return theta


class PointTnf(object):
    """
    
    Class with functions for transforming a set of points with affine/tps transformations
    
    """

    def __init__(self, tps_grid_size=3, tps_reg_factor=0, use_cuda=True):
        self.use_cuda = use_cuda
        self.tpsTnf = TpsGridGen(grid_size=tps_grid_size, reg_factor=
            tps_reg_factor, use_cuda=self.use_cuda)

    def tpsPointTnf(self, theta, points):
        points = points.unsqueeze(3).transpose(1, 3)
        warped_points = self.tpsTnf.apply_transformation(theta, points)
        warped_points = warped_points.transpose(3, 1).squeeze(3)
        return warped_points

    def homPointTnf(self, theta, points, eps=1e-05):
        b = theta.size(0)
        if theta.size(1) == 9:
            H = theta
        else:
            H = homography_mat_from_4_pts(theta)
        h0 = H[:, (0)].unsqueeze(1).unsqueeze(2)
        h1 = H[:, (1)].unsqueeze(1).unsqueeze(2)
        h2 = H[:, (2)].unsqueeze(1).unsqueeze(2)
        h3 = H[:, (3)].unsqueeze(1).unsqueeze(2)
        h4 = H[:, (4)].unsqueeze(1).unsqueeze(2)
        h5 = H[:, (5)].unsqueeze(1).unsqueeze(2)
        h6 = H[:, (6)].unsqueeze(1).unsqueeze(2)
        h7 = H[:, (7)].unsqueeze(1).unsqueeze(2)
        h8 = H[:, (8)].unsqueeze(1).unsqueeze(2)
        X = points[:, (0), :].unsqueeze(1)
        Y = points[:, (1), :].unsqueeze(1)
        Xp = X * h0 + Y * h1 + h2
        Yp = X * h3 + Y * h4 + h5
        k = X * h6 + Y * h7 + h8
        k = k + torch.sign(k) * eps
        Xp /= k
        Yp /= k
        return torch.cat((Xp, Yp), 1)

    def affPointTnf(self, theta, points):
        theta_mat = theta.view(-1, 2, 3)
        warped_points = torch.bmm(theta_mat[:, :, :2], points)
        warped_points += theta_mat[:, :, (2)].unsqueeze(2).expand_as(
            warped_points)
        return warped_points


class TransformedGridLoss(nn.Module):

    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model
        axis_coords = np.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        X, Y = np.meshgrid(axis_coords, axis_coords)
        X = np.reshape(X, (1, 1, self.N))
        Y = np.reshape(Y, (1, 1, self.N))
        P = np.concatenate((X, Y), 1)
        self.P = Variable(torch.FloatTensor(P), requires_grad=False)
        self.pointTnf = PointTnf(use_cuda=use_cuda)
        if use_cuda:
            self.P = self.P

    def forward(self, theta, theta_GT):
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size, 2, self.N)
        if self.geometric_model == 'affine':
            P_prime = self.pointTnf.affPointTnf(theta, P)
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT, P)
        elif self.geometric_model == 'hom':
            P_prime = self.pointTnf.homPointTnf(theta, P)
            P_prime_GT = self.pointTnf.homPointTnf(theta_GT, P)
        elif self.geometric_model == 'tps':
            P_prime = self.pointTnf.tpsPointTnf(theta.unsqueeze(2).
                unsqueeze(3), P)
            P_prime_GT = self.pointTnf.tpsPointTnf(theta_GT, P)
        loss = torch.sum(torch.pow(P_prime - P_prime_GT, 2), 1)
        loss = torch.mean(loss)
        return loss


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ignacio_rocco_cnngeometric_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(AffineGridGen(*[], **{}), [torch.rand([4, 2, 3])], {})

    @_fails_compile()
    def test_001(self):
        self._check(AffineGridGenV2(*[], **{}), [torch.rand([4, 6])], {})

    @_fails_compile()
    def test_002(self):
        self._check(FeatureCorrelation(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(FeatureExtraction(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(HomographyGridGen(*[], **{}), [torch.rand([4, 8])], {})

    @_fails_compile()
    def test_005(self):
        self._check(TransformedGridLoss(*[], **{}), [torch.rand([4, 2, 3]), torch.rand([4, 2, 3])], {})

