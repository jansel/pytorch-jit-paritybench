import sys
_module = sys.modules[__name__]
del sys
main_experiment = _module
VAE = _module
models = _module
flows = _module
layers = _module
optimization = _module
loss = _module
training = _module
utils = _module
distributions = _module
load_data = _module
log_likelihood = _module
plotting = _module
visual_evaluation = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
yaml = logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
yaml.load.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import time


import torch


import torch.utils.data


import torch.optim as optim


import numpy as np


import math


import random


import torch.nn as nn


from torch.autograd import Variable


import torch.nn.functional as F


from torch.nn.parameter import Parameter


import torch.utils.data as data_utils


from scipy.io import loadmat


class GatedConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g


class GatedConvTranspose2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, output_padding=0, dilation=1, activation=None):
        super(GatedConvTranspose2d, self).__init__()
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding, dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride, padding, output_padding, dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))
        return h * g


class VAE(nn.Module):
    """
    The base VAE class containing gated convolutional encoder and decoder architecture.
    Can be used as a base class for VAE's with normalizing flows.
    """

    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_size = args.z_size
        self.input_size = args.input_size
        self.input_type = args.input_type
        if self.input_size == [1, 28, 28] or self.input_size == [3, 28, 28]:
            self.last_kernel_size = 7
        elif self.input_size == [1, 28, 20]:
            self.last_kernel_size = 7, 5
        else:
            raise ValueError('invalid input size!!')
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        self.q_z_nn_output_dim = 256
        if args.cuda:
            self.FloatTensor = torch.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor
        self.log_det_j = Variable(self.FloatTensor(1).zero_())

    def create_encoder(self):
        """
        Helper function to create the elemental blocks for the encoder. Creates a gated convnet encoder.
        the encoder expects data as input of shape (batch_size, num_channels, width, height).
        """
        if self.input_type == 'binary':
            q_z_nn = nn.Sequential(GatedConv2d(self.input_size[0], 32, 5, 1, 2), GatedConv2d(32, 32, 5, 2, 2), GatedConv2d(32, 64, 5, 1, 2), GatedConv2d(64, 64, 5, 2, 2), GatedConv2d(64, 64, 5, 1, 2), GatedConv2d(64, 256, self.last_kernel_size, 1, 0))
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(nn.Linear(256, self.z_size), nn.Softplus())
            return q_z_nn, q_z_mean, q_z_var
        elif self.input_type == 'multinomial':
            act = None
            q_z_nn = nn.Sequential(GatedConv2d(self.input_size[0], 32, 5, 1, 2, activation=act), GatedConv2d(32, 32, 5, 2, 2, activation=act), GatedConv2d(32, 64, 5, 1, 2, activation=act), GatedConv2d(64, 64, 5, 2, 2, activation=act), GatedConv2d(64, 64, 5, 1, 2, activation=act), GatedConv2d(64, 256, self.last_kernel_size, 1, 0, activation=act))
            q_z_mean = nn.Linear(256, self.z_size)
            q_z_var = nn.Sequential(nn.Linear(256, self.z_size), nn.Softplus(), nn.Hardtanh(min_val=0.01, max_val=7.0))
            return q_z_nn, q_z_mean, q_z_var

    def create_decoder(self):
        """
        Helper function to create the elemental blocks for the decoder. Creates a gated convnet decoder.
        """
        num_classes = 256
        if self.input_type == 'binary':
            p_x_nn = nn.Sequential(GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0), GatedConvTranspose2d(64, 64, 5, 1, 2), GatedConvTranspose2d(64, 32, 5, 2, 2, 1), GatedConvTranspose2d(32, 32, 5, 1, 2), GatedConvTranspose2d(32, 32, 5, 2, 2, 1), GatedConvTranspose2d(32, 32, 5, 1, 2))
            p_x_mean = nn.Sequential(nn.Conv2d(32, self.input_size[0], 1, 1, 0), nn.Sigmoid())
            return p_x_nn, p_x_mean
        elif self.input_type == 'multinomial':
            act = None
            p_x_nn = nn.Sequential(GatedConvTranspose2d(self.z_size, 64, self.last_kernel_size, 1, 0, activation=act), GatedConvTranspose2d(64, 64, 5, 1, 2, activation=act), GatedConvTranspose2d(64, 32, 5, 2, 2, 1, activation=act), GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act), GatedConvTranspose2d(32, 32, 5, 2, 2, 1, activation=act), GatedConvTranspose2d(32, 32, 5, 1, 2, activation=act))
            p_x_mean = nn.Sequential(nn.Conv2d(32, 256, 5, 1, 2), nn.Conv2d(256, self.input_size[0] * num_classes, 1, 1, 0))
            return p_x_nn, p_x_mean
        else:
            raise ValueError('invalid input type!!')

    def reparameterize(self, mu, var):
        """
        Samples z from a multivariate Gaussian with diagonal covariance matrix using the
         reparameterization trick.
        """
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x):
        """
        Encoder expects following data shapes as input: shape = (batch_size, num_channels, width, height)
        """
        h = self.q_z_nn(x)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z):
        """
        Decoder outputs reconstructed image in the following shapes:
        x_mean.shape = (batch_size, num_channels, width, height)
        """
        z = z.view(z.size(0), self.z_size, 1, 1)
        h = self.p_x_nn(z)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x):
        """
        Evaluates the model as a whole, encodes and decodes. Note that the log det jacobian is zero
         for a plain VAE (without flows), and z_0 = z_k.
        """
        z_mu, z_var = self.encode(x)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z)
        return x_mean, z_mu, z_var, self.log_det_j, z, z


class PlanarVAE(VAE):
    """
    Variational auto-encoder with planar flows in the encoder.
    """

    def __init__(self, args):
        super(PlanarVAE, self).__init__(args)
        self.log_det_j = 0.0
        flow = flows.Planar
        self.num_flows = args.num_flows
        self.amor_u = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_w = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows)
        for k in range(self.num_flows):
            flow_k = flow()
            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)
        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        u = self.amor_u(h).view(batch_size, self.num_flows, self.z_size, 1)
        w = self.amor_w(h).view(batch_size, self.num_flows, 1, self.z_size)
        b = self.amor_b(h).view(batch_size, self.num_flows, 1, 1)
        return mean_z, var_z, u, w, b

    def forward(self, x):
        """
        Forward pass with planar flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\\sum_k log |det dz_k/dz_k-1| ].
        """
        self.log_det_j = 0.0
        z_mu, z_var, u, w, b = self.encode(x)
        z = [self.reparameterize(z_mu, z_var)]
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], u[:, (k), :, :], w[:, (k), :, :], b[:, (k), :, :])
            z.append(z_k)
            self.log_det_j += log_det_jacobian
        x_mean = self.decode(z[-1])
        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class OrthogonalSylvesterVAE(VAE):
    """
    Variational auto-encoder with orthogonal flows in the encoder.
    """

    def __init__(self, args):
        super(OrthogonalSylvesterVAE, self).__init__(args)
        self.log_det_j = 0.0
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.num_ortho_vecs = args.num_ortho_vecs
        assert self.num_ortho_vecs <= self.z_size and self.num_ortho_vecs > 0
        if self.num_ortho_vecs == self.z_size:
            self.cond = 1e-05
        else:
            self.cond = 1e-06
        self.steps = 100
        identity = torch.eye(self.num_ortho_vecs, self.num_ortho_vecs)
        identity = identity.unsqueeze(0)
        self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False
        triu_mask = torch.triu(torch.ones(self.num_ortho_vecs, self.num_ortho_vecs), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.num_ortho_vecs).long()
        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)
        self.diag_activation = nn.Tanh()
        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs * self.num_ortho_vecs)
        self.amor_diag1 = nn.Sequential(nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs), self.diag_activation)
        self.amor_diag2 = nn.Sequential(nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs), self.diag_activation)
        self.amor_q = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_ortho_vecs)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.num_ortho_vecs)
        for k in range(self.num_flows):
            flow_k = flow(self.num_ortho_vecs)
            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size * num_flows, z_size * num_ortho_vecs)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, num_ortho_vecs)
        """
        q = q.view(-1, self.z_size * self.num_ortho_vecs)
        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.resize(dim0, self.z_size, self.num_ortho_vecs)
        max_norm = 0.0
        for s in range(self.steps):
            tmp = torch.bmm(amat.transpose(2, 1), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = torch.bmm(amat, tmp)
            test = torch.bmm(amat.transpose(2, 1), amat) - self._eye
            norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
            norms = torch.sqrt(norms2)
            max_norm = torch.max(norms).data[0]
            if max_norm <= self.cond:
                break
        if max_norm > self.cond:
            None
            None
            None
        amat = amat.view(-1, self.num_flows, self.z_size, self.num_ortho_vecs)
        amat = amat.transpose(0, 1)
        return amat

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)
        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)
        full_d = full_d.resize(batch_size, self.num_ortho_vecs, self.num_ortho_vecs, self.num_flows)
        diag1 = diag1.resize(batch_size, self.num_ortho_vecs, self.num_flows)
        diag2 = diag2.resize(batch_size, self.num_ortho_vecs, self.num_flows)
        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask
        r1[:, (self.diag_idx), (self.diag_idx), :] = diag1
        r2[:, (self.diag_idx), (self.diag_idx), :] = diag2
        q = self.amor_q(h)
        b = self.amor_b(h)
        b = b.resize(batch_size, 1, self.num_ortho_vecs, self.num_flows)
        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        """
        Forward pass with orthogonal sylvester flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\\sum_k log |det dz_k/dz_k-1| ].
        """
        self.log_det_j = 0.0
        z_mu, z_var, r1, r2, q, b = self.encode(x)
        q_ortho = self.batch_construct_orthogonal(q)
        z = [self.reparameterize(z_mu, z_var)]
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, (k)], r2[:, :, :, (k)], q_ortho[(k), :, :, :], b[:, :, :, (k)])
            z.append(z_k)
            self.log_det_j += log_det_jacobian
        x_mean = self.decode(z[-1])
        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class HouseholderSylvesterVAE(VAE):
    """
    Variational auto-encoder with householder sylvester flows in the encoder.
    """

    def __init__(self, args):
        super(HouseholderSylvesterVAE, self).__init__(args)
        self.log_det_j = 0.0
        flow = flows.Sylvester
        self.num_flows = args.num_flows
        self.num_householder = args.num_householder
        assert self.num_householder > 0
        identity = torch.eye(self.z_size, self.z_size)
        identity = identity.unsqueeze(0)
        self.register_buffer('_eye', Variable(identity))
        self._eye.requires_grad = False
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()
        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)
        self.diag_activation = nn.Tanh()
        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)
        self.amor_diag1 = nn.Sequential(nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size), self.diag_activation)
        self.amor_diag2 = nn.Sequential(nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size), self.diag_activation)
        self.amor_q = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.num_householder)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)
            self.add_module('flow_' + str(k), flow_k)

    def batch_construct_orthogonal(self, q):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size, num_flows * z_size * num_householder)
        :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, z_size)
        """
        q = q.view(-1, self.z_size)
        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        v = torch.div(q, norm)
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))
        amat = self._eye - 2 * vvT
        amat = amat.view(-1, self.num_householder, self.z_size, self.z_size)
        tmp = amat[:, (0)]
        for k in range(1, self.num_householder):
            tmp = torch.bmm(amat[:, (k)], tmp)
        amat = tmp.view(-1, self.num_flows, self.z_size, self.z_size)
        amat = amat.transpose(0, 1)
        return amat

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)
        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)
        full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)
        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask
        r1[:, (self.diag_idx), (self.diag_idx), :] = diag1
        r2[:, (self.diag_idx), (self.diag_idx), :] = diag2
        q = self.amor_q(h)
        b = self.amor_b(h)
        b = b.resize(batch_size, 1, self.z_size, self.num_flows)
        return mean_z, var_z, r1, r2, q, b

    def forward(self, x):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\\sum_k log |det dz_k/dz_k-1| ].
        """
        self.log_det_j = 0.0
        batch_size = x.size(0)
        z_mu, z_var, r1, r2, q, b = self.encode(x)
        q_ortho = self.batch_construct_orthogonal(q)
        z = [self.reparameterize(z_mu, z_var)]
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            q_k = q_ortho[k]
            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, (k)], r2[:, :, :, (k)], q_k, b[:, :, :, (k)], sum_ldj=True)
            z.append(z_k)
            self.log_det_j += log_det_jacobian
        x_mean = self.decode(z[-1])
        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class TriangularSylvesterVAE(VAE):
    """
    Variational auto-encoder with triangular Sylvester flows in the encoder. Alternates between setting
    the orthogonal matrix equal to permutation and identity matrix for each flow.
    """

    def __init__(self, args):
        super(TriangularSylvesterVAE, self).__init__(args)
        self.log_det_j = 0.0
        flow = flows.TriangularSylvester
        self.num_flows = args.num_flows
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)
        triu_mask = torch.triu(torch.ones(self.z_size, self.z_size), diagonal=1)
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)
        diag_idx = torch.arange(0, self.z_size).long()
        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)
        self.diag_activation = nn.Tanh()
        self.amor_d = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size * self.z_size)
        self.amor_diag1 = nn.Sequential(nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size), self.diag_activation)
        self.amor_diag2 = nn.Sequential(nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size), self.diag_activation)
        self.amor_b = nn.Linear(self.q_z_nn_output_dim, self.num_flows * self.z_size)
        for k in range(self.num_flows):
            flow_k = flow(self.z_size)
            self.add_module('flow_' + str(k), flow_k)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """
        batch_size = x.size(0)
        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        full_d = self.amor_d(h)
        diag1 = self.amor_diag1(h)
        diag2 = self.amor_diag2(h)
        full_d = full_d.resize(batch_size, self.z_size, self.z_size, self.num_flows)
        diag1 = diag1.resize(batch_size, self.z_size, self.num_flows)
        diag2 = diag2.resize(batch_size, self.z_size, self.num_flows)
        r1 = full_d * self.triu_mask
        r2 = full_d.transpose(2, 1) * self.triu_mask
        r1[:, (self.diag_idx), (self.diag_idx), :] = diag1
        r2[:, (self.diag_idx), (self.diag_idx), :] = diag2
        b = self.amor_b(h)
        b = b.resize(batch_size, 1, self.z_size, self.num_flows)
        return mean_z, var_z, r1, r2, b

    def forward(self, x):
        """
        Forward pass with orthogonal flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\\sum_k log |det dz_k/dz_k-1| ].
        """
        self.log_det_j = 0.0
        z_mu, z_var, r1, r2, b = self.encode(x)
        z = [self.reparameterize(z_mu, z_var)]
        for k in range(self.num_flows):
            flow_k = getattr(self, 'flow_' + str(k))
            if k % 2 == 1:
                permute_z = self.flip_idx
            else:
                permute_z = None
            z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, (k)], r2[:, :, :, (k)], b[:, :, :, (k)], permute_z, sum_ldj=True)
            z.append(z_k)
            self.log_det_j += log_det_jacobian
        x_mean = self.decode(z[-1])
        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]


class IAFVAE(VAE):
    """
    Variational auto-encoder with inverse autoregressive flows in the encoder.
    """

    def __init__(self, args):
        super(IAFVAE, self).__init__(args)
        self.log_det_j = 0.0
        self.h_size = args.made_h_size
        self.h_context = nn.Linear(self.q_z_nn_output_dim, self.h_size)
        self.num_flows = args.num_flows
        self.flow = flows.IAF(z_size=self.z_size, num_flows=self.num_flows, num_hidden=1, h_size=self.h_size, conv2d=False)

    def encode(self, x):
        """
        Encoder that ouputs parameters for base distribution of z and context h for flows.
        """
        h = self.q_z_nn(x)
        h = h.view(-1, self.q_z_nn_output_dim)
        mean_z = self.q_z_mean(h)
        var_z = self.q_z_var(h)
        h_context = self.h_context(h)
        return mean_z, var_z, h_context

    def forward(self, x):
        """
        Forward pass with inverse autoregressive flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\\sum_k log |det dz_k/dz_k-1| ].
        """
        z_mu, z_var, h_context = self.encode(x)
        z_0 = self.reparameterize(z_mu, z_var)
        z_k, self.log_det_j = self.flow(z_0, h_context)
        x_mean = self.decode(z_k)
        return x_mean, z_mu, z_var, self.log_det_j, z_0, z_k


class Planar(nn.Module):
    """
    PyTorch implementation of planar flows as presented in "Variational Inference with Normalizing Flows"
    by Danilo Jimenez Rezende, Shakir Mohamed. Model assumes amortized flow parameters.
    """

    def __init__(self):
        super(Planar, self).__init__()
        self.h = nn.Tanh()
        self.softplus = nn.Softplus()

    def der_h(self, x):
        """ Derivative of tanh """
        return 1 - self.h(x) ** 2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """
        zk = zk.unsqueeze(2)
        uw = torch.bmm(w, u)
        m_uw = -1.0 + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + (m_uw - uw) * w.transpose(2, 1) / w_norm_sq
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.h(wzb)
        z = z.squeeze(2)
        psi = w * self.der_h(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat)))
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)
        return z, log_det_jacobian


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):
        super(Sylvester, self).__init__()
        self.num_ortho_vecs = num_ortho_vecs
        self.h = nn.Tanh()
        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()
        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size , num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """
        zk = zk.unsqueeze(1)
        diag_r1 = r1[:, (self.diag_idx), (self.diag_idx)]
        diag_r2 = r2[:, (self.diag_idx), (self.diag_idx)]
        r1_hat = r1
        r2_hat = r2
        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)
        r2qzb = torch.bmm(zk, qr2) + b
        z = torch.bmm(self.h(r2qzb), qr1.transpose(2, 1)) + zk
        z = z.squeeze(1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.0
        log_diag_j = diag_j.abs().log()
        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j
        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):
        super(TriangularSylvester, self).__init__()
        self.z_size = z_size
        self.h = nn.Tanh()
        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.h(x) ** 2

    def _forward(self, zk, r1, r2, b, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """
        zk = zk.unsqueeze(1)
        diag_r1 = r1[:, (self.diag_idx), (self.diag_idx)]
        diag_r2 = r2[:, (self.diag_idx), (self.diag_idx)]
        if permute_z is not None:
            z_per = zk[:, :, (permute_z)]
        else:
            z_per = zk
        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.h(r2qzb), r1.transpose(2, 1))
        if permute_z is not None:
            z = z[:, :, (permute_z)]
        z += zk
        z = z.squeeze(1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.0
        log_diag_j = diag_j.abs().log()
        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j
        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):
        return self._forward(zk, r1, r2, q_ortho, b, sum_ldj)


class MaskedConv2d(nn.Module):
    """
    Creates masked convolutional autoregressive layer for pixelCNN.
    For input (x) to hidden (h) or hidden to hidden layers choose diagonal_zeros = False.
    For hidden to output (y) layers:
    If output depends on input through y_i = f(x_{<i}) set diagonal_zeros = True.
    Else if output depends on input through y_i = f(x_{<=i}) set diagonal_zeros = False.
    """

    def __init__(self, in_features, out_features, size_kernel=(3, 3), diagonal_zeros=False, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.size_kernel = size_kernel
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(out_features, in_features, *self.size_kernel))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_out % n_in == 0 or n_in % n_out == 0, '%d - %d' % (n_in, n_out)
        l = (self.size_kernel[0] - 1) // 2
        m = (self.size_kernel[1] - 1) // 2
        mask = np.ones((n_out, n_in, self.size_kernel[0], self.size_kernel[1]), dtype=np.float32)
        mask[:, :, :l, :] = 0
        mask[:, :, (l), :m] = 0
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i * k:(i + 1) * k, i + 1:, (l), (m)] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k, i:i + 1, (l), (m)] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[i:i + 1, (i + 1) * k:, (l), (m)] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k, (l), (m)] = 0
        return mask

    def forward(self, x):
        output = F.conv2d(x, self.mask * self.weight, bias=self.bias, padding=(1, 1))
        return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ', diagonal_zeros=' + str(self.diagonal_zeros) + ', bias=' + str(bias) + ', size_kernel=' + str(self.size_kernel) + ')'


class MaskedLinear(nn.Module):
    """
    Creates masked linear layer for MLP MADE.
    For input (x) to hidden (h) or hidden to hidden layers choose diagonal_zeros = False.
    For hidden to output (y) layers:
    If output depends on input through y_i = f(x_{<i}) set diagonal_zeros = True.
    Else if output depends on input through y_i = f(x_{<=i}) set diagonal_zeros = False.
    """

    def __init__(self, in_features, out_features, diagonal_zeros=False, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_in % n_out == 0 or n_out % n_in == 0
        mask = np.ones((n_in, n_out), dtype=np.float32)
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i + 1:, i * k:(i + 1) * k] = 0
                if self.diagonal_zeros:
                    mask[i:i + 1, i * k:(i + 1) * k] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[(i + 1) * k:, i:i + 1] = 0
                if self.diagonal_zeros:
                    mask[i * k:(i + 1) * k, i:i + 1] = 0
        return mask

    def forward(self, x):
        output = x.mm(self.mask * self.weight)
        if self.bias is not None:
            return output.add(self.bias.expand_as(output))
        else:
            return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ', diagonal_zeros=' + str(self.diagonal_zeros) + ', bias=' + str(bias) + ')'


class IAF(nn.Module):
    """
    PyTorch implementation of inverse autoregressive flows as presented in
    "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, Tim Salimans,
    Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling.
    Inverse Autoregressive Flow with either MADE MLPs or Pixel CNNs. Contains several flows. Each transformation
     takes as an input the previous stochastic z, and a context h. The structure of each flow is then as follows:
     z <- autoregressive_layer(z) + h, allow for diagonal connections
     z <- autoregressive_layer(z), allow for diagonal connections
     :
     z <- autoregressive_layer(z), do not allow for diagonal connections.

     Note that the size of h needs to be the same as h_size, which is the width of the MADE layers.
     """

    def __init__(self, z_size, num_flows=2, num_hidden=0, h_size=50, forget_bias=1.0, conv2d=False):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.num_flows = num_flows
        self.num_hidden = num_hidden
        self.h_size = h_size
        self.conv2d = conv2d
        if not conv2d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv2d
        self.activation = torch.nn.ELU
        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)
        for k in range(num_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())
            if torch.cuda.is_available():
                z_feats = z_feats
                zh_feats = zh_feats
                linear_mean = linear_mean
                linear_std = linear_std
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))
        self.param_list = torch.nn.ParameterList(self.param_list)

    def forward(self, z, h_context):
        logdets = 0.0
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv2d:
                z = z[:, (self.flip_idx)]
            h = flow[0](z)
            h = h + h_context
            h = flow[1](h)
            mean = flow[2](h)
            gate = F.sigmoid(flow[3](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(gate.log().view(gate.size(0), -1), 1)
        return z, logdets


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (GatedConv2d,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GatedConvTranspose2d,
     lambda: ([], {'input_channels': 4, 'output_channels': 4, 'kernel_size': 4, 'stride': 1, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedConv2d,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MaskedLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Planar,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])], {}),
     True),
]

class Test_riannevdberg_sylvester_flows(_paritybench_base):
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

