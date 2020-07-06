import sys
_module = sys.modules[__name__]
del sys
master = _module
dataset = _module
get_graph_data = _module
get_qm8_data = _module
graph_data = _module
qm8 = _module
model = _module
ada_lanczos_net = _module
cheby_net = _module
dcnn = _module
gat = _module
gcn = _module
gcnfp = _module
ggnn = _module
gpnn = _module
graph_sage = _module
lanczos_net = _module
lanczos_net_general = _module
mpnn = _module
set2set = _module
operators = _module
build_segment_reduction = _module
functions = _module
unsorted_segment_sum = _module
modules = _module
unsorted_segment_sum = _module
run_exp = _module
runner = _module
graph_runner = _module
qm8_runner = _module
utils = _module
arg_helper = _module
data_helper = _module
logger = _module
spectral_graph_partition = _module
train_helper = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import torch


import numpy as np


import torch.nn as nn


import torch.nn.functional as F


import math


from torch.autograd import Function


from torch.autograd import Variable


from torch.nn.parameter import Parameter


import logging


from collections import defaultdict


import torch.utils.data


import torch.optim as optim


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


EPS = float(np.finfo(np.float32).eps)


def check_dist(dist):
    for dd in dist:
        if not isinstance(dd, int) and dd != 'inf':
            raise ValueError('Non-supported value of diffusion distance')
    return dist


class AdaLanczosNet(nn.Module):

    def __init__(self, config):
        super(AdaLanczosNet, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.short_diffusion_dist = check_dist(config.model.short_diffusion_dist)
        self.long_diffusion_dist = check_dist(config.model.long_diffusion_dist)
        self.max_short_diffusion_dist = max(self.short_diffusion_dist) if self.short_diffusion_dist else None
        self.max_long_diffusion_dist = max(self.long_diffusion_dist) if self.long_diffusion_dist else None
        self.num_scale_short = len(self.short_diffusion_dist)
        self.num_scale_long = len(self.long_diffusion_dist)
        self.num_eig_vec = config.model.num_eig_vec
        self.spectral_filter_kind = config.model.spectral_filter_kind
        self.use_reorthogonalization = config.model.use_reorthogonalization if hasattr(config, 'use_reorthogonalization') else True
        self.use_power_iteration_cap = config.model.use_power_iteration_cap if hasattr(config, 'use_power_iteration_cap') else True
        self.input_dim = self.num_atom
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([nn.Linear(dim_list[tt] * (self.num_scale_short + self.num_scale_long + self.num_edgetype + 1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
            self.spectral_filter = nn.ModuleList([nn.Sequential(*[nn.Linear(self.num_eig_vec * self.num_eig_vec * self.num_scale_long, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, self.num_eig_vec * self.num_eig_vec * self.num_scale_long)]) for _ in range(self.num_layer)])
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
            for f in self.spectral_filter:
                for ff in f:
                    if isinstance(ff, nn.Linear):
                        nn.init.xavier_uniform_(ff.weight.data)
                        if ff.bias is not None:
                            ff.bias.data.zero_()

    def _get_graph_laplacian(self, node_feat, adj_mask):
        """ Compute graph Laplacian

      Args:
        node_feat: float tensor, shape B X N X D
        adj_mask: float tensor, shape B X N X N, binary mask, should contain self-loop

      Returns:
        L: float tensor, shape B X N X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        dim_feat = node_feat.shape[2]
        idx_row, idx_col = np.meshgrid(range(num_node), range(num_node))
        idx_row, idx_col = torch.Tensor(idx_row.reshape(-1)).long(), torch.Tensor(idx_col.reshape(-1)).long()
        diff = node_feat[:, (idx_row), :] - node_feat[:, (idx_col), :]
        dist2 = (diff * diff).sum(dim=2)
        sigma2 = torch.mean(dist2, dim=1, keepdim=True)
        A = torch.exp(-dist2 / sigma2)
        A = A.reshape(batch_size, num_node, num_node) * adj_mask
        row_sum = torch.sum(A, dim=2, keepdim=True)
        pad_row_sum = torch.zeros_like(row_sum)
        pad_row_sum[row_sum == 0.0] = 1.0
        alpha = 0.5
        D = 1.0 / (row_sum + pad_row_sum).pow(alpha)
        L = D * A * D.transpose(1, 2)
        return L

    def _lanczos_layer(self, A, mask=None):
        """ Lanczos layer for symmetric matrix A
    
      Args:
        A: float tensor, shape B X N X N
        mask: float tensor, shape B X N

      Returns:
        T: float tensor, shape B X K X K
        Q: float tensor, shape B X N X K
    """
        batch_size = A.shape[0]
        num_node = A.shape[1]
        lanczos_iter = min(num_node, self.num_eig_vec)
        alpha = [None] * (lanczos_iter + 1)
        beta = [None] * (lanczos_iter + 1)
        Q = [None] * (lanczos_iter + 2)
        beta[0] = torch.zeros(batch_size, 1, 1)
        Q[0] = torch.zeros(batch_size, num_node, 1)
        Q[1] = torch.randn(batch_size, num_node, 1)
        if mask is not None:
            mask = mask.unsqueeze(dim=2).float()
            Q[1] = Q[1] * mask
        Q[1] = Q[1] / torch.norm(Q[1], 2, dim=1, keepdim=True)
        lb = 0.0001
        valid_mask = []
        for ii in range(1, lanczos_iter + 1):
            z = torch.bmm(A, Q[ii])
            alpha[ii] = torch.sum(Q[ii] * z, dim=1, keepdim=True)
            z = z - alpha[ii] * Q[ii] - beta[ii - 1] * Q[ii - 1]
            if self.use_reorthogonalization and ii > 1:

                def _gram_schmidt(xx, tt):
                    for jj in range(1, tt):
                        xx = xx - torch.sum(xx * Q[jj], dim=1, keepdim=True) / (torch.sum(Q[jj] * Q[jj], dim=1, keepdim=True) + EPS) * Q[jj]
                    return xx
                for _ in range(2):
                    z = _gram_schmidt(z, ii)
            beta[ii] = torch.norm(z, p=2, dim=1, keepdim=True)
            tmp_valid_mask = (beta[ii] >= lb).float()
            if ii == 1:
                valid_mask += [tmp_valid_mask]
            else:
                valid_mask += [valid_mask[-1] * tmp_valid_mask]
            Q[ii + 1] = z * valid_mask[-1] / (beta[ii] + EPS)
        alpha = torch.cat(alpha[1:], dim=1).squeeze(dim=2)
        beta = torch.cat(beta[1:-1], dim=1).squeeze(dim=2)
        valid_mask = torch.cat(valid_mask, dim=1).squeeze(dim=2)
        idx_mask = torch.sum(valid_mask, dim=1).long()
        if mask is not None:
            idx_mask = torch.min(idx_mask, torch.sum(mask, dim=1).squeeze().long())
        for ii in range(batch_size):
            if idx_mask[ii] < valid_mask.shape[1]:
                valid_mask[(ii), idx_mask[ii]:] = 0.0
        alpha = alpha * valid_mask
        beta = beta * valid_mask[:, :-1]
        T = []
        for ii in range(batch_size):
            T += [torch.diag(alpha[ii]) + torch.diag(beta[ii], diagonal=1) + torch.diag(beta[ii], diagonal=-1)]
        T = torch.stack(T, dim=0)
        Q = torch.cat(Q[1:-1], dim=2)
        Q_mask = valid_mask.unsqueeze(dim=1).repeat(1, Q.shape[1], 1)
        for ii in range(batch_size):
            if idx_mask[ii] < Q_mask.shape[1]:
                Q_mask[(ii), idx_mask[ii]:, :] = 0.0
        Q = Q * Q_mask
        if lanczos_iter < self.num_eig_vec:
            pad = 0, self.num_eig_vec - lanczos_iter, 0, self.num_eig_vec - lanczos_iter
            T = F.pad(T, pad)
            pad = 0, self.num_eig_vec - lanczos_iter
            Q = F.pad(Q, pad)
        return T, Q

    def _get_spectral_filters(self, T, Q, layer_idx):
        """ Construct Spectral Filters based on Lanczos Outputs

      Args:
        T: shape B X K X K, tridiagonal matrix
        Q: shape B X N X K, orthonormal matrix
        layer_idx: int, index of layer

      Returns:
        L: shape B X N X N X num_scale
    """
        L = []
        T_list = []
        TT = T
        for ii in range(1, self.max_long_diffusion_dist + 1):
            if ii in self.long_diffusion_dist:
                T_list += [TT]
            TT = torch.bmm(TT, T)
        if self.spectral_filter_kind == 'MLP':
            DD = self.spectral_filter[layer_idx](torch.cat(T_list, dim=2).view(T.shape[0], -1))
            DD = DD.view(T.shape[0], T.shape[1], T.shape[2], self.num_scale_long)
            DD = (DD + DD.transpose(1, 2)) * 0.5
            for ii in range(self.num_scale_long):
                L += [Q.bmm(DD[:, :, :, (ii)]).bmm(Q.transpose(1, 2))]
        else:
            for ii in range(self.num_scale_long):
                L += [Q.bmm(T_list[ii]).bmm(Q.transpose(1, 2))]
        return torch.stack(L, dim=3)

    def forward(self, node_feat, L, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        if self.num_scale_long > 0:
            adj = torch.zeros_like(L[:, :, :, (0)])
            adj[L[:, :, :, (0)] != 0.0] = 1.0
            Le = self._get_graph_laplacian(state, adj)
            T, Q = self._lanczos_layer(Le, mask)
        for tt in range(self.num_layer):
            msg = []
            if self.num_scale_long > 0:
                Lf = self._get_spectral_filters(T, Q, tt)
            if self.num_scale_short > 0:
                tmp_state = state
                for ii in range(1, self.max_short_diffusion_dist + 1):
                    tmp_state = torch.bmm(L[:, :, :, (0)], tmp_state)
                    if ii in self.short_diffusion_dist:
                        msg += [tmp_state]
            if self.num_scale_long > 0:
                for ii in range(self.num_scale_long):
                    msg += [torch.bmm(Lf[:, :, :, (ii)], state)]
            for ii in range(self.num_edgetype + 1):
                msg += [torch.bmm(L[:, :, :, (ii)], state)]
            msg = torch.cat(msg, dim=2).view(num_node * batch_size, -1)
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class ChebyNet(nn.Module):
    """ Chebyshev Network, see reference below for more information

      Defferrard, M., Bresson, X. and Vandergheynst, P., 2016.
      Convolutional neural networks on graphs with fast localized spectral
      filtering. In NIPS.
  """

    def __init__(self, config):
        super(ChebyNet, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.polynomial_order = config.model.polynomial_order
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([nn.Linear(dim_list[tt] * (self.polynomial_order + self.num_edgetype + 1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

    def forward(self, node_feat, L, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        for tt in range(self.num_layer):
            state_scale = [None] * (self.polynomial_order + 1)
            state_scale[-1] = state
            state_scale[0] = torch.bmm(L[:, :, :, (0)], state)
            for kk in range(1, self.polynomial_order):
                state_scale[kk] = 2.0 * torch.bmm(L[:, :, :, (0)], state_scale[kk - 1]) - state_scale[kk - 2]
            msg = []
            for ii in range(1, self.num_edgetype + 1):
                msg += [torch.bmm(L[:, :, :, (ii)], state)]
            msg = torch.cat(msg + state_scale, dim=2).view(num_node * batch_size, -1)
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class DCNN(nn.Module):
    """ Diffusion-convolutional neural networks,
      see reference below for more information

      Atwood, J. and Towsley, D., 2016.
      Diffusion-convolutional neural networks. In NIPS.
  """

    def __init__(self, config):
        super(DCNN, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.diffusion_dist = config.model.diffusion_dist
        self.num_scale = len(self.diffusion_dist)
        self.max_dist = max(config.model.diffusion_dist)
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([nn.Linear(dim_list[tt] * (self.num_scale + self.num_edgetype + 1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

    def forward(self, node_feat, L, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        for tt in range(self.num_layer):
            state_scale = []
            tmp_state = state
            for ii in range(1, self.max_dist + 1):
                tmp_state = torch.bmm(L[:, :, :, (0)], tmp_state)
                if ii in self.diffusion_dist:
                    state_scale += [tmp_state]
            msg = []
            for ii in range(self.num_edgetype + 1):
                msg += [torch.bmm(L[:, :, :, (ii)], state)]
            msg = torch.cat(msg + state_scale, dim=2).view(num_node * batch_size, -1)
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class GAT(nn.Module):
    """ Graph Attention Networks,
      see reference below for more information

      Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P.
      and Bengio, Y., 2018. Graph attention networks. In ICLR.
  """

    def __init__(self, config):
        super(GAT, self).__init__()
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.num_heads = config.model.num_heads
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Linear(dim_list[tt] * (int(tt == 0) + int(tt != 0) * self.num_heads[tt] * (self.num_edgetype + 1)), dim_list[tt + 1], bias=False) for _ in range(self.num_heads[tt])]) for _ in range(self.num_edgetype + 1)]) for tt in range(self.num_layer)])
        self.att_net_1 = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Linear(dim_list[tt + 1], 1) for _ in range(self.num_heads[tt])]) for _ in range(self.num_edgetype + 1)]) for tt in range(self.num_layer)])
        self.att_net_2 = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Linear(dim_list[tt + 1], 1) for _ in range(self.num_heads[tt])]) for _ in range(self.num_edgetype + 1)]) for tt in range(self.num_layer)])
        self.state_bias = [([[None] * self.num_heads[tt]] * (self.num_edgetype + 1)) for tt in range(self.num_layer)]
        for tt in range(self.num_layer):
            for jj in range(self.num_edgetype + 1):
                for ii in range(self.num_heads[tt]):
                    self.state_bias[tt][jj][ii] = torch.nn.Parameter(torch.zeros(dim_list[tt + 1]))
                    self.register_parameter('bias_{}_{}_{}'.format(ii, jj, tt), self.state_bias[tt][jj][ii])
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        self.output_func = nn.Sequential(*[nn.Linear(dim_list[-2], dim_list[-1])])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for ff in self.output_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for f in self.filter:
            for ff in f:
                for fff in ff:
                    if isinstance(fff, nn.Linear):
                        nn.init.xavier_uniform_(fff.weight.data)
                        if fff.bias is not None:
                            fff.bias.data.zero_()
        for f in self.att_net_1:
            for ff in f:
                for fff in ff:
                    if isinstance(fff, nn.Linear):
                        nn.init.xavier_uniform_(fff.weight.data)
                        if fff.bias is not None:
                            fff.bias.data.zero_()
        for f in self.att_net_2:
            for ff in f:
                for fff in ff:
                    if isinstance(fff, nn.Linear):
                        nn.init.xavier_uniform_(fff.weight.data)
                        if fff.bias is not None:
                            fff.bias.data.zero_()

    def forward(self, node_feat, L, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        for tt in range(self.num_layer):
            h = []
            for jj in range(self.num_edgetype + 1):
                for ii in range(self.num_heads[tt]):
                    state_head = F.dropout(state, self.dropout, training=self.training)
                    Wh = self.filter[tt][jj][ii](state_head.view(batch_size * num_node, -1)).view(batch_size, num_node, -1)
                    att_weights_1 = self.att_net_1[tt][jj][ii](Wh)
                    att_weights_2 = self.att_net_2[tt][jj][ii](Wh)
                    att_weights = att_weights_1 + att_weights_2.transpose(1, 2)
                    att_weights = F.softmax(F.leaky_relu(att_weights, negative_slope=0.2) + L[:, :, :, (jj)], dim=1)
                    att_weights = F.dropout(att_weights, self.dropout, training=self.training)
                    Wh = F.dropout(Wh, self.dropout, training=self.training)
                    if tt == self.num_layer - 1:
                        h += [torch.bmm(att_weights, Wh) + self.state_bias[tt][jj][ii].view(1, 1, -1)]
                    else:
                        h += [F.elu(torch.bmm(att_weights, Wh) + self.state_bias[tt][jj][ii].view(1, 1, -1))]
            if tt == self.num_layer - 1:
                state = torch.mean(torch.stack(h, dim=0), dim=0)
            else:
                state = torch.cat(h, dim=2)
        state = state.view(batch_size * num_node, -1)
        y = self.output_func(state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class GCN(nn.Module):
    """ Graph Convolutional Networks,
      see reference below for more information

      Kipf, T.N. and Welling, M., 2016.
      Semi-supervised classification with graph convolutional networks.
      arXiv preprint arXiv:1609.02907.
  """

    def __init__(self, config):
        super(GCN, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([nn.Linear(dim_list[tt] * (self.num_edgetype + 1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

    def forward(self, node_feat, L, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        for tt in range(self.num_layer):
            msg = []
            for ii in range(self.num_edgetype + 1):
                msg += [torch.bmm(L[:, :, :, (ii)], state)]
            msg = torch.cat(msg, dim=2).view(num_node * batch_size, -1)
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class GCNFP(nn.Module):
    """ Graph Convolutional Networks for fingerprints,
      see reference below for more information

      Duvenaud, D.K., Maclaurin, D., Iparraguirre, J., Bombarell,
      R., Hirzel, T., Aspuru-Guzik, A. and Adams, R.P., 2015.
      Convolutional networks on graphs for learning molecular
      fingerprints. In NIPS.

      N.B.: the difference with GCN is, Duvenaud et. al. use
      binary adjacency matrix rather than graph Laplacian
  """

    def __init__(self, config):
        super(GCNFP, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([nn.Linear(dim_list[tt] * (self.num_edgetype + 1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

    def forward(self, node_feat, L, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        L[L != 0] = 1.0
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        for tt in range(self.num_layer):
            msg = []
            for ii in range(self.num_edgetype + 1):
                msg += [torch.bmm(L[:, :, :, (ii)], state)]
            msg = torch.cat(msg, dim=2).view(num_node * batch_size, -1)
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class GGNN(nn.Module):

    def __init__(self, config):
        """ Gated Graph Neural Networks,
        see reference below for more information

        Li, Y., Tarlow, D., Brockschmidt, M. and Zemel, R., 2015. 
        Gated graph sequence neural networks. 
        arXiv preprint arXiv:1511.05493.
    """
        super(GGNN, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.num_prop = config.model.num_prop
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.aggregate_type = config.model.aggregate_type
        assert self.num_layer == 1, 'not implemented'
        assert self.aggregate_type in ['avg', 'sum'], 'not implemented'
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        if config.model.update_func == 'RNN':
            self.update_func = nn.RNNCell(input_size=self.hidden_dim * (self.num_edgetype + 1), hidden_size=self.hidden_dim, nonlinearity='relu')
        elif config.model.update_func == 'GRU':
            self.update_func = nn.GRUCell(input_size=self.hidden_dim * (self.num_edgetype + 1), hidden_size=self.hidden_dim)
        elif config.model.update_func == 'MLP':
            self.update_func = nn.Sequential(*[nn.Linear(self.hidden_dim * (self.num_edgetype + 1), self.hidden_dim), nn.Tanh()])
        if config.model.msg_func == 'MLP':
            self.msg_func = nn.ModuleList([nn.Sequential(*[nn.Linear(self.hidden_dim, 128), nn.ReLU(), nn.Linear(128, self.hidden_dim)]) for _ in range(self.num_edgetype + 1)])
        else:
            self.msg_func = None
        self.att_func = nn.Sequential(*[nn.Linear(self.hidden_dim, 1), nn.Sigmoid()])
        self.input_func = nn.Sequential(*[nn.Linear(self.input_dim, self.hidden_dim)])
        self.output_func = nn.Sequential(*[nn.Linear(self.hidden_dim, self.output_dim)])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        mlp_modules = [xx for xx in [self.input_func, self.msg_func, self.att_func, self.output_func] if xx is not None]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.config.model.update_func in ['GRU', 'RNN']:
            for m in [self.update_func]:
                nn.init.xavier_uniform_(m.weight_hh.data)
                nn.init.xavier_uniform_(m.weight_ih.data)
                if m.bias:
                    m.bias_hh.data.zero_()
                    m.bias_ih.data.zero_()
        elif self.config.model.update_func == 'MLP':
            for m in self.update_func:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, node_feat, L, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        L[L != 0] = 1.0
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        state = self.input_func(state)

        def _prop(state_old):
            msg = []
            for ii in range(self.num_edgetype + 1):
                if self.msg_func is not None:
                    tmp_msg = self.msg_func[ii](state_old.view(batch_size * num_node, -1)).view(batch_size, num_node, -1)
                if self.aggregate_type == 'sum':
                    tmp_msg = torch.bmm(L[:, :, :, (ii)], tmp_msg)
                elif self.aggregate_type == 'avg':
                    denom = torch.sum(L[:, :, :, (ii)], dim=2, keepdim=True) + EPS
                    tmp_msg = torch.bmm(L[:, :, :, (ii)] / denom, tmp_msg)
                else:
                    pass
                msg += [tmp_msg]
            msg = torch.cat(msg, dim=2).view(batch_size * num_node, -1)
            state_old = state_old.view(batch_size * num_node, -1)
            state_new = self.update_func(msg, state_old).view(batch_size, num_node, -1)
            return state_new
        for tt in range(self.num_prop):
            state = _prop(state)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.output_func(state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class GPNN(nn.Module):

    def __init__(self, config):
        """ Graph Partition Neural Networks,
        see reference below for more information

        Liao, R., Brockschmidt, M., Tarlow, D., Gaunt, A.L., 
        Urtasun, R. and Zemel, R., 2018. 
        Graph Partition Neural Networks for Semi-Supervised 
        Classification. arXiv preprint arXiv:1803.06272.
    """
        super(GPNN, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.num_prop = config.model.num_prop
        self.num_partition = config.model.num_partition
        self.num_prop_cluster = config.model.num_prop_cluster
        self.num_prop_cut = config.model.num_prop_cut
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.aggregate_type = config.model.aggregate_type
        assert self.num_layer == 1, 'not implemented'
        assert self.aggregate_type in ['avg', 'sum'], 'not implemented'
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        if config.model.update_func == 'RNN':
            self.update_func = nn.RNNCell(input_size=self.hidden_dim * (self.num_edgetype + 1), hidden_size=self.hidden_dim, nonlinearity='relu')
            self.update_func_partition = nn.RNNCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim, nonlinearity='relu')
        elif config.model.update_func == 'GRU':
            self.update_func = nn.GRUCell(input_size=self.hidden_dim * (self.num_edgetype + 1), hidden_size=self.hidden_dim)
            self.update_func_partition = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        elif config.model.update_func == 'MLP':
            self.update_func = nn.Sequential(*[nn.Linear(self.hidden_dim * (self.num_edgetype + 1), self.hidden_dim), nn.Tanh()])
            self.update_func_partition = nn.Sequential(*[nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh()])
        self.state_func = nn.Sequential(*[nn.Linear(3 * self.hidden_dim, 512), nn.ReLU(), nn.Linear(512, self.hidden_dim)])
        if config.model.msg_func == 'MLP':
            self.msg_func = nn.ModuleList([nn.Sequential(*[nn.Linear(self.hidden_dim, 128), nn.ReLU(), nn.Linear(128, self.hidden_dim)]) for _ in range(self.num_edgetype + 1)])
        else:
            self.msg_func = None
        self.att_func = nn.Sequential(*[nn.Linear(self.hidden_dim, 1), nn.Sigmoid()])
        self.input_func = nn.Sequential(*[nn.Linear(self.input_dim, self.hidden_dim)])
        self.output_func = nn.Sequential(*[nn.Linear(self.hidden_dim, self.output_dim)])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        mlp_modules = [xx for xx in [self.input_func, self.state_func, self.msg_func, self.att_func, self.output_func] if xx is not None]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.config.model.update_func in ['GRU', 'RNN']:
            for m in [self.update_func, self.update_func_partition]:
                nn.init.xavier_uniform_(m.weight_hh.data)
                nn.init.xavier_uniform_(m.weight_ih.data)
                if m.bias:
                    m.bias_hh.data.zero_()
                    m.bias_ih.data.zero_()
        elif self.config.model.update_func == 'MLP':
            for m in [self.update_func, self.update_func_partition]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, node_feat, L, L_cluster, L_cut, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P

      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        L_cluster: float tensor, shape B X N X N
        L_cut: float tensor, shape B X N X N
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        L[L != 0] = 1.0
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        state = self.input_func(state)

        def _prop(state_old):
            msg = []
            for ii in range(self.num_edgetype + 1):
                if self.msg_func is not None:
                    tmp_msg = self.msg_func[ii](state_old.view(batch_size * num_node, -1)).view(batch_size, num_node, -1)
                if self.aggregate_type == 'sum':
                    tmp_msg = torch.bmm(L[:, :, :, (ii)], tmp_msg)
                elif self.aggregate_type == 'avg':
                    denom = torch.sum(L[:, :, :, (ii)], dim=2, keepdim=True) + EPS
                    tmp_msg = torch.bmm(L[:, :, :, (ii)] / denom, tmp_msg)
                else:
                    pass
                msg += [tmp_msg]
            msg = torch.cat(msg, dim=2).view(batch_size * num_node, -1)
            state_old = state_old.view(batch_size * num_node, -1)
            state_new = self.update_func(msg, state_old).view(batch_size, num_node, -1)
            return state_new

        def _prop_partition(state_old, L_step):
            if self.msg_func is not None:
                msg = self.msg_func[0](state_old.view(batch_size * num_node, -1)).view(batch_size, num_node, -1)
            if self.aggregate_type == 'sum':
                msg = torch.bmm(L_step, msg)
            elif self.aggregate_type == 'avg':
                denom = torch.sum(L_step, dim=2, keepdim=True) + EPS
                msg = torch.bmm(L_step / denom, msg)
            else:
                pass
            msg = msg.view(batch_size * num_node, -1)
            state_old = state_old.view(batch_size * num_node, -1)
            state_new = self.update_func_partition(msg, state_old).view(batch_size, num_node, -1)
            return state_new
        for tt in range(self.num_prop):
            state_cluster = state
            for ii in range(self.num_prop_cluster):
                state_cluster = _prop_partition(state_cluster, L_cluster)
            state_cut = state
            for ii in range(self.num_prop_cut):
                state_cut = _prop_partition(state_cut, L_cut)
            state = self.state_func(torch.cat([state, state_cluster, state_cut], dim=2))
            state = _prop(state)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.output_func(state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class GraphSAGE(nn.Module):

    def __init__(self, config):
        """ GraphSAGE,
        see reference below for more information

        Hamilton, W., Ying, Z. and Leskovec, J., 2017. Inductive
        representation learning on large graphs. In NIPS.
    """
        super(GraphSAGE, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.num_sample_neighbors = config.model.num_sample_neighbors
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        assert self.num_layer == len(self.hidden_dim)
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        self.agg_func_name = config.model.agg_func
        if self.agg_func_name == 'LSTM':
            self.agg_func = nn.ModuleList([nn.LSTMCell(input_size=dim_list[tt], hidden_size=dim_list[tt]) for tt in range(self.num_layer - 1)])
        elif self.agg_func_name == 'Mean':
            self.agg_func = torch.mean
        elif self.agg_func_name == 'Max':
            self.agg_func = torch.max
        else:
            self.agg_func = None
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        self.filter = nn.ModuleList([nn.Linear(dim_list[tt] * (self.num_edgetype + 1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        mlp_modules = [xx for xx in [self.att_func] if xx is not None]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.config.model.agg_func == 'LSTM':
            for m in self.agg_func:
                nn.init.xavier_uniform_(m.weight_hh.data)
                nn.init.xavier_uniform_(m.weight_ih.data)
                if m.bias:
                    m.bias_hh.data.zero_()
                    m.bias_ih.data.zero_()
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()

    def forward(self, node_feat, nn_idx, nonempty_mask, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        neighborhood size = K
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: float tensor, shape B X N X D
        nn_idx: float tensor, shape B X N X K X E
        nonempty_mask: float tensor, shape B X N X 1
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.embedding(node_feat)
        for ii in range(self.num_layer - 1):
            msg = []
            for jj in range(self.num_edgetype + 1):
                nn_state = []
                for bb in range(batch_size):
                    nn_state += [state[(bb), (nn_idx[(bb), :, :, (jj)]), :]]
                nn_state = torch.stack(nn_state, dim=0)
                if self.agg_func_name == 'LSTM':
                    cx = torch.zeros_like(state).view(batch_size * num_node, -1)
                    hx = torch.zeros_like(state).view(batch_size * num_node, -1)
                    for tt in range(self.num_sample_neighbors):
                        ix = nn_state[:, :, (tt), :]
                        hx, cx = self.agg_func[ii](ix.view(batch_size * num_node, -1), (hx, cx))
                    agg_state = hx.view(batch_size, num_node, -1)
                elif self.agg_func_name == 'Max':
                    agg_state, _ = self.agg_func(nn_state, dim=2)
                else:
                    agg_state = self.agg_func(nn_state, dim=2)
                msg += [agg_state * nonempty_mask]
            state = F.relu(self.filter[ii](torch.cat(msg, dim=2).view(batch_size * num_node, -1)))
            state = (state / (torch.norm(state, 2, dim=1, keepdim=True) + EPS)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        if mask is not None:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        else:
            for bb in range(batch_size):
                score += [torch.mean(y[(bb), :, :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class LanczosNet(nn.Module):

    def __init__(self, config):
        super(LanczosNet, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.short_diffusion_dist = check_dist(config.model.short_diffusion_dist)
        self.long_diffusion_dist = check_dist(config.model.long_diffusion_dist)
        self.max_short_diffusion_dist = max(self.short_diffusion_dist) if self.short_diffusion_dist else None
        self.max_long_diffusion_dist = max(self.long_diffusion_dist) if self.long_diffusion_dist else None
        self.num_scale_short = len(self.short_diffusion_dist)
        self.num_scale_long = len(self.long_diffusion_dist)
        self.num_eig_vec = config.model.num_eig_vec
        self.spectral_filter_kind = config.model.spectral_filter_kind
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([nn.Linear(dim_list[tt] * (self.num_scale_short + self.num_scale_long + self.num_edgetype + 1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        self.embedding = nn.Embedding(self.num_atom, self.input_dim)
        if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
            self.spectral_filter = nn.ModuleList([nn.Sequential(*[nn.Linear(self.num_scale_long, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.num_scale_long)]) for _ in range(self.num_layer)])
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
            for ff in self.spectral_filter:
                for f in ff:
                    if isinstance(f, nn.Linear):
                        nn.init.xavier_uniform_(f.weight.data)
                        if f.bias is not None:
                            f.bias.data.zero_()

    def _get_spectral_filters(self, T_list, Q, layer_idx):
        """ Construct Spectral Filters based on Lanczos Outputs

      Args:
        T_list: each element is of shape B X K
        Q: shape B X N X K

      Returns:
        L: shape B X N X N X num_scale
    """
        L = []
        if self.spectral_filter_kind == 'MLP':
            DD = torch.stack(T_list, dim=2).view(Q.shape[0] * Q.shape[2], -1)
            DD = self.spectral_filter[layer_idx](DD).view(Q.shape[0], Q.shape[2], -1)
            for ii in range(self.num_scale_long):
                tmp_DD = DD[:, :, (ii)].unsqueeze(1).repeat(1, Q.shape[1], 1)
                L += [(Q * tmp_DD).bmm(Q.transpose(1, 2))]
        else:
            for ii in range(self.num_scale_long):
                DD = T_list[ii].unsqueeze(1).repeat(1, Q.shape[1], 1)
                L += [(Q * DD).bmm(Q.transpose(1, 2))]
        return torch.stack(L, dim=3)

    def forward(self, node_feat, L, D, V, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
        number of approximated eigenvalues, i.e., Ritz values = K
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        D: float tensor, Ritz values, shape B X K
        V: float tensor, Ritz vectors, shape B X N X K
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        D_pow_list = []
        for ii in self.long_diffusion_dist:
            D_pow_list += [torch.pow(D, ii)]
        state = self.embedding(node_feat)
        for tt in range(self.num_layer):
            msg = []
            if self.num_scale_long > 0:
                Lf = self._get_spectral_filters(D_pow_list, V, tt)
            if self.num_scale_short > 0:
                tmp_state = state
                for ii in range(1, self.max_short_diffusion_dist + 1):
                    tmp_state = torch.bmm(L[:, :, :, (0)], tmp_state)
                    if ii in self.short_diffusion_dist:
                        msg += [tmp_state]
            if self.num_scale_long > 0:
                for ii in range(self.num_scale_long):
                    msg += [torch.bmm(Lf[:, :, :, (ii)], state)]
            for ii in range(self.num_edgetype + 1):
                msg += [torch.bmm(L[:, :, :, (ii)], state)]
            msg = torch.cat(msg, dim=2).view(num_node * batch_size, -1)
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        for bb in range(batch_size):
            score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class LanczosNetGeneral(nn.Module):

    def __init__(self, config):
        super(LanczosNetGeneral, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.node_emb_dim = config.dataset.node_emb_dim
        self.graph_emb_dim = config.dataset.graph_emb_dim
        self.num_edgetype = config.dataset.num_edge_type
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.short_diffusion_dist = check_dist(config.model.short_diffusion_dist)
        self.long_diffusion_dist = check_dist(config.model.long_diffusion_dist)
        self.max_short_diffusion_dist = max(self.short_diffusion_dist) if self.short_diffusion_dist else None
        self.max_long_diffusion_dist = max(self.long_diffusion_dist) if self.long_diffusion_dist else None
        self.num_scale_short = len(self.short_diffusion_dist)
        self.num_scale_long = len(self.long_diffusion_dist)
        self.num_eig_vec = config.model.num_eig_vec
        self.spectral_filter_kind = config.model.spectral_filter_kind
        dim_list = [self.input_dim] + self.hidden_dim + [self.output_dim]
        self.filter = nn.ModuleList([nn.Linear(dim_list[tt] * (self.num_scale_short + self.num_scale_long + self.num_edgetype + 1), dim_list[tt + 1]) for tt in range(self.num_layer)] + [nn.Linear(dim_list[-2], dim_list[-1])])
        assert self.input_dim == self.node_emb_dim
        assert self.output_dim == self.graph_emb_dim
        if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
            self.spectral_filter = nn.ModuleList([nn.Sequential(*[nn.Linear(self.num_scale_long, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.num_scale_long)]) for _ in range(self.num_layer)])
        self.att_func = nn.Sequential(*[nn.Linear(dim_list[-2], 1), nn.Sigmoid()])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        for ff in self.filter:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        for ff in self.att_func:
            if isinstance(ff, nn.Linear):
                nn.init.xavier_uniform_(ff.weight.data)
                if ff.bias is not None:
                    ff.bias.data.zero_()
        if self.spectral_filter_kind == 'MLP' and self.num_scale_long > 0:
            for ff in self.spectral_filter:
                for f in ff:
                    if isinstance(f, nn.Linear):
                        nn.init.xavier_uniform_(f.weight.data)
                        if f.bias is not None:
                            f.bias.data.zero_()

    def _get_spectral_filters(self, T_list, Q, layer_idx):
        """ Construct Spectral Filters based on Lanczos Outputs

      Args:
        T_list: each element is of shape B X K
        Q: shape B X N X K

      Returns:
        L: shape B X N X N X num_scale
    """
        L = []
        if self.spectral_filter_kind == 'MLP':
            DD = torch.stack(T_list, dim=2).view(Q.shape[0] * Q.shape[2], -1)
            DD = self.spectral_filter[layer_idx](DD).view(Q.shape[0], Q.shape[2], -1)
            for ii in range(self.num_scale_long):
                tmp_DD = DD[:, :, (ii)].unsqueeze(1).repeat(1, Q.shape[1], 1)
                L += [(Q * tmp_DD).bmm(Q.transpose(1, 2))]
        else:
            for ii in range(self.num_scale_long):
                DD = T_list[ii].unsqueeze(1).repeat(1, Q.shape[1], 1)
                L += [(Q * DD).bmm(Q.transpose(1, 2))]
        return torch.stack(L, dim=3)

    def forward(self, node_feat, L, D, V, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
        number of approximated eigenvalues, i.e., Ritz values = K
      
      Args:
        node_feat: long tensor, shape B X N X D
        L: float tensor, shape B X N X N X (E + 1)
        D: float tensor, Ritz values, shape B X K
        V: float tensor, Ritz vectors, shape B X N X K
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        D_pow_list = []
        for ii in self.long_diffusion_dist:
            D_pow_list += [torch.pow(D, ii)]
        state = node_feat
        for tt in range(self.num_layer):
            msg = []
            if self.num_scale_long > 0:
                Lf = self._get_spectral_filters(D_pow_list, V, tt)
            if self.num_scale_short > 0:
                tmp_state = state
                for ii in range(1, self.max_short_diffusion_dist + 1):
                    tmp_state = torch.bmm(L[:, :, :, (0)], tmp_state)
                    if ii in self.short_diffusion_dist:
                        msg += [tmp_state]
            if self.num_scale_long > 0:
                for ii in range(self.num_scale_long):
                    msg += [torch.bmm(Lf[:, :, :, (ii)], state)]
            for ii in range(self.num_edgetype + 1):
                msg += [torch.bmm(L[:, :, :, (ii)], state)]
            msg = torch.cat(msg, dim=2).view(num_node * batch_size, -1)
            state = F.relu(self.filter[tt](msg)).view(batch_size, num_node, -1)
            state = F.dropout(state, self.dropout, training=self.training)
        state = state.view(batch_size * num_node, -1)
        y = self.filter[-1](state)
        att_weight = self.att_func(state)
        y = (att_weight * y).view(batch_size, num_node, -1)
        score = []
        for bb in range(batch_size):
            score += [torch.mean(y[(bb), (mask[bb]), :], dim=0)]
        score = torch.stack(score)
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class Set2SetLSTM(nn.Module):

    def __init__(self, hidden_dim):
        """ Implementation of customized LSTM for set2set """
        super(Set2SetLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.forget_gate = nn.Sequential(*[nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Sigmoid()])
        self.input_gate = nn.Sequential(*[nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Sigmoid()])
        self.output_gate = nn.Sequential(*[nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Sigmoid()])
        self.memory_gate = nn.Sequential(*[nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Tanh()])
        self._init_param()

    def _init_param(self):
        for m in [self.forget_gate, self.input_gate, self.output_gate, self.memory_gate]:
            for mm in m:
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight.data)
                    if mm.bias is not None:
                        mm.bias.data.zero_()

    def forward(self, hidden, memory):
        """
      Args:
        hidden: shape N X 2D
        memory: shape N X D

      Returns:
        hidden: shape N X D
        memory: shape N X D
    """
        ft = self.forget_gate(hidden)
        it = self.input_gate(hidden)
        ot = self.output_gate(hidden)
        ct = self.memory_gate(hidden)
        memory = ft * memory + it * ct
        hidden = ot * torch.tanh(memory)
        return hidden, memory


class Set2Vec(nn.Module):

    def __init__(self, element_dim, num_step_encoder):
        """ Implementation of Set2Vec """
        super(Set2Vec, self).__init__()
        self.element_dim = element_dim
        self.num_step_encoder = num_step_encoder
        self.LSTM = Set2SetLSTM(element_dim)
        self.W_1 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
        self.W_2 = nn.Parameter(torch.ones(self.element_dim, 1))
        self.register_parameter('W_1', self.W_1)
        self.register_parameter('W_2', self.W_2)
        self._init_param()

    def _init_param(self):
        nn.init.xavier_uniform_(self.W_1.data)
        nn.init.xavier_uniform_(self.W_2.data)

    def forward(self, input_set):
        """
      Args:
        input_set: shape N X D

      Returns:
        output_vec: shape 1 X 2D
    """
        num_element = input_set.shape[0]
        element_dim = input_set.shape[1]
        assert element_dim == self.element_dim
        hidden = torch.zeros(1, 2 * self.element_dim)
        memory = torch.zeros(1, self.element_dim)
        for tt in range(self.num_step_encoder):
            hidden, memory = self.LSTM(hidden, memory)
            energy = torch.tanh(torch.mm(hidden, self.W_1) + input_set).mm(self.W_2)
            att_weight = F.softmax(energy, dim=0)
            read = (input_set * att_weight).sum(dim=0, keepdim=True)
            hidden = torch.cat([hidden, read], dim=1)
        return hidden


class MPNN(nn.Module):

    def __init__(self, config):
        """ Message Passing Neural Networks,
        see reference below for more information

        Gilmer, J., Schoenholz, S.S., Riley, P.F., Vinyals, O. and Dahl,
        G.E., 2017. Neural message passing for quantum chemistry. In ICML.
    """
        super(MPNN, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.output_dim = config.model.output_dim
        self.num_layer = config.model.num_layer
        self.num_prop = config.model.num_prop
        self.msg_func_name = config.model.msg_func
        self.num_step_set2vec = config.model.num_step_set2vec
        self.dropout = config.model.dropout if hasattr(config.model, 'dropout') else 0.0
        self.num_atom = config.dataset.num_atom
        self.num_edgetype = config.dataset.num_bond_type
        self.aggregate_type = config.model.aggregate_type
        assert self.num_layer == 1, 'not implemented'
        assert self.aggregate_type in ['avg', 'sum'], 'not implemented'
        self.node_embedding = nn.Embedding(self.num_atom, self.input_dim)
        self.input_func = nn.Sequential(*[nn.Linear(self.input_dim, self.hidden_dim)])
        self.update_func = nn.GRUCell(input_size=self.hidden_dim * (self.num_edgetype + 1), hidden_size=self.hidden_dim)
        if config.model.msg_func == 'embedding':
            self.edge_embedding = nn.Embedding(self.num_edgetype + 1, self.hidden_dim ** 2)
        elif config.model.msg_func == 'MLP':
            self.edge_func = nn.ModuleList([nn.Sequential(*[nn.Linear(self.hidden_dim * 2, 64), nn.ReLU(), nn.Linear(64, self.hidden_dim)]) for _ in range(self.num_edgetype + 1)])
        else:
            raise ValueError('Non-supported message function')
        self.att_func = Set2Vec(self.hidden_dim, self.num_step_set2vec)
        self.output_func = nn.Sequential(*[nn.Linear(2 * self.hidden_dim, self.output_dim)])
        if config.model.loss == 'CrossEntropy':
            self.loss_func = torch.nn.CrossEntropyLoss()
        elif config.model.loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif config.model.loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            raise ValueError('Non-supported loss function!')
        self._init_param()

    def _init_param(self):
        mlp_modules = [xx for xx in [self.input_func, self.output_func, self.att_func] if xx is not None]
        for m in mlp_modules:
            if isinstance(m, nn.Sequential):
                for mm in m:
                    if isinstance(mm, nn.Linear):
                        nn.init.xavier_uniform_(mm.weight.data)
                        if mm.bias is not None:
                            mm.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in [self.update_func]:
            nn.init.xavier_uniform_(m.weight_hh.data)
            nn.init.xavier_uniform_(m.weight_ih.data)
            if m.bias:
                m.bias_hh.data.zero_()
                m.bias_ih.data.zero_()

    def forward(self, node_feat, L, label=None, mask=None):
        """
      shape parameters:
        batch size = B
        embedding dim = D
        max number of nodes within one mini batch = N
        number of edge types = E
        number of predicted properties = P
      
      Args:
        node_feat: long tensor, shape B X N
        L: float tensor, shape B X N X N X (E + 1)
        label: float tensor, shape B X P
        mask: float tensor, shape B X N
    """
        L[L != 0] = 1.0
        batch_size = node_feat.shape[0]
        num_node = node_feat.shape[1]
        state = self.node_embedding(node_feat)
        state = self.input_func(state)
        if self.msg_func_name == 'MLP':
            idx_row, idx_col = np.meshgrid(range(num_node), range(num_node))
            idx_row, idx_col = idx_row.flatten().astype(np.int64), idx_col.flatten().astype(np.int64)

        def _prop(state_old):
            state_dim = state_old.shape[2]
            msg = []
            for ii in range(self.num_edgetype + 1):
                if self.msg_func_name == 'embedding':
                    idx_edgetype = torch.Tensor([ii]).long()
                    edge_em = self.edge_embedding(idx_edgetype).view(state_dim, state_dim)
                    node_state = state_old.view(batch_size * num_node, -1)
                    tmp_msg = node_state.mm(edge_em).view(batch_size, num_node, -1)
                    if self.aggregate_type == 'sum':
                        tmp_msg = torch.bmm(L[:, :, :, (ii)], tmp_msg)
                    elif self.aggregate_type == 'avg':
                        denom = torch.sum(L[:, :, :, (ii)], dim=2, keepdim=True) + EPS
                        tmp_msg = torch.bmm(L[:, :, :, (ii)] / denom, tmp_msg)
                    else:
                        pass
                elif self.msg_func_name == 'MLP':
                    state_in = state_old[:, (idx_col), :]
                    state_out = state_old[:, (idx_row), :]
                    tmp_msg = self.edge_func[ii](torch.cat([state_out, state_in], dim=2).view(batch_size * num_node * num_node, -1)).view(batch_size, num_node, num_node, -1)
                    if self.aggregate_type == 'sum':
                        tmp_msg = torch.matmul(tmp_msg.permute(0, 1, 3, 2), L[:, :, :, (ii)].unsqueeze(dim=3)).squeeze()
                    elif self.aggregate_type == 'avg':
                        denom = torch.sum(L[:, :, :, (ii)], dim=2, keepdim=True) + EPS
                        tmp_msg = torch.matmul(tmp_msg.permute(0, 1, 3, 2), L[:, :, :, (ii)].unsqueeze(dim=3)).squeeze()
                        tmp_msg = tmp_msg / denom
                    else:
                        pass
                msg += [tmp_msg]
            msg = torch.cat(msg, dim=2).view(batch_size * num_node, -1)
            state_old = state_old.view(batch_size * num_node, -1)
            state_new = self.update_func(msg, state_old).view(batch_size, num_node, -1)
            return state_new
        for tt in range(self.num_prop):
            state = _prop(state)
            state = F.dropout(state, self.dropout, training=self.training)
        y = []
        if mask is not None:
            for bb in range(batch_size):
                y += [self.att_func(state[(bb), (mask[bb]), :])]
        else:
            for bb in range(batch_size):
                y += [self.att_func(state[(bb), :, :])]
        score = self.output_func(torch.cat(y, dim=0))
        if label is not None:
            return score, self.loss_func(score, label)
        else:
            return score


class Set2Set(nn.Module):

    def __init__(self, element_dim, num_step_encoder):
        """ Implementation of Set2Set """
        super(Set2Set, self).__init__()
        self.element_dim = element_dim
        self.num_step_encoder = num_step_encoder
        self.LSTM_encoder = Set2SetLSTM(element_dim)
        self.LSTM_decoder = Set2SetLSTM(element_dim)
        self.W_1 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
        self.W_2 = nn.Parameter(torch.ones(self.element_dim, 1))
        self.W_3 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
        self.W_4 = nn.Parameter(torch.ones(self.element_dim, 1))
        self.W_5 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
        self.W_6 = nn.Parameter(torch.ones(self.element_dim, self.element_dim))
        self.W_7 = nn.Parameter(torch.ones(self.element_dim, 1))
        self.register_parameter('W_1', self.W_1)
        self.register_parameter('W_2', self.W_2)
        self.register_parameter('W_3', self.W_3)
        self.register_parameter('W_4', self.W_4)
        self.register_parameter('W_5', self.W_5)
        self.register_parameter('W_6', self.W_6)
        self.register_parameter('W_7', self.W_7)
        self._init_param()

    def _init_param(self):
        for xx in [self.W_1, self.W_2, self.W_3, self.W_4, self.W_5, self.W_6, self.W_7]:
            nn.init.xavier_uniform_(xx.data)

    def forward(self, input_set):
        """
      Args:
        input_set: shape N X D

      Returns:
        output_set: shape N X 1
    """
        num_element = input_set.shape[0]
        element_dim = input_set.shape[1]
        assert element_dim == self.element_dim
        hidden = torch.zeros(1, 2 * self.element_dim)
        memory = torch.zeros(1, self.element_dim)
        for tt in range(self.num_step_encoder):
            hidden, memory = self.LSTM_encoder(hidden, memory)
            energy = torch.tanh(torch.mm(hidden, self.W_1) + input_set).mm(self.W_2)
            att_weight = F.softmax(energy, dim=0)
            read = (input_set * att_weight).sum(dim=0, keepdim=True)
            hidden = torch.cat([hidden, read], dim=1)
        memory = torch.zeros_like(memory)
        output_set = []
        for tt in range(num_element):
            hidden, memory = self.LSTM_decoder(hidden, memory)
            energy = torch.tanh(torch.mm(hidden, self.W_3) + input_set).mm(self.W_4)
            att_weight = F.softmax(energy, dim=0)
            read = (input_set * att_weight).sum(dim=0, keepdim=True)
            hidden = torch.cat([hidden, read], dim=1)
            energy = torch.tanh(torch.mm(read, self.W_5) + torch.mm(input_set, self.W_6)).mm(self.W_7)
            output_set += [torch.argmax(energy)]
        return torch.stack(output_set)


class UnsortedSegmentSumFunction(Function):

    @staticmethod
    def forward(ctx, data, segment_index, num_segments):
        ctx.save_for_backward(data, segment_index)
        if not data.is_cuda:
            output = torch.FloatTensor(data.size(0), num_segments, data.size(2)).zero_()
            segment_reduction.unsorted_segment_sum_forward(data, segment_index, data.size(), output)
        else:
            output = torch.FloatTensor(data.size(0), num_segments, data.size(2)).zero_()
            segment_reduction.unsorted_segment_sum_forward_gpu(data, segment_index, data.size(), output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        data, segment_index = ctx.saved_tensors
        grad_data = data.new().resize_as_(data).zero_()
        if not data.is_cuda:
            segment_reduction.unsorted_segment_sum_backward(grad_output.data, segment_index, grad_data.size(), grad_data)
        else:
            segment_reduction.unsorted_segment_sum_backward_gpu(grad_output.data, segment_index, grad_data.size(), grad_data)
        return Variable(grad_data), None, None


class UnsortedSegmentSum(nn.Module):

    def __init__(self, num_segments):
        super(UnsortedSegmentSum, self).__init__()
        self.num_segments = num_segments

    def forward(self, data, segment_index):
        return UnsortedSegmentSumFunction.apply(data, segment_index, self.num_segments)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Set2Set,
     lambda: ([], {'element_dim': 4, 'num_step_encoder': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Set2SetLSTM,
     lambda: ([], {'hidden_dim': 4}),
     lambda: ([torch.rand([8, 8]), torch.rand([4, 4, 8, 4])], {}),
     True),
    (Set2Vec,
     lambda: ([], {'element_dim': 4, 'num_step_encoder': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
]

class Test_lrjconan_LanczosNetwork(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

