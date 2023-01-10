import sys
_module = sys.modules[__name__]
del sys
dataset = _module
humanact12poses = _module
uestc = _module
get_data = _module
quaternion = _module
skeleton = _module
data = _module
dataset = _module
motion_loaders = _module
comp_v6_model_dataset = _module
dataset_motion_loader = _module
model_motion_loaders = _module
networks = _module
evaluator_wrapper = _module
modules = _module
trainers = _module
motion_process = _module
get_opt = _module
metrics = _module
paramUtil = _module
plot_script = _module
utils = _module
word_vectorizer = _module
humanml_utils = _module
tensors = _module
fp16_util = _module
gaussian_diffusion = _module
logger = _module
losses = _module
nn = _module
resample = _module
respace = _module
a2m = _module
accuracy = _module
diversity = _module
evaluate = _module
fid = _module
models = _module
gru_eval = _module
stgcn = _module
graph = _module
tgcn = _module
accuracy = _module
diversity = _module
evaluate = _module
stgcn_eval = _module
tools = _module
eval_humanact12_uestc = _module
eval_humanml = _module
evaluate = _module
kid = _module
precision_recall = _module
stgcn = _module
cfg_sampler = _module
mdm = _module
rotation2xyz = _module
smpl = _module
edit = _module
generate = _module
train_mdm = _module
train_platforms = _module
training_loop = _module
config = _module
dist_util = _module
fixseed = _module
misc = _module
model_util = _module
parser_util = _module
rotation_conversions = _module
fit_seq = _module
customloss = _module
prior = _module
smplify = _module
render_mesh = _module
simplify_loc2rot = _module
vis_utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, matplotlib, numbers, numpy, pandas, queue, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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
xrange = range
wraps = functools.wraps


import random


import numpy as np


import torch


from torch.utils.data import DataLoader


import scipy.ndimage.filters as filters


from torch.utils import data


from torch.utils.data._utils.collate import default_collate


from torch.utils.data import Dataset


import torch.nn as nn


import time


import math


from torch.nn.utils.rnn import pack_padded_sequence


from torch.nn.utils.rnn import pad_packed_sequence


import torch.nn.functional as F


import torch.optim as optim


from torch.nn.utils import clip_grad_norm_


from collections import OrderedDict


import torch as th


from torch._utils import _flatten_dense_tensors


from torch._utils import _unflatten_dense_tensors


import enum


from copy import deepcopy


from abc import ABC


from abc import abstractmethod


import torch.distributed as dist


import copy


import functools


import re


import pandas as pd


from matplotlib import pyplot as plt


from sklearn.metrics.pairwise import polynomial_kernel


from types import SimpleNamespace


from torch.optim import AdamW


from typing import Optional


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MovementConvEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(nn.Conv1d(input_size, hidden_size, 4, 2, 1), nn.Dropout(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True), nn.Conv1d(hidden_size, output_size, 4, 2, 1), nn.Dropout(0.2, inplace=True), nn.LeakyReLU(0.2, inplace=True))
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class TextVAEDecoder(nn.Module):

    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextVAEDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True))
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)
        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_size, output_size))
        self.output.apply(init_weight)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs, last_pred, hidden, p):
        h_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).detach()
        h_in = h_in + pos_enc
        for i in range(self.n_layers):
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        return pose_pred, hidden


def reparameterize(mu, logvar):
    s_var = logvar.mul(0.5).exp_()
    eps = s_var.data.new(s_var.size()).normal_()
    return eps.mul(s_var).add_(mu)


class TextDecoder(nn.Module):

    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True))
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.positional_encoder = PositionalEncoding(hidden_size)
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        self.mu_net.apply(init_weight)
        self.logvar_net.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def forward(self, inputs, hidden, p):
        x_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).detach()
        x_in = x_in + pos_enc
        for i in range(self.n_layers):
            hidden[i] = self.gru[i](x_in, hidden[i])
            h_in = hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = reparameterize(mu, logvar)
        return z, mu, logvar, hidden


class AttLayer(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim
        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, query, key_mat):
        """
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        """
        query_vec = self.W_q(query).unsqueeze(-1)
        val_set = self.W_v(key_mat)
        key_set = self.W_k(key_mat)
        weights = torch.matmul(key_set, query_vec) / np.sqrt(self.dim)
        co_weights = self.softmax(weights)
        values = val_set * co_weights
        pred = values.sum(dim=1)
        return pred, co_weights

    def short_cut(self, querys, keys):
        return self.W_q(querys), self.W_k(keys)


class TextEncoderBiGRU(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, device):
        super(TextEncoderBiGRU, self).__init__()
        self.device = device
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        gru_seq = pad_packed_sequence(gru_seq, batch_first=True)[0]
        forward_seq = gru_seq[..., :self.hidden_size]
        backward_seq = gru_seq[..., self.hidden_size:].clone()
        for i, length in enumerate(cap_lens):
            backward_seq[i:i + 1, :length] = torch.flip(backward_seq[i:i + 1, :length].clone(), dims=[1])
        gru_seq = torch.cat([forward_seq, backward_seq], dim=-1)
        return gru_seq, gru_last


class TextEncoderBiGRUCo(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_size, output_size))
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2, inplace=True), nn.Linear(hidden_size, output_size))
        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output_net(gru_last)


class MotionLenEstimatorBiGRU(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super(MotionLenEstimatorBiGRU, self).__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        nd = 512
        self.output = nn.Sequential(nn.Linear(hidden_size * 2, nd), nn.LayerNorm(nd), nn.LeakyReLU(0.2, inplace=True), nn.Linear(nd, nd // 2), nn.LayerNorm(nd // 2), nn.LeakyReLU(0.2, inplace=True), nn.Linear(nd // 2, nd // 4), nn.LayerNorm(nd // 4), nn.LeakyReLU(0.2, inplace=True), nn.Linear(nd // 4, output_size))
        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]
        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)
        gru_seq, gru_last = self.gru(emb, hidden)
        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)
        return self.output(gru_last)


class SiLU(nn.Module):

    def forward(self, x):
        return x * th.sigmoid(x)


class GroupNorm32(nn.GroupNorm):

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class MotionDiscriminator(nn.Module):

    def __init__(self, input_size, hidden_size, hidden_layer, device, output_size=12, use_noise=None):
        super(MotionDiscriminator, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.use_noise = use_noise
        self.recurrent = nn.GRU(input_size, hidden_size, hidden_layer)
        self.linear1 = nn.Linear(hidden_size, 30)
        self.linear2 = nn.Linear(30, output_size)

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints * nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        out = gru_o[tuple(torch.stack((lengths - 1, torch.arange(bs, device=self.device))))]
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        lin2 = self.linear2(lin1)
        return lin2

    def initHidden(self, num_samples, layer):
        return torch.randn(layer, num_samples, self.hidden_size, device=self.device, requires_grad=False)


class MotionDiscriminatorForFID(MotionDiscriminator):

    def forward(self, motion_sequence, lengths=None, hidden_unit=None):
        bs, njoints, nfeats, num_frames = motion_sequence.shape
        motion_sequence = motion_sequence.reshape(bs, njoints * nfeats, num_frames)
        motion_sequence = motion_sequence.permute(2, 0, 1)
        if hidden_unit is None:
            hidden_unit = self.initHidden(motion_sequence.size(1), self.hidden_layer)
        gru_o, _ = self.recurrent(motion_sequence.float(), hidden_unit)
        out = gru_o[tuple(torch.stack((lengths - 1, torch.arange(bs, device=self.device))))]
        lin1 = self.linear1(out)
        lin1 = torch.tanh(lin1)
        return lin1


SMPL_DATA_PATH = './body_models/smpl'


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    AD = np.dot(A, Dn)
    return AD


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1), padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A


class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0, residual=True):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = (kernel_size[0] - 1) // 2, 0
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), (stride, 1), padding), nn.BatchNorm2d(out_channels), nn.Dropout(dropout, inplace=True))
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class STGCN(nn.Module):
    """Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args, edge_importance_weighting, device, **kwargs):
        super().__init__()
        self.device = device
        self.num_class = num_class
        self.losses = ['accuracy', 'cross_entropy', 'mixed']
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = temporal_kernel_size, spatial_kernel_size
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0), st_gcn(64, 64, kernel_size, 1, **kwargs), st_gcn(64, 64, kernel_size, 1, **kwargs), st_gcn(64, 128, kernel_size, 2, **kwargs), st_gcn(128, 128, kernel_size, 1, **kwargs), st_gcn(128, 256, kernel_size, 2, **kwargs)))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for i in self.st_gcn_networks])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, batch):
        x = batch['x'].permute(0, 2, 3, 1).unsqueeze(4).contiguous()
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        batch['features'] = x.squeeze()
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        batch['yhat'] = x
        return batch

    def compute_accuracy(self, batch):
        confusion = torch.zeros(self.num_class, self.num_class, dtype=int)
        yhat = batch['yhat'].max(dim=1).indices
        ygt = batch['y']
        for label, pred in zip(ygt, yhat):
            confusion[label][pred] += 1
        accuracy = torch.trace(confusion) / torch.sum(confusion)
        return accuracy

    def compute_loss(self, batch):
        cross_entropy = self.criterion(batch['yhat'], batch['y'])
        mixed_loss = cross_entropy
        acc = self.compute_accuracy(batch)
        losses = {'cross_entropy': cross_entropy.item(), 'mixed': mixed_loss.item(), 'accuracy': acc.item()}
        return mixed_loss, losses


class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'
        self.rot2xyz = self.model.rot2xyz
        self.translation = self.model.translation
        self.njoints = self.model.njoints
        self.nfeats = self.model.nfeats
        self.data_rep = self.model.data_rep
        self.cond_mode = self.model.cond_mode

    def forward(self, x, timesteps, y=None):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['text', 'action']
        y_uncond = deepcopy(y)
        y_uncond['uncond'] = True
        out = self.model(x, timesteps, y)
        out_uncond = self.model(x, timesteps, y_uncond)
        return out_uncond + y['scale'].view(-1, 1, 1, 1) * (out - out_uncond)


class EmbedAction(nn.Module):

    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0]
        output = self.action_embedding[idx]
        return output


class InputProcess(nn.Module):

    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]
            first_pose = self.poseEmbedding(first_pose)
            vel = x[1:]
            vel = self.velEmbedding(vel)
            return torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError


class OutputProcess(nn.Module):

    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]
            first_pose = self.poseFinal(first_pose)
            vel = output[1:]
            vel = self.velFinal(vel)
            output = torch.cat((first_pose, vel), axis=0)
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)
        return output


JOINTSTYPES = ['a2m', 'a2mpl', 'smpl', 'vibe', 'vertices']


JOINTSTYPE_ROOT = {'a2m': 0, 'smpl': 0, 'a2mpl': 0, 'vibe': 8}


JOINT_MAP = {'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17, 'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16, 'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0, 'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8, 'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7, 'OP REye': 25, 'OP LEye': 26, 'OP REar': 27, 'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30, 'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34, 'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45, 'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7, 'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17, 'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20, 'Neck (LSP)': 47, 'Top of Head (LSP)': 48, 'Pelvis (MPII)': 49, 'Thorax (MPII)': 50, 'Spine (H36M)': 51, 'Jaw (H36M)': 52, 'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26, 'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27}


JOINT_NAMES = ['OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow', 'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist', 'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle', 'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye', 'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe', 'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel', 'Right Ankle', 'Right Knee', 'Right Hip', 'Left Hip', 'Left Knee', 'Left Ankle', 'Right Wrist', 'Right Elbow', 'Right Shoulder', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Neck (LSP)', 'Top of Head (LSP)', 'Pelvis (MPII)', 'Thorax (MPII)', 'Spine (H36M)', 'Jaw (H36M)', 'Head (H36M)', 'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear']


action2motion_joints = [8, 1, 2, 3, 4, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 21, 24, 38]


class Rotation2xyz:

    def __init__(self, device, dataset='amass'):
        self.device = device
        self.dataset = dataset
        self.smpl_model = SMPL().eval()

    def __call__(self, x, mask, pose_rep, translation, glob, jointstype, vertstrans, betas=None, beta=0, glob_rot=None, get_rotations_back=False, **kwargs):
        if pose_rep == 'xyz':
            return x
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]), dtype=bool, device=x.device)
        if not glob and glob_rot is None:
            raise TypeError('You must specify global rotation if glob is False')
        if jointstype not in JOINTSTYPES:
            raise NotImplementedError('This jointstype is not implemented.')
        if translation:
            x_translations = x[:, -1, :3]
            x_rotations = x[:, :-1]
        else:
            x_rotations = x
        x_rotations = x_rotations.permute(0, 3, 1, 2)
        nsamples, time, njoints, feats = x_rotations.shape
        if pose_rep == 'rotvec':
            rotations = geometry.axis_angle_to_matrix(x_rotations[mask])
        elif pose_rep == 'rotmat':
            rotations = x_rotations[mask].view(-1, njoints, 3, 3)
        elif pose_rep == 'rotquat':
            rotations = geometry.quaternion_to_matrix(x_rotations[mask])
        elif pose_rep == 'rot6d':
            rotations = geometry.rotation_6d_to_matrix(x_rotations[mask])
        else:
            raise NotImplementedError('No geometry for this one.')
        if not glob:
            global_orient = torch.tensor(glob_rot, device=x.device)
            global_orient = geometry.axis_angle_to_matrix(global_orient).view(1, 1, 3, 3)
            global_orient = global_orient.repeat(len(rotations), 1, 1, 1)
        else:
            global_orient = rotations[:, 0]
            rotations = rotations[:, 1:]
        if betas is None:
            betas = torch.zeros([rotations.shape[0], self.smpl_model.num_betas], dtype=rotations.dtype, device=rotations.device)
            betas[:, 1] = beta
        out = self.smpl_model(body_pose=rotations, global_orient=global_orient, betas=betas)
        joints = out[jointstype]
        x_xyz = torch.empty(nsamples, time, joints.shape[1], 3, device=x.device, dtype=x.dtype)
        x_xyz[~mask] = 0
        x_xyz[mask] = joints
        x_xyz = x_xyz.permute(0, 2, 3, 1).contiguous()
        if jointstype != 'vertices':
            rootindex = JOINTSTYPE_ROOT[jointstype]
            x_xyz = x_xyz - x_xyz[:, [rootindex], :, :]
        if translation and vertstrans:
            x_translations = x_translations - x_translations[:, :, [0]]
            x_xyz = x_xyz + x_translations[:, None, :, :]
        if get_rotations_back:
            return x_xyz, rotations, global_orient
        else:
            return x_xyz


class TimestepEmbedder(nn.Module):

    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(nn.Linear(self.latent_dim, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class MDM(nn.Module):

    def __init__(self, modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, ablation=None, activation='gelu', legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512, arch='trans_enc', emb_trans_dec=False, clip_version=None, **kargs):
        super().__init__()
        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_actions = num_actions
        self.data_rep = data_rep
        self.dataset = dataset
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.action_emb = kargs.get('action_emb', None)
        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)
        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.0)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        if self.arch == 'trans_enc':
            None
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            None
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size, dropout=self.dropout, activation=activation)
            self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer, num_layers=self.num_layers)
        elif self.arch == 'gru':
            None
            self.gru = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        if self.cond_mode != 'no_cond':
            if 'text' in self.cond_mode:
                self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
                None
                None
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            if 'action' in self.cond_mode:
                self.embed_action = EmbedAction(self.num_actions, self.latent_dim)
                None
        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)
        self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.0:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)
            return cond * (1.0 - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2
            assert context_length < default_context_length
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True)
            zero_pad = torch.zeros([texts.shape[0], default_context_length - context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
        else:
            texts = clip.tokenize(raw_text, truncate=True)
        return self.clip_model.encode_text(texts).float()

    def forward(self, x, timesteps, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)
        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)
        if self.arch == 'gru':
            x_reshaped = x.reshape(bs, njoints * nfeats, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)
            emb_gru = emb_gru.permute(1, 2, 0)
            emb_gru = emb_gru.reshape(bs, self.latent_dim, 1, nframes)
            x = torch.cat((x_reshaped, emb_gru), axis=1)
        x = self.input_process(x)
        if self.arch == 'trans_enc':
            xseq = torch.cat((emb, x), axis=0)
            xseq = self.sequence_pos_encoder(xseq)
            output = self.seqTransEncoder(xseq)[1:]
        elif self.arch == 'trans_dec':
            if self.emb_trans_dec:
                xseq = torch.cat((emb, x), axis=0)
            else:
                xseq = x
            xseq = self.sequence_pos_encoder(xseq)
            if self.emb_trans_dec:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)[1:]
            else:
                output = self.seqTransDecoder(tgt=xseq, memory=emb)
        elif self.arch == 'gru':
            xseq = x
            xseq = self.sequence_pos_encoder(xseq)
            output, _ = self.gru(xseq)
        output = self.output_process(output)
        return output

    def _apply(self, fn):
        super()._apply(fn)
        self.rot2xyz.smpl_model._apply(fn)

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.rot2xyz.smpl_model.train(*args, **kwargs)


class SMPLifyAnglePrior(nn.Module):

    def __init__(self, dtype=torch.float32, **kwargs):
        super(SMPLifyAnglePrior, self).__init__()
        angle_prior_idxs = np.array([55, 58, 12, 15], dtype=np.int64)
        angle_prior_idxs = torch.tensor(angle_prior_idxs, dtype=torch.long)
        self.register_buffer('angle_prior_idxs', angle_prior_idxs)
        angle_prior_signs = np.array([1, -1, -1, -1], dtype=np.float32 if dtype == torch.float32 else np.float64)
        angle_prior_signs = torch.tensor(angle_prior_signs, dtype=dtype)
        self.register_buffer('angle_prior_signs', angle_prior_signs)

    def forward(self, pose, with_global_pose=False):
        """ Returns the angle prior loss for the given pose

        Args:
            pose: (Bx[23 + 1] * 3) torch tensor with the axis-angle
            representation of the rotations of the joints of the SMPL model.
        Kwargs:
            with_global_pose: Whether the pose vector also contains the global
            orientation of the SMPL model. If not then the indices must be
            corrected.
        Returns:
            A sze (B) tensor containing the angle prior loss for each element
            in the batch.
        """
        angle_prior_idxs = self.angle_prior_idxs - (not with_global_pose) * 3
        return torch.exp(pose[:, angle_prior_idxs] * self.angle_prior_signs).pow(2)


DEFAULT_DTYPE = torch.float32


class L2Prior(nn.Module):

    def __init__(self, dtype=DEFAULT_DTYPE, reduction='sum', **kwargs):
        super(L2Prior, self).__init__()

    def forward(self, module_input, *args):
        return torch.sum(module_input.pow(2))


class MaxMixturePrior(nn.Module):

    def __init__(self, prior_folder='prior', num_gaussians=6, dtype=DEFAULT_DTYPE, epsilon=1e-16, use_merged=True, **kwargs):
        super(MaxMixturePrior, self).__init__()
        if dtype == DEFAULT_DTYPE:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            None
            sys.exit(-1)
        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = 'gmm_{:02d}.pkl'.format(num_gaussians)
        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            None
            sys.exit(-1)
        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')
        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            None
            sys.exit(-1)
        self.register_buffer('means', torch.tensor(means, dtype=dtype))
        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))
        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)
        self.register_buffer('precisions', torch.tensor(precisions, dtype=dtype))
        sqrdets = np.array([np.sqrt(np.linalg.det(c)) for c in gmm['covars']])
        const = (2 * np.pi) ** (69 / 2.0)
        nll_weights = np.asarray(gmm['weights'] / (const * (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)
        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)
        self.register_buffer('pi_term', torch.log(torch.tensor(2 * np.pi, dtype=dtype)))
        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon) for cov in covs]
        self.register_buffer('cov_dets', torch.tensor(cov_dets, dtype=dtype))
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        """ Returns the mean of the mixture """
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose, betas):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means
        prec_diff_prod = torch.einsum('mij,bmj->bmi', [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)
        curr_loglikelihood = 0.5 * diff_prec_quadratic - torch.log(self.nll_weights)
        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose, betas, *args, **kwargs):
        """ Create graph operation for negative log-likelihood calculation
        """
        likelihoods = []
        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean
            curr_loglikelihood = torch.einsum('bj,ji->bi', [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b', [curr_loglikelihood, diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term + self.random_var_dim * self.pi_term)
            likelihoods.append(curr_loglikelihood)
        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)
        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose, betas):
        if self.use_merged:
            return self.merged_log_likelihood(pose, betas)
        else:
            return self.log_likelihood(pose, betas)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttLayer,
     lambda: ([], {'query_dim': 4, 'key_dim': 4, 'value_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (ContrastiveLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     True),
    (ConvTemporalGraphical,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (EmbedAction,
     lambda: ([], {'num_actions': 4, 'latent_dim': 4}),
     lambda: ([torch.ones([4, 4], dtype=torch.int64)], {}),
     True),
    (GroupNorm32,
     lambda: ([], {'num_groups': 1, 'num_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (L2Prior,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (MotionEncoderBiGRUCo,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (MotionLenEstimatorBiGRU,
     lambda: ([], {'word_size': 4, 'pos_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (MovementConvDecoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (MovementConvEncoder,
     lambda: ([], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     True),
    (PositionalEncoding,
     lambda: ([], {'d_model': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SiLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TextEncoderBiGRU,
     lambda: ([], {'word_size': 4, 'pos_size': 4, 'hidden_size': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
    (TextEncoderBiGRUCo,
     lambda: ([], {'word_size': 4, 'pos_size': 4, 'hidden_size': 4, 'output_size': 4, 'device': 0}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.ones([4], dtype=torch.int64)], {}),
     False),
]

class Test_GuyTevet_motion_diffusion_model(_paritybench_base):
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

    def test_007(self):
        self._check(*TESTCASES[7])

    def test_008(self):
        self._check(*TESTCASES[8])

    def test_009(self):
        self._check(*TESTCASES[9])

    def test_010(self):
        self._check(*TESTCASES[10])

    def test_011(self):
        self._check(*TESTCASES[11])

    def test_012(self):
        self._check(*TESTCASES[12])

    def test_013(self):
        self._check(*TESTCASES[13])

