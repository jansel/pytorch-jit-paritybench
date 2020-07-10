import sys
_module = sys.modules[__name__]
del sys
data_io = _module
filt = _module
gen_dct_mat = _module
gen_hamm_mat = _module
gen_splice = _module
make_blstm_proto = _module
make_cnn2d_proto = _module
make_cnn_proto = _module
make_lstm_proto = _module
make_nnet_proto = _module
reverse_arpa = _module
neural_networks = _module
run_TIMIT_fast = _module
run_TIMIT_full_decoding = _module
run_minichime5_fast = _module
utils = _module
waveminionet = _module
dataset = _module
losses = _module
models = _module
core = _module
decoders = _module
encoders = _module
frontend = _module
minions = _module
modules = _module
transforms = _module
utils = _module
master = _module
chime5_utils = _module
kaldi_data_dir = _module
prepare_openslr_rirs_cfg = _module
prepare_segmented_dataset_ami = _module
prepare_segmented_dataset_libri = _module
prepare_segmented_dataset_swbd = _module
unsupervised_data_cfg_ami = _module
unsupervised_data_cfg_librispeech = _module
unsupervised_data_cfg_vctk = _module
get_voxforge_lid_data = _module
prep_voxceleb = _module
prep_voxforge = _module
arff2npy = _module
neural_networks = _module
prepare_iemocap = _module
run_IEMOCAP_fast = _module
train = _module
make_trainset_statistics = _module
pase = _module
dataset = _module
log = _module
losses = _module
Minions = _module
cls_minions = _module
minions = _module
WorkerScheduler = _module
encoder = _module
lr_scheduler = _module
min_norm_solvers = _module
radam = _module
trainer = _module
worker_scheduler = _module
aspp = _module
attention_block = _module
classifiers = _module
core = _module
decoders = _module
discriminator = _module
encoders = _module
frontend = _module
modules = _module
neural_networks = _module
pase = _module
tdnn = _module
sbatch_writer = _module
dataset = _module
transforms = _module
utils = _module
precompute_aco_data = _module
setup = _module
knn = _module
make_fefeats_cfg = _module
mfcc_baseline = _module
neural_networks = _module
nnet = _module
run_minivox_fast = _module
select_supervised_ckpt = _module
utils = _module
train = _module
clusterize_frontend = _module
encode_codec2 = _module
eval_ckpts = _module
forward_chunk = _module
make_contaminated_trainset = _module
make_fbanks = _module
project_features = _module
prosodic_eval = _module
vadproc = _module

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
xrange = range
wraps = functools.wraps


import torch


import torch.nn.functional as F


import torch.nn as nn


import numpy as np


import math


import warnings


import torch.optim as optim


import random


import re


from torch.utils.data import Dataset


from collections import defaultdict


import torch.optim.lr_scheduler as lr_scheduler


from torch.autograd import Variable


from torch.nn.utils.spectral_norm import spectral_norm


from random import shuffle


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from torchvision.transforms import Compose


from torch.utils.data import ConcatDataset


import torchaudio


from torch.optim.optimizer import Optimizer


from torch.optim.optimizer import required


import torchvision.models as models


from torch.distributions import Binomial


from torch.nn.utils.weight_norm import weight_norm


from scipy import interpolate


from scipy import signal


from scipy.signal import decimate


from scipy.io import loadmat


from scipy.signal import lfilter


from scipy.signal import resample


from scipy.interpolate import interp1d


from torch.autograd import Function


from sklearn.cluster import KMeans


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-06):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def act_fun(act_type):
    if act_type == 'relu':
        return nn.ReLU()
    if act_type == 'tanh':
        return nn.Tanh()
    if act_type == 'sigmoid':
        return nn.Sigmoid()
    if act_type == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    if act_type == 'elu':
        return nn.ELU()
    if act_type == 'softmax':
        return nn.LogSoftmax(dim=1)
    if act_type == 'linear':
        return nn.LeakyReLU(1)


class MLP(nn.Module):

    def __init__(self, options, inp_dim):
        super(MLP, self).__init__()
        self.input_dim = inp_dim
        self.dnn_lay = list(map(int, options['dnn_lay'].split(',')))
        self.dnn_drop = list(map(float, options['dnn_drop'].split(',')))
        self.dnn_use_batchnorm = list(map(strtobool, options['dnn_use_batchnorm'].split(',')))
        self.dnn_use_laynorm = list(map(strtobool, options['dnn_use_laynorm'].split(',')))
        self.dnn_use_laynorm_inp = strtobool(options['dnn_use_laynorm_inp'])
        self.dnn_use_batchnorm_inp = strtobool(options['dnn_use_batchnorm_inp'])
        self.dnn_act = options['dnn_act'].split(',')
        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)
        self.N_dnn_lay = len(self.dnn_lay)
        current_input = self.input_dim
        for i in range(self.N_dnn_lay):
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]))
            self.act.append(act_fun(self.dnn_act[i]))
            add_bias = True
            self.ln.append(LayerNorm(self.dnn_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05))
            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias = False
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias))
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.dnn_lay[i], current_input).uniform_(-np.sqrt(0.01 / (current_input + self.dnn_lay[i])), np.sqrt(0.01 / (current_input + self.dnn_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))
            current_input = self.dnn_lay[i]
        self.out_dim = current_input

    def forward(self, x):
        if bool(self.dnn_use_laynorm_inp):
            x = self.ln0(x)
        if bool(self.dnn_use_batchnorm_inp):
            x = self.bn0(x)
        for i in range(self.N_dnn_lay):
            if self.dnn_use_laynorm[i] and not self.dnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
            if self.dnn_use_batchnorm[i] and not self.dnn_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))
            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.wx[i](x)))
        return x


class LSTM_cudnn(nn.Module):

    def __init__(self, options, inp_dim):
        super(LSTM_cudnn, self).__init__()
        self.input_dim = inp_dim
        self.hidden_size = int(options['hidden_size'])
        self.num_layers = int(options['num_layers'])
        self.bias = bool(strtobool(options['bias']))
        self.batch_first = bool(strtobool(options['batch_first']))
        self.dropout = float(options['dropout'])
        self.bidirectional = bool(strtobool(options['bidirectional']))
        self.lstm = nn.ModuleList([nn.LSTM(self.input_dim, self.hidden_size, self.num_layers, bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional)])
        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
            c0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
        if x.is_cuda:
            h0 = h0
            c0 = c0
        output, (hn, cn) = self.lstm[0](x, (h0, c0))
        return output


class GRU_cudnn(nn.Module):

    def __init__(self, options, inp_dim):
        super(GRU_cudnn, self).__init__()
        self.input_dim = inp_dim
        self.hidden_size = int(options['hidden_size'])
        self.num_layers = int(options['num_layers'])
        self.bias = bool(strtobool(options['bias']))
        self.batch_first = bool(strtobool(options['batch_first']))
        self.dropout = float(options['dropout'])
        self.bidirectional = bool(strtobool(options['bidirectional']))
        self.gru = nn.ModuleList([nn.GRU(self.input_dim, self.hidden_size, self.num_layers, bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional)])
        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
        if x.is_cuda:
            h0 = h0
        output, hn = self.gru[0](x, h0)
        return output


class RNN_cudnn(nn.Module):

    def __init__(self, options, inp_dim):
        super(RNN_cudnn, self).__init__()
        self.input_dim = inp_dim
        self.hidden_size = int(options['hidden_size'])
        self.num_layers = int(options['num_layers'])
        self.nonlinearity = options['nonlinearity']
        self.bias = bool(strtobool(options['bias']))
        self.batch_first = bool(strtobool(options['batch_first']))
        self.dropout = float(options['dropout'])
        self.bidirectional = bool(strtobool(options['bidirectional']))
        self.rnn = nn.ModuleList([nn.RNN(self.input_dim, self.hidden_size, self.num_layers, nonlinearity=self.nonlinearity, bias=self.bias, dropout=self.dropout, bidirectional=self.bidirectional)])
        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers * 2, x.shape[1], self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
        if x.is_cuda:
            h0 = h0
        output, hn = self.rnn[0](x, h0)
        return output


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, (getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long()), :]
    return x.view(xsize)


class LSTM(nn.Module):

    def __init__(self, options, inp_dim):
        super(LSTM, self).__init__()
        self.input_dim = inp_dim
        self.lstm_lay = list(map(int, options['lstm_lay'].split(',')))
        self.lstm_drop = list(map(float, options['lstm_drop'].split(',')))
        self.lstm_use_batchnorm = list(map(strtobool, options['lstm_use_batchnorm'].split(',')))
        self.lstm_use_laynorm = list(map(strtobool, options['lstm_use_laynorm'].split(',')))
        self.lstm_use_laynorm_inp = strtobool(options['lstm_use_laynorm_inp'])
        self.lstm_use_batchnorm_inp = strtobool(options['lstm_use_batchnorm_inp'])
        self.lstm_act = options['lstm_act'].split(',')
        self.lstm_orthinit = strtobool(options['lstm_orthinit'])
        self.bidir = strtobool(options['lstm_bidir'])
        self.use_cuda = strtobool(options['use_cuda'])
        self.to_do = options['to_do']
        if self.to_do == 'train':
            self.test_flag = False
        else:
            self.test_flag = True
        self.wfx = nn.ModuleList([])
        self.ufh = nn.ModuleList([])
        self.wix = nn.ModuleList([])
        self.uih = nn.ModuleList([])
        self.wox = nn.ModuleList([])
        self.uoh = nn.ModuleList([])
        self.wcx = nn.ModuleList([])
        self.uch = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.bn_wfx = nn.ModuleList([])
        self.bn_wix = nn.ModuleList([])
        self.bn_wox = nn.ModuleList([])
        self.bn_wcx = nn.ModuleList([])
        self.act = nn.ModuleList([])
        if self.lstm_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
        if self.lstm_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)
        self.N_lstm_lay = len(self.lstm_lay)
        current_input = self.input_dim
        for i in range(self.N_lstm_lay):
            self.act.append(act_fun(self.lstm_act[i]))
            add_bias = True
            if self.lstm_use_laynorm[i] or self.lstm_use_batchnorm[i]:
                add_bias = False
            self.wfx.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wix.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wox.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.wcx.append(nn.Linear(current_input, self.lstm_lay[i], bias=add_bias))
            self.ufh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uih.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uoh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            self.uch.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i], bias=False))
            if self.lstm_orthinit:
                nn.init.orthogonal_(self.ufh[i].weight)
                nn.init.orthogonal_(self.uih[i].weight)
                nn.init.orthogonal_(self.uoh[i].weight)
                nn.init.orthogonal_(self.uch[i].weight)
            self.bn_wfx.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wix.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wox.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.bn_wcx.append(nn.BatchNorm1d(self.lstm_lay[i], momentum=0.05))
            self.ln.append(LayerNorm(self.lstm_lay[i]))
            if self.bidir:
                current_input = 2 * self.lstm_lay[i]
            else:
                current_input = self.lstm_lay[i]
        self.out_dim = self.lstm_lay[i] + self.bidir * self.lstm_lay[i]

    def forward(self, x):
        if bool(self.lstm_use_laynorm_inp):
            x = self.ln0(x)
        if bool(self.lstm_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
        for i in range(self.N_lstm_lay):
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.lstm_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.lstm_lay[i])
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.lstm_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.lstm_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wfx_out = self.wfx[i](x)
            wix_out = self.wix[i](x)
            wox_out = self.wox[i](x)
            wcx_out = self.wcx[i](x)
            if self.lstm_use_batchnorm[i]:
                wfx_out_bn = self.bn_wfx[i](wfx_out.view(wfx_out.shape[0] * wfx_out.shape[1], wfx_out.shape[2]))
                wfx_out = wfx_out_bn.view(wfx_out.shape[0], wfx_out.shape[1], wfx_out.shape[2])
                wix_out_bn = self.bn_wix[i](wix_out.view(wix_out.shape[0] * wix_out.shape[1], wix_out.shape[2]))
                wix_out = wix_out_bn.view(wix_out.shape[0], wix_out.shape[1], wix_out.shape[2])
                wox_out_bn = self.bn_wox[i](wox_out.view(wox_out.shape[0] * wox_out.shape[1], wox_out.shape[2]))
                wox_out = wox_out_bn.view(wox_out.shape[0], wox_out.shape[1], wox_out.shape[2])
                wcx_out_bn = self.bn_wcx[i](wcx_out.view(wcx_out.shape[0] * wcx_out.shape[1], wcx_out.shape[2]))
                wcx_out = wcx_out_bn.view(wcx_out.shape[0], wcx_out.shape[1], wcx_out.shape[2])
            hiddens = []
            ct = h_init
            ht = h_init
            for k in range(x.shape[0]):
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                ct = it * self.act[i](wcx_out[k] + self.uch[i](ht)) * drop_mask + ft * ct
                ht = ot * self.act[i](ct)
                if self.lstm_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class GRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(GRU, self).__init__()
        self.input_dim = inp_dim
        self.gru_lay = list(map(int, options['gru_lay'].split(',')))
        self.gru_drop = list(map(float, options['gru_drop'].split(',')))
        self.gru_use_batchnorm = list(map(strtobool, options['gru_use_batchnorm'].split(',')))
        self.gru_use_laynorm = list(map(strtobool, options['gru_use_laynorm'].split(',')))
        self.gru_use_laynorm_inp = strtobool(options['gru_use_laynorm_inp'])
        self.gru_use_batchnorm_inp = strtobool(options['gru_use_batchnorm_inp'])
        self.gru_orthinit = strtobool(options['gru_orthinit'])
        self.gru_act = options['gru_act'].split(',')
        self.bidir = strtobool(options['gru_bidir'])
        self.use_cuda = strtobool(options['use_cuda'])
        self.to_do = options['to_do']
        if self.to_do == 'train':
            self.test_flag = False
        else:
            self.test_flag = True
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])
        self.wz = nn.ModuleList([])
        self.uz = nn.ModuleList([])
        self.wr = nn.ModuleList([])
        self.ur = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.bn_wh = nn.ModuleList([])
        self.bn_wz = nn.ModuleList([])
        self.bn_wr = nn.ModuleList([])
        self.act = nn.ModuleList([])
        if self.gru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
        if self.gru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)
        self.N_gru_lay = len(self.gru_lay)
        current_input = self.input_dim
        for i in range(self.N_gru_lay):
            self.act.append(act_fun(self.gru_act[i]))
            add_bias = True
            if self.gru_use_laynorm[i] or self.gru_use_batchnorm[i]:
                add_bias = False
            self.wh.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.wr.append(nn.Linear(current_input, self.gru_lay[i], bias=add_bias))
            self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias=False))
            if self.gru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
                nn.init.orthogonal_(self.ur[i].weight)
            self.bn_wh.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.bn_wr.append(nn.BatchNorm1d(self.gru_lay[i], momentum=0.05))
            self.ln.append(LayerNorm(self.gru_lay[i]))
            if self.bidir:
                current_input = 2 * self.gru_lay[i]
            else:
                current_input = self.gru_lay[i]
        self.out_dim = self.gru_lay[i] + self.bidir * self.gru_lay[i]

    def forward(self, x):
        if bool(self.gru_use_laynorm_inp):
            x = self.ln0(x)
        if bool(self.gru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
        for i in range(self.N_gru_lay):
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.gru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.gru_lay[i])
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.gru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.gru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            wr_out = self.wr[i](x)
            if self.gru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])
                wr_out_bn = self.bn_wr[i](wr_out.view(wr_out.shape[0] * wr_out.shape[1], wr_out.shape[2]))
                wr_out = wr_out_bn.view(wr_out.shape[0], wr_out.shape[1], wr_out.shape[2])
            hiddens = []
            ht = h_init
            for k in range(x.shape[0]):
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                rt = torch.sigmoid(wr_out[k] + self.ur[i](ht))
                at = wh_out[k] + self.uh[i](rt * ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand
                if self.gru_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class liGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(liGRU, self).__init__()
        self.input_dim = inp_dim
        self.ligru_lay = list(map(int, options['ligru_lay'].split(',')))
        self.ligru_drop = list(map(float, options['ligru_drop'].split(',')))
        self.ligru_use_batchnorm = list(map(strtobool, options['ligru_use_batchnorm'].split(',')))
        self.ligru_use_laynorm = list(map(strtobool, options['ligru_use_laynorm'].split(',')))
        self.ligru_use_laynorm_inp = strtobool(options['ligru_use_laynorm_inp'])
        self.ligru_use_batchnorm_inp = strtobool(options['ligru_use_batchnorm_inp'])
        self.ligru_orthinit = strtobool(options['ligru_orthinit'])
        self.ligru_act = options['ligru_act'].split(',')
        self.bidir = strtobool(options['ligru_bidir'])
        self.use_cuda = strtobool(options['use_cuda'])
        self.to_do = options['to_do']
        if self.to_do == 'train':
            self.test_flag = False
        else:
            self.test_flag = True
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])
        self.wz = nn.ModuleList([])
        self.uz = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.bn_wh = nn.ModuleList([])
        self.bn_wz = nn.ModuleList([])
        self.act = nn.ModuleList([])
        if self.ligru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
        if self.ligru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)
        self.N_ligru_lay = len(self.ligru_lay)
        current_input = self.input_dim
        for i in range(self.N_ligru_lay):
            self.act.append(act_fun(self.ligru_act[i]))
            add_bias = True
            if self.ligru_use_laynorm[i] or self.ligru_use_batchnorm[i]:
                add_bias = False
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias=add_bias))
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i], bias=False))
            if self.ligru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
            self.bn_wh.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.ligru_lay[i], momentum=0.05))
            self.ln.append(LayerNorm(self.ligru_lay[i]))
            if self.bidir:
                current_input = 2 * self.ligru_lay[i]
            else:
                current_input = self.ligru_lay[i]
        self.out_dim = self.ligru_lay[i] + self.bidir * self.ligru_lay[i]

    def forward(self, x):
        if bool(self.ligru_use_laynorm_inp):
            x = self.ln0(x)
        if bool(self.ligru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
        for i in range(self.N_ligru_lay):
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.ligru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.ligru_lay[i])
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.ligru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.ligru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])
            hiddens = []
            ht = h_init
            for k in range(x.shape[0]):
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand
                if self.ligru_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class minimalGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(minimalGRU, self).__init__()
        self.input_dim = inp_dim
        self.minimalgru_lay = list(map(int, options['minimalgru_lay'].split(',')))
        self.minimalgru_drop = list(map(float, options['minimalgru_drop'].split(',')))
        self.minimalgru_use_batchnorm = list(map(strtobool, options['minimalgru_use_batchnorm'].split(',')))
        self.minimalgru_use_laynorm = list(map(strtobool, options['minimalgru_use_laynorm'].split(',')))
        self.minimalgru_use_laynorm_inp = strtobool(options['minimalgru_use_laynorm_inp'])
        self.minimalgru_use_batchnorm_inp = strtobool(options['minimalgru_use_batchnorm_inp'])
        self.minimalgru_orthinit = strtobool(options['minimalgru_orthinit'])
        self.minimalgru_act = options['minimalgru_act'].split(',')
        self.bidir = strtobool(options['minimalgru_bidir'])
        self.use_cuda = strtobool(options['use_cuda'])
        self.to_do = options['to_do']
        if self.to_do == 'train':
            self.test_flag = False
        else:
            self.test_flag = True
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])
        self.wz = nn.ModuleList([])
        self.uz = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.bn_wh = nn.ModuleList([])
        self.bn_wz = nn.ModuleList([])
        self.act = nn.ModuleList([])
        if self.minimalgru_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
        if self.minimalgru_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)
        self.N_minimalgru_lay = len(self.minimalgru_lay)
        current_input = self.input_dim
        for i in range(self.N_minimalgru_lay):
            self.act.append(act_fun(self.minimalgru_act[i]))
            add_bias = True
            if self.minimalgru_use_laynorm[i] or self.minimalgru_use_batchnorm[i]:
                add_bias = False
            self.wh.append(nn.Linear(current_input, self.minimalgru_lay[i], bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.minimalgru_lay[i], bias=add_bias))
            self.uh.append(nn.Linear(self.minimalgru_lay[i], self.minimalgru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.minimalgru_lay[i], self.minimalgru_lay[i], bias=False))
            if self.minimalgru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
            self.bn_wh.append(nn.BatchNorm1d(self.minimalgru_lay[i], momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.minimalgru_lay[i], momentum=0.05))
            self.ln.append(LayerNorm(self.minimalgru_lay[i]))
            if self.bidir:
                current_input = 2 * self.minimalgru_lay[i]
            else:
                current_input = self.minimalgru_lay[i]
        self.out_dim = self.minimalgru_lay[i] + self.bidir * self.minimalgru_lay[i]

    def forward(self, x):
        if bool(self.minimalgru_use_laynorm_inp):
            x = self.ln0(x)
        if bool(self.minimalgru_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
        for i in range(self.N_minimalgru_lay):
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.minimalgru_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.minimalgru_lay[i])
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.minimalgru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.minimalgru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.minimalgru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] * wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1], wz_out.shape[2])
            hiddens = []
            ht = h_init
            for k in range(x.shape[0]):
                zt = torch.sigmoid(wz_out[k] + self.uz[i](ht))
                at = wh_out[k] + self.uh[i](zt * ht)
                hcand = self.act[i](at) * drop_mask
                ht = zt * ht + (1 - zt) * hcand
                if self.minimalgru_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class RNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(RNN, self).__init__()
        self.input_dim = inp_dim
        self.rnn_lay = list(map(int, options['rnn_lay'].split(',')))
        self.rnn_drop = list(map(float, options['rnn_drop'].split(',')))
        self.rnn_use_batchnorm = list(map(strtobool, options['rnn_use_batchnorm'].split(',')))
        self.rnn_use_laynorm = list(map(strtobool, options['rnn_use_laynorm'].split(',')))
        self.rnn_use_laynorm_inp = strtobool(options['rnn_use_laynorm_inp'])
        self.rnn_use_batchnorm_inp = strtobool(options['rnn_use_batchnorm_inp'])
        self.rnn_orthinit = strtobool(options['rnn_orthinit'])
        self.rnn_act = options['rnn_act'].split(',')
        self.bidir = strtobool(options['rnn_bidir'])
        self.use_cuda = strtobool(options['use_cuda'])
        self.to_do = options['to_do']
        if self.to_do == 'train':
            self.test_flag = False
        else:
            self.test_flag = True
        self.wh = nn.ModuleList([])
        self.uh = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.bn_wh = nn.ModuleList([])
        self.act = nn.ModuleList([])
        if self.rnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
        if self.rnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05)
        self.N_rnn_lay = len(self.rnn_lay)
        current_input = self.input_dim
        for i in range(self.N_rnn_lay):
            self.act.append(act_fun(self.rnn_act[i]))
            add_bias = True
            if self.rnn_use_laynorm[i] or self.rnn_use_batchnorm[i]:
                add_bias = False
            self.wh.append(nn.Linear(current_input, self.rnn_lay[i], bias=add_bias))
            self.uh.append(nn.Linear(self.rnn_lay[i], self.rnn_lay[i], bias=False))
            if self.rnn_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
            self.bn_wh.append(nn.BatchNorm1d(self.rnn_lay[i], momentum=0.05))
            self.ln.append(LayerNorm(self.rnn_lay[i]))
            if self.bidir:
                current_input = 2 * self.rnn_lay[i]
            else:
                current_input = self.rnn_lay[i]
        self.out_dim = self.rnn_lay[i] + self.bidir * self.rnn_lay[i]

    def forward(self, x):
        if bool(self.rnn_use_laynorm_inp):
            x = self.ln0(x)
        if bool(self.rnn_use_batchnorm_inp):
            x_bn = self.bn0(x.view(x.shape[0] * x.shape[1], x.shape[2]))
            x = x_bn.view(x.shape[0], x.shape[1], x.shape[2])
        for i in range(self.N_rnn_lay):
            if self.bidir:
                h_init = torch.zeros(2 * x.shape[1], self.rnn_lay[i])
                x = torch.cat([x, flip(x, 0)], 1)
            else:
                h_init = torch.zeros(x.shape[1], self.rnn_lay[i])
            if self.test_flag == False:
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0], h_init.shape[1]).fill_(1 - self.rnn_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.rnn_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            if self.rnn_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] * wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1], wh_out.shape[2])
            hiddens = []
            ht = h_init
            for k in range(x.shape[0]):
                at = wh_out[k] + self.uh[i](ht)
                ht = self.act[i](at) * drop_mask
                if self.rnn_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0)
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class CNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options['cnn_N_filt'].split(',')))
        self.cnn_len_filt = list(map(int, options['cnn_len_filt'].split(',')))
        self.cnn_max_pool_len = list(map(int, options['cnn_max_pool_len'].split(',')))
        self.cnn_act = options['cnn_act'].split(',')
        self.cnn_drop = list(map(float, options['cnn_drop'].split(',')))
        self.cnn_use_laynorm = list(map(strtobool, options['cnn_use_laynorm'].split(',')))
        self.cnn_use_batchnorm = list(map(strtobool, options['cnn_use_batchnorm'].split(',')))
        self.cnn_use_laynorm_inp = strtobool(options['cnn_use_laynorm_inp'])
        self.cnn_use_batchnorm_inp = strtobool(options['cnn_use_batchnorm_inp'])
        self.N_cnn_lay = len(self.cnn_N_filt)
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
        if self.cnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
        if self.cnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)
        current_input = self.input_dim
        for i in range(self.N_cnn_lay):
            N_filt = int(self.cnn_N_filt[i])
            len_filt = int(self.cnn_len_filt[i])
            self.drop.append(nn.Dropout(p=self.cnn_drop[i]))
            self.act.append(act_fun(self.cnn_act[i]))
            self.ln.append(LayerNorm([N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]), momentum=0.05))
            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))
            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.cnn_N_filt[i], self.cnn_len_filt[i]))
            current_input = int((current_input - self.cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])
        self.out_dim = current_input * N_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]
        if bool(self.cnn_use_laynorm_inp):
            x = self.ln0(x)
        if bool(self.cnn_use_batchnorm_inp):
            x = self.bn0(x)
        x = x.view(batch, 1, seq_len)
        for i in range(self.N_cnn_lay):
            if self.cnn_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.cnn_max_pool_len[i])))
        x = x.view(batch, -1)
        return x


class SincConv(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, groups=1, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()
        if in_channels != 1:
            msg = 'SincConv only support one input channel (here, in_channels = {%i})' % in_channels
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n + 1).view(1, -1) / self.sample_rate

    def sinc(self, x):
        x_left = x[:, 0:int((x.shape[1] - 1) / 2)]
        y_left = torch.sin(x_left) / x_left
        y_right = torch.flip(y_left, dims=[1])
        sinc = torch.cat([y_left, torch.ones([x.shape[0], 1]), y_right], dim=1)
        return sinc

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        self.n_ = self.n_
        self.window_ = self.window_
        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz / self.sample_rate + torch.abs(self.band_hz_)
        f_times_t = torch.matmul(low, self.n_)
        low_pass1 = 2 * low * self.sinc(2 * math.pi * f_times_t * self.sample_rate)
        f_times_t = torch.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(2 * math.pi * f_times_t * self.sample_rate)
        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_
        self.filters = (band_pass * self.window_).view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None, groups=1)


class SincNet(nn.Module):

    def __init__(self, options, inp_dim):
        super(SincNet, self).__init__()
        self.input_dim = inp_dim
        self.sinc_N_filt = list(map(int, options['sinc_N_filt'].split(',')))
        self.sinc_len_filt = list(map(int, options['sinc_len_filt'].split(',')))
        self.sinc_max_pool_len = list(map(int, options['sinc_max_pool_len'].split(',')))
        self.sinc_act = options['sinc_act'].split(',')
        self.sinc_drop = list(map(float, options['sinc_drop'].split(',')))
        self.sinc_use_laynorm = list(map(strtobool, options['sinc_use_laynorm'].split(',')))
        self.sinc_use_batchnorm = list(map(strtobool, options['sinc_use_batchnorm'].split(',')))
        self.sinc_use_laynorm_inp = strtobool(options['sinc_use_laynorm_inp'])
        self.sinc_use_batchnorm_inp = strtobool(options['sinc_use_batchnorm_inp'])
        self.N_sinc_lay = len(self.sinc_N_filt)
        self.sinc_sample_rate = int(options['sinc_sample_rate'])
        self.sinc_min_low_hz = int(options['sinc_min_low_hz'])
        self.sinc_min_band_hz = int(options['sinc_min_band_hz'])
        self.conv = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
        if self.sinc_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim)
        if self.sinc_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d([self.input_dim], momentum=0.05)
        current_input = self.input_dim
        for i in range(self.N_sinc_lay):
            N_filt = int(self.sinc_N_filt[i])
            len_filt = int(self.sinc_len_filt[i])
            self.drop.append(nn.Dropout(p=self.sinc_drop[i]))
            self.act.append(act_fun(self.sinc_act[i]))
            self.ln.append(LayerNorm([N_filt, int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i]), momentum=0.05))
            if i == 0:
                self.conv.append(SincConv(1, N_filt, len_filt, sample_rate=self.sinc_sample_rate, min_low_hz=self.sinc_min_low_hz, min_band_hz=self.sinc_min_band_hz))
            else:
                self.conv.append(nn.Conv1d(self.sinc_N_filt[i - 1], self.sinc_N_filt[i], self.sinc_len_filt[i]))
            current_input = int((current_input - self.sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])
        self.out_dim = current_input * N_filt

    def forward(self, x):
        batch = x.shape[0]
        seq_len = x.shape[1]
        if bool(self.sinc_use_laynorm_inp):
            x = self.ln0(x)
        if bool(self.sinc_use_batchnorm_inp):
            x = self.bn0(x)
        x = x.view(batch, 1, seq_len)
        for i in range(self.N_sinc_lay):
            if self.sinc_use_laynorm[i]:
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i] == False and self.sinc_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x), self.sinc_max_pool_len[i])))
        x = x.view(batch, -1)
        return x


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False, groups=1, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()
        if in_channels != 1:
            msg = 'SincConv only support one input channel (here, in_channels = {%i})' % in_channels
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size / 2 - 1, steps=int(self.kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        self.n_ = self.n_
        self.window_ = self.window_
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, (0)]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, (None)])
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None, groups=1)


class SRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(SRU, self).__init__()
        self.input_dim = inp_dim
        self.hidden_size = int(options['sru_hidden_size'])
        self.num_layers = int(options['sru_num_layers'])
        self.dropout = float(options['sru_dropout'])
        self.rnn_dropout = float(options['sru_rnn_dropout'])
        self.use_tanh = bool(strtobool(options['sru_use_tanh']))
        self.use_relu = bool(strtobool(options['sru_use_relu']))
        self.use_selu = bool(strtobool(options['sru_use_selu']))
        self.weight_norm = bool(strtobool(options['sru_weight_norm']))
        self.layer_norm = bool(strtobool(options['sru_layer_norm']))
        self.bidirectional = bool(strtobool(options['sru_bidirectional']))
        self.is_input_normalized = bool(strtobool(options['sru_is_input_normalized']))
        self.has_skip_term = bool(strtobool(options['sru_has_skip_term']))
        self.rescale = bool(strtobool(options['sru_rescale']))
        self.highway_bias = float(options['sru_highway_bias'])
        self.n_proj = int(options['sru_n_proj'])
        self.sru = sru.SRU(self.input_dim, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout, rnn_dropout=self.rnn_dropout, bidirectional=self.bidirectional, n_proj=self.n_proj, use_tanh=self.use_tanh, use_selu=self.use_selu, use_relu=self.use_relu, weight_norm=self.weight_norm, layer_norm=self.layer_norm, has_skip_term=self.has_skip_term, is_input_normalized=self.is_input_normalized, highway_bias=self.highway_bias, rescale=self.rescale)
        self.out_dim = self.hidden_size + self.bidirectional * self.hidden_size

    def forward(self, x):
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size * 2)
        else:
            h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_size)
        if x.is_cuda:
            h0 = h0
        output, hn = self.sru(x, c0=h0)
        return output


class SpectrumLM(nn.Module):
    """ RNN lang model for spectrum frame preds """

    def __init__(self, rnn_size, rnn_layers, out_dim, dropout, cuda, rnn_type='LSTM', bidirectional=False):
        super().__init__()
        self.do_cuda = cuda
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.out_dim = out_dim
        self.dropout = dropout
        self.bidirectional = bidirectional
        if bidirectional:
            self.dirs = 2
        else:
            self.dirs = 1
        assert rnn_type == 'LSTM' or rnn_type == 'GRU', rnn_type
        self.rnn = getattr(nn, rnn_type)(self.out_dim, self.rnn_size, self.rnn_layers, batch_first=True, dropout=self.dropout, bidirectional=bidirectional)
        self.out_fc = nn.Linear(self.rnn_size, self.out_dim)

    def forward(self, x, dec_steps, state=None, dec_cps={}):
        assert len(x.size()) == 2, x.size()
        if state is None:
            state = self.init_hidden(x.size(0))
        assert isinstance(dec_cps, dict), type(dec_cps)
        x = x.unsqueeze(1)
        ht = x
        frames = []
        for t in range(dec_steps):
            if t in dec_cps:
                ht = dec_cps[t]
                if len(ht.size()) == 2:
                    ht = ht.unsqueeze(1)
            ht, state = self.rnn(ht, state)
            ht = self.out_fc(ht)
            frames.append(ht)
        frames = torch.cat(frames, 1)
        return frames, state

    def init_hidden(self, bsz):
        h0 = Variable(torch.randn(self.dirs * self.rnn_layers, bsz, self.rnn_size))
        if self.do_cuda:
            h0 = h0
        if self.rnn_type == 'LSTM':
            c0 = h0.clone()
            return h0, c0
        else:
            return h0


class AhoCNNEncoder(nn.Module):

    def __init__(self, input_dim, kwidth=3, dropout=0.5, layer_norm=False):
        super().__init__()
        pad = (kwidth - 1) // 2
        if layer_norm:
            norm_layer = LayerNorm
        else:
            norm_layer = nn.BatchNorm1d
        self.enc = nn.Sequential(nn.Conv1d(input_dim, 256, kwidth, stride=1, padding=pad), norm_layer(256), nn.PReLU(256), nn.Conv1d(256, 256, kwidth, stride=1, padding=pad), norm_layer(256), nn.PReLU(256), nn.MaxPool1d(2), nn.Dropout(0.2), nn.Conv1d(256, 512, kwidth, stride=1, padding=pad), norm_layer(512), nn.PReLU(512), nn.Conv1d(512, 512, kwidth, stride=1, padding=pad), norm_layer(512), nn.PReLU(512), nn.MaxPool1d(2), nn.Dropout(0.2), nn.Conv1d(512, 1024, kwidth, stride=1, padding=pad), norm_layer(1024), nn.PReLU(1024), nn.Conv1d(1024, 1024, kwidth, stride=1, padding=pad), norm_layer(1024), nn.PReLU(1024), nn.MaxPool1d(2), nn.Dropout(0.2), nn.Conv1d(1024, 1024, kwidth, stride=1, padding=pad))

    def forward(self, x):
        return self.enc(x)


class AhoCNNHourGlassEncoder(nn.Module):

    def __init__(self, input_dim, kwidth=3, dropout=0.5, layer_norm=False):
        super().__init__()
        pad = (kwidth - 1) // 2
        if layer_norm:
            norm_layer = LayerNorm
        else:
            norm_layer = nn.BatchNorm1d
        self.enc = nn.Sequential(nn.Conv1d(input_dim, 64, kwidth, stride=1, padding=pad), norm_layer(64), nn.PReLU(64), nn.Conv1d(64, 128, kwidth, stride=1, padding=pad), norm_layer(128), nn.PReLU(128), nn.MaxPool1d(2), nn.Dropout(dropout), nn.Conv1d(128, 256, kwidth, stride=1, padding=pad), norm_layer(256), nn.PReLU(256), nn.Conv1d(256, 512, kwidth, stride=1, padding=pad), norm_layer(512), nn.PReLU(512), nn.MaxPool1d(2), nn.Dropout(dropout), nn.Conv1d(512, 256, kwidth, stride=1, padding=pad), norm_layer(256), nn.PReLU(256), nn.Conv1d(256, 128, kwidth, stride=1, padding=pad), norm_layer(128), nn.PReLU(128), nn.MaxPool1d(2), nn.Dropout(dropout), nn.Conv1d(128, 64, kwidth, stride=1, padding=pad), norm_layer(64), nn.PReLU(64))

    def forward(self, x):
        return self.enc(x)


class NeuralBlock(nn.Module):

    def __init__(self, name='NeuralBlock'):
        super().__init__()
        self.name = name

    def describe_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        None
        None
        None
        None
        return pp


class Saver(object):

    def __init__(self, model, save_path, max_ckpts=5, optimizer=None, prefix=''):
        self.model = model
        self.save_path = save_path
        self.ckpt_path = os.path.join(save_path, '{}checkpoints'.format(prefix))
        self.max_ckpts = max_ckpts
        self.optimizer = optimizer
        self.prefix = prefix

    def save(self, model_name, step, best_val=False):
        save_path = self.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ckpt_path = self.ckpt_path
        if os.path.exists(ckpt_path):
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
        else:
            ckpts = {'latest': [], 'current': []}
        model_path = '{}-{}.ckpt'.format(model_name, step)
        if best_val:
            model_path = 'best_' + model_path
        model_path = '{}{}'.format(self.prefix, model_path)
        latest = ckpts['latest']
        if len(latest) > 0:
            todel = latest[0]
            if self.max_ckpts is not None:
                if len(latest) > self.max_ckpts:
                    try:
                        None
                        os.remove(os.path.join(save_path, 'weights_' + todel))
                        latest = latest[1:]
                    except FileNotFoundError:
                        None
        latest += [model_path]
        ckpts['latest'] = latest
        ckpts['current'] = model_path
        with open(ckpt_path, 'w') as ckpt_f:
            ckpt_f.write(json.dumps(ckpts, indent=2))
        st_dict = {'step': step, 'state_dict': self.model.state_dict()}
        if self.optimizer is not None:
            st_dict['optimizer'] = self.optimizer.state_dict()
        torch.save(st_dict, os.path.join(save_path, 'weights_' + model_path))

    def read_latest_checkpoint(self):
        ckpt_path = self.ckpt_path
        None
        if not os.path.exists(ckpt_path):
            None
            return None
        else:
            with open(ckpt_path, 'r') as ckpt_f:
                ckpts = json.load(ckpt_f)
            curr_ckpt = ckpts['current']
            return curr_ckpt

    def load_weights(self):
        save_path = self.save_path
        curr_ckpt = self.read_latest_checkpoint()
        if curr_ckpt is None:
            None
            return False
        else:
            st_dict = torch.load(os.path.join(save_path, 'weights_' + curr_ckpt))
            if 'state_dict' in st_dict:
                model_state = st_dict['state_dict']
                self.model.load_state_dict(model_state)
                if self.optimizer is not None and 'optimizer' in st_dict:
                    self.optimizer.load_state_dict(st_dict['optimizer'])
            else:
                self.model.load_state_dict(st_dict)
            None
            return True

    def load_ckpt_step(self, curr_ckpt):
        ckpt = torch.load(os.path.join(self.save_path, 'weights_' + curr_ckpt), map_location='cpu')
        step = ckpt['step']
        return step

    def load_pretrained_ckpt(self, ckpt_file, load_last=False, load_opt=True, verbose=True):
        model_dict = self.model.state_dict()
        st_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        if 'state_dict' in st_dict:
            pt_dict = st_dict['state_dict']
        else:
            pt_dict = st_dict
        all_pt_keys = list(pt_dict.keys())
        if not load_last:
            allowed_keys = all_pt_keys[:-2]
        else:
            allowed_keys = all_pt_keys[:]
        pt_dict = {k: v for k, v in pt_dict.items() if k in model_dict and k in allowed_keys and v.size() == model_dict[k].size()}
        if verbose:
            None
            None
            None
        if len(pt_dict.keys()) != len(model_dict.keys()):
            raise ValueError('WARNING: LOADING DIFFERENT NUM OF KEYS')
            None
        model_dict.update(pt_dict)
        self.model.load_state_dict(model_dict)
        for k in model_dict.keys():
            if k not in allowed_keys:
                None
        if self.optimizer is not None and 'optimizer' in st_dict and load_opt:
            self.optimizer.load_state_dict(st_dict['optimizer'])


class Model(NeuralBlock):

    def __init__(self, max_ckpts=5, name='BaseModel'):
        super().__init__()
        self.name = name
        self.optim = None
        self.max_ckpts = max_ckpts

    def save(self, save_path, step, best_val=False, saver=None):
        model_name = self.name
        if not hasattr(self, 'saver') and saver is None:
            self.saver = Saver(self, save_path, optimizer=self.optim, prefix=model_name + '-', max_ckpts=self.max_ckpts)
        if saver is None:
            self.saver.save(model_name, step, best_val=best_val)
        else:
            saver.save(model_name, step, best_val=best_val)

    def load(self, save_path):
        if os.path.isdir(save_path):
            if not hasattr(self, 'saver'):
                self.saver = Saver(self, save_path, optimizer=self.optim, prefix=self.name + '-', max_ckpts=self.max_ckpts)
            self.saver.load_weights()
        else:
            None
            self.load_pretrained(save_path)

    def load_pretrained(self, ckpt_path, load_last=False, verbose=True):
        saver = Saver(self, '.', optimizer=self.optim)
        saver.load_pretrained_ckpt(ckpt_path, load_last, verbose=verbose)

    def activation(self, name):
        return getattr(nn, name)()

    def parameters(self):
        return filter(lambda p: p.requires_grad, super().parameters())

    def get_total_params(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def describe_params(self):
        pp = 0
        if hasattr(self, 'blocks'):
            for b in self.blocks:
                p = b.describe_params()
                pp += p
        else:
            None
            None
        total_params = self.get_total_params()
        None
        return total_params


def build_activation(activation, params, init=0):
    if activation == 'prelu' or activation is None:
        return nn.PReLU(params, init=init)
    if isinstance(activation, str):
        return getattr(nn, activation)()
    else:
        return activation


def build_norm_layer(norm_type, param=None, num_feats=None):
    if norm_type == 'bnorm':
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'snorm':
        spectral_norm(param)
        return None
    elif norm_type == 'bsnorm':
        spectral_norm(param)
        return nn.BatchNorm1d(num_feats)
    elif norm_type == 'lnorm':
        return nn.LayerNorm(num_feats)
    elif norm_type == 'wnorm':
        weight_norm(param)
        return None
    elif norm_type == 'inorm':
        return nn.InstanceNorm1d(num_feats, affine=False)
    elif norm_type == 'affinorm':
        return nn.InstanceNorm1d(num_feats, affine=True)
    elif norm_type is None:
        return None
    else:
        raise TypeError('Unrecognized norm type: ', norm_type)


def forward_activation(activation, tensor):
    if activation == 'glu':
        z, g = torch.chunk(tensor, 2, dim=1)
        y = z * torch.sigmoid(g)
        return y
    else:
        return activation(tensor)


def forward_norm(x, norm_layer):
    if norm_layer is not None:
        if isinstance(norm_layer, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = norm_layer(x)
        if isinstance(norm_layer, nn.LayerNorm):
            x = x.transpose(1, 2)
        return x
    else:
        return x


class GConv1DBlock(NeuralBlock):

    def __init__(self, ninp, fmaps, kwidth, stride=1, norm_type=None, act='prelu', name='GConv1DBlock'):
        super().__init__(name=name)
        if act is not None and act == 'glu':
            Wfmaps = 2 * fmaps
        else:
            Wfmaps = fmaps
        self.conv = nn.Conv1d(ninp, Wfmaps, kwidth, stride=stride)
        self.norm = build_norm_layer(norm_type, self.conv, fmaps)
        self.act = build_activation(act, fmaps)
        self.kwidth = kwidth
        self.stride = stride

    def forward(self, x):
        if self.stride > 1 or self.kwidth % 2 == 0:
            P = self.kwidth // 2 - 1, self.kwidth // 2
        else:
            P = self.kwidth // 2, self.kwidth // 2
        x_p = F.pad(x, P, mode='reflect')
        h = self.conv(x_p)
        h = forward_activation(self.act, h)
        h = forward_norm(h, self.norm)
        return h


class GDeconv1DBlock(NeuralBlock):

    def __init__(self, ninp, fmaps, kwidth, stride=4, norm_type=None, act=None, bias=True, name='GDeconv1DBlock'):
        super().__init__(name=name)
        if act is not None and act == 'glu':
            Wfmaps = 2 * fmaps
        else:
            Wfmaps = fmaps
        pad = max(0, (stride - kwidth) // -2)
        self.deconv = nn.ConvTranspose1d(ninp, Wfmaps, kwidth, stride=stride, padding=pad, bias=bias)
        self.norm = build_norm_layer(norm_type, self.deconv, Wfmaps)
        self.act = build_activation(act, fmaps)
        self.kwidth = kwidth
        self.stride = stride

    def forward(self, x):
        h = self.deconv(x)
        if self.stride % 2 != 0 and self.kwidth % 2 == 0 or self.stride % 2 == 0 and self.kwidth % 2 != 0:
            h = h[:, :, :-1]
        h = forward_norm(h, self.norm)
        h = forward_activation(self.act, h)
        return h


class ResARModule(NeuralBlock):

    def __init__(self, ninp, fmaps, res_fmaps, kwidth, dilation, norm_type=None, act=None, name='ResARModule'):
        super().__init__(name=name)
        self.dil_conv = nn.Conv1d(ninp, fmaps, kwidth, dilation=dilation)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv, fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, self.conv_1x1_skip, ninp)
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1)
        self.conv_1x1_res_norm = build_norm_layer(norm_type, self.conv_1x1_res, res_fmaps)

    def forward(self, x):
        kw__1 = self.kwidth - 1
        P = kw__1 + kw__1 * (self.dilation - 1)
        x_p = F.pad(x, (P, 0))
        h = self.dil_conv(x_p)
        h = forward_norm(h, self.dil_norm)
        h = self.act(h)
        a = h
        h = self.conv_1x1_skip(h)
        h = forward_norm(h, self.conv_1x1_skip_norm)
        y = x + h
        sh = self.conv_1x1_res(a)
        sh = forward_norm(sh, self.conv_1x1_res_norm)
        return y, sh


class FeBlock(NeuralBlock):

    def __init__(self, num_inputs, fmaps, kwidth, stride, dilation, pad_mode='reflect', act=None, norm_type=None, sincnet=False, sr=16000, name='FeBlock'):
        super().__init__(name=name)
        if act is not None and act == 'glu':
            Wfmaps = 2 * fmaps
        else:
            Wfmaps = fmaps
        self.num_inputs = num_inputs
        self.fmaps = fmaps
        self.kwidth = kwidth
        self.stride = stride
        self.dilation = dilation
        self.pad_mode = pad_mode
        self.sincnet = sincnet
        if sincnet:
            assert num_inputs == 1, num_inputs
            self.conv = SincConv_fast(1, Wfmaps, kwidth, sample_rate=sr, padding='SAME', stride=stride, pad_mode=pad_mode)
        else:
            self.conv = nn.Conv1d(num_inputs, Wfmaps, kwidth, stride, dilation=dilation)
        if not (norm_type == 'snorm' and sincnet):
            self.norm = build_norm_layer(norm_type, self.conv, Wfmaps)
        self.act = build_activation(act, fmaps)

    def forward(self, x):
        if self.kwidth > 1 and not self.sincnet:
            if self.stride > 1 or self.kwidth % 2 == 0:
                if self.dilation > 1:
                    raise ValueError('Cannot make dilated convolution with stride > 1')
                P = self.kwidth // 2 - 1, self.kwidth // 2
            else:
                pad = self.kwidth // 2 * (self.dilation - 1) + self.kwidth // 2
                P = pad, pad
            x = F.pad(x, P, mode=self.pad_mode)
        h = self.conv(x)
        if hasattr(self, 'norm'):
            h = forward_norm(h, self.norm)
        h = forward_activation(self.act, h)
        return h


class VQEMA(nn.Module):
    """ VQ w/ Exp. Moving Averages,
        as in (https://arxiv.org/pdf/1711.00937.pdf A.1).
        Partly based on
        https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    """

    def __init__(self, emb_K, emb_dim, beta, gamma, eps=1e-05):
        super().__init__()
        self.emb_K = emb_K
        self.emb_dim = emb_dim
        self.emb = nn.Embedding(self.emb_K, self.emb_dim)
        self.emb.weight.data.normal_()
        self.beta = beta
        self.gamma = gamma
        self.register_buffer('ema_cluster_size', torch.zeros(emb_K))
        self.ema_w = nn.Parameter(torch.Tensor(emb_K, emb_dim))
        self.ema_w.data.normal_()
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.emb_dim)
        device = 'cuda' if inputs.is_cuda else 'cpu'
        dist = torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(self.emb.weight ** 2, dim=1) - 2 * torch.matmul(flat_input, self.emb.weight.t())
        enc_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        enc = torch.zeros(enc_indices.shape[0], self.emb_K)
        enc.scatter_(1, enc_indices, 1)
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.gamma + (1 - self.gamma) * torch.sum(enc, 0)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (self.ema_cluster_size + self.eps) / (n + self.emb_K * self.eps) * n
            dw = torch.matmul(enc.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.gamma + (1 - self.gamma) * dw)
            self.emb.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        Q = torch.matmul(enc, self.emb.weight).view(input_shape)
        e_latent_loss = torch.mean((Q.detach() - inputs) ** 2)
        loss = self.beta * e_latent_loss
        Q = inputs + (Q - inputs).detach()
        avg_probs = torch.mean(enc, dim=0)
        PP = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, Q.permute(0, 2, 1).contiguous(), PP, enc


class WaveAdversarialLoss(nn.Module):

    def __init__(self, discriminator, d_optimizer, size_average=True, loss='L2', batch_acum=1, device='cpu'):
        super().__init__()
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.batch_acum = batch_acum
        if loss == 'L2':
            self.loss = nn.MSELoss(size_average)
            self.labels = [1, -1, 0]
        elif loss == 'BCE':
            self.loss = nn.BCEWithLogitsLoss()
            self.labels = [1, 0, 1]
        elif loss == 'Hinge':
            self.loss = None
        else:
            raise ValueError('Urecognized loss: {}'.format(loss))
        self.device = device

    def retrieve_label(self, y, lab_value, name=''):
        label = torch.ones(y.size()) * lab_value
        label = label
        return label

    def forward(self, iteration, x_fake, x_real, c_real=None, c_fake=None, grad=True):
        if grad:
            d_real = self.discriminator(x_real, cond=c_real)
            if self.loss:
                rl_lab = self.retrieve_label(d_real, self.labels[0], 'rl_lab')
                d_real_loss = self.loss(d_real, rl_lab)
            else:
                d_real_loss = F.relu(1.0 - d_real).mean()
            d_fake = self.discriminator(x_fake.detach(), cond=c_real)
            if self.loss:
                fk_lab = self.retrieve_label(d_fake, self.labels[1], 'fk_lab')
                d_fake_loss = self.loss(d_fake, fk_lab)
            else:
                d_fake_loss = F.relu(1.0 + d_fake).mean()
            if c_fake is not None:
                d_fake_lab = self.discriminator(x_real, cond=c_fake)
                if self.loss:
                    d_fake_lab_loss = self.loss(d_fake_lab, fk_lab)
                else:
                    d_fake_lab_loss = F.relu(1.0 + d_fake_lab).mean()
                d_loss = d_real_loss + d_fake_loss + d_fake_lab_loss
            else:
                d_loss = d_real_loss + d_fake_loss
            d_loss.backward(retain_graph=True)
            if iteration % self.batch_acum == 0:
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()
        g_real = self.discriminator(x_fake, cond=c_real)
        if self.loss:
            grl_lab = self.retrieve_label(g_real, self.labels[2], 'grl_lab')
            g_real_loss = self.loss(g_real, grl_lab)
        else:
            g_real_loss = -g_real.mean()
        if grad:
            return {'g_loss': g_real_loss, 'd_real_loss': d_real_loss, 'd_fake_loss': d_fake_loss}
        else:
            return {'g_loss': g_real_loss}


def make_labels(y):
    bsz = y.size(0) // 2
    slen = y.size(2)
    label = torch.cat((torch.ones(bsz, 1, slen, requires_grad=False), torch.zeros(bsz, 1, slen, requires_grad=False)), dim=0)
    return label


def make_samples(x, augment):
    x_pos = torch.cat((x[0], x[1]), dim=1)
    x_neg = torch.cat((x[0], x[2]), dim=1)
    if augment:
        x_pos2 = torch.cat((x[1], x[0]), dim=1)
        x_neg2 = torch.cat((x[1], x[2]), dim=1)
        x_pos = torch.cat((x_pos, x_pos2), dim=0)
        x_neg = torch.cat((x_neg, x_neg2), dim=0)
    return x_pos, x_neg


class PatternedDropout(nn.Module):

    def __init__(self, emb_size, p=0.5, dropout_mode=['fixed_rand'], ratio_fixed=None, range_fixed=None, drop_whole_channels=False):
        """Applies a fixed pattern of dropout for the whole training
        session (i.e applies different only among pre-specified dimensions)
        """
        super(PatternedDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError('dropout probability has to be between 0 and 1, but got {}'.format(p))
        self.p = p
        if self.p > 0:
            d_modes = ['std', 'fixed_rand', 'fixed_given']
            assert dropout_mode in d_modes, 'Expected dropout mode in {}, got {}'.format(d_modes, dropout_mode)
            self.drop_whole_channels = drop_whole_channels
            self.dropout_fixed = False
            if dropout_mode == 'fixed_rand':
                self.dropout_fixed = True
                assert ratio_fixed is not None, "{} needs 'ratio_fixed' arg set.".format(dropout_mode)
                if ratio_fixed <= 0 or ratio_fixed > 1:
                    raise ValueError("{} mode needs 'ratio_fixed' to be set and in (0, 1) range, got {}".format(dropout_mode, ratio_fixed))
                self.ratio_fixed = ratio_fixed
                self.dropped_dimsize = int(emb_size - emb_size * ratio_fixed)
                tot_idx = np.arange(emb_size)
                sel_idx = np.sort(np.random.choice(tot_idx, size=self.dropped_dimsize, replace=False))
            elif dropout_mode == 'fixed_given':
                self.dropout_fixed = True
                if range_fixed is None or not isinstance(range_fixed, str) or len(range_fixed.split(':')) < 2:
                    raise ValueError("{} mode needs 'range_dropped' to be set (i.e. 10:20)".format(dropout_mode))
                rng = range_fixed.split(':')
                beg = int(rng[0])
                end = int(rng[1])
                assert beg < end and end <= emb_size, 'Incorrect range {}'.format(range_fixed)
                self.dropped_dimsize = int(emb_size - (end - beg))
                tot_idx = np.arange(emb_size)
                fixed_idx = np.arange(beg, end, 1)
                sel_idx = np.setdiff1d(tot_idx, fixed_idx, assume_unique=True)
            if self.dropout_fixed:
                assert len(sel_idx) > 0, 'Asked for fixed dropout, but sel_idx {}'.format(sel_idx)
                None
                self.dindexes = torch.LongTensor(sel_idx)
                self.p = p
                self.p_scale = 1.0 / (1.0 - self.p)
            else:
                self.p = p
                None
        else:
            None

    def forward(self, x):
        if self.p == 0 or not self.training:
            return x
        if self.dropout_fixed and self.training:
            self.dindexes = self.dindexes
            assert len(x.size()) == 3, 'Expected to get 3 dimensional tensor, got {}'.format(len(x.size()))
            bsize, emb_size, tsize = x.size()
            if self.drop_whole_channels:
                batch_mask = torch.full(size=(bsize, emb_size), fill_value=1.0, device=x.device)
                probs = torch.full(size=(bsize, self.dropped_dimsize), fill_value=1.0 - self.p)
                b = Binomial(total_count=1, probs=probs)
                mask = b.sample()
                mask = mask
                batch_mask[:, (self.dindexes)] *= mask * self.p_scale
                x = x * batch_mask.view(bsize, emb_size, -1)
            else:
                batch_mask = torch.ones_like(x, device=x.device)
                probs = torch.full(size=(bsize, self.dropped_dimsize, tsize), fill_value=1.0 - self.p)
                b = Binomial(total_count=1, probs=probs)
                mask = b.sample()
                mask = mask
                batch_mask[:, (self.dindexes), :] *= mask * self.p_scale
                x = x * batch_mask
            return x
        else:
            return F.dropout(x, p=self.p, training=self.training)


class MLPBlock(NeuralBlock):

    def __init__(self, ninp, fmaps, din=0, dout=0, context=1, tie_context_weights=False, name='MLPBlock', ratio_fixed=None, range_fixed=None, dropin_mode='std', drop_channels=False, emb_size=100):
        super().__init__(name=name)
        self.ninp = ninp
        self.fmaps = fmaps
        self.tie_context_weights = tie_context_weights
        assert context % 2 != 0, context
        if tie_context_weights:
            self.W = nn.Conv1d(ninp, fmaps, 1)
            self.pool = nn.AvgPool1d(kernel_size=context, stride=1, padding=context // 2, count_include_pad=False)
        else:
            self.W = nn.Conv1d(ninp, fmaps, context, padding=context // 2)
        self.din = PatternedDropout(emb_size=emb_size, p=din, dropout_mode=dropin_mode, range_fixed=range_fixed, ratio_fixed=ratio_fixed, drop_whole_channels=drop_channels)
        self.act = nn.PReLU(fmaps)
        self.dout = nn.Dropout(dout)

    def forward(self, x, device=None):
        if self.tie_context_weights:
            return self.dout(self.act(self.pool(self.W(self.din(x)))))
        return self.dout(self.act(self.W(self.din(x))))


class ScaleGrad(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * ctx.alpha
        return output, None


class DecoderMinion(Model):

    def __init__(self, num_inputs, num_outputs, dropout, dropout_time=0.0, shuffle=False, shuffle_depth=7, hidden_size=256, hidden_layers=2, fmaps=[256, 256, 128, 128, 128, 64, 64], strides=[2, 2, 2, 2, 2, 5], kwidths=[2, 2, 2, 2, 2, 5], norm_type=None, skip=False, loss=None, loss_weight=1.0, keys=None, name='DecoderMinion'):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.dropout_time = dropout_time
        self.shuffle = shuffle
        self.shuffle_depth = shuffle_depth
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.fmaps = fmaps
        self.strides = strides
        self.kwidths = kwidths
        self.norm_type = norm_type
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        for fmap, kw, stride in zip(fmaps, kwidths, strides):
            block = GDeconv1DBlock(ninp, fmap, kw, stride, norm_type=norm_type)
            self.blocks.append(block)
            ninp = fmap
        for _ in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp, hidden_size, dropout))
            ninp = hidden_size
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        if self.dropout_time > 0:
            mask = (torch.FloatTensor(x.shape[0], x.shape[2]).uniform_() > self.dropout_time).float().unsqueeze(1)
            x = x * mask
        if self.shuffle:
            x = torch.split(x, self.shuffle_depth, dim=2)
            shuffled_x = []
            for elem in x:
                r = torch.randperm(elem.shape[2])
                shuffled_x.append(elem[:, :, (r)])
            x = torch.cat(shuffled_x, dim=2)
        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h_ = h
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y


class GRUMinion(Model):

    def __init__(self, num_inputs, num_outputs, dropout, hidden_size=256, hidden_layers=2, skip=True, loss=None, loss_weight=1.0, keys=None, name='GRUMinion'):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        if keys is None:
            keys = [name]
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        self.rnn = nn.GRU(ninp, hidden_size, num_layers=hidden_layers, batch_first=True, dropout=dropout)
        self.W = nn.Conv1d(hidden_size, num_outputs, 1)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        h, _ = self.rnn(x.transpose(1, 2))
        h = h.transpose(1, 2)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y


class MLPMinion(Model):

    def __init__(self, num_inputs, num_outputs, dropout, dropout_time=0.0, hidden_size=256, dropin=0.0, hidden_layers=2, context=1, tie_context_weights=False, skip=True, loss=None, loss_weight=1.0, keys=None, augment=False, r=1, name='MLPMinion', ratio_fixed=None, range_fixed=None, dropin_mode='std', drop_channels=False, emb_size=100):
        super().__init__(name=name)
        self.num_inputs = num_inputs
        assert context % 2 != 0, context
        self.context = context
        self.tie_context_weights = tie_context_weights
        self.dropout = dropout
        self.dropout_time = dropout_time
        self.skip = skip
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        if keys is None:
            keys = [name]
        self.r = r
        self.num_outputs = num_outputs * r
        self.blocks = nn.ModuleList()
        ninp = num_inputs
        for hi in range(hidden_layers):
            self.blocks.append(MLPBlock(ninp, hidden_size, din=dropin, dout=dropout, context=context, tie_context_weights=tie_context_weights, emb_size=emb_size, dropin_mode=dropin_mode, range_fixed=range_fixed, ratio_fixed=ratio_fixed, drop_channels=drop_channels))
            ninp = hidden_size
            context = 1
        self.W = nn.Conv1d(ninp, self.num_outputs, context, padding=context // 2)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        if self.dropout_time > 0 and self.context > 1:
            mask = (torch.FloatTensor(x.shape[0], x.shape[2]).uniform_() > self.dropout_time).float().unsqueeze(1)
            x = x * mask
        h = x
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y


class GapMinion(MLPMinion):

    def __init__(self, num_inputs, num_outputs, dropout, hidden_size=256, hidden_layers=2, skip=True, loss=None, loss_weight=1.0, keys=None, name='GapMinion'):
        super().__init__(num_inputs=num_inputs, num_outputs=num_outputs, dropout=dropout, hidden_size=hidden_size, hidden_layers=hidden_layers, skip=skip, loss=loss, loss_weight=loss_weight, keys=keys, name=name)
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        T = x.shape[2]
        aidx = torch.LongTensor(np.random.randint(0, T, size=x.shape[0]))
        bidx = torch.LongTensor(np.random.randint(0, T, size=x.shape[0]))
        x_a = []
        x_b = []
        dists = []
        for i_, (aidx_, bidx_) in enumerate(zip(aidx, bidx)):
            x_a.append(x[(i_), :, (aidx_)].unsqueeze(0))
            x_b.append(x[(i_), :, (bidx_)].unsqueeze(0))
            dist = torch.abs(aidx_ - bidx_) / (T - 1)
            dists.append(dist)
        x_a = torch.cat(x_a, dim=0)
        x_b = torch.cat(x_b, dim=0)
        x_full = torch.cat((x_a, x_b), dim=1).unsqueeze(2)
        dists = torch.LongTensor(dists)
        dists = dists.view(-1, 1, 1)
        h = x_full
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h, dists
        else:
            return y, dists


class RegularizerMinion(object):

    def __init__(self, num_inputs=None, loss='MSELoss', loss_weight=1.0, name=''):
        if isinstance(loss, str):
            self.loss = getattr(nn, loss)()
        else:
            self.loss = loss
        self.loss_weight = loss_weight
        self.name = name

    def __call__(self, x, alpha=1, device=None):
        return self.forward(x, alpha=alpha, device=device)

    def forward(self, x, alpha=1, device=None):
        return x


class SPCMinion(MLPMinion):

    def __init__(self, num_inputs, num_outputs, dropout, hidden_size=256, hidden_layers=2, ctxt_frames=5, seq_pad=16, skip=True, loss=None, loss_weight=1.0, keys=None, name='SPCMinion'):
        None
        None
        num_inputs = (ctxt_frames + 1) * num_inputs
        None
        super().__init__(num_inputs=num_inputs, num_outputs=num_outputs, dropout=dropout, hidden_size=hidden_size, hidden_layers=hidden_layers, skip=skip, loss=loss, loss_weight=loss_weight, keys=keys, name=name)
        self.ctxt_frames = ctxt_frames
        self.seq_pad = seq_pad
        self.sg = ScaleGrad()

    def forward(self, x, alpha=1, device=None):
        self.sg.apply(x, alpha)
        seq_pad = self.seq_pad
        N = self.ctxt_frames
        M = seq_pad + N
        idxs_t = list(range(M + 1, x.size(2) - M))
        t = random.choice(idxs_t)
        bsz = x.size(0)
        idxs_ft = list(range(t + seq_pad, x.size(2) - N))
        future_t = random.choice(idxs_ft)
        idxs_pt = list(range(N, t - seq_pad))
        past_t = random.choice(idxs_pt)
        future = x[:, :, future_t:future_t + N].contiguous().view(bsz, -1)
        past = x[:, :, past_t - N:past_t].contiguous().view(bsz, -1)
        current = x[:, :, (t)].contiguous()
        pos = torch.cat((current, future), dim=1)
        neg = torch.cat((current, past), dim=1)
        x_full = torch.cat((pos, neg), dim=0).unsqueeze(2)
        h = x_full
        for bi, block in enumerate(self.blocks, start=1):
            h = block(h)
        y = self.W(h)
        if self.skip:
            return y, h
        else:
            return y


class SimpleResBlock1D(nn.Module):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(SimpleResBlock1D(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, feat_dims, upsample_scales=[4, 4, 10], compute_dims=128, res_blocks=10, res_out_dims=128, pad=2):
        super().__init__()
        self.num_outputs = res_out_dims
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = 1, scale * 2 + 1
            padding = 0, scale
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """
    https://github.com/fatchord/WaveRNN/blob/master/utils/distribution.py
    Sample from discretized mixture of logistic distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-05, 1.0 - 1e-05)
    temp = logit_probs.data - torch.log(-torch.log(temp))
    _, argmax = temp.max(dim=-1)
    one_hot = F.one_hot(argmax, nr_mix).float()
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    u = means.data.new(means.size()).uniform_(1e-05, 1.0 - 1e-05)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1.0 - u))
    x = torch.clamp(torch.clamp(x, min=-1.0), max=1.0)
    return x


class WaveRNNMinion(Model):
    """ Based on WaveRNN a publicly available WaveRNN implementation:
        https://github.com/fatchord/WaveRNN/blob/master/models/fatchord_version.py
    """

    def __init__(self, num_inputs, rnn_dims=512, fc_dims=512, bits=9, sample_rate=16000, hop_length=160, mode='RAW', pad=2, upsample_cfg={}, loss=None, loss_weight=1.0, keys=None, name='WaveRNNMinion'):
        super().__init__(name=name)
        feat_dims = num_inputs
        self.num_inputs = num_inputs
        self.loss = loss
        self.loss_weight = loss_weight
        self.keys = keys
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW':
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError('Unknown model mode value - ', self.mode)
        upsample_cfg['feat_dims'] = num_inputs
        upsample_cfg['pad'] = pad
        self.upsample = UpsampleNetwork(**upsample_cfg)
        self.rnn_dims = rnn_dims
        self.aux_dims = self.upsample.num_outputs // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)
        self.step = nn.Parameter(torch.zeros(1).long(), requires_grad=False)
        if keys is None:
            keys = [name]
        self.sg = ScaleGrad()

    def forward(self, x, mels, alpha=1, device=None):
        self.sg.apply(x, alpha)
        device = next(self.parameters()).device
        self.step += 1
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        h2 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        mels, aux = self.upsample(mels)
        aux_idx = [(self.aux_dims * i) for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)
        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)
        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))
        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def generate(self, mels, save_path, batched, target, overlap, mu_law):
        device = next(self.parameters()).device
        mu_law = mu_law if self.mode == 'RAW' else False
        self.eval()
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)
        with torch.no_grad():
            mels = torch.as_tensor(mels, device=device)
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))
            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)
            b_size, seq_len, _ = mels.size()
            h1 = torch.zeros(b_size, self.rnn_dims, device=device)
            h2 = torch.zeros(b_size, self.rnn_dims, device=device)
            x = torch.zeros(b_size, 1, device=device)
            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]
            for i in range(seq_len):
                m_t = mels[:, (i), :]
                a1_t, a2_t, a3_t, a4_t = (a[:, (i), :] for a in aux_split)
                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)
                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)
                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))
                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                logits = self.fc3(x)
                if self.mode == 'MOL':
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    x = sample.transpose(0, 1)
                elif self.mode == 'RAW':
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)
                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.0) - 1.0
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError('Unknown model mode value - ', self.mode)
        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)
        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]
        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out
        save_wav(output, save_path)
        self.train()
        return output

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side='both'):
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c, device=x.device)
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):
        """ Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()
        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)
        Details:
            x = [[h1, h2, ... hn]]
            Where each h is a vector of conditioning features
            Eg: target=2, overlap=1 with x.size(1)=10
            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        """
        _, total_len, features = x.size()
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')
        folded = torch.zeros(num_folds, target + 2 * overlap, features, device=x.device)
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]
        return folded

    def xfade_and_unfold(self, y, target, overlap):
        """ Applies a crossfade and unfolds into a 1d array.
        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64
        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]
            Apply a gain envelope at both ends of the sequences
            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]
            Stagger and add up the groups of samples:
            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]
        """
        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros(silence_len, dtype=np.float64)
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out
        unfolded = np.zeros(total_len, dtype=np.float64)
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]
        return unfolded


def minion_maker(cfg):
    if isinstance(cfg, str):
        with open(cfg, 'r') as f:
            cfg = json.load(f)
    None
    None
    None
    mtype = cfg.pop('type', 'mlp')
    if mtype == 'mlp':
        minion = MLPMinion(**cfg)
    elif mtype == 'decoder':
        minion = DecoderMinion(**cfg)
    elif mtype == 'wavernn':
        minion = WaveRNNMinion(**cfg)
    elif mtype == 'spc':
        minion = SPCMinion(**cfg)
    elif mtype == 'gap':
        minion = GapMinion(**cfg)
    elif mtype == 'gru':
        minion = GRUMinion(**cfg)
    elif mtype == 'regularizer':
        minion = RegularizerMinion(**cfg)
    else:
        raise TypeError('Unrecognized minion type {}'.format(mtype))
    return minion


class LIM(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])
        cfg['num_inputs'] = 2 * emb_dim
        if 'augment' in cfg.keys():
            self.augment = cfg['augment']
        else:
            self.augment = False
        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        x_pos, x_neg = make_samples(x, self.augment)
        x = torch.cat((x_pos, x_neg), dim=0)
        y = self.minion(x, alpha)
        label = make_labels(y)
        return y, label


class GIM(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])
        cfg['num_inputs'] = 2 * emb_dim
        if 'augment' in cfg.keys():
            self.augment = cfg['augment']
        else:
            self.augment = False
        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        x_pos, x_neg = make_samples(x, self.augment)
        x = torch.cat((x_pos, x_neg), dim=0)
        x = torch.mean(x, dim=2, keepdim=True)
        y = self.minion(x, alpha)
        label = make_labels(y)
        return y, label


class SPC(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])
        cfg['num_inputs'] = emb_dim
        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        y = self.minion(x, alpha)
        label = make_labels(y)
        return y, label


class Gap(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])
        cfg['num_inputs'] = 2 * emb_dim
        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        y, label = self.minion(x, alpha)
        label = label.float()
        return y, label


class AdversarialChunk(Model):

    def __init__(self, cfg, emb_dim):
        super().__init__(name=cfg['name'])
        self.minion = minion_maker(cfg)
        self.loss = self.minion.loss
        self.loss_weight = self.minion.loss_weight

    def forward(self, x, alpha=1, device=None):
        y, label = self.minion(x, alpha)
        label = label.float()
        return y, label


class encoder(Model):

    def __init__(self, frontend, name='encoder'):
        super().__init__(name)
        self.frontend = frontend
        self.emb_dim = self.frontend.emb_dim

    def forward(self, batch, device):
        if type(batch) == dict:
            x = torch.cat((batch['chunk'], batch['chunk_ctxt'], batch['chunk_rand']), dim=0)
        else:
            x = batch
        y = self.frontend(x)
        if type(batch) == dict:
            embedding = torch.chunk(y, 3, dim=0)
            chunk = embedding[0]
            return embedding, chunk
        else:
            return y


class _ASPPModule(Model):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(Model):

    def __init__(self, inplanes, emb_dim, dilations=[1, 6, 12, 18], fmaps=48, dense=False):
        super(ASPP, self).__init__()
        if not dense:
            self.aspp1 = _ASPPModule(inplanes, fmaps, 1, padding=0, dilation=dilations[0])
            self.aspp2 = _ASPPModule(inplanes, fmaps, 3, padding=dilations[1], dilation=dilations[1])
            self.aspp3 = _ASPPModule(inplanes, fmaps, 3, padding=dilations[2], dilation=dilations[2])
            self.aspp4 = _ASPPModule(inplanes, fmaps, 3, padding=dilations[3], dilation=dilations[3])
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(inplanes, fmaps, 1, stride=1, bias=False), nn.BatchNorm1d(fmaps), nn.ReLU())
        else:
            self.aspp1 = _ASPPModule(inplanes, fmaps, dilations[0], padding=0, dilation=1)
            self.aspp2 = _ASPPModule(inplanes, fmaps, dilations[1], padding=dilations[1] // 2, dilation=1)
            self.aspp3 = _ASPPModule(inplanes, fmaps, dilations[2], padding=dilations[2] // 2, dilation=1)
            self.aspp4 = _ASPPModule(inplanes, fmaps, dilations[3], padding=dilations[3] // 2, dilation=1)
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(inplanes, fmaps, 1, stride=1, bias=False), nn.BatchNorm1d(fmaps), nn.ReLU())
        self.conv1 = nn.Conv1d(fmaps * 5, emb_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='linear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class _ASPPModule2d(Model):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule2d, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP2d(Model):

    def __init__(self, inplanes, emb_dim, dilations=[1, 6, 12, 18], fmaps=48, dense=False):
        super(ASPP2d, self).__init__()
        if not dense:
            self.aspp1 = _ASPPModule2d(inplanes, fmaps, 1, padding=0, dilation=dilations[0])
            self.aspp2 = _ASPPModule2d(inplanes, fmaps, 3, padding=dilations[1], dilation=dilations[1])
            self.aspp3 = _ASPPModule2d(inplanes, fmaps, 3, padding=dilations[2], dilation=dilations[2])
            self.aspp4 = _ASPPModule2d(inplanes, fmaps, 3, padding=dilations[3], dilation=dilations[3])
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(inplanes, fmaps, 1, stride=1, bias=False), nn.BatchNorm2d(fmaps), nn.ReLU())
        self.conv1 = nn.Conv2d(fmaps * 5, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x).squeeze(1)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class aspp_resblock(Model):

    def __init__(self, in_channel, out_channel, kernel_size, stride, dilations, fmaps, pool2d=False, dense=False):
        super().__init__(name='aspp_resblock')
        padding = kernel_size // 2
        if pool2d:
            self.block1 = nn.Sequential(ASPP2d(1, out_channel, dilations, fmaps, dense), nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm1d(out_channel), nn.ReLU(out_channel))
            self.block2 = nn.Sequential(ASPP2d(1, out_channel, dilations, fmaps, dense), nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False), nn.BatchNorm1d(out_channel), nn.ReLU(out_channel))
        else:
            self.block1 = nn.Sequential(ASPP(in_channel, out_channel, dilations, fmaps, dense), nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm1d(out_channel), nn.ReLU(out_channel))
            self.block2 = nn.Sequential(ASPP(out_channel, out_channel, dilations, fmaps, dense), nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False), nn.BatchNorm1d(out_channel), nn.ReLU(out_channel))
        self._init_weight()

    def forward(self, x):
        out_1 = self.block1(x)
        out_2 = self.block2(out_1)
        y = out_1 + out_2
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_rnn_block(in_size, rnn_size, rnn_layers, rnn_type, bidirectional=True, dropout=0, use_cuda=True):
    if rnn_type.lower() == 'qrnn' and QRNN is not None:
        if bidirectional:
            None
            rnn_size = 2 * rnn_size
        rnn = QRNN(in_size, rnn_size, rnn_layers, dropout=dropout, window=2, use_cuda=use_cuda)
    elif rnn_type.lower() == 'lstm' or rnn_type.lower() == 'gru':
        rnn = getattr(nn, rnn_type.upper())(in_size, rnn_size, rnn_layers, dropout=dropout, bidirectional=bidirectional)
    else:
        raise TypeError('Unrecognized rnn type: ', rnn_type)
    return rnn


def format_frontend_chunk(batch, device='cpu'):
    if type(batch) == dict:
        if 'chunk_ctxt' and 'chunk_rand' in batch:
            keys = ['chunk', 'chunk_ctxt', 'chunk_rand', 'cchunk']
            batches = [batch[k] for k in keys if k in batch]
            x = torch.cat(batches, dim=0)
            data_fmt = len(batches)
        else:
            x = batch['chunk']
            data_fmt = 1
    else:
        x = batch
        data_fmt = 0
    return x, data_fmt


def select_output(h, mode=None):
    if mode == 'avg_norm':
        return h - torch.mean(h, dim=2, keepdim=True)
    elif mode == 'avg_concat':
        global_avg = torch.mean(h, dim=2, keepdim=True).repeat(1, 1, h.shape[-1])
        return torch.cat((h, global_avg), dim=1)
    elif mode == 'avg_norm_concat':
        global_avg = torch.mean(h, dim=2, keepdim=True)
        h = h - global_avg
        global_feature = global_avg.repeat(1, 1, h.shape[-1])
        return torch.cat((h, global_feature), dim=1)
    else:
        return h


def format_frontend_output(y, data_fmt, mode):
    if data_fmt > 1:
        embedding = torch.chunk(y, data_fmt, dim=0)
        chunk = embedding[0]
        return embedding, chunk
    elif data_fmt == 1:
        chunk = embedding = y
        return embedding, chunk
    else:
        return select_output(y, mode=mode)


class aspp_res_encoder(Model):

    def __init__(self, sinc_out, hidden_dim, kernel_sizes=[11, 11, 11, 11], sinc_kernel=251, sinc_stride=1, strides=[10, 4, 2, 2], dilations=[1, 6, 12, 18], fmaps=48, name='aspp_encoder', pool2d=False, rnn_pool=False, rnn_add=False, concat=[False, False, False, True], dense=False):
        super().__init__(name=name)
        self.sinc = SincConv_fast(1, sinc_out, sinc_kernel, sample_rate=16000, padding='SAME', stride=sinc_stride, pad_mode='reflect')
        self.ASPP_blocks = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            if i == 0:
                self.ASPP_blocks.append(aspp_resblock(sinc_out, hidden_dim, kernel_sizes[i], strides[i], dilations, fmaps[i], pool2d[i], dense))
            else:
                self.ASPP_blocks.append(aspp_resblock(hidden_dim, hidden_dim, kernel_sizes[i], strides[i], dilations, fmaps[i], pool2d[i], dense))
        self.rnn_pool = rnn_pool
        self.rnn_add = rnn_add
        self.concat = concat
        assert (self.rnn_pool and self.rnn_add or not self.rnn_pool) or self.rnn_pool
        if rnn_pool:
            self.rnn = build_rnn_block(hidden_dim, hidden_dim // 2, rnn_layers=1, rnn_type='qrnn', bidirectional=True, dropout=0)
            self.W = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.emb_dim = hidden_dim

    def forward(self, batch, device=None, mode=None):
        x, data_fmt = format_frontend_chunk(batch, device)
        sinc_out = self.sinc(x)
        out = []
        input = sinc_out
        for i, block in enumerate(self.ASPP_blocks, 0):
            input = block(input)
            if self.concat[i]:
                out.append(input)
        if len(out) > 1:
            out = self.fuse(out)
            out = torch.cat(out, dim=1)
        else:
            out = out[0]
        if self.rnn_pool:
            rnn_out = out.transpose(1, 2).transpose(0, 1)
            rnn_out, _ = self.rnn(rnn_out)
            rnn_out = rnn_out.transpose(0, 1).transpose(1, 2)
        if self.rnn_pool and self.rnn_add:
            h = out + rnn_out
        elif self.rnn_pool and not self.rnn_add:
            h = rnn_out
        else:
            h = out
        return format_frontend_output(h, data_fmt, mode)

    def fuse(self, out):
        last_feature = out[-1]
        for i in range(len(out) - 1):
            out[i] = F.adaptive_avg_pool1d(out[i], last_feature.shape[-1])
        return out


class attention_block(Model):

    def __init__(self, emb_dim, name, options, K, strides, chunksize, avg_factor=0, mode='concat'):
        super().__init__(name=name)
        self.name = name
        self.mode = mode
        self.emb_dim = emb_dim
        self.avg_factor = avg_factor
        nn_input = self.cal_nn_input_dim(strides, chunksize)
        self.mlp = MLP(options=options, inp_dim=nn_input)
        self.K = K
        self.avg_init = True

    def forward(self, hidden, device):
        batch_size = hidden.shape[0]
        feature_length = hidden.shape[2]
        hidden = hidden.contiguous()
        if self.mode == 'concat':
            hidden_att = hidden.view(hidden.shape[0], self.emb_dim * feature_length)
        if self.mode == 'avg_time':
            hidden_att = hidden.mean(-1)
        if self.mode == 'avg_time_batch':
            hidden_att = hidden.mean(-1).mean(0).unsqueeze(0)
        distribution = self.mlp(hidden_att)
        if self.avg_init:
            self.running_dist = self.init_running_avg(batch_size).detach()
            self.avg_init = False
        self.running_dist = self.running_dist.detach() * self.avg_factor + distribution * (1 - self.avg_factor)
        distribution = self.running_dist
        _, indices = torch.topk(distribution, dim=1, k=self.K, largest=True, sorted=False)
        mask = torch.zeros(distribution.size(), requires_grad=False).detach()
        mask = mask.scatter(1, indices, 1).unsqueeze(-1).repeat(1, 1, feature_length)
        selection = mask * hidden
        return selection, mask

    def cal_nn_input_dim(self, strides, chunk_size):
        if self.mode == 'concat':
            compress_factor = 1
            for s in strides:
                compress_factor = compress_factor * s
            if chunk_size % compress_factor != 0:
                raise ValueError('chunk_size should be divisible by the product of the strides factors!')
            nn_input = int(chunk_size // compress_factor) * self.emb_dim
            None
            return nn_input
        if self.mode == 'avg_time' or self.mode == 'avg_time_batch':
            return self.emb_dim

    def init_running_avg(self, batch_size):
        dist = torch.randn(self.emb_dim).float()
        dist = dist.unsqueeze(0).repeat(batch_size, 1)
        dist = F.softmax(dist, dim=1)
        return dist


class ResBasicBlock1D(NeuralBlock):
    """ Adapted from
        https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, kwidth=3, dilation=1, norm_layer=None, name='ResBasicBlock1D'):
        super().__init__(name=name)
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        P = kwidth // 2 * dilation
        self.conv1 = nn.Conv1d(inplanes, planes, kwidth, stride=1, padding=P, bias=False, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kwidth, padding=P, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class EmoDRNLSTM(Model):
    """ Based on https://ieeexplore.ieee.org/document/8682154 
        (Li et al. 2019), without MHA
    """

    def __init__(self, num_inputs, num_outputs, max_ckpts=5, frontend=None, ft_fe=False, dropout=0, rnn_dropout=0, att=False, att_heads=4, att_dropout=0, name='EmoDRNMHA'):
        super().__init__(max_ckpts=max_ckpts, name=name)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.frontend = frontend
        self.ft_fe = ft_fe
        self.drn = nn.Sequential(nn.Conv1d(num_inputs, 32, 10), nn.Conv1d(32, 64, 2, stride=2), ResBasicBlock1D(64, 64, kwidth=5, att=att, att_heads=att_heads, att_dropout=att_dropout), ResBasicBlock1D(64, 64, kwidth=5, att=att, att_heads=att_heads, att_dropout=att_dropout), nn.Dropout2d(dropout), nn.Conv1d(64, 128, 2, stride=2), ResBasicBlock1D(128, 128, kwidth=5, att=att, att_heads=att_heads, att_dropout=att_dropout), ResBasicBlock1D(128, 128, kwidth=5, att=att, att_heads=att_heads, att_dropout=att_dropout), nn.Dropout2d(dropout), nn.Conv1d(128, 256, 1, stride=1), ResBasicBlock1D(256, 256, kwidth=3, dilation=2, att=att, att_heads=att_heads, att_dropout=att_dropout), ResBasicBlock1D(256, 256, kwidth=3, dilation=2, att=att, att_heads=att_heads, att_dropout=att_dropout), nn.Dropout2d(dropout), nn.Conv1d(256, 512, 1, stride=1), ResBasicBlock1D(512, 512, kwidth=3, dilation=4, att=att, att_heads=att_heads, att_dropout=att_dropout), ResBasicBlock1D(512, 512, kwidth=3, dilation=4, att=att, att_heads=att_heads, att_dropout=att_dropout), nn.Dropout2d(dropout))
        self.rnn = nn.LSTM(512, 512, num_layers=2, batch_first=True, dropout=rnn_dropout)
        self.mlp = nn.Sequential(nn.Conv1d(512, 200, 1), nn.ReLU(inplace=True), nn.Conv1d(200, 200, 1), nn.ReLU(inplace=True), nn.Conv1d(200, num_outputs, 1), nn.LogSoftmax(dim=1))

    def forward(self, x):
        if self.frontend is not None:
            x = self.frontend(x)
            if not self.ft_fe:
                x = x.detach()
        x = F.pad(x, (4, 5))
        x = self.drn(x)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        xt = torch.chunk(x, x.shape[1], dim=1)
        x = xt[-1].transpose(1, 2)
        x = self.mlp(x)
        return x


class MLPClassifier(Model):

    def __init__(self, frontend, num_spks=None, ft_fe=False, hidden_size=2048, hidden_layers=1, z_bnorm=False, name='MLP'):
        super().__init__(name=name, max_ckpts=1000)
        self.frontend = frontend
        self.ft_fe = ft_fe
        if ft_fe:
            None
        if z_bnorm:
            self.z_bnorm = nn.BatchNorm1d(frontend.emb_dim, affine=False)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        layers = [nn.Conv1d(frontend.emb_dim, hidden_size, 1), nn.LeakyReLU(), nn.BatchNorm1d(hidden_size)]
        for n in range(1, hidden_layers):
            layers += [nn.Conv1d(hidden_size, hidden_size, 1), nn.LeakyReLU(), nn.BatchNorm1d(hidden_size)]
        layers += [nn.Conv1d(hidden_size, num_spks, 1), nn.LogSoftmax(dim=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        h = self.frontend(x)
        if not self.ft_fe:
            h = h.detach()
        if hasattr(self, 'z_bnorm'):
            h = self.z_bnorm(h)
        return self.model(h)


class RNNClassifier(Model):

    def __init__(self, frontend, num_spks=None, ft_fe=False, hidden_size=1300, z_bnorm=False, uni=False, return_sequence=False, name='RNN'):
        super().__init__(name=name, max_ckpts=1000)
        self.frontend = frontend
        self.ft_fe = ft_fe
        if ft_fe:
            None
        if z_bnorm:
            self.z_bnorm = nn.BatchNorm1d(frontend.emb_dim, affine=False)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        if uni:
            hsize = hidden_size
        else:
            hsize = hidden_size // 2
        self.rnn = nn.GRU(frontend.emb_dim, hsize, bidirectional=not uni, batch_first=True)
        self.model = nn.Sequential(nn.Conv1d(hidden_size, num_spks, 1), nn.LogSoftmax(dim=1))
        self.return_sequence = return_sequence
        self.uni = uni

    def forward(self, x):
        h = self.frontend(x)
        if not self.ft_fe:
            h = h.detach()
        if hasattr(self, 'z_bnorm'):
            h = self.z_bnorm(h)
        ht, state = self.rnn(h.transpose(1, 2))
        if self.return_sequence:
            ht = ht.transpose(1, 2)
        elif not self.uni:
            bsz, slen, feats = ht.size()
            ht = torch.chunk(ht.view(bsz, slen, 2, feats // 2), 2, dim=2)
            ht_fwd = ht[0][:, (-1), (0), :].unsqueeze(2)
            ht_bwd = ht[1][:, (0), (0), :].unsqueeze(2)
            ht = torch.cat((ht_fwd, ht_bwd), dim=1)
        else:
            ht = ht[:, (-1), :].unsqueeze(2)
        y = self.model(ht)
        return y


class AuxiliarSuperviser(object):

    def __init__(self, cmd_file, save_path='.'):
        self.cmd_file = cmd_file
        with open(cmd_file, 'r') as cmd_f:
            self.cmd = [l.rstrip() for l in cmd_f]
        self.save_path = save_path

    def __call__(self, iteration, ckpt_path, cfg_path):
        assert isinstance(iteration, int)
        assert isinstance(ckpt_path, str)
        assert isinstance(cfg_path, str)
        for cmd in self.cmd:
            sub_cmd = cmd.replace('$model', ckpt_path)
            sub_cmd = sub_cmd.replace('$iteration', str(iteration))
            sub_cmd = sub_cmd.replace('$cfg', cfg_path)
            sub_cmd = sub_cmd.replace('$save_path', self.save_path)
            None
            p = subprocess.Popen(sub_cmd, shell=True)


class PklWriter(object):

    def __init__(self, save_path):
        curr_time = datetime.now().strftime('%b%d_%H-%M-%S')
        fname = 'losses_{}.pkl'.format(curr_time)
        self.save_path = os.path.join(save_path, fname)
        self.losses = {}

    def add_scalar(self, tag, scalar_value, global_step=None):
        if tag not in self.losses:
            self.losses[tag] = {'global_step': [], 'scalar_value': []}
        if torch.is_tensor(scalar_value):
            scalar_value = scalar_value.item()
        self.losses[tag]['scalar_value'].append(scalar_value)
        self.losses[tag]['global_step'].append(global_step)
        with open(self.save_path, 'wb') as out_f:
            pickle.dump(self.losses, out_f)

    def add_histogram(self, tag, values, global_step=None, bins='sturges'):
        pass


class LogWriter(object):

    def __init__(self, save_path, log_types=['tensorboard', 'pkl']):
        self.save_path = save_path
        if len(log_types) == 0:
            raise ValueError('Please specify at least one log_type file to write to in the LogWriter!')
        self.writers = []
        for log_type in log_types:
            if 'tensorboard' == log_type:
                self.writers.append(SummaryWriter(save_path))
            elif 'pkl' == log_type:
                self.writers.append(PklWriter(save_path))
            else:
                raise TypeError('Unrecognized log_writer type: ', log_writer)

    def add_scalar(self, tag, scalar_value, global_step=None):
        for writer in self.writers:
            writer.add_scalar(tag, scalar_value=scalar_value, global_step=global_step)

    def add_histogram(self, tag, values, global_step=None, bins='sturges'):
        for writer in self.writers:
            writer.add_histogram(tag, values=values, global_step=global_step, bins=bins)


def get_padding(kwidth, dilation):
    return kwidth // 2 * dilation


class FeResBlock(NeuralBlock):

    def __init__(self, num_inputs, fmaps, kwidth, dilations=[1, 2], downsample=1, pad_mode='constant', act=None, norm_type=None, name='FeResBlock'):
        super().__init__(name=name)
        if act is not None and act == 'glu':
            Wfmaps = 2 * fmaps
        else:
            Wfmaps = fmaps
        self.num_inputs = num_inputs
        self.fmaps = fmaps
        self.kwidth = kwidth
        downscale = 1.0 / downsample
        self.downscale = downscale
        self.stride = 1
        dilation = dilations[0]
        self.conv1 = nn.Conv1d(num_inputs, Wfmaps, kwidth, dilation=dilation, padding=get_padding(kwidth, dilation))
        self.norm1 = build_norm_layer(norm_type, self.conv1, fmaps)
        self.act1 = build_activation(act, fmaps)
        dilation = dilations[1]
        self.conv2 = nn.Conv1d(fmaps, Wfmaps, kwidth, dilation=dilation, padding=get_padding(kwidth, dilation))
        self.norm2 = build_norm_layer(norm_type, self.conv2, fmaps)
        self.act2 = build_activation(act, fmaps)
        if self.num_inputs != self.fmaps:
            self.resproj = nn.Conv1d(self.num_inputs, self.fmaps, 1)

    def forward(self, x):
        """
        # compute pad factor
        if self.kwidth % 2 == 0:
            if self.dilation > 1:
                raise ValueError('Not supported dilation with even kwdith')
            P = (self.kwidth // 2 - 1,
                 self.kwidth // 2)
        else:
            pad = (self.kwidth // 2) * (self.dilation - 1) +                     (self.kwidth // 2)
            P = (pad, pad)
        """
        identity = x
        if self.downscale < 1:
            x = F.interpolate(x, scale_factor=self.downscale)
        x = self.conv1(x)
        x = forward_norm(x, self.norm1)
        x = forward_activation(self.act1, x)
        x = self.conv2(x)
        x = forward_activation(self.act2, x)
        if hasattr(self, 'resproj'):
            identity = self.resproj(identity)
        if self.downscale < 1:
            identity = F.interpolate(identity, scale_factor=self.downscale)
        x = x + identity
        x = forward_norm(x, self.norm2)
        return x


class WaveFe(Model):
    """ Convolutional front-end to process waveforms
        into a decimated intermediate representation 
    """

    def __init__(self, num_inputs=1, sincnet=True, kwidths=[251, 10, 5, 5, 5, 5, 5, 5], strides=[1, 10, 2, 1, 2, 1, 2, 2], dilations=[1, 1, 1, 1, 1, 1, 1, 1], fmaps=[64, 64, 128, 128, 256, 256, 512, 512], norm_type='bnorm', pad_mode='reflect', sr=16000, emb_dim=256, rnn_dim=None, activation=None, rnn_pool=False, rnn_layers=1, rnn_dropout=0, rnn_type='qrnn', vq_K=None, vq_beta=0.25, vq_gamma=0.99, norm_out=False, tanh_out=False, resblocks=False, denseskips=False, densemerge='sum', name='WaveFe'):
        super().__init__(name=name)
        self.sincnet = sincnet
        self.kwidths = kwidths
        self.strides = strides
        self.fmaps = fmaps
        self.densemerge = densemerge
        if denseskips:
            self.denseskips = nn.ModuleList()
        self.blocks = nn.ModuleList()
        assert len(kwidths) == len(strides)
        assert len(strides) == len(fmaps)
        concat_emb_dim = emb_dim
        ninp = num_inputs
        for n, (kwidth, stride, dilation, fmap) in enumerate(zip(kwidths, strides, dilations, fmaps), start=1):
            if n > 1:
                sincnet = False
            if resblocks and not sincnet:
                feblock = FeResBlock(ninp, fmap, kwidth, downsample=stride, act=activation, pad_mode=pad_mode, norm_type=norm_type)
            else:
                feblock = FeBlock(ninp, fmap, kwidth, stride, dilation, act=activation, pad_mode=pad_mode, norm_type=norm_type, sincnet=sincnet, sr=sr)
            self.blocks.append(feblock)
            if denseskips and n < len(kwidths):
                self.denseskips.append(nn.Conv1d(fmap, emb_dim, 1, bias=False))
                if densemerge == 'concat':
                    concat_emb_dim += emb_dim
            ninp = fmap
        if rnn_pool:
            if rnn_dim is None:
                rnn_dim = emb_dim
            self.rnn = build_rnn_block(fmap, rnn_dim // 2, rnn_layers=rnn_layers, rnn_type=rnn_type, bidirectional=True, dropout=rnn_dropout)
            self.W = nn.Conv1d(rnn_dim, emb_dim, 1)
        else:
            self.W = nn.Conv1d(fmap, emb_dim, 1)
        self.emb_dim = concat_emb_dim
        self.rnn_pool = rnn_pool
        if vq_K is not None and vq_K > 0:
            self.quantizer = VQEMA(vq_K, self.emb_dim, vq_beta, vq_gamma)
        else:
            self.quantizer = None
        if norm_out:
            if norm_type == 'bnorm':
                self.norm_out = nn.BatchNorm1d(self.emb_dim, affine=False)
            else:
                self.norm_out = nn.InstanceNorm1d(self.emb_dim)
        self.tanh_out = tanh_out

    def fuse_skip(self, input_, skip):
        dfactor = skip.shape[2] // input_.shape[2]
        if dfactor > 1:
            maxlen = input_.shape[2] * dfactor
            skip = skip[:, :, :maxlen]
            bsz, feats, slen = skip.shape
            skip_re = skip.view(bsz, feats, slen // dfactor, dfactor)
            skip = torch.mean(skip_re, dim=3)
        if self.densemerge == 'concat':
            return torch.cat((input_, skip), dim=1)
        elif self.densemerge == 'sum':
            return input_ + skip
        else:
            raise TypeError('Unknown densemerge: ', self.densemerge)

    def forward(self, batch, device=None, mode=None):
        x, data_fmt = format_frontend_chunk(batch, device)
        h = x
        denseskips = hasattr(self, 'denseskips')
        if denseskips:
            dskips = None
            dskips = []
        for n, block in enumerate(self.blocks):
            h = block(h)
            if denseskips and n + 1 < len(self.blocks):
                proj = self.denseskips[n]
                dskips.append(proj(h))
                """
                if dskips is None:
                    dskips = proj(h)
                else:
                    h_proj = proj(h)
                    dskips = self.fuse_skip(h_proj, dskips)
                """
        if self.rnn_pool:
            h = h.transpose(1, 2).transpose(0, 1)
            h, _ = self.rnn(h)
            h = h.transpose(0, 1).transpose(1, 2)
        y = self.W(h)
        if denseskips:
            for dskip in dskips:
                y = self.fuse_skip(y, dskip)
        if hasattr(self, 'norm_out'):
            y = self.norm_out(y)
        if self.tanh_out:
            y = torch.tanh(y)
        if self.quantizer is not None:
            qloss, y, pp, enc = self.quantizer(y)
            if self.training:
                return qloss, y, pp, enc
            else:
                return y
        return format_frontend_output(y, data_fmt, mode)


class ZAdversarialLoss(object):

    def __init__(self, z_gen=torch.randn, batch_acum=1, grad_reverse=False, loss='L2'):
        self.z_gen = z_gen
        self.batch_acum = batch_acum
        self.grad_reverse = grad_reverse
        self.loss = loss
        if loss == 'L2':
            self.criterion = nn.MSELoss()
        elif loss == 'BCE':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Unrecognized loss ', loss)

    def register_DNet(self, Dnet):
        self.Dnet = Dnet

    def forward_grad_reverse(self, step, fake, optim, real, true_lab, fake_lab):
        dreal = self.Dnet(real)
        dreal_loss = self.criterion(dreal, true_lab)
        dfake = self.Dnet(fake)
        dfake_loss = self.criterion(dfake, fake_lab)
        d_loss = dreal_loss + dfake_loss
        d_loss.backward(retain_graph=True)
        if step % self.batch_acum == 0:
            optim.step()
            optim.zero_grad()
        return {'afake_loss': dfake_loss, 'areal_loss': dreal_loss}

    def forward_alternate(self, step, fake, optim, real, true_lab, fake_lab, gfake_exists=False):
        dreal = self.Dnet(real.detach())
        dreal_loss = self.criterion(dreal, true_lab)
        dfake = self.Dnet(fake.detach())
        dfake_loss = self.criterion(dfake, fake_lab)
        d_loss = dreal_loss + dfake_loss
        d_loss.backward()
        if step % self.batch_acum == 0:
            optim.step()
            optim.zero_grad()
        greal = self.Dnet(fake)
        greal_loss = self.criterion(greal, true_lab)
        ret_losses = {'dfake_loss': dfake_loss, 'dreal_loss': dreal_loss, 'd_loss': d_loss, 'greal_loss': greal_loss}
        if gfake_exists:
            gfake = self.Dnet(real)
            gfake_loss = self.criterion(gfake, fake_lab)
            g_loss = greal_loss + gfake_loss
            ret_losses['gfake_loss'] = gfake_loss
        else:
            g_loss = greal_loss
        ret_losses['g_loss'] = g_loss
        return ret_losses

    def __call__(self, step, fake, optim, z_true=None, z_true_trainable=False):
        if not hasattr(self, 'Dnet'):
            raise ValueError('Please register Dnet first prior to using L2Adversarial Loss.')
        if z_true is None:
            real = self.z_gen(fake.size())
        else:
            real = z_true
        lab_1 = torch.ones(real.shape[0], 1, real.shape[2])
        lab_0 = torch.zeros(lab_1.shape)
        if fake.is_cuda:
            real = real
            lab_1 = lab_1
            lab_0 = lab_0
        if self.grad_reverse:
            losses = self.forward_grad_reverse(step, fake, optim, z_true, lab_1, lab_0)
        else:
            losses = self.forward_alternate(step, fake, optim, z_true, lab_1, lab_0, z_true_trainable)
        return losses


def get_grad_norms(model, keys=[]):
    grads = {}
    for i, (k, param) in enumerate(dict(model.named_parameters()).items()):
        accept = False
        for key in keys:
            if key in k:
                accept = True
                break
        if not accept:
            continue
        if param.grad is None:
            None
            continue
        grads[k] = torch.norm(param.grad).cpu().item()
    return grads


class Waveminionet(Model):

    def __init__(self, frontend=None, frontend_cfg=None, minions_cfg=None, z_minion=True, z_cfg=None, adv_loss='BCE', num_devices=1, pretrained_ckpts=None, name='Waveminionet'):
        super().__init__(name=name)
        if minions_cfg is None or len(minions_cfg) < 1:
            raise ValueError('Please specify a stack of minions config with at least 1 minion. GIMME SOMETHING TO DO.')
        if frontend is not None:
            self.frontend = frontend
        elif frontend_cfg is None:
            self.frontend = WaveFe()
        else:
            self.frontend = WaveFe(**frontend_cfg)
        if self.frontend.quantizer is not None:
            self.vq = True
        else:
            self.vq = False
        self.minions = nn.ModuleList()
        self.mi_fwd = False
        ninp = self.frontend.emb_dim
        self.min2idx = {}
        for minion_cfg in minions_cfg:
            if 'mi' in minion_cfg['name'] and not self.mi_fwd:
                ninp += self.frontend.emb_dim
            minion_cfg['num_inputs'] = ninp
            minion = minion_maker(minion_cfg)
            self.minions.append(minion)
            self.min2idx[minion.name] = len(self.min2idx)
            if hasattr(minion, 'skip') and minion.skip:
                nouts = minion.hidden_size
                ninp += nouts
            if 'mi' in minion.name:
                self.mi_fwd = True
        if z_minion:
            if z_cfg is None:
                z_cfg = {'num_inputs': self.frontend.emb_dim, 'num_outputs': 1, 'hidden_layers': 3, 'hidden_size': 1024, 'norm_type': 'bnorm', 'dropout': 0.0, 'kwidths': [31, 11, 5], 'name': 'z', 'grad_reverse': False, 'skip': False}
            self.z_cfg = z_cfg
            self.z_adv_loss = adv_loss
        if pretrained_ckpts is not None:
            self.load_checkpoints(pretrained_ckpts)
        if num_devices > 1:
            self.frontend_dp = nn.DataParallel(self.frontend)
            self.minions_dp = nn.ModuleList([nn.DataParallel(m) for m in self.minions])

    def build_z_minion(self, cfg):
        None
        device = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        self.z_cfg['loss'] = ZAdversarialLoss(loss=self.z_adv_loss, batch_acum=cfg['batch_acum'])
        self.z_minion = minion_maker(self.z_cfg)
        self.z_minion.loss.register_DNet(self.z_minion)
        self.z_minion

    def forward(self, x):
        raise NotImplementedError
        fe_h = self.frontend(x)
        h = fe_h
        outs = {}
        for mi, minion in enumerate(self.minions, start=1):
            y, h_ = minion(h)
            if minion.skip:
                h_c = torch.cat((h, h_), dim=1)
                h = h_c
            else:
                h = h
            outs[minion.name] = y
        return outs, h

    def join_skip(self, x, skip):
        if skip is None:
            return x
        else:
            return torch.cat((x, skip), dim=1)

    def load_checkpoints(self, load_path):
        savers = [Saver(self.frontend, load_path, prefix='PASE-')]
        if hasattr(self, 'z_minion'):
            savers.append(Saver(self.z_minion, load_path, prefix='Zminion-'))
        for mi, minion in enumerate(self.minions, start=1):
            savers.append(Saver(minion, load_path, prefix='M-{}-'.format(minion.name)))
        giters = 0
        for saver in savers:
            try:
                state = saver.read_latest_checkpoint()
                giter_ = saver.load_ckpt_step(state)
                None
                if giters == 0:
                    giters = giter_
                else:
                    assert giters == giter_, giter_
                saver.load_pretrained_ckpt(os.path.join(load_path, 'weights_' + state), load_last=True)
            except TypeError:
                break

    def forward_chunk(self, frontend, batch, chunk_name, device):
        if self.vq:
            vq_loss, fe_Q, vq_pp, vq_idx = frontend(batch[chunk_name])
            return fe_Q
        else:
            return frontend(batch[chunk_name])

    def train_(self, dloader, cfg, device='cpu', va_dloader=None):
        epoch = cfg['epoch']
        bsize = cfg['batch_size']
        batch_acum = cfg['batch_acum']
        save_path = cfg['save_path']
        log_freq = cfg['log_freq']
        sup_freq = cfg['sup_freq']
        grad_keys = cfg['log_grad_keys']
        if cfg['sup_exec'] is not None:
            aux_save_path = os.path.join(cfg['save_path'], 'sup_aux')
            if not os.path.exists(aux_save_path):
                os.makedirs(aux_save_path)
            self.aux_sup = AuxiliarSuperviser(cfg['sup_exec'], aux_save_path)
        warmup_epoch = cfg['warmup']
        zinit_weight = cfg['zinit_weight']
        zinc = cfg['zinc']
        zweight = 0
        if hasattr(self, 'frontend_dp'):
            frontend = self.frontend_dp
        else:
            frontend = self.frontend
        self.build_z_minion(cfg)
        writer = LogWriter(save_path, log_types=cfg['log_types'])
        bpe = cfg['bpe'] if 'bpe' in cfg else len(dloader)
        None
        None
        None
        rndmin_train = cfg['rndmin_train']
        None
        feopt = getattr(optim, cfg['fe_opt'])(self.frontend.parameters(), lr=cfg['fe_lr'])
        savers = [Saver(self.frontend, save_path, max_ckpts=cfg['max_ckpts'], optimizer=feopt, prefix='PASE-')]
        lrdecay = cfg['lrdecay']
        if lrdecay > 0:
            fesched = optim.lr_scheduler.StepLR(feopt, step_size=cfg['lrdec_step'], gamma=cfg['lrdecay'])
        if hasattr(self, 'z_minion'):
            z_lr = cfg['z_lr']
            zopt = getattr(optim, cfg['min_opt'])(self.z_minion.parameters(), lr=z_lr)
            if lrdecay > 0:
                zsched = optim.lr_scheduler.StepLR(zopt, step_size=cfg['lrdec_step'], gamma=cfg['lrdecay'])
            savers.append(Saver(self.z_minion, save_path, max_ckpts=cfg['max_ckpts'], optimizer=zopt, prefix='Zminion-'))
        None
        if 'min_lrs' in cfg:
            min_lrs = cfg['min_lrs']
        else:
            min_lrs = None
        minopts = {}
        minscheds = {}
        for mi, minion in enumerate(self.minions, start=1):
            min_opt = cfg['min_opt']
            min_lr = cfg['min_lr']
            if min_lrs is not None and minion.name in min_lrs:
                min_lr = min_lrs[minion.name]
                None
            minopts[minion.name] = getattr(optim, min_opt)(minion.parameters(), lr=min_lr)
            if lrdecay > 0:
                minsched = lr_scheduler.StepLR(minopts[minion.name], step_size=cfg['lrdec_step'], gamma=cfg['lrdecay'])
                minscheds[minion.name] = minsched
            savers.append(Saver(minion, save_path, max_ckpts=cfg['max_ckpts'], optimizer=minopts[minion.name], prefix='M-{}-'.format(minion.name)))
        minions_run = self.minions
        if hasattr(self, 'minions_dp'):
            minions_run = self.minions_dp
        min_global_steps = {}
        if cfg['ckpt_continue']:
            giters = 0
            for saver in savers:
                try:
                    state = saver.read_latest_checkpoint()
                    giter_ = saver.load_ckpt_step(state)
                    None
                    if giters == 0:
                        giters = giter_
                    else:
                        assert giters == giter_, giter_
                    saver.load_pretrained_ckpt(os.path.join(save_path, 'weights_' + state), load_last=True)
                except TypeError:
                    break
            global_step = giters
            epoch_beg = int(global_step / bpe)
            epoch = epoch - epoch_beg
        else:
            epoch_beg = 0
            global_step = 0
        z_losses = None
        None
        None
        for epoch_ in range(epoch_beg, epoch_beg + epoch):
            self.train()
            timings = []
            beg_t = timeit.default_timer()
            min_loss = {}
            if epoch_ + 1 == warmup_epoch and hasattr(self, 'z_minion'):
                zweight = zinit_weight
            iterator = iter(dloader)
            for bidx in range(1, bpe + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dloader)
                    batch = next(iterator)
                fe_h = {}
                fe_forwards = [batch['chunk']]
                if self.mi_fwd:
                    fe_forwards.extend([batch['chunk_ctxt'], batch['chunk_rand']])
                fe_forwards.append(batch['cchunk'])
                fe_forwards_b = torch.cat(fe_forwards, dim=0)
                if self.vq:
                    vq_loss, fe_Q, vq_pp, vq_idx = frontend(fe_forwards_b)
                    fe_h['all'] = fe_Q
                else:
                    fe_h['all'] = frontend(fe_forwards_b)
                all_feh = torch.chunk(fe_h['all'], len(fe_forwards), dim=0)
                fe_h['chunk'] = all_feh[0]
                fe_h['cchunk'] = all_feh[-1]
                min_h = {}
                h = fe_h['chunk']
                skip_acum = None
                for mi, minion in enumerate(minions_run, start=1):
                    min_name = self.minions[mi - 1].name
                    if 'mi' in min_name:
                        triplet_P = self.join_skip(torch.cat((all_feh[0], all_feh[1]), dim=1), skip_acum)
                        triplet_N = self.join_skip(torch.cat((all_feh[0], all_feh[2]), dim=1), skip_acum)
                        triplet_all = torch.cat((triplet_P, triplet_N), dim=0)
                        if min_name == 'cmi':
                            triplet_all = torch.mean(triplet_all, dim=2, keepdim=True)
                        y = minion(triplet_all)
                        bsz = y.size(0) // 2
                        slen = y.size(2)
                        batch[min_name] = torch.cat((torch.ones(bsz, 1, slen), torch.zeros(bsz, 1, slen)), dim=0)
                    else:
                        if self.minions[mi - 1].skip:
                            y, h_ = minion(self.join_skip(h, skip_acum))
                            if skip_acum is None:
                                skip_acum = h_
                            else:
                                skip_acum = torch.cat((skip_acum, h_), dim=1)
                        else:
                            y = minion(self.join_skip(h, skip_acum))
                        if min_name == 'spc':
                            bsz = y.size(0) // 2
                            slen = y.size(2)
                            batch['spc'] = torch.cat((torch.ones(bsz, 1, slen), torch.zeros(bsz, 1, slen)), dim=0)
                    min_h[min_name] = y
                if epoch_ + 1 >= warmup_epoch and hasattr(self, 'z_minion'):
                    if cfg['cchunk_prior']:
                        z_real = fe_h['cchunk']
                        z_true_trainable = True
                    else:
                        z_real = None
                        z_true_trainable = False
                    z_losses = self.z_minion.loss(global_step, fe_h['chunk'], zopt, z_true=z_real, z_true_trainable=z_true_trainable)
                    zweight = min(1, zweight + zinc)
                global_step += 1
                t_loss = torch.zeros(1)
                if z_losses is not None and 'g_loss' in z_losses:
                    t_loss += z_losses['g_loss']
                if self.vq:
                    t_loss += vq_loss
                if rndmin_train:
                    if rnd_min not in min_global_steps:
                        min_global_steps[rnd_min] = 0
                    min_names = list(min_h.keys())
                    rnd_min = random.choice(min_names)
                    y_ = min_h[rnd_min]
                    minion = minions_run[self.min2idx[rnd_min]]
                    y_lab = batch[rnd_min]
                    lweight = minion.loss_weight
                    if isinstance(minion.loss, WaveAdversarialLoss):
                        loss = minion.loss(min_global_steps[rnd_min], y_, y_lab, c_real=fe_h['chunk'])
                        d_real_loss = loss['d_real_loss']
                        d_fake_loss = loss['d_fake_loss']
                        if not '{}_Dreal'.format(rnd_min) in min_loss:
                            min_loss['{}_Dreal'.format(rnd_min)] = []
                            min_loss['{}_Dfake'.format(rnd_min)] = []
                        if not '{}_Dreal'.format(rnd_min) in min_global_steps:
                            min_global_steps['{}_Dreal'.format(rnd_min)] = 0
                            min_global_steps['{}_Dfake'.format(rnd_min)] = 0
                        min_loss['{}_Dreal'.format(rnd_min)] = d_real_loss.item()
                        min_loss['{}_Dfake'.format(rnd_min)] = d_fake_loss.item()
                        loss = loss['g_loss']
                        loss = lweight * loss
                        loss.backward()
                    else:
                        loss = minion.loss(y_, y_lab)
                        loss = lweight * loss
                        loss.backward()
                    if rnd_min not in min_loss:
                        min_loss[rnd_min] = []
                    min_loss[rnd_min].append(loss.item())
                    min_global_steps[rnd_min] += 1
                    if '{}_Dreal'.format(rnd_min) in min_global_steps:
                        min_global_steps['{}_Dreal'.format(rnd_min)] += 1
                        min_global_steps['{}_Dfake'.format(rnd_min)] += 1
                else:
                    for min_name, y_ in min_h.items():
                        if min_name not in min_global_steps:
                            min_global_steps[min_name] = 0
                        minion = minions_run[self.min2idx[min_name]]
                        minopts[min_name].zero_grad()
                        y_lab = batch[min_name]
                        lweight = minion.loss_weight
                        if isinstance(minion.loss, WaveAdversarialLoss):
                            loss = minion.loss(min_global_steps[min_name], y_, y_lab, c_real=fe_h['chunk'])
                            d_real_loss = loss['d_real_loss']
                            d_fake_loss = loss['d_fake_loss']
                            if not '{}_Dreal'.format(min_name) in min_loss:
                                min_loss['{}_Dreal'.format(min_name)] = []
                                min_loss['{}_Dfake'.format(min_name)] = []
                            if not '{}_Dreal'.format(min_name) in min_global_steps:
                                min_global_steps['{}_Dreal'.format(min_name)] = 0
                                min_global_steps['{}_Dfake'.format(min_name)] = 0
                            min_loss['{}_Dreal'.format(min_name)].append(d_real_loss.item())
                            min_loss['{}_Dfake'.format(min_name)].append(d_fake_loss.item())
                            loss = loss['g_loss']
                            loss = lweight * loss
                        else:
                            loss = minion.loss(y_, y_lab)
                            loss = lweight * loss
                        t_loss += loss
                        if min_name not in min_loss:
                            min_loss[min_name] = []
                        min_loss[min_name].append(loss.item())
                        min_global_steps[min_name] += 1
                        if '{}_Dreal'.format(min_name) in min_global_steps:
                            min_global_steps['{}_Dreal'.format(min_name)] += 1
                            min_global_steps['{}_Dfake'.format(min_name)] += 1
                    t_loss.backward()
                if bidx % batch_acum == 0 or bidx >= bpe:
                    grads = get_grad_norms(self, grad_keys)
                    for min_name, y_ in min_h.items():
                        minopts[min_name].step()
                        minopts[min_name].zero_grad()
                    feopt.step()
                    feopt.zero_grad()
                    if epoch_ + 1 >= warmup_epoch and hasattr(self, 'z_minion'):
                        zopt.step()
                        zopt.zero_grad()
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if bidx % log_freq == 0 or bidx >= bpe:
                    None
                    None
                    for min_name, losses in min_loss.items():
                        None
                        writer.add_scalar('train/{}_loss'.format(min_name), losses[-1], min_global_steps[min_name])
                        if min_name in min_h:
                            writer.add_histogram('train/{}'.format(min_name), min_h[min_name].data, bins='sturges', global_step=min_global_steps[min_name])
                        if min_name in min_h:
                            writer.add_histogram('train/gtruth_{}'.format(min_name), batch[min_name].data, bins='sturges', global_step=min_global_steps[min_name])
                    if z_losses is not None:
                        z_log = 'ZMinion '
                        if 'dfake_loss' in z_losses:
                            dfake_loss = z_losses['dfake_loss'].item()
                            z_log += 'dfake_loss: {:.3f},'.format(dfake_loss)
                            writer.add_scalar('train/dfake_loss', dfake_loss, global_step)
                        if 'dreal_loss' in z_losses:
                            dreal_loss = z_losses['dreal_loss'].item()
                            writer.add_scalar('train/dreal_loss', dreal_loss, global_step)
                            z_log += ' dreal_loss: {:.3f},'.format(dreal_loss)
                        if 'greal_loss' in z_losses:
                            greal_loss = z_losses['greal_loss'].item()
                            z_log += ', greal_loss: {:.3f},'.format(greal_loss)
                            writer.add_scalar('train/greal_loss', greal_loss, global_step)
                        if 'gfake_loss' in z_losses:
                            gfake_loss = z_losses['gfake_loss'].item()
                            z_log += ', gfake_loss: {:.3f},'.format(gfake_loss)
                            writer.add_scalar('train/gfake_loss', gfake_loss, global_step)
                        None
                        if z_true_trainable:
                            writer.add_histogram('train/z_real', fe_h['cchunk'], bins='sturges', global_step=global_step)
                        writer.add_histogram('train/z_fake', fe_h['chunk'], bins='sturges', global_step=global_step)
                    if self.vq:
                        None
                        writer.add_scalar('train/vq_loss', vq_loss.item(), global_step=global_step)
                        writer.add_scalar('train/vq_pp', vq_pp.item(), global_step=global_step)
                    for kgrad, vgrad in grads.items():
                        writer.add_scalar('train/GRAD/{}'.format(kgrad), vgrad, global_step)
                    None
                    None
            if va_dloader is not None:
                va_bpe = cfg['va_bpe']
                eloss = self.eval_(va_dloader, bsize, va_bpe, log_freq=log_freq, epoch_idx=epoch_, writer=writer, device=device)
                """
                if lrdecay > 0:
                    # update frontend lr
                    fesched.step(eloss)
                    # update Z minion lr
                    if hasattr(self, 'z_minion'):
                        zsched.step(eloss)
                    # update each minion lr
                    for mi, minion in enumerate(self.minions, start=1):
                        minscheds[minion.name].step(eloss)
                """
            if lrdecay > 0:
                fesched.step()
                if hasattr(self, 'z_minion'):
                    zsched.step()
                for mi, minion in enumerate(self.minions, start=1):
                    minscheds[minion.name].step()
            fe_path = os.path.join(save_path, 'FE_e{}.ckpt'.format(epoch_))
            torch.save(self.frontend.state_dict(), fe_path)
            for saver in savers:
                saver.save(saver.prefix[:-1], global_step)
            if (epoch_ + 1) % sup_freq == 0 or epoch_ + 1 >= epoch_beg + epoch:
                if hasattr(self, 'aux_sup'):
                    self.aux_sup(epoch_, fe_path, cfg['fe_cfg'])

    def eval_(self, dloader, batch_size, bpe, log_freq, epoch_idx=0, writer=None, device='cpu'):
        self.eval()
        with torch.no_grad():
            bsize = batch_size
            frontend = self.frontend
            minions_run = self.minions
            None
            None
            timings = []
            beg_t = timeit.default_timer()
            min_loss = {}
            iterator = iter(dloader)
            for bidx in range(1, bpe + 1):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(dloader)
                    batch = next(iterator)
                chunk_keys = ['chunk']
                if self.mi_fwd:
                    chunk_keys += ['chunk_ctxt', 'chunk_rand']
                fe_h = {}
                for k in chunk_keys:
                    fe_h[k] = frontend(batch[k])
                min_h = {}
                h = fe_h['chunk']
                skip_acum = None
                for mi, minion in enumerate(minions_run, start=1):
                    min_name = self.minions[mi - 1].name
                    if 'mi' in min_name:
                        triplet_P = self.join_skip(torch.cat((fe_h['chunk'], fe_h['chunk_ctxt']), dim=1), skip_acum)
                        triplet_N = self.join_skip(torch.cat((fe_h['chunk'], fe_h['chunk_rand']), dim=1), skip_acum)
                        triplet_all = torch.cat((triplet_P, triplet_N), dim=0)
                        if min_name == 'cmi':
                            triplet_all = torch.mean(triplet_all, dim=2, keepdim=True)
                        y = minion(triplet_all)
                        bsz = y.size(0) // 2
                        slen = y.size(2)
                        batch[min_name] = torch.cat((torch.ones(bsz, 1, slen), torch.zeros(bsz, 1, slen)), dim=0)
                    else:
                        if self.minions[mi - 1].skip:
                            y, h_ = minion(self.join_skip(h, skip_acum))
                            if skip_acum is None:
                                skip_acum = h_
                            else:
                                skip_acum = torch.cat((skip_acum, h_), dim=1)
                        else:
                            y = minion(self.join_skip(h, skip_acum))
                        if min_name == 'spc':
                            bsz = y.size(0) // 2
                            slen = y.size(2)
                            batch['spc'] = torch.cat((torch.ones(bsz, 1, slen), torch.zeros(bsz, 1, slen)), dim=0)
                    min_h[min_name] = y
                for min_name, y_ in min_h.items():
                    y_lab = batch[min_name]
                    minion = self.minions[self.min2idx[min_name]]
                    lweight = minion.loss_weight
                    if isinstance(minion.loss, WaveAdversarialLoss):
                        loss = minion.loss(bidx, y_, y_lab, c_real=fe_h['chunk'], grad=False)
                        loss = loss['g_loss']
                    else:
                        loss = minion.loss(y_, y_lab)
                    loss = lweight * loss
                    if min_name not in min_loss:
                        min_loss[min_name] = []
                    min_loss[min_name].append(loss.item())
                end_t = timeit.default_timer()
                timings.append(end_t - beg_t)
                beg_t = timeit.default_timer()
                if bidx % log_freq == 0 or bidx >= bpe:
                    None
                    None
                    for min_name, losses in min_loss.items():
                        None
                    None
            aggregate = 0
            for min_name, losses in min_loss.items():
                mlosses = np.mean(losses)
                writer.add_scalar('eval/{}_loss'.format(min_name), mlosses, epoch_idx)
                aggregate += mlosses
            writer.add_scalar('eval/total_loss', aggregate, epoch_idx)
            return aggregate

    def state_dict(self):
        sdict = {}
        for k, v in super().state_dict().items():
            if '_dp.' in k:
                continue
            sdict[k] = v
        return sdict


class SpectrogramDecoder(Model):

    def __init__(self, num_inputs, nfft=1024, strides=[1, 1, 1], kwidths=[3, 3, 3], fmaps=[256, 256, 256], norm_type=None, name='SpectrogramDecoder'):
        super().__init__(name=name)
        ninp = num_inputs
        self.dec = nn.ModuleList()
        for di, (kwidth, stride, fmap) in enumerate(zip(kwidths, strides, fmaps), start=1):
            if stride > 1:
                self.dec.append(GDeconv1DBlock(ninp, fmap, kwidth, stride, norm_type=norm_type))
            else:
                self.dec.append(GConv1DBlock(ninp, fmap, kwidth, 1, norm_type=norm_type))
            ninp = fmap
        self.dec.append(nn.Conv1d(ninp, nfft // 2 + 1, 1))

    def forward(self, x):
        for dec in self.dec:
            x = dec(x)
        return x


class WaveDiscriminator(nn.Module):

    def __init__(self, ninputs=1, fmaps=[128, 128, 256, 256, 512, 100], strides=[10, 4, 4, 1, 1, 1], kwidths=[30, 30, 30, 3, 3, 3], norm_type='snorm'):
        super().__init__()
        self.aco_decimator = nn.ModuleList()
        ninp = ninputs
        for fmap, kwidth, stride in zip(fmaps, kwidths, strides):
            self.aco_decimator.append(GConv1DBlock(ninp, fmap, kwidth, stride, norm_type=norm_type))
            ninp = fmap
        self.out_fc = nn.Conv1d(fmaps[-1], 1, 1)
        if norm_type == 'snorm':
            nn.utils.spectral_norm(self.out_fc)
        self.norm_type = norm_type

    def build_conditionW(self, cond):
        if cond is not None:
            cond_dim = cond.size(1)
            if not hasattr(self, 'proj_W'):
                self.proj_W = nn.Linear(cond_dim, cond_dim, bias=False)
                if self.norm_type == 'snorm':
                    nn.utils.spectral_norm(self.proj_W)
                if cond.is_cuda:
                    self.proj_W

    def forward(self, x, cond=None):
        self.build_conditionW(cond)
        h = x
        for di in range(len(self.aco_decimator)):
            dec_layer = self.aco_decimator[di]
            h = dec_layer(h)
        bsz, nfeats, slen = h.size()
        if cond is not None:
            cond = torch.mean(cond, dim=2)
            cond = self.proj_W(cond)
            h = torch.mean(h, dim=2)
            h = h.view(-1, nfeats)
            cond = cond.view(-1, nfeats)
            cls = torch.bmm(h.unsqueeze(1), cond.unsqueeze(2)).squeeze(2)
            cls = cls.view(bsz, 1)
        y = self.out_fc(h.unsqueeze(2)).squeeze(2)
        y = y + cls
        return y.squeeze(1)


class StatisticalPooling(nn.Module):

    def forward(self, x):
        mu = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return torch.cat((mu, std), dim=1)


class TDNN(Model):

    def __init__(self, num_inputs, num_outputs, method='cls', name='TDNN'):
        super().__init__(name=name)
        self.method = method
        self.model = nn.Sequential(nn.Conv1d(num_inputs, 512, 5, padding=2), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, 512, 3, dilation=2, padding=2), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, 512, 3, dilation=3, padding=3), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, 1500, 1), nn.BatchNorm1d(1500), nn.ReLU(inplace=True), StatisticalPooling(), nn.Conv1d(3000, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Conv1d(512, num_outputs, 1), nn.LogSoftmax(dim=1))
        if method == 'cls':
            None
        elif method == 'xvector':
            self.model = nn.Sequential(*list(self.model.children())[:-5])
            None
        elif method == 'unpooled':
            self.model = nn.Sequential(*list(self.model.children())[:-9])
            None
        else:
            raise TypeError('Unrecognized TDNN method: ', method)
        self.emb_dim = 1500

    def forward(self, x):
        return self.model(x)

    def load_pretrained(self, ckpt_path, verbose=True):
        if self.method != 'cls':
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            sdict = ckpt['state_dict']
            curr_keys = list(dict(self.named_parameters()).keys())
            del_keys = [k for k in sdict.keys() if k not in curr_keys]
            for k in del_keys:
                del sdict[k]
            self.load_state_dict(sdict)
        else:
            super().load_pretrained(ckpt_path, load_last=True, verbose=verbose)


class TDNNFe(Model):
    """ Time-Delayed Neural Network front-end
    """

    def __init__(self, num_inputs=1, sincnet=True, kwidth=641, stride=160, fmaps=128, norm_type='bnorm', pad_mode='reflect', sr=16000, emb_dim=256, activation=None, rnn_pool=False, rnn_layers=1, rnn_dropout=0, rnn_type='qrnn', name='TDNNFe'):
        super().__init__(name=name)
        self.sincnet = sincnet
        self.emb_dim = emb_dim
        ninp = num_inputs
        if self.sincnet:
            self.feblock = FeBlock(ninp, fmaps, kwidth, stride, 1, act=activation, pad_mode=pad_mode, norm_type=norm_type, sincnet=True, sr=sr)
            ninp = fmaps
        self.tdnn = TDNN(ninp, 2, method='unpooled')
        fmap = self.tdnn.emb_dim
        if rnn_pool:
            self.rnn = build_rnn_block(fmap, emb_dim // 2, rnn_layers=rnn_layers, rnn_type=rnn_type, bidirectional=True, dropout=rnn_dropout)
            self.W = nn.Conv1d(emb_dim, emb_dim, 1)
        else:
            self.W = nn.Conv1d(fmap, emb_dim, 1)
        self.rnn_pool = rnn_pool

    def forward(self, batch, device=None, mode=None):
        x, data_fmt = format_frontend_chunk(batch, device)
        if hasattr(self, 'feblock'):
            h = self.feblock(x)
        h = self.tdnn(h)
        if self.rnn_pool:
            h = h.transpose(1, 2).transpose(0, 1)
            h, _ = self.rnn(h)
            h = h.transpose(0, 1).transpose(1, 2)
        y = self.W(h)
        return format_frontend_output(y, data_fmt, mode)
        """
        if self.training:
            if batched:
                embedding = torch.chunk(y, 3, dim=0)
                chunk = embedding[0]
            else:
                chunk = y
            return embedding, chunk
        else:
            return select_output(h, mode=mode)
        """


class Resnet50_encoder(Model):

    def __init__(self, sinc_out, hidden_dim, sinc_kernel=251, sinc_stride=1, conv_stride=5, kernel_size=21, pretrained=True, name='Resnet50'):
        super().__init__(name=name)
        self.sinc = SincConv_fast(1, sinc_out, sinc_kernel, sample_rate=16000, padding='SAME', stride=sinc_stride, pad_mode='reflect')
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=kernel_size, stride=conv_stride, padding=kernel_size // 2, bias=False), nn.BatchNorm2d(64), nn.ReLU(64))
        resnet = models.resnet34(pretrained=pretrained)
        self.resnet = nn.Sequential(resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.conv2 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=[2, 1], stride=1, bias=False))
        self.emb_dim = hidden_dim

    def forward(self, batch, device=None, mode=None):
        x, data_fmt = format_frontend_chunk(batch, device)
        sinc_out = self.sinc(x).unsqueeze(1)
        conv_out = self.conv1(sinc_out)
        res_out = self.resnet(conv_out)
        h = self.conv2(res_out).squeeze(2)
        return format_frontend_output(h, data_fmt, mode)


class ResDilatedModule(NeuralBlock):

    def __init__(self, ninp, fmaps, res_fmaps, kwidth, dilation, norm_type=None, act=None, causal=False, name='ResDilatedModule'):
        super().__init__(name=name)
        assert kwidth % 2 != 0
        self.causal = causal
        self.dil_conv = nn.Conv1d(ninp, fmaps, kwidth, dilation=dilation)
        if act is not None:
            self.act = getattr(nn, act)()
        else:
            self.act = nn.PReLU(fmaps, init=0)
        self.dil_norm = build_norm_layer(norm_type, self.dil_conv, fmaps)
        self.kwidth = kwidth
        self.dilation = dilation
        self.conv_1x1_skip = nn.Conv1d(fmaps, ninp, 1)
        self.conv_1x1_skip_norm = build_norm_layer(norm_type, self.conv_1x1_skip, ninp)
        self.conv_1x1_res = nn.Conv1d(fmaps, res_fmaps, 1)
        self.conv_1x1_res_norm = build_norm_layer(norm_type, self.conv_1x1_res, res_fmaps)

    def forward(self, x):
        if self.causal:
            kw__1 = self.kwidth - 1
            P = kw__1 + kw__1 * (self.dilation - 1)
            x_p = F.pad(x, (P, 0))
        else:
            kw_2 = self.kwidth // 2
            P = kw_2 * self.dilation
            x_p = F.pad(x, (P, P))
        h = self.dil_conv(x_p)
        h = forward_norm(h, self.dil_norm)
        h = self.act(h)
        a = h
        h = self.conv_1x1_skip(h)
        h = forward_norm(h, self.conv_1x1_skip_norm)
        y = x + h
        sh = self.conv_1x1_res(a)
        sh = forward_norm(sh, self.conv_1x1_res_norm)
        return y, sh


def cls_worker_maker(cfg, emb_dim):
    None
    None
    None
    if cfg['name'] == 'mi':
        return LIM(cfg, emb_dim)
    elif cfg['name'] == 'cmi':
        return GIM(cfg, emb_dim)
    elif cfg['name'] == 'spc':
        return SPC(cfg, emb_dim)
    elif cfg['name'] == 'gap':
        return Gap(cfg, emb_dim)
    else:
        return minion_maker(cfg)


def wf_builder(cfg_path):
    if cfg_path is not None:
        if isinstance(cfg_path, str):
            with open(cfg_path, 'r') as cfg_f:
                cfg = json.load(cfg_f)
                return wf_builder(cfg)
        elif isinstance(cfg_path, dict):
            if 'name' in cfg_path.keys():
                model_name = cfg_path['name']
                if cfg_path['name'] == 'asppRes':
                    return aspp_res_encoder(**cfg_path)
                elif model_name == 'Resnet50':
                    return Resnet50_encoder(**cfg_path)
                elif model_name == 'tdnn':
                    return TDNNFe(**cfg_path)
                else:
                    raise TypeError('Unrecognized frontend type: ', model_name)
            else:
                return WaveFe(**cfg_path)
        else:
            TypeError('Unexpected config for WaveFe')
    else:
        raise ValueError('cfg cannot be None!')


class pase_attention(Model):

    def __init__(self, frontend=None, frontend_cfg=None, att_cfg=None, minions_cfg=None, cls_lst=['mi', 'cmi', 'spc'], regr_lst=['chunk', 'lps', 'mfcc', 'prosody'], adv_lst=[], K=40, att_mode='concat', avg_factor=0, chunk_size=16000, pretrained_ckpt=None, name='adversarial'):
        super().__init__(name=name)
        if minions_cfg is None or len(minions_cfg) < 1:
            raise ValueError('Please specify a stack of minions config with at least 1 minion. GIMME SOMETHING TO DO.')
        None
        self.frontend = wf_builder(frontend_cfg)
        self.cls_lst = cls_lst
        self.reg_lst = regr_lst
        self.adv_lst = adv_lst
        ninp = self.frontend.emb_dim
        self.regression_workers = nn.ModuleList()
        self.classification_workers = nn.ModuleList()
        self.adversarial_workers = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        att_cfg['dnn_lay'] += ',' + str(ninp)
        for type, cfg_lst in minions_cfg.items():
            for cfg in cfg_lst:
                if type == 'cls':
                    cfg['num_inputs'] = ninp
                    self.classification_workers.append(cls_worker_maker(cfg, ninp))
                    self.attention_blocks.append(attention_block(ninp, cfg['name'], att_cfg, K, frontend_cfg['strides'], chunk_size, avg_factor, att_mode))
                elif type == 'regr':
                    cfg['num_inputs'] = ninp
                    minion = minion_maker(cfg)
                    self.regression_workers.append(minion)
                    self.attention_blocks.append(attention_block(ninp, cfg['name'], att_cfg, K, frontend_cfg['strides'], chunk_size, avg_factor, att_mode))
                elif type == 'adv':
                    cfg['num_inputs'] = ninp
                    minion = minion_maker(cfg)
                    self.adversarial_workers.append(minion)
                    self.attention_blocks.append(attention_block(ninp, cfg['name'], att_cfg, K, frontend_cfg['strides'], chunk_size, avg_factor, att_mode))
                else:
                    raise TypeError('Unrecognized worker type: ', type)
        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt, load_last=True)

    def forward(self, x, alpha=1, device=None):
        h, chunk = self.frontend(x, device)
        new_hidden = {}
        for att_block in self.attention_blocks:
            hidden, indices = att_block(chunk, device)
            new_hidden[att_block.name] = hidden, indices
        preds = {}
        labels = {}
        for worker in self.regression_workers:
            hidden, _ = new_hidden[worker.name]
            y = worker(hidden, alpha)
            preds[worker.name] = y
            labels[worker.name] = x[worker.name].detach()
            if worker.name == 'chunk':
                labels[worker.name] = x['cchunk'].detach()
        for worker in self.classification_workers:
            hidden, mask = new_hidden[worker.name]
            h = [hidden, h[1] * mask, h[2] * mask]
            if worker.name == 'spc':
                y, label = worker(hidden, alpha, device)
            elif worker.name == 'overlap':
                y = worker(hidden, alpha)
                label = x[worker.name].detach()
            else:
                y, label = worker(h, alpha, device=device)
            preds[worker.name] = y
            labels[worker.name] = label
        return h, chunk, preds, labels


class pase_chunking(Model):

    def __init__(self, frontend=None, frontend_cfg=None, minions_cfg=None, cls_lst=['mi', 'cmi', 'spc'], regr_lst=['chunk', 'lps', 'mfcc', 'prosody'], chunk_size=None, batch_size=None, pretrained_ckpt=None, name='adversarial'):
        super().__init__(name=name)
        if minions_cfg is None or len(minions_cfg) < 1:
            raise ValueError('Please specify a stack of minions config with at least 1 minion. GIMME SOMETHING TO DO.')
        if 'aspp' in frontend_cfg.keys():
            self.frontend = aspp_encoder(sinc_out=frontend_cfg['sinc_out'], hidden_dim=frontend_cfg['hidden_dim'])
        elif 'aspp_res' in frontend_cfg.keys():
            self.frontend = aspp_res_encoder(sinc_out=frontend_cfg['sinc_out'], hidden_dim=frontend_cfg['hidden_dim'], stride=frontend_cfg['strides'], rnn_pool=frontend_cfg['rnn_pool'])
        else:
            self.frontend = encoder(WaveFe(**frontend_cfg))
        self.cls_lst = cls_lst
        self.reg_lst = regr_lst
        self.ninp = self.frontend.emb_dim
        self.regression_workers = nn.ModuleList()
        self.classification_workers = nn.ModuleList()
        self.K = chunk_size
        self.chunk_masks = None
        for cfg in minions_cfg:
            if cfg['name'] in self.cls_lst:
                self.classification_workers.append(cls_worker_maker(cfg, ninp))
            elif cfg['name'] in self.reg_lst:
                cfg['num_inputs'] = self.ninp
                minion = minion_maker(cfg)
                self.regression_workers.append(minion)
        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt, load_last=True)

    def forward(self, x, device):
        if self.chunk_masks is None:
            for worker in self.regression_workers:
                self.chunk_masks[worker.name] = self.generate_mask(worker.name, x)
            for worker in self.classification_workers:
                self.chunk_masks[worker.name] = self.generate_mask(worker.name, x)
        h, chunk = self.frontend(x, device)
        preds = {}
        labels = {}
        for worker in self.regression_workers:
            chunk = chunk * self.chunk_masks[worker.name]
            y = worker(chunk)
            preds[worker.name] = y
            labels[worker.name] = x[worker.name].detach()
            if worker.name == 'chunk':
                labels[worker.name] = x['cchunk'].detach()
        for worker in self.classification_workers:
            h = [h[0] * self.chunk_masks[worker.name], h[1] * self.chunk_masks[worker.name], h[2] * self.chunk_masks[worker.name]]
            chunk = h[0]
            if worker.name == 'spc':
                y, label = worker(chunk, device)
            else:
                y, label = worker(h, device)
            preds[worker.name] = y
            labels[worker.name] = label
        return h, chunk, preds, labels

    def generate_mask(self, name, x):
        selection_mask = np.zeros(self.ninp)
        selection_mask[:self.K] = 1
        selection_mask = np.random.shuffle(selection_mask)
        mask = torch.zeros(x.size())
        for i in range(self.K):
            mask[:, (selection_mask[i]), :] = 1
        None
        return mask


class LinearClassifier(Model):

    def __init__(self, frontend, num_spks=None, ft_fe=False, z_bnorm=False, name='CLS'):
        super().__init__(name=name)
        self.frontend = frontend
        self.ft_fe = ft_fe
        if z_bnorm:
            self.z_bnorm = nn.BatchNorm1d(frontend.emb_dim, affine=False)
        if num_spks is None:
            raise ValueError('Please specify a number of spks.')
        self.fc = nn.Conv1d(frontend.emb_dim, num_spks, 1)
        self.act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h = self.frontend(x)
        if not self.ft_fe:
            h = h.detach()
        if hasattr(self, 'z_bnorm'):
            h = self.z_bnorm(h)
        h = self.fc(h)
        y = self.act(h)
        return y


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ASPP,
     lambda: ([], {'inplanes': 4, 'emb_dim': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (AhoCNNEncoder,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (AhoCNNHourGlassEncoder,
     lambda: ([], {'input_dim': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (DecoderMinion,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (FeBlock,
     lambda: ([], {'num_inputs': 4, 'fmaps': 4, 'kwidth': 4, 'stride': 1, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GConv1DBlock,
     lambda: ([], {'ninp': 4, 'fmaps': 4, 'kwidth': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GDeconv1DBlock,
     lambda: ([], {'ninp': 4, 'fmaps': 4, 'kwidth': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (GRUMinion,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (GapMinion,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 2, 64])], {}),
     False),
    (LayerNorm,
     lambda: ([], {'features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (MLPBlock,
     lambda: ([], {'ninp': 4, 'fmaps': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (MLPMinion,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4, 'dropout': 0.5}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (MelResNet,
     lambda: ([], {'res_blocks': 4, 'in_dims': 4, 'compute_dims': 4, 'res_out_dims': 4, 'pad': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (ResARModule,
     lambda: ([], {'ninp': 4, 'fmaps': 4, 'res_fmaps': 4, 'kwidth': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (ResBasicBlock1D,
     lambda: ([], {'inplanes': 4, 'planes': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (SimpleResBlock1D,
     lambda: ([], {'dims': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (SpectrogramDecoder,
     lambda: ([], {'num_inputs': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (StatisticalPooling,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TDNN,
     lambda: ([], {'num_inputs': 4, 'num_outputs': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (UpsampleNetwork,
     lambda: ([], {'feat_dims': 4}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     False),
    (VQEMA,
     lambda: ([], {'emb_K': 4, 'emb_dim': 4, 'beta': 4, 'gamma': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (WaveRNNMinion,
     lambda: ([], {'num_inputs': 4}),
     lambda: ([torch.rand([4, 9600]), torch.rand([4, 4, 64])], {}),
     False),
    (_ASPPModule,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 64])], {}),
     True),
    (_ASPPModule2d,
     lambda: ([], {'inplanes': 4, 'planes': 4, 'kernel_size': 4, 'padding': 4, 'dilation': 1}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_santi_pdp_pase(_paritybench_base):
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

    def test_014(self):
        self._check(*TESTCASES[14])

    def test_015(self):
        self._check(*TESTCASES[15])

    def test_016(self):
        self._check(*TESTCASES[16])

    def test_017(self):
        self._check(*TESTCASES[17])

    def test_018(self):
        self._check(*TESTCASES[18])

    def test_019(self):
        self._check(*TESTCASES[19])

    def test_020(self):
        self._check(*TESTCASES[20])

    def test_021(self):
        self._check(*TESTCASES[21])

    def test_022(self):
        self._check(*TESTCASES[22])

    def test_023(self):
        self._check(*TESTCASES[23])

