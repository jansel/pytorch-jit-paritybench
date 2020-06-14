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
decoders = _module
discriminator = _module
encoders = _module
frontend = _module
modules = _module
neural_networks = _module
tdnn = _module
sbatch_writer = _module
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


import torch.nn.functional as F


import torch.nn as nn


import numpy as np


import math


import warnings


import torch.optim as optim


import random


import re


from torch.autograd import Variable


from torch.nn.utils.spectral_norm import spectral_norm


from random import shuffle


from torch.utils.data import DataLoader


from torch.optim.lr_scheduler import ReduceLROnPlateau


from torch.optim.lr_scheduler import StepLR


from torch.utils.data import Dataset


from torch.utils.data import ConcatDataset


from collections import defaultdict


import torch.optim.lr_scheduler as lr_scheduler


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
        self.dnn_use_batchnorm = list(map(strtobool, options[
            'dnn_use_batchnorm'].split(',')))
        self.dnn_use_laynorm = list(map(strtobool, options[
            'dnn_use_laynorm'].split(',')))
        self.dnn_use_laynorm_inp = strtobool(options['dnn_use_laynorm_inp'])
        self.dnn_use_batchnorm_inp = strtobool(options['dnn_use_batchnorm_inp']
            )
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
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=
                add_bias))
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.
                dnn_lay[i], current_input).uniform_(-np.sqrt(0.01 / (
                current_input + self.dnn_lay[i])), np.sqrt(0.01 / (
                current_input + self.dnn_lay[i]))))
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
            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i
                ] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[
                    i](x)))))
            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i
                ] == False:
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
        self.lstm = nn.ModuleList([nn.LSTM(self.input_dim, self.hidden_size,
            self.num_layers, bias=self.bias, dropout=self.dropout,
            bidirectional=self.bidirectional)])
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
        self.gru = nn.ModuleList([nn.GRU(self.input_dim, self.hidden_size,
            self.num_layers, bias=self.bias, dropout=self.dropout,
            bidirectional=self.bidirectional)])
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
        self.rnn = nn.ModuleList([nn.RNN(self.input_dim, self.hidden_size,
            self.num_layers, nonlinearity=self.nonlinearity, bias=self.bias,
            dropout=self.dropout, bidirectional=self.bidirectional)])
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
    x = x.view(x.size(0), x.size(1), -1)[:, (getattr(torch.arange(x.size(1) -
        1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long()), :]
    return x.view(xsize)


class LSTM(nn.Module):

    def __init__(self, options, inp_dim):
        super(LSTM, self).__init__()
        self.input_dim = inp_dim
        self.lstm_lay = list(map(int, options['lstm_lay'].split(',')))
        self.lstm_drop = list(map(float, options['lstm_drop'].split(',')))
        self.lstm_use_batchnorm = list(map(strtobool, options[
            'lstm_use_batchnorm'].split(',')))
        self.lstm_use_laynorm = list(map(strtobool, options[
            'lstm_use_laynorm'].split(',')))
        self.lstm_use_laynorm_inp = strtobool(options['lstm_use_laynorm_inp'])
        self.lstm_use_batchnorm_inp = strtobool(options[
            'lstm_use_batchnorm_inp'])
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
            self.wfx.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wix.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wox.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wcx.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.ufh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uih.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uoh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uch.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.lstm_drop[i]))
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
                wfx_out_bn = self.bn_wfx[i](wfx_out.view(wfx_out.shape[0] *
                    wfx_out.shape[1], wfx_out.shape[2]))
                wfx_out = wfx_out_bn.view(wfx_out.shape[0], wfx_out.shape[1
                    ], wfx_out.shape[2])
                wix_out_bn = self.bn_wix[i](wix_out.view(wix_out.shape[0] *
                    wix_out.shape[1], wix_out.shape[2]))
                wix_out = wix_out_bn.view(wix_out.shape[0], wix_out.shape[1
                    ], wix_out.shape[2])
                wox_out_bn = self.bn_wox[i](wox_out.view(wox_out.shape[0] *
                    wox_out.shape[1], wox_out.shape[2]))
                wox_out = wox_out_bn.view(wox_out.shape[0], wox_out.shape[1
                    ], wox_out.shape[2])
                wcx_out_bn = self.bn_wcx[i](wcx_out.view(wcx_out.shape[0] *
                    wcx_out.shape[1], wcx_out.shape[2]))
                wcx_out = wcx_out_bn.view(wcx_out.shape[0], wcx_out.shape[1
                    ], wcx_out.shape[2])
            hiddens = []
            ct = h_init
            ht = h_init
            for k in range(x.shape[0]):
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                ct = it * self.act[i](wcx_out[k] + self.uch[i](ht)
                    ) * drop_mask + ft * ct
                ht = ot * self.act[i](ct)
                if self.lstm_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class GRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(GRU, self).__init__()
        self.input_dim = inp_dim
        self.gru_lay = list(map(int, options['gru_lay'].split(',')))
        self.gru_drop = list(map(float, options['gru_drop'].split(',')))
        self.gru_use_batchnorm = list(map(strtobool, options[
            'gru_use_batchnorm'].split(',')))
        self.gru_use_laynorm = list(map(strtobool, options[
            'gru_use_laynorm'].split(',')))
        self.gru_use_laynorm_inp = strtobool(options['gru_use_laynorm_inp'])
        self.gru_use_batchnorm_inp = strtobool(options['gru_use_batchnorm_inp']
            )
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
            self.wh.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.wz.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.wr.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
            self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
            self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.gru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.gru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            wr_out = self.wr[i](x)
            if self.gru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
                wr_out_bn = self.bn_wr[i](wr_out.view(wr_out.shape[0] *
                    wr_out.shape[1], wr_out.shape[2]))
                wr_out = wr_out_bn.view(wr_out.shape[0], wr_out.shape[1],
                    wr_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class liGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(liGRU, self).__init__()
        self.input_dim = inp_dim
        self.ligru_lay = list(map(int, options['ligru_lay'].split(',')))
        self.ligru_drop = list(map(float, options['ligru_drop'].split(',')))
        self.ligru_use_batchnorm = list(map(strtobool, options[
            'ligru_use_batchnorm'].split(',')))
        self.ligru_use_laynorm = list(map(strtobool, options[
            'ligru_use_laynorm'].split(',')))
        self.ligru_use_laynorm_inp = strtobool(options['ligru_use_laynorm_inp']
            )
        self.ligru_use_batchnorm_inp = strtobool(options[
            'ligru_use_batchnorm_inp'])
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
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias
                =add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias
                =add_bias))
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i],
                bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i],
                bias=False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.ligru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.ligru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class minimalGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(minimalGRU, self).__init__()
        self.input_dim = inp_dim
        self.minimalgru_lay = list(map(int, options['minimalgru_lay'].split
            (',')))
        self.minimalgru_drop = list(map(float, options['minimalgru_drop'].
            split(',')))
        self.minimalgru_use_batchnorm = list(map(strtobool, options[
            'minimalgru_use_batchnorm'].split(',')))
        self.minimalgru_use_laynorm = list(map(strtobool, options[
            'minimalgru_use_laynorm'].split(',')))
        self.minimalgru_use_laynorm_inp = strtobool(options[
            'minimalgru_use_laynorm_inp'])
        self.minimalgru_use_batchnorm_inp = strtobool(options[
            'minimalgru_use_batchnorm_inp'])
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
            if self.minimalgru_use_laynorm[i] or self.minimalgru_use_batchnorm[
                i]:
                add_bias = False
            self.wh.append(nn.Linear(current_input, self.minimalgru_lay[i],
                bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.minimalgru_lay[i],
                bias=add_bias))
            self.uh.append(nn.Linear(self.minimalgru_lay[i], self.
                minimalgru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.minimalgru_lay[i], self.
                minimalgru_lay[i], bias=False))
            if self.minimalgru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
            self.bn_wh.append(nn.BatchNorm1d(self.minimalgru_lay[i],
                momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.minimalgru_lay[i],
                momentum=0.05))
            self.ln.append(LayerNorm(self.minimalgru_lay[i]))
            if self.bidir:
                current_input = 2 * self.minimalgru_lay[i]
            else:
                current_input = self.minimalgru_lay[i]
        self.out_dim = self.minimalgru_lay[i
            ] + self.bidir * self.minimalgru_lay[i]

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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.minimalgru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.minimalgru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.minimalgru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class RNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(RNN, self).__init__()
        self.input_dim = inp_dim
        self.rnn_lay = list(map(int, options['rnn_lay'].split(',')))
        self.rnn_drop = list(map(float, options['rnn_drop'].split(',')))
        self.rnn_use_batchnorm = list(map(strtobool, options[
            'rnn_use_batchnorm'].split(',')))
        self.rnn_use_laynorm = list(map(strtobool, options[
            'rnn_use_laynorm'].split(',')))
        self.rnn_use_laynorm_inp = strtobool(options['rnn_use_laynorm_inp'])
        self.rnn_use_batchnorm_inp = strtobool(options['rnn_use_batchnorm_inp']
            )
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
            self.wh.append(nn.Linear(current_input, self.rnn_lay[i], bias=
                add_bias))
            self.uh.append(nn.Linear(self.rnn_lay[i], self.rnn_lay[i], bias
                =False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.rnn_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.rnn_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            if self.rnn_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class CNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options['cnn_N_filt'].split(',')))
        self.cnn_len_filt = list(map(int, options['cnn_len_filt'].split(',')))
        self.cnn_max_pool_len = list(map(int, options['cnn_max_pool_len'].
            split(',')))
        self.cnn_act = options['cnn_act'].split(',')
        self.cnn_drop = list(map(float, options['cnn_drop'].split(',')))
        self.cnn_use_laynorm = list(map(strtobool, options[
            'cnn_use_laynorm'].split(',')))
        self.cnn_use_batchnorm = list(map(strtobool, options[
            'cnn_use_batchnorm'].split(',')))
        self.cnn_use_laynorm_inp = strtobool(options['cnn_use_laynorm_inp'])
        self.cnn_use_batchnorm_inp = strtobool(options['cnn_use_batchnorm_inp']
            )
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
            self.ln.append(LayerNorm([N_filt, int((current_input - self.
                cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self
                .cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]), momentum
                =0.05))
            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))
            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.
                    cnn_N_filt[i], self.cnn_len_filt[i]))
            current_input = int((current_input - self.cnn_len_filt[i] + 1) /
                self.cnn_max_pool_len[i])
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
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.
                    conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.
                    conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i
                ] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x),
                    self.cnn_max_pool_len[i])))
        x = x.view(batch, -1)
        return x


class SincNet(nn.Module):

    def __init__(self, options, inp_dim):
        super(SincNet, self).__init__()
        self.input_dim = inp_dim
        self.sinc_N_filt = list(map(int, options['sinc_N_filt'].split(',')))
        self.sinc_len_filt = list(map(int, options['sinc_len_filt'].split(','))
            )
        self.sinc_max_pool_len = list(map(int, options['sinc_max_pool_len']
            .split(',')))
        self.sinc_act = options['sinc_act'].split(',')
        self.sinc_drop = list(map(float, options['sinc_drop'].split(',')))
        self.sinc_use_laynorm = list(map(strtobool, options[
            'sinc_use_laynorm'].split(',')))
        self.sinc_use_batchnorm = list(map(strtobool, options[
            'sinc_use_batchnorm'].split(',')))
        self.sinc_use_laynorm_inp = strtobool(options['sinc_use_laynorm_inp'])
        self.sinc_use_batchnorm_inp = strtobool(options[
            'sinc_use_batchnorm_inp'])
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
            self.ln.append(LayerNorm([N_filt, int((current_input - self.
                sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self
                .sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i]),
                momentum=0.05))
            if i == 0:
                self.conv.append(SincConv(1, N_filt, len_filt, sample_rate=
                    self.sinc_sample_rate, min_low_hz=self.sinc_min_low_hz,
                    min_band_hz=self.sinc_min_band_hz))
            else:
                self.conv.append(nn.Conv1d(self.sinc_N_filt[i - 1], self.
                    sinc_N_filt[i], self.sinc_len_filt[i]))
            current_input = int((current_input - self.sinc_len_filt[i] + 1) /
                self.sinc_max_pool_len[i])
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
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.
                    conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.
                    conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i] == False and self.sinc_use_laynorm[i
                ] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x),
                    self.sinc_max_pool_len[i])))
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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=False, groups=1, sample_rate=16000,
        min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n + 1).view(1, -1) / self.sample_rate

    def sinc(self, x):
        x_left = x[:, 0:int((x.shape[1] - 1) / 2)]
        y_left = torch.sin(x_left) / x_left
        y_right = torch.flip(y_left, dims=[1])
        sinc = torch.cat([y_left, torch.ones([x.shape[0], 1]).to(x.device),
            y_right], dim=1)
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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz / self.sample_rate + torch.abs(self.
            band_hz_)
        f_times_t = torch.matmul(low, self.n_)
        low_pass1 = 2 * low * self.sinc(2 * math.pi * f_times_t * self.
            sample_rate)
        f_times_t = torch.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(2 * math.pi * f_times_t * self.
            sample_rate)
        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_
        self.filters = (band_pass * self.window_).view(self.out_channels, 1,
            self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=None, groups=1)


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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=False, groups=1, sample_rate=16000,
        min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size / 2 - 1, steps=int(self.
            kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1
            ) / self.sample_rate

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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_
            ), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, (0)]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)
            ) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center,
            band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, (None)])
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=None, groups=1)


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
        self.is_input_normalized = bool(strtobool(options[
            'sru_is_input_normalized']))
        self.has_skip_term = bool(strtobool(options['sru_has_skip_term']))
        self.rescale = bool(strtobool(options['sru_rescale']))
        self.highway_bias = float(options['sru_highway_bias'])
        self.n_proj = int(options['sru_n_proj'])
        self.sru = sru.SRU(self.input_dim, self.hidden_size, num_layers=
            self.num_layers, dropout=self.dropout, rnn_dropout=self.
            rnn_dropout, bidirectional=self.bidirectional, n_proj=self.
            n_proj, use_tanh=self.use_tanh, use_selu=self.use_selu,
            use_relu=self.use_relu, weight_norm=self.weight_norm,
            layer_norm=self.layer_norm, has_skip_term=self.has_skip_term,
            is_input_normalized=self.is_input_normalized, highway_bias=self
            .highway_bias, rescale=self.rescale)
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

    def __init__(self, rnn_size, rnn_layers, out_dim, dropout, cuda,
        rnn_type='LSTM', bidirectional=False):
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
        self.rnn = getattr(nn, rnn_type)(self.out_dim, self.rnn_size, self.
            rnn_layers, batch_first=True, dropout=self.dropout,
            bidirectional=bidirectional)
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
        h0 = Variable(torch.randn(self.dirs * self.rnn_layers, bsz, self.
            rnn_size))
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
        self.enc = nn.Sequential(nn.Conv1d(input_dim, 256, kwidth, stride=1,
            padding=pad), norm_layer(256), nn.PReLU(256), nn.Conv1d(256, 
            256, kwidth, stride=1, padding=pad), norm_layer(256), nn.PReLU(
            256), nn.MaxPool1d(2), nn.Dropout(0.2), nn.Conv1d(256, 512,
            kwidth, stride=1, padding=pad), norm_layer(512), nn.PReLU(512),
            nn.Conv1d(512, 512, kwidth, stride=1, padding=pad), norm_layer(
            512), nn.PReLU(512), nn.MaxPool1d(2), nn.Dropout(0.2), nn.
            Conv1d(512, 1024, kwidth, stride=1, padding=pad), norm_layer(
            1024), nn.PReLU(1024), nn.Conv1d(1024, 1024, kwidth, stride=1,
            padding=pad), norm_layer(1024), nn.PReLU(1024), nn.MaxPool1d(2),
            nn.Dropout(0.2), nn.Conv1d(1024, 1024, kwidth, stride=1,
            padding=pad))

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
        self.enc = nn.Sequential(nn.Conv1d(input_dim, 64, kwidth, stride=1,
            padding=pad), norm_layer(64), nn.PReLU(64), nn.Conv1d(64, 128,
            kwidth, stride=1, padding=pad), norm_layer(128), nn.PReLU(128),
            nn.MaxPool1d(2), nn.Dropout(dropout), nn.Conv1d(128, 256,
            kwidth, stride=1, padding=pad), norm_layer(256), nn.PReLU(256),
            nn.Conv1d(256, 512, kwidth, stride=1, padding=pad), norm_layer(
            512), nn.PReLU(512), nn.MaxPool1d(2), nn.Dropout(dropout), nn.
            Conv1d(512, 256, kwidth, stride=1, padding=pad), norm_layer(256
            ), nn.PReLU(256), nn.Conv1d(256, 128, kwidth, stride=1, padding
            =pad), norm_layer(128), nn.PReLU(128), nn.MaxPool1d(2), nn.
            Dropout(dropout), nn.Conv1d(128, 64, kwidth, stride=1, padding=
            pad), norm_layer(64), nn.PReLU(64))

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


def sinc(band, t_right, cuda=False):
    y_right = torch.sin(2 * math.pi * band * t_right) / (2 * math.pi * band *
        t_right)
    y_left = flip(y_right, 0)
    ones = torch.ones(1)
    if cuda:
        ones = ones.to('cuda')
    y = torch.cat([y_left, ones, y_right])
    return y


class SincConv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs, stride=1, padding='VALID',
        pad_mode='reflect'):
        super(SincConv, self).__init__()
        low_freq_mel = 80
        high_freq_mel = 2595 * np.log10(1 + fs / 2 / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)
        f_cos = 700 * (10 ** (mel_points / 2595) - 1)
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = fs / 2 - 100
        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.
            freq_scale))
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.padding = padding
        self.stride = stride
        self.pad_mode = pad_mode

    def forward(self, x):
        cuda = x.is_cuda
        filters = torch.zeros((self.N_filt, self.Filt_dim))
        N = self.Filt_dim
        t_right = torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)
            ) / self.fs
        if cuda:
            filters = filters.to('cuda')
            t_right = t_right.to('cuda')
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + 
            min_band / self.freq_scale)
        n = torch.linspace(0, N, steps=N)
        window = (0.54 - 0.46 * torch.cos(2 * math.pi * n / N)).float()
        if cuda:
            window = window.to('cuda')
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i
                ].float() * self.freq_scale, t_right, cuda)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i
                ].float() * self.freq_scale, t_right, cuda)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / torch.max(band_pass)
            if cuda:
                band_pass = band_pass.to('cuda')
            filters[(i), :] = band_pass * window
        if self.padding == 'SAME':
            if self.stride > 1:
                x_p = F.pad(x, (self.Filt_dim // 2 - 1, self.Filt_dim // 2),
                    mode=self.pad_mode)
            else:
                x_p = F.pad(x, (self.Filt_dim // 2, self.Filt_dim // 2),
                    mode=self.pad_mode)
        else:
            x_p = x
        out = F.conv1d(x_p, filters.view(self.N_filt, 1, self.Filt_dim),
            stride=self.stride)
        return out


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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding='VALID', pad_mode='reflect', dilation=1, bias=False, groups
        =1, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.pad_mode = pad_mode
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size / 2 - 1, steps=int(self.
            kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1
            ) / self.sample_rate

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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_
            ), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, (0)]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)
            ) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center,
            band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, (None)])
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        x = waveforms
        if self.padding == 'SAME':
            if self.stride > 1:
                x_p = F.pad(x, (self.kernel_size // 2 - 1, self.kernel_size //
                    2), mode=self.pad_mode)
            else:
                x_p = F.pad(x, (self.kernel_size // 2, self.kernel_size // 
                    2), mode=self.pad_mode)
        else:
            x_p = x
        return F.conv1d(x_p, self.filters, stride=self.stride, padding=0,
            dilation=self.dilation, bias=None, groups=1)


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
        dist = torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(
            self.emb.weight ** 2, dim=1) - 2 * torch.matmul(flat_input,
            self.emb.weight.t())
        enc_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        enc = torch.zeros(enc_indices.shape[0], self.emb_K).to(device)
        enc.scatter_(1, enc_indices, 1)
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.gamma + (1 -
                self.gamma) * torch.sum(enc, 0)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (self.ema_cluster_size + self.eps) / (n +
                self.emb_K * self.eps) * n
            dw = torch.matmul(enc.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.gamma + (1 - self.
                gamma) * dw)
            self.emb.weight = nn.Parameter(self.ema_w / self.
                ema_cluster_size.unsqueeze(1))
        Q = torch.matmul(enc, self.emb.weight).view(input_shape)
        e_latent_loss = torch.mean((Q.detach() - inputs) ** 2)
        loss = self.beta * e_latent_loss
        Q = inputs + (Q - inputs).detach()
        avg_probs = torch.mean(enc, dim=0)
        PP = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, Q.permute(0, 2, 1).contiguous(), PP, enc


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


class MLP(nn.Module):

    def __init__(self, options, inp_dim):
        super(MLP, self).__init__()
        self.input_dim = inp_dim
        self.dnn_lay = list(map(int, options['dnn_lay'].split(',')))
        self.dnn_drop = list(map(float, options['dnn_drop'].split(',')))
        self.dnn_use_batchnorm = list(map(strtobool, options[
            'dnn_use_batchnorm'].split(',')))
        self.dnn_use_laynorm = list(map(strtobool, options[
            'dnn_use_laynorm'].split(',')))
        self.dnn_use_laynorm_inp = strtobool(options['dnn_use_laynorm_inp'])
        self.dnn_use_batchnorm_inp = strtobool(options['dnn_use_batchnorm_inp']
            )
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
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=
                add_bias))
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.
                dnn_lay[i], current_input).uniform_(-np.sqrt(0.01 / (
                current_input + self.dnn_lay[i])), np.sqrt(0.01 / (
                current_input + self.dnn_lay[i]))))
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
            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i
                ] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[
                    i](x)))))
            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i
                ] == False:
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
        self.lstm = nn.ModuleList([nn.LSTM(self.input_dim, self.hidden_size,
            self.num_layers, bias=self.bias, dropout=self.dropout,
            bidirectional=self.bidirectional)])
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
        self.gru = nn.ModuleList([nn.GRU(self.input_dim, self.hidden_size,
            self.num_layers, bias=self.bias, dropout=self.dropout,
            bidirectional=self.bidirectional)])
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
        self.rnn = nn.ModuleList([nn.RNN(self.input_dim, self.hidden_size,
            self.num_layers, nonlinearity=self.nonlinearity, bias=self.bias,
            dropout=self.dropout, bidirectional=self.bidirectional)])
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


class LSTM(nn.Module):

    def __init__(self, options, inp_dim):
        super(LSTM, self).__init__()
        self.input_dim = inp_dim
        self.lstm_lay = list(map(int, options['lstm_lay'].split(',')))
        self.lstm_drop = list(map(float, options['lstm_drop'].split(',')))
        self.lstm_use_batchnorm = list(map(strtobool, options[
            'lstm_use_batchnorm'].split(',')))
        self.lstm_use_laynorm = list(map(strtobool, options[
            'lstm_use_laynorm'].split(',')))
        self.lstm_use_laynorm_inp = strtobool(options['lstm_use_laynorm_inp'])
        self.lstm_use_batchnorm_inp = strtobool(options[
            'lstm_use_batchnorm_inp'])
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
            self.wfx.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wix.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wox.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wcx.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.ufh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uih.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uoh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uch.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.lstm_drop[i]))
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
                wfx_out_bn = self.bn_wfx[i](wfx_out.view(wfx_out.shape[0] *
                    wfx_out.shape[1], wfx_out.shape[2]))
                wfx_out = wfx_out_bn.view(wfx_out.shape[0], wfx_out.shape[1
                    ], wfx_out.shape[2])
                wix_out_bn = self.bn_wix[i](wix_out.view(wix_out.shape[0] *
                    wix_out.shape[1], wix_out.shape[2]))
                wix_out = wix_out_bn.view(wix_out.shape[0], wix_out.shape[1
                    ], wix_out.shape[2])
                wox_out_bn = self.bn_wox[i](wox_out.view(wox_out.shape[0] *
                    wox_out.shape[1], wox_out.shape[2]))
                wox_out = wox_out_bn.view(wox_out.shape[0], wox_out.shape[1
                    ], wox_out.shape[2])
                wcx_out_bn = self.bn_wcx[i](wcx_out.view(wcx_out.shape[0] *
                    wcx_out.shape[1], wcx_out.shape[2]))
                wcx_out = wcx_out_bn.view(wcx_out.shape[0], wcx_out.shape[1
                    ], wcx_out.shape[2])
            hiddens = []
            ct = h_init
            ht = h_init
            for k in range(x.shape[0]):
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                ct = it * self.act[i](wcx_out[k] + self.uch[i](ht)
                    ) * drop_mask + ft * ct
                ht = ot * self.act[i](ct)
                if self.lstm_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class GRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(GRU, self).__init__()
        self.input_dim = inp_dim
        self.gru_lay = list(map(int, options['gru_lay'].split(',')))
        self.gru_drop = list(map(float, options['gru_drop'].split(',')))
        self.gru_use_batchnorm = list(map(strtobool, options[
            'gru_use_batchnorm'].split(',')))
        self.gru_use_laynorm = list(map(strtobool, options[
            'gru_use_laynorm'].split(',')))
        self.gru_use_laynorm_inp = strtobool(options['gru_use_laynorm_inp'])
        self.gru_use_batchnorm_inp = strtobool(options['gru_use_batchnorm_inp']
            )
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
            self.wh.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.wz.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.wr.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
            self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
            self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.gru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.gru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            wr_out = self.wr[i](x)
            if self.gru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
                wr_out_bn = self.bn_wr[i](wr_out.view(wr_out.shape[0] *
                    wr_out.shape[1], wr_out.shape[2]))
                wr_out = wr_out_bn.view(wr_out.shape[0], wr_out.shape[1],
                    wr_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class liGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(liGRU, self).__init__()
        self.input_dim = inp_dim
        self.ligru_lay = list(map(int, options['ligru_lay'].split(',')))
        self.ligru_drop = list(map(float, options['ligru_drop'].split(',')))
        self.ligru_use_batchnorm = list(map(strtobool, options[
            'ligru_use_batchnorm'].split(',')))
        self.ligru_use_laynorm = list(map(strtobool, options[
            'ligru_use_laynorm'].split(',')))
        self.ligru_use_laynorm_inp = strtobool(options['ligru_use_laynorm_inp']
            )
        self.ligru_use_batchnorm_inp = strtobool(options[
            'ligru_use_batchnorm_inp'])
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
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias
                =add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias
                =add_bias))
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i],
                bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i],
                bias=False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.ligru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.ligru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class minimalGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(minimalGRU, self).__init__()
        self.input_dim = inp_dim
        self.minimalgru_lay = list(map(int, options['minimalgru_lay'].split
            (',')))
        self.minimalgru_drop = list(map(float, options['minimalgru_drop'].
            split(',')))
        self.minimalgru_use_batchnorm = list(map(strtobool, options[
            'minimalgru_use_batchnorm'].split(',')))
        self.minimalgru_use_laynorm = list(map(strtobool, options[
            'minimalgru_use_laynorm'].split(',')))
        self.minimalgru_use_laynorm_inp = strtobool(options[
            'minimalgru_use_laynorm_inp'])
        self.minimalgru_use_batchnorm_inp = strtobool(options[
            'minimalgru_use_batchnorm_inp'])
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
            if self.minimalgru_use_laynorm[i] or self.minimalgru_use_batchnorm[
                i]:
                add_bias = False
            self.wh.append(nn.Linear(current_input, self.minimalgru_lay[i],
                bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.minimalgru_lay[i],
                bias=add_bias))
            self.uh.append(nn.Linear(self.minimalgru_lay[i], self.
                minimalgru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.minimalgru_lay[i], self.
                minimalgru_lay[i], bias=False))
            if self.minimalgru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
            self.bn_wh.append(nn.BatchNorm1d(self.minimalgru_lay[i],
                momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.minimalgru_lay[i],
                momentum=0.05))
            self.ln.append(LayerNorm(self.minimalgru_lay[i]))
            if self.bidir:
                current_input = 2 * self.minimalgru_lay[i]
            else:
                current_input = self.minimalgru_lay[i]
        self.out_dim = self.minimalgru_lay[i
            ] + self.bidir * self.minimalgru_lay[i]

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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.minimalgru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.minimalgru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.minimalgru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class RNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(RNN, self).__init__()
        self.input_dim = inp_dim
        self.rnn_lay = list(map(int, options['rnn_lay'].split(',')))
        self.rnn_drop = list(map(float, options['rnn_drop'].split(',')))
        self.rnn_use_batchnorm = list(map(strtobool, options[
            'rnn_use_batchnorm'].split(',')))
        self.rnn_use_laynorm = list(map(strtobool, options[
            'rnn_use_laynorm'].split(',')))
        self.rnn_use_laynorm_inp = strtobool(options['rnn_use_laynorm_inp'])
        self.rnn_use_batchnorm_inp = strtobool(options['rnn_use_batchnorm_inp']
            )
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
            self.wh.append(nn.Linear(current_input, self.rnn_lay[i], bias=
                add_bias))
            self.uh.append(nn.Linear(self.rnn_lay[i], self.rnn_lay[i], bias
                =False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.rnn_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.rnn_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            if self.rnn_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class CNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options['cnn_N_filt'].split(',')))
        self.cnn_len_filt = list(map(int, options['cnn_len_filt'].split(',')))
        self.cnn_max_pool_len = list(map(int, options['cnn_max_pool_len'].
            split(',')))
        self.cnn_act = options['cnn_act'].split(',')
        self.cnn_drop = list(map(float, options['cnn_drop'].split(',')))
        self.cnn_use_laynorm = list(map(strtobool, options[
            'cnn_use_laynorm'].split(',')))
        self.cnn_use_batchnorm = list(map(strtobool, options[
            'cnn_use_batchnorm'].split(',')))
        self.cnn_use_laynorm_inp = strtobool(options['cnn_use_laynorm_inp'])
        self.cnn_use_batchnorm_inp = strtobool(options['cnn_use_batchnorm_inp']
            )
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
            self.ln.append(LayerNorm([N_filt, int((current_input - self.
                cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self
                .cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]), momentum
                =0.05))
            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))
            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.
                    cnn_N_filt[i], self.cnn_len_filt[i]))
            current_input = int((current_input - self.cnn_len_filt[i] + 1) /
                self.cnn_max_pool_len[i])
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
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.
                    conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.
                    conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i
                ] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x),
                    self.cnn_max_pool_len[i])))
        x = x.view(batch, -1)
        return x


class SincNet(nn.Module):

    def __init__(self, options, inp_dim):
        super(SincNet, self).__init__()
        self.input_dim = inp_dim
        self.sinc_N_filt = list(map(int, options['sinc_N_filt'].split(',')))
        self.sinc_len_filt = list(map(int, options['sinc_len_filt'].split(','))
            )
        self.sinc_max_pool_len = list(map(int, options['sinc_max_pool_len']
            .split(',')))
        self.sinc_act = options['sinc_act'].split(',')
        self.sinc_drop = list(map(float, options['sinc_drop'].split(',')))
        self.sinc_use_laynorm = list(map(strtobool, options[
            'sinc_use_laynorm'].split(',')))
        self.sinc_use_batchnorm = list(map(strtobool, options[
            'sinc_use_batchnorm'].split(',')))
        self.sinc_use_laynorm_inp = strtobool(options['sinc_use_laynorm_inp'])
        self.sinc_use_batchnorm_inp = strtobool(options[
            'sinc_use_batchnorm_inp'])
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
            self.ln.append(LayerNorm([N_filt, int((current_input - self.
                sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self
                .sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i]),
                momentum=0.05))
            if i == 0:
                self.conv.append(SincConv(1, N_filt, len_filt, sample_rate=
                    self.sinc_sample_rate, min_low_hz=self.sinc_min_low_hz,
                    min_band_hz=self.sinc_min_band_hz))
            else:
                self.conv.append(nn.Conv1d(self.sinc_N_filt[i - 1], self.
                    sinc_N_filt[i], self.sinc_len_filt[i]))
            current_input = int((current_input - self.sinc_len_filt[i] + 1) /
                self.sinc_max_pool_len[i])
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
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.
                    conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.
                    conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i] == False and self.sinc_use_laynorm[i
                ] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x),
                    self.sinc_max_pool_len[i])))
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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=False, groups=1, sample_rate=16000,
        min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n + 1).view(1, -1) / self.sample_rate

    def sinc(self, x):
        x_left = x[:, 0:int((x.shape[1] - 1) / 2)]
        y_left = torch.sin(x_left) / x_left
        y_right = torch.flip(y_left, dims=[1])
        sinc = torch.cat([y_left, torch.ones([x.shape[0], 1]).to(x.device),
            y_right], dim=1)
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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz / self.sample_rate + torch.abs(self.
            band_hz_)
        f_times_t = torch.matmul(low, self.n_)
        low_pass1 = 2 * low * self.sinc(2 * math.pi * f_times_t * self.
            sample_rate)
        f_times_t = torch.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(2 * math.pi * f_times_t * self.
            sample_rate)
        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_
        self.filters = (band_pass * self.window_).view(self.out_channels, 1,
            self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=None, groups=1)


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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=False, groups=1, sample_rate=16000,
        min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size / 2 - 1, steps=int(self.
            kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1
            ) / self.sample_rate

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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_
            ), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, (0)]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)
            ) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center,
            band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, (None)])
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=None, groups=1)


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
        self.is_input_normalized = bool(strtobool(options[
            'sru_is_input_normalized']))
        self.has_skip_term = bool(strtobool(options['sru_has_skip_term']))
        self.rescale = bool(strtobool(options['sru_rescale']))
        self.highway_bias = float(options['sru_highway_bias'])
        self.n_proj = int(options['sru_n_proj'])
        self.sru = sru.SRU(self.input_dim, self.hidden_size, num_layers=
            self.num_layers, dropout=self.dropout, rnn_dropout=self.
            rnn_dropout, bidirectional=self.bidirectional, n_proj=self.
            n_proj, use_tanh=self.use_tanh, use_selu=self.use_selu,
            use_relu=self.use_relu, weight_norm=self.weight_norm,
            layer_norm=self.layer_norm, has_skip_term=self.has_skip_term,
            is_input_normalized=self.is_input_normalized, highway_bias=self
            .highway_bias, rescale=self.rescale)
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


class WaveAdversarialLoss(nn.Module):

    def __init__(self, discriminator, d_optimizer, size_average=True, loss=
        'L2', batch_acum=1, device='cpu'):
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
        label = label.to(self.device)
        return label

    def forward(self, iteration, x_fake, x_real, c_real=None, c_fake=None,
        grad=True):
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
            return {'g_loss': g_real_loss, 'd_real_loss': d_real_loss,
                'd_fake_loss': d_fake_loss}
        else:
            return {'g_loss': g_real_loss}


class SpectrumLM(nn.Module):
    """ RNN lang model for spectrum frame preds """

    def __init__(self, rnn_size, rnn_layers, out_dim, dropout, cuda,
        rnn_type='LSTM', bidirectional=False):
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
        self.rnn = getattr(nn, rnn_type)(self.out_dim, self.rnn_size, self.
            rnn_layers, batch_first=True, dropout=self.dropout,
            bidirectional=bidirectional)
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
        h0 = Variable(torch.randn(self.dirs * self.rnn_layers, bsz, self.
            rnn_size))
        if self.do_cuda:
            h0 = h0
        if self.rnn_type == 'LSTM':
            c0 = h0.clone()
            return h0, c0
        else:
            return h0


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

    def __init__(self, ninp, fmaps, kwidth, stride=1, norm_type=None, act=
        'prelu', name='GConv1DBlock'):
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


class WaveDiscriminator(nn.Module):

    def __init__(self, ninputs=1, fmaps=[128, 128, 256, 256, 512, 100],
        strides=[10, 4, 4, 1, 1, 1], kwidths=[30, 30, 30, 3, 3, 3],
        norm_type='snorm'):
        super().__init__()
        self.aco_decimator = nn.ModuleList()
        ninp = ninputs
        for fmap, kwidth, stride in zip(fmaps, kwidths, strides):
            self.aco_decimator.append(GConv1DBlock(ninp, fmap, kwidth,
                stride, norm_type=norm_type))
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


class AhoCNNEncoder(nn.Module):

    def __init__(self, input_dim, kwidth=3, dropout=0.5, layer_norm=False):
        super().__init__()
        pad = (kwidth - 1) // 2
        if layer_norm:
            norm_layer = LayerNorm
        else:
            norm_layer = nn.BatchNorm1d
        self.enc = nn.Sequential(nn.Conv1d(input_dim, 256, kwidth, stride=1,
            padding=pad), norm_layer(256), nn.PReLU(256), nn.Conv1d(256, 
            256, kwidth, stride=1, padding=pad), norm_layer(256), nn.PReLU(
            256), nn.MaxPool1d(2), nn.Dropout(0.2), nn.Conv1d(256, 512,
            kwidth, stride=1, padding=pad), norm_layer(512), nn.PReLU(512),
            nn.Conv1d(512, 512, kwidth, stride=1, padding=pad), norm_layer(
            512), nn.PReLU(512), nn.MaxPool1d(2), nn.Dropout(0.2), nn.
            Conv1d(512, 1024, kwidth, stride=1, padding=pad), norm_layer(
            1024), nn.PReLU(1024), nn.Conv1d(1024, 1024, kwidth, stride=1,
            padding=pad), norm_layer(1024), nn.PReLU(1024), nn.MaxPool1d(2),
            nn.Dropout(0.2), nn.Conv1d(1024, 1024, kwidth, stride=1,
            padding=pad))

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
        self.enc = nn.Sequential(nn.Conv1d(input_dim, 64, kwidth, stride=1,
            padding=pad), norm_layer(64), nn.PReLU(64), nn.Conv1d(64, 128,
            kwidth, stride=1, padding=pad), norm_layer(128), nn.PReLU(128),
            nn.MaxPool1d(2), nn.Dropout(dropout), nn.Conv1d(128, 256,
            kwidth, stride=1, padding=pad), norm_layer(256), nn.PReLU(256),
            nn.Conv1d(256, 512, kwidth, stride=1, padding=pad), norm_layer(
            512), nn.PReLU(512), nn.MaxPool1d(2), nn.Dropout(dropout), nn.
            Conv1d(512, 256, kwidth, stride=1, padding=pad), norm_layer(256
            ), nn.PReLU(256), nn.Conv1d(256, 128, kwidth, stride=1, padding
            =pad), norm_layer(128), nn.PReLU(128), nn.MaxPool1d(2), nn.
            Dropout(dropout), nn.Conv1d(128, 64, kwidth, stride=1, padding=
            pad), norm_layer(64), nn.PReLU(64))

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


class PatternedDropout(nn.Module):

    def __init__(self, emb_size, p=0.5, dropout_mode=['fixed_rand'],
        ratio_fixed=None, range_fixed=None, drop_whole_channels=False):
        """Applies a fixed pattern of dropout for the whole training
        session (i.e applies different only among pre-specified dimensions)
        """
        super(PatternedDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                'dropout probability has to be between 0 and 1, but got {}'
                .format(p))
        self.p = p
        if self.p > 0:
            d_modes = ['std', 'fixed_rand', 'fixed_given']
            assert dropout_mode in d_modes, 'Expected dropout mode in {}, got {}'.format(
                d_modes, dropout_mode)
            self.drop_whole_channels = drop_whole_channels
            self.dropout_fixed = False
            if dropout_mode == 'fixed_rand':
                self.dropout_fixed = True
                assert ratio_fixed is not None, "{} needs 'ratio_fixed' arg set.".format(
                    dropout_mode)
                if ratio_fixed <= 0 or ratio_fixed > 1:
                    raise ValueError(
                        "{} mode needs 'ratio_fixed' to be set and in (0, 1) range, got {}"
                        .format(dropout_mode, ratio_fixed))
                self.ratio_fixed = ratio_fixed
                self.dropped_dimsize = int(emb_size - emb_size * ratio_fixed)
                tot_idx = np.arange(emb_size)
                sel_idx = np.sort(np.random.choice(tot_idx, size=self.
                    dropped_dimsize, replace=False))
            elif dropout_mode == 'fixed_given':
                self.dropout_fixed = True
                if range_fixed is None or not isinstance(range_fixed, str
                    ) or len(range_fixed.split(':')) < 2:
                    raise ValueError(
                        "{} mode needs 'range_dropped' to be set (i.e. 10:20)"
                        .format(dropout_mode))
                rng = range_fixed.split(':')
                beg = int(rng[0])
                end = int(rng[1])
                assert beg < end and end <= emb_size, 'Incorrect range {}'.format(
                    range_fixed)
                self.dropped_dimsize = int(emb_size - (end - beg))
                tot_idx = np.arange(emb_size)
                fixed_idx = np.arange(beg, end, 1)
                sel_idx = np.setdiff1d(tot_idx, fixed_idx, assume_unique=True)
            if self.dropout_fixed:
                assert len(sel_idx
                    ) > 0, 'Asked for fixed dropout, but sel_idx {}'.format(
                    sel_idx)
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
            self.dindexes = self.dindexes.to(x.device)
            assert len(x.size()
                ) == 3, 'Expected to get 3 dimensional tensor, got {}'.format(
                len(x.size()))
            bsize, emb_size, tsize = x.size()
            if self.drop_whole_channels:
                batch_mask = torch.full(size=(bsize, emb_size), fill_value=
                    1.0, device=x.device)
                probs = torch.full(size=(bsize, self.dropped_dimsize),
                    fill_value=1.0 - self.p)
                b = Binomial(total_count=1, probs=probs)
                mask = b.sample()
                mask = mask.to(x.device)
                batch_mask[:, (self.dindexes)] *= mask * self.p_scale
                x = x * batch_mask.view(bsize, emb_size, -1)
            else:
                batch_mask = torch.ones_like(x, device=x.device)
                probs = torch.full(size=(bsize, self.dropped_dimsize, tsize
                    ), fill_value=1.0 - self.p)
                b = Binomial(total_count=1, probs=probs)
                mask = b.sample()
                mask = mask.to(x.device)
                batch_mask[:, (self.dindexes), :] *= mask * self.p_scale
                x = x * batch_mask
            return x
        else:
            return F.dropout(x, p=self.p, training=self.training)


class SincConv(nn.Module):

    def __init__(self, N_filt, Filt_dim, fs, stride=1, padding='VALID',
        pad_mode='reflect'):
        super(SincConv, self).__init__()
        low_freq_mel = 80
        high_freq_mel = 2595 * np.log10(1 + fs / 2 / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, N_filt)
        f_cos = 700 * (10 ** (mel_points / 2595) - 1)
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = fs / 2 - 100
        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy((b2 - b1) / self.
            freq_scale))
        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.padding = padding
        self.stride = stride
        self.pad_mode = pad_mode

    def forward(self, x):
        cuda = x.is_cuda
        filters = torch.zeros((self.N_filt, self.Filt_dim))
        N = self.Filt_dim
        t_right = torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)
            ) / self.fs
        if cuda:
            filters = filters.to('cuda')
            t_right = t_right.to('cuda')
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = torch.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + (torch.abs(self.filt_band) + 
            min_band / self.freq_scale)
        n = torch.linspace(0, N, steps=N)
        window = (0.54 - 0.46 * torch.cos(2 * math.pi * n / N)).float()
        if cuda:
            window = window.to('cuda')
        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i
                ].float() * self.freq_scale, t_right, cuda)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i
                ].float() * self.freq_scale, t_right, cuda)
            band_pass = low_pass2 - low_pass1
            band_pass = band_pass / torch.max(band_pass)
            if cuda:
                band_pass = band_pass.to('cuda')
            filters[(i), :] = band_pass * window
        if self.padding == 'SAME':
            if self.stride > 1:
                x_p = F.pad(x, (self.Filt_dim // 2 - 1, self.Filt_dim // 2),
                    mode=self.pad_mode)
            else:
                x_p = F.pad(x, (self.Filt_dim // 2, self.Filt_dim // 2),
                    mode=self.pad_mode)
        else:
            x_p = x
        out = F.conv1d(x_p, filters.view(self.N_filt, 1, self.Filt_dim),
            stride=self.stride)
        return out


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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding='VALID', pad_mode='reflect', dilation=1, bias=False, groups
        =1, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.pad_mode = pad_mode
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size / 2 - 1, steps=int(self.
            kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1
            ) / self.sample_rate

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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_
            ), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, (0)]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)
            ) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center,
            band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, (None)])
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        x = waveforms
        if self.padding == 'SAME':
            if self.stride > 1:
                x_p = F.pad(x, (self.kernel_size // 2 - 1, self.kernel_size //
                    2), mode=self.pad_mode)
            else:
                x_p = F.pad(x, (self.kernel_size // 2, self.kernel_size // 
                    2), mode=self.pad_mode)
        else:
            x_p = x
        return F.conv1d(x_p, self.filters, stride=self.stride, padding=0,
            dilation=self.dilation, bias=None, groups=1)


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
        dist = torch.sum(flat_input ** 2, dim=1, keepdim=True) + torch.sum(
            self.emb.weight ** 2, dim=1) - 2 * torch.matmul(flat_input,
            self.emb.weight.t())
        enc_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        enc = torch.zeros(enc_indices.shape[0], self.emb_K).to(device)
        enc.scatter_(1, enc_indices, 1)
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.gamma + (1 -
                self.gamma) * torch.sum(enc, 0)
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (self.ema_cluster_size + self.eps) / (n +
                self.emb_K * self.eps) * n
            dw = torch.matmul(enc.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.gamma + (1 - self.
                gamma) * dw)
            self.emb.weight = nn.Parameter(self.ema_w / self.
                ema_cluster_size.unsqueeze(1))
        Q = torch.matmul(enc, self.emb.weight).view(input_shape)
        e_latent_loss = torch.mean((Q.detach() - inputs) ** 2)
        loss = self.beta * e_latent_loss
        Q = inputs + (Q - inputs).detach()
        avg_probs = torch.mean(enc, dim=0)
        PP = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return loss, Q.permute(0, 2, 1).contiguous(), PP, enc


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
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size,
            bias=False)
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

    def __init__(self, feat_dims, upsample_scales=[4, 4, 10], compute_dims=
        128, res_blocks=10, res_out_dims=128, pad=2):
        super().__init__()
        self.num_outputs = res_out_dims
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims,
            res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = 1, scale * 2 + 1
            padding = 0, scale
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding,
                bias=False)
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


class MLP(nn.Module):

    def __init__(self, options, inp_dim):
        super(MLP, self).__init__()
        self.input_dim = inp_dim
        self.dnn_lay = list(map(int, options['dnn_lay'].split(',')))
        self.dnn_drop = list(map(float, options['dnn_drop'].split(',')))
        self.dnn_use_batchnorm = list(map(strtobool, options[
            'dnn_use_batchnorm'].split(',')))
        self.dnn_use_laynorm = list(map(strtobool, options[
            'dnn_use_laynorm'].split(',')))
        self.dnn_use_laynorm_inp = strtobool(options['dnn_use_laynorm_inp'])
        self.dnn_use_batchnorm_inp = strtobool(options['dnn_use_batchnorm_inp']
            )
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
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=
                add_bias))
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.
                dnn_lay[i], current_input).uniform_(-np.sqrt(0.01 / (
                current_input + self.dnn_lay[i])), np.sqrt(0.01 / (
                current_input + self.dnn_lay[i]))))
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
            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i
                ] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[
                    i](x)))))
            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i
                ] == False:
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
        self.lstm = nn.ModuleList([nn.LSTM(self.input_dim, self.hidden_size,
            self.num_layers, bias=self.bias, dropout=self.dropout,
            bidirectional=self.bidirectional)])
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
        self.gru = nn.ModuleList([nn.GRU(self.input_dim, self.hidden_size,
            self.num_layers, bias=self.bias, dropout=self.dropout,
            bidirectional=self.bidirectional)])
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
        self.rnn = nn.ModuleList([nn.RNN(self.input_dim, self.hidden_size,
            self.num_layers, nonlinearity=self.nonlinearity, bias=self.bias,
            dropout=self.dropout, bidirectional=self.bidirectional)])
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


class LSTM(nn.Module):

    def __init__(self, options, inp_dim):
        super(LSTM, self).__init__()
        self.input_dim = inp_dim
        self.lstm_lay = list(map(int, options['lstm_lay'].split(',')))
        self.lstm_drop = list(map(float, options['lstm_drop'].split(',')))
        self.lstm_use_batchnorm = list(map(strtobool, options[
            'lstm_use_batchnorm'].split(',')))
        self.lstm_use_laynorm = list(map(strtobool, options[
            'lstm_use_laynorm'].split(',')))
        self.lstm_use_laynorm_inp = strtobool(options['lstm_use_laynorm_inp'])
        self.lstm_use_batchnorm_inp = strtobool(options[
            'lstm_use_batchnorm_inp'])
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
            self.wfx.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wix.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wox.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wcx.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.ufh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uih.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uoh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uch.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.lstm_drop[i]))
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
                wfx_out_bn = self.bn_wfx[i](wfx_out.view(wfx_out.shape[0] *
                    wfx_out.shape[1], wfx_out.shape[2]))
                wfx_out = wfx_out_bn.view(wfx_out.shape[0], wfx_out.shape[1
                    ], wfx_out.shape[2])
                wix_out_bn = self.bn_wix[i](wix_out.view(wix_out.shape[0] *
                    wix_out.shape[1], wix_out.shape[2]))
                wix_out = wix_out_bn.view(wix_out.shape[0], wix_out.shape[1
                    ], wix_out.shape[2])
                wox_out_bn = self.bn_wox[i](wox_out.view(wox_out.shape[0] *
                    wox_out.shape[1], wox_out.shape[2]))
                wox_out = wox_out_bn.view(wox_out.shape[0], wox_out.shape[1
                    ], wox_out.shape[2])
                wcx_out_bn = self.bn_wcx[i](wcx_out.view(wcx_out.shape[0] *
                    wcx_out.shape[1], wcx_out.shape[2]))
                wcx_out = wcx_out_bn.view(wcx_out.shape[0], wcx_out.shape[1
                    ], wcx_out.shape[2])
            hiddens = []
            ct = h_init
            ht = h_init
            for k in range(x.shape[0]):
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                ct = it * self.act[i](wcx_out[k] + self.uch[i](ht)
                    ) * drop_mask + ft * ct
                ht = ot * self.act[i](ct)
                if self.lstm_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class GRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(GRU, self).__init__()
        self.input_dim = inp_dim
        self.gru_lay = list(map(int, options['gru_lay'].split(',')))
        self.gru_drop = list(map(float, options['gru_drop'].split(',')))
        self.gru_use_batchnorm = list(map(strtobool, options[
            'gru_use_batchnorm'].split(',')))
        self.gru_use_laynorm = list(map(strtobool, options[
            'gru_use_laynorm'].split(',')))
        self.gru_use_laynorm_inp = strtobool(options['gru_use_laynorm_inp'])
        self.gru_use_batchnorm_inp = strtobool(options['gru_use_batchnorm_inp']
            )
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
            self.wh.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.wz.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.wr.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
            self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
            self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.gru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.gru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            wr_out = self.wr[i](x)
            if self.gru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
                wr_out_bn = self.bn_wr[i](wr_out.view(wr_out.shape[0] *
                    wr_out.shape[1], wr_out.shape[2]))
                wr_out = wr_out_bn.view(wr_out.shape[0], wr_out.shape[1],
                    wr_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class liGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(liGRU, self).__init__()
        self.input_dim = inp_dim
        self.ligru_lay = list(map(int, options['ligru_lay'].split(',')))
        self.ligru_drop = list(map(float, options['ligru_drop'].split(',')))
        self.ligru_use_batchnorm = list(map(strtobool, options[
            'ligru_use_batchnorm'].split(',')))
        self.ligru_use_laynorm = list(map(strtobool, options[
            'ligru_use_laynorm'].split(',')))
        self.ligru_use_laynorm_inp = strtobool(options['ligru_use_laynorm_inp']
            )
        self.ligru_use_batchnorm_inp = strtobool(options[
            'ligru_use_batchnorm_inp'])
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
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias
                =add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias
                =add_bias))
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i],
                bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i],
                bias=False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.ligru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.ligru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class minimalGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(minimalGRU, self).__init__()
        self.input_dim = inp_dim
        self.minimalgru_lay = list(map(int, options['minimalgru_lay'].split
            (',')))
        self.minimalgru_drop = list(map(float, options['minimalgru_drop'].
            split(',')))
        self.minimalgru_use_batchnorm = list(map(strtobool, options[
            'minimalgru_use_batchnorm'].split(',')))
        self.minimalgru_use_laynorm = list(map(strtobool, options[
            'minimalgru_use_laynorm'].split(',')))
        self.minimalgru_use_laynorm_inp = strtobool(options[
            'minimalgru_use_laynorm_inp'])
        self.minimalgru_use_batchnorm_inp = strtobool(options[
            'minimalgru_use_batchnorm_inp'])
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
            if self.minimalgru_use_laynorm[i] or self.minimalgru_use_batchnorm[
                i]:
                add_bias = False
            self.wh.append(nn.Linear(current_input, self.minimalgru_lay[i],
                bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.minimalgru_lay[i],
                bias=add_bias))
            self.uh.append(nn.Linear(self.minimalgru_lay[i], self.
                minimalgru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.minimalgru_lay[i], self.
                minimalgru_lay[i], bias=False))
            if self.minimalgru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
            self.bn_wh.append(nn.BatchNorm1d(self.minimalgru_lay[i],
                momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.minimalgru_lay[i],
                momentum=0.05))
            self.ln.append(LayerNorm(self.minimalgru_lay[i]))
            if self.bidir:
                current_input = 2 * self.minimalgru_lay[i]
            else:
                current_input = self.minimalgru_lay[i]
        self.out_dim = self.minimalgru_lay[i
            ] + self.bidir * self.minimalgru_lay[i]

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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.minimalgru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.minimalgru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.minimalgru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class RNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(RNN, self).__init__()
        self.input_dim = inp_dim
        self.rnn_lay = list(map(int, options['rnn_lay'].split(',')))
        self.rnn_drop = list(map(float, options['rnn_drop'].split(',')))
        self.rnn_use_batchnorm = list(map(strtobool, options[
            'rnn_use_batchnorm'].split(',')))
        self.rnn_use_laynorm = list(map(strtobool, options[
            'rnn_use_laynorm'].split(',')))
        self.rnn_use_laynorm_inp = strtobool(options['rnn_use_laynorm_inp'])
        self.rnn_use_batchnorm_inp = strtobool(options['rnn_use_batchnorm_inp']
            )
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
            self.wh.append(nn.Linear(current_input, self.rnn_lay[i], bias=
                add_bias))
            self.uh.append(nn.Linear(self.rnn_lay[i], self.rnn_lay[i], bias
                =False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.rnn_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.rnn_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            if self.rnn_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class CNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options['cnn_N_filt'].split(',')))
        self.cnn_len_filt = list(map(int, options['cnn_len_filt'].split(',')))
        self.cnn_max_pool_len = list(map(int, options['cnn_max_pool_len'].
            split(',')))
        self.cnn_act = options['cnn_act'].split(',')
        self.cnn_drop = list(map(float, options['cnn_drop'].split(',')))
        self.cnn_use_laynorm = list(map(strtobool, options[
            'cnn_use_laynorm'].split(',')))
        self.cnn_use_batchnorm = list(map(strtobool, options[
            'cnn_use_batchnorm'].split(',')))
        self.cnn_use_laynorm_inp = strtobool(options['cnn_use_laynorm_inp'])
        self.cnn_use_batchnorm_inp = strtobool(options['cnn_use_batchnorm_inp']
            )
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
            self.ln.append(LayerNorm([N_filt, int((current_input - self.
                cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self
                .cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]), momentum
                =0.05))
            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))
            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.
                    cnn_N_filt[i], self.cnn_len_filt[i]))
            current_input = int((current_input - self.cnn_len_filt[i] + 1) /
                self.cnn_max_pool_len[i])
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
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.
                    conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.
                    conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i
                ] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x),
                    self.cnn_max_pool_len[i])))
        x = x.view(batch, -1)
        return x


class SincNet(nn.Module):

    def __init__(self, options, inp_dim):
        super(SincNet, self).__init__()
        self.input_dim = inp_dim
        self.sinc_N_filt = list(map(int, options['sinc_N_filt'].split(',')))
        self.sinc_len_filt = list(map(int, options['sinc_len_filt'].split(','))
            )
        self.sinc_max_pool_len = list(map(int, options['sinc_max_pool_len']
            .split(',')))
        self.sinc_act = options['sinc_act'].split(',')
        self.sinc_drop = list(map(float, options['sinc_drop'].split(',')))
        self.sinc_use_laynorm = list(map(strtobool, options[
            'sinc_use_laynorm'].split(',')))
        self.sinc_use_batchnorm = list(map(strtobool, options[
            'sinc_use_batchnorm'].split(',')))
        self.sinc_use_laynorm_inp = strtobool(options['sinc_use_laynorm_inp'])
        self.sinc_use_batchnorm_inp = strtobool(options[
            'sinc_use_batchnorm_inp'])
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
            self.ln.append(LayerNorm([N_filt, int((current_input - self.
                sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self
                .sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i]),
                momentum=0.05))
            if i == 0:
                self.conv.append(SincConv(1, N_filt, len_filt, sample_rate=
                    self.sinc_sample_rate, min_low_hz=self.sinc_min_low_hz,
                    min_band_hz=self.sinc_min_band_hz))
            else:
                self.conv.append(nn.Conv1d(self.sinc_N_filt[i - 1], self.
                    sinc_N_filt[i], self.sinc_len_filt[i]))
            current_input = int((current_input - self.sinc_len_filt[i] + 1) /
                self.sinc_max_pool_len[i])
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
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.
                    conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.
                    conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i] == False and self.sinc_use_laynorm[i
                ] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x),
                    self.sinc_max_pool_len[i])))
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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=False, groups=1, sample_rate=16000,
        min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n + 1).view(1, -1) / self.sample_rate

    def sinc(self, x):
        x_left = x[:, 0:int((x.shape[1] - 1) / 2)]
        y_left = torch.sin(x_left) / x_left
        y_right = torch.flip(y_left, dims=[1])
        sinc = torch.cat([y_left, torch.ones([x.shape[0], 1]).to(x.device),
            y_right], dim=1)
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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz / self.sample_rate + torch.abs(self.
            band_hz_)
        f_times_t = torch.matmul(low, self.n_)
        low_pass1 = 2 * low * self.sinc(2 * math.pi * f_times_t * self.
            sample_rate)
        f_times_t = torch.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(2 * math.pi * f_times_t * self.
            sample_rate)
        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_
        self.filters = (band_pass * self.window_).view(self.out_channels, 1,
            self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=None, groups=1)


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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=False, groups=1, sample_rate=16000,
        min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size / 2 - 1, steps=int(self.
            kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1
            ) / self.sample_rate

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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_
            ), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, (0)]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)
            ) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center,
            band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, (None)])
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=None, groups=1)


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
        self.is_input_normalized = bool(strtobool(options[
            'sru_is_input_normalized']))
        self.has_skip_term = bool(strtobool(options['sru_has_skip_term']))
        self.rescale = bool(strtobool(options['sru_rescale']))
        self.highway_bias = float(options['sru_highway_bias'])
        self.n_proj = int(options['sru_n_proj'])
        self.sru = sru.SRU(self.input_dim, self.hidden_size, num_layers=
            self.num_layers, dropout=self.dropout, rnn_dropout=self.
            rnn_dropout, bidirectional=self.bidirectional, n_proj=self.
            n_proj, use_tanh=self.use_tanh, use_selu=self.use_selu,
            use_relu=self.use_relu, weight_norm=self.weight_norm,
            layer_norm=self.layer_norm, has_skip_term=self.has_skip_term,
            is_input_normalized=self.is_input_normalized, highway_bias=self
            .highway_bias, rescale=self.rescale)
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


class StatisticalPooling(nn.Module):

    def forward(self, x):
        mu = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        return torch.cat((mu, std), dim=1)


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


class MLP(nn.Module):

    def __init__(self, options, inp_dim):
        super(MLP, self).__init__()
        self.input_dim = inp_dim
        self.dnn_lay = list(map(int, options['dnn_lay'].split(',')))
        self.dnn_drop = list(map(float, options['dnn_drop'].split(',')))
        self.dnn_use_batchnorm = list(map(strtobool, options[
            'dnn_use_batchnorm'].split(',')))
        self.dnn_use_laynorm = list(map(strtobool, options[
            'dnn_use_laynorm'].split(',')))
        self.dnn_use_laynorm_inp = strtobool(options['dnn_use_laynorm_inp'])
        self.dnn_use_batchnorm_inp = strtobool(options['dnn_use_batchnorm_inp']
            )
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
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=
                add_bias))
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.
                dnn_lay[i], current_input).uniform_(-np.sqrt(0.01 / (
                current_input + self.dnn_lay[i])), np.sqrt(0.01 / (
                current_input + self.dnn_lay[i]))))
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
            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i
                ] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[
                    i](x)))))
            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i
                ] == False:
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
        self.lstm = nn.ModuleList([nn.LSTM(self.input_dim, self.hidden_size,
            self.num_layers, bias=self.bias, dropout=self.dropout,
            bidirectional=self.bidirectional)])
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
        self.gru = nn.ModuleList([nn.GRU(self.input_dim, self.hidden_size,
            self.num_layers, bias=self.bias, dropout=self.dropout,
            bidirectional=self.bidirectional)])
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
        self.rnn = nn.ModuleList([nn.RNN(self.input_dim, self.hidden_size,
            self.num_layers, nonlinearity=self.nonlinearity, bias=self.bias,
            dropout=self.dropout, bidirectional=self.bidirectional)])
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


class LSTM(nn.Module):

    def __init__(self, options, inp_dim):
        super(LSTM, self).__init__()
        self.input_dim = inp_dim
        self.lstm_lay = list(map(int, options['lstm_lay'].split(',')))
        self.lstm_drop = list(map(float, options['lstm_drop'].split(',')))
        self.lstm_use_batchnorm = list(map(strtobool, options[
            'lstm_use_batchnorm'].split(',')))
        self.lstm_use_laynorm = list(map(strtobool, options[
            'lstm_use_laynorm'].split(',')))
        self.lstm_use_laynorm_inp = strtobool(options['lstm_use_laynorm_inp'])
        self.lstm_use_batchnorm_inp = strtobool(options[
            'lstm_use_batchnorm_inp'])
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
            self.wfx.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wix.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wox.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.wcx.append(nn.Linear(current_input, self.lstm_lay[i], bias
                =add_bias))
            self.ufh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uih.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uoh.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
            self.uch.append(nn.Linear(self.lstm_lay[i], self.lstm_lay[i],
                bias=False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.lstm_drop[i]))
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
                wfx_out_bn = self.bn_wfx[i](wfx_out.view(wfx_out.shape[0] *
                    wfx_out.shape[1], wfx_out.shape[2]))
                wfx_out = wfx_out_bn.view(wfx_out.shape[0], wfx_out.shape[1
                    ], wfx_out.shape[2])
                wix_out_bn = self.bn_wix[i](wix_out.view(wix_out.shape[0] *
                    wix_out.shape[1], wix_out.shape[2]))
                wix_out = wix_out_bn.view(wix_out.shape[0], wix_out.shape[1
                    ], wix_out.shape[2])
                wox_out_bn = self.bn_wox[i](wox_out.view(wox_out.shape[0] *
                    wox_out.shape[1], wox_out.shape[2]))
                wox_out = wox_out_bn.view(wox_out.shape[0], wox_out.shape[1
                    ], wox_out.shape[2])
                wcx_out_bn = self.bn_wcx[i](wcx_out.view(wcx_out.shape[0] *
                    wcx_out.shape[1], wcx_out.shape[2]))
                wcx_out = wcx_out_bn.view(wcx_out.shape[0], wcx_out.shape[1
                    ], wcx_out.shape[2])
            hiddens = []
            ct = h_init
            ht = h_init
            for k in range(x.shape[0]):
                ft = torch.sigmoid(wfx_out[k] + self.ufh[i](ht))
                it = torch.sigmoid(wix_out[k] + self.uih[i](ht))
                ot = torch.sigmoid(wox_out[k] + self.uoh[i](ht))
                ct = it * self.act[i](wcx_out[k] + self.uch[i](ht)
                    ) * drop_mask + ft * ct
                ht = ot * self.act[i](ct)
                if self.lstm_use_laynorm[i]:
                    ht = self.ln[i](ht)
                hiddens.append(ht)
            h = torch.stack(hiddens)
            if self.bidir:
                h_f = h[:, 0:int(x.shape[1] / 2)]
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class GRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(GRU, self).__init__()
        self.input_dim = inp_dim
        self.gru_lay = list(map(int, options['gru_lay'].split(',')))
        self.gru_drop = list(map(float, options['gru_drop'].split(',')))
        self.gru_use_batchnorm = list(map(strtobool, options[
            'gru_use_batchnorm'].split(',')))
        self.gru_use_laynorm = list(map(strtobool, options[
            'gru_use_laynorm'].split(',')))
        self.gru_use_laynorm_inp = strtobool(options['gru_use_laynorm_inp'])
        self.gru_use_batchnorm_inp = strtobool(options['gru_use_batchnorm_inp']
            )
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
            self.wh.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.wz.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.wr.append(nn.Linear(current_input, self.gru_lay[i], bias=
                add_bias))
            self.uh.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
            self.uz.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
            self.ur.append(nn.Linear(self.gru_lay[i], self.gru_lay[i], bias
                =False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.gru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.gru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            wr_out = self.wr[i](x)
            if self.gru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
                wr_out_bn = self.bn_wr[i](wr_out.view(wr_out.shape[0] *
                    wr_out.shape[1], wr_out.shape[2]))
                wr_out = wr_out_bn.view(wr_out.shape[0], wr_out.shape[1],
                    wr_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class liGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(liGRU, self).__init__()
        self.input_dim = inp_dim
        self.ligru_lay = list(map(int, options['ligru_lay'].split(',')))
        self.ligru_drop = list(map(float, options['ligru_drop'].split(',')))
        self.ligru_use_batchnorm = list(map(strtobool, options[
            'ligru_use_batchnorm'].split(',')))
        self.ligru_use_laynorm = list(map(strtobool, options[
            'ligru_use_laynorm'].split(',')))
        self.ligru_use_laynorm_inp = strtobool(options['ligru_use_laynorm_inp']
            )
        self.ligru_use_batchnorm_inp = strtobool(options[
            'ligru_use_batchnorm_inp'])
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
            self.wh.append(nn.Linear(current_input, self.ligru_lay[i], bias
                =add_bias))
            self.wz.append(nn.Linear(current_input, self.ligru_lay[i], bias
                =add_bias))
            self.uh.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i],
                bias=False))
            self.uz.append(nn.Linear(self.ligru_lay[i], self.ligru_lay[i],
                bias=False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.ligru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.ligru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.ligru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class minimalGRU(nn.Module):

    def __init__(self, options, inp_dim):
        super(minimalGRU, self).__init__()
        self.input_dim = inp_dim
        self.minimalgru_lay = list(map(int, options['minimalgru_lay'].split
            (',')))
        self.minimalgru_drop = list(map(float, options['minimalgru_drop'].
            split(',')))
        self.minimalgru_use_batchnorm = list(map(strtobool, options[
            'minimalgru_use_batchnorm'].split(',')))
        self.minimalgru_use_laynorm = list(map(strtobool, options[
            'minimalgru_use_laynorm'].split(',')))
        self.minimalgru_use_laynorm_inp = strtobool(options[
            'minimalgru_use_laynorm_inp'])
        self.minimalgru_use_batchnorm_inp = strtobool(options[
            'minimalgru_use_batchnorm_inp'])
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
            if self.minimalgru_use_laynorm[i] or self.minimalgru_use_batchnorm[
                i]:
                add_bias = False
            self.wh.append(nn.Linear(current_input, self.minimalgru_lay[i],
                bias=add_bias))
            self.wz.append(nn.Linear(current_input, self.minimalgru_lay[i],
                bias=add_bias))
            self.uh.append(nn.Linear(self.minimalgru_lay[i], self.
                minimalgru_lay[i], bias=False))
            self.uz.append(nn.Linear(self.minimalgru_lay[i], self.
                minimalgru_lay[i], bias=False))
            if self.minimalgru_orthinit:
                nn.init.orthogonal_(self.uh[i].weight)
                nn.init.orthogonal_(self.uz[i].weight)
            self.bn_wh.append(nn.BatchNorm1d(self.minimalgru_lay[i],
                momentum=0.05))
            self.bn_wz.append(nn.BatchNorm1d(self.minimalgru_lay[i],
                momentum=0.05))
            self.ln.append(LayerNorm(self.minimalgru_lay[i]))
            if self.bidir:
                current_input = 2 * self.minimalgru_lay[i]
            else:
                current_input = self.minimalgru_lay[i]
        self.out_dim = self.minimalgru_lay[i
            ] + self.bidir * self.minimalgru_lay[i]

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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.minimalgru_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.minimalgru_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            wz_out = self.wz[i](x)
            if self.minimalgru_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
                wz_out_bn = self.bn_wz[i](wz_out.view(wz_out.shape[0] *
                    wz_out.shape[1], wz_out.shape[2]))
                wz_out = wz_out_bn.view(wz_out.shape[0], wz_out.shape[1],
                    wz_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class RNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(RNN, self).__init__()
        self.input_dim = inp_dim
        self.rnn_lay = list(map(int, options['rnn_lay'].split(',')))
        self.rnn_drop = list(map(float, options['rnn_drop'].split(',')))
        self.rnn_use_batchnorm = list(map(strtobool, options[
            'rnn_use_batchnorm'].split(',')))
        self.rnn_use_laynorm = list(map(strtobool, options[
            'rnn_use_laynorm'].split(',')))
        self.rnn_use_laynorm_inp = strtobool(options['rnn_use_laynorm_inp'])
        self.rnn_use_batchnorm_inp = strtobool(options['rnn_use_batchnorm_inp']
            )
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
            self.wh.append(nn.Linear(current_input, self.rnn_lay[i], bias=
                add_bias))
            self.uh.append(nn.Linear(self.rnn_lay[i], self.rnn_lay[i], bias
                =False))
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
                drop_mask = torch.bernoulli(torch.Tensor(h_init.shape[0],
                    h_init.shape[1]).fill_(1 - self.rnn_drop[i]))
            else:
                drop_mask = torch.FloatTensor([1 - self.rnn_drop[i]])
            if self.use_cuda:
                h_init = h_init
                drop_mask = drop_mask
            wh_out = self.wh[i](x)
            if self.rnn_use_batchnorm[i]:
                wh_out_bn = self.bn_wh[i](wh_out.view(wh_out.shape[0] *
                    wh_out.shape[1], wh_out.shape[2]))
                wh_out = wh_out_bn.view(wh_out.shape[0], wh_out.shape[1],
                    wh_out.shape[2])
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
                h_b = flip(h[:, int(x.shape[1] / 2):x.shape[1]].contiguous(), 0
                    )
                h = torch.cat([h_f, h_b], 2)
            x = h
        return x


class CNN(nn.Module):

    def __init__(self, options, inp_dim):
        super(CNN, self).__init__()
        self.input_dim = inp_dim
        self.cnn_N_filt = list(map(int, options['cnn_N_filt'].split(',')))
        self.cnn_len_filt = list(map(int, options['cnn_len_filt'].split(',')))
        self.cnn_max_pool_len = list(map(int, options['cnn_max_pool_len'].
            split(',')))
        self.cnn_act = options['cnn_act'].split(',')
        self.cnn_drop = list(map(float, options['cnn_drop'].split(',')))
        self.cnn_use_laynorm = list(map(strtobool, options[
            'cnn_use_laynorm'].split(',')))
        self.cnn_use_batchnorm = list(map(strtobool, options[
            'cnn_use_batchnorm'].split(',')))
        self.cnn_use_laynorm_inp = strtobool(options['cnn_use_laynorm_inp'])
        self.cnn_use_batchnorm_inp = strtobool(options['cnn_use_batchnorm_inp']
            )
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
            self.ln.append(LayerNorm([N_filt, int((current_input - self.
                cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self
                .cnn_len_filt[i] + 1) / self.cnn_max_pool_len[i]), momentum
                =0.05))
            if i == 0:
                self.conv.append(nn.Conv1d(1, N_filt, len_filt))
            else:
                self.conv.append(nn.Conv1d(self.cnn_N_filt[i - 1], self.
                    cnn_N_filt[i], self.cnn_len_filt[i]))
            current_input = int((current_input - self.cnn_len_filt[i] + 1) /
                self.cnn_max_pool_len[i])
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
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.
                    conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.
                    conv[i](x), self.cnn_max_pool_len[i]))))
            if self.cnn_use_batchnorm[i] == False and self.cnn_use_laynorm[i
                ] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x),
                    self.cnn_max_pool_len[i])))
        x = x.view(batch, -1)
        return x


class SincNet(nn.Module):

    def __init__(self, options, inp_dim):
        super(SincNet, self).__init__()
        self.input_dim = inp_dim
        self.sinc_N_filt = list(map(int, options['sinc_N_filt'].split(',')))
        self.sinc_len_filt = list(map(int, options['sinc_len_filt'].split(','))
            )
        self.sinc_max_pool_len = list(map(int, options['sinc_max_pool_len']
            .split(',')))
        self.sinc_act = options['sinc_act'].split(',')
        self.sinc_drop = list(map(float, options['sinc_drop'].split(',')))
        self.sinc_use_laynorm = list(map(strtobool, options[
            'sinc_use_laynorm'].split(',')))
        self.sinc_use_batchnorm = list(map(strtobool, options[
            'sinc_use_batchnorm'].split(',')))
        self.sinc_use_laynorm_inp = strtobool(options['sinc_use_laynorm_inp'])
        self.sinc_use_batchnorm_inp = strtobool(options[
            'sinc_use_batchnorm_inp'])
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
            self.ln.append(LayerNorm([N_filt, int((current_input - self.
                sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i])]))
            self.bn.append(nn.BatchNorm1d(N_filt, int((current_input - self
                .sinc_len_filt[i] + 1) / self.sinc_max_pool_len[i]),
                momentum=0.05))
            if i == 0:
                self.conv.append(SincConv(1, N_filt, len_filt, sample_rate=
                    self.sinc_sample_rate, min_low_hz=self.sinc_min_low_hz,
                    min_band_hz=self.sinc_min_band_hz))
            else:
                self.conv.append(nn.Conv1d(self.sinc_N_filt[i - 1], self.
                    sinc_N_filt[i], self.sinc_len_filt[i]))
            current_input = int((current_input - self.sinc_len_filt[i] + 1) /
                self.sinc_max_pool_len[i])
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
                x = self.drop[i](self.act[i](self.ln[i](F.max_pool1d(self.
                    conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i]:
                x = self.drop[i](self.act[i](self.bn[i](F.max_pool1d(self.
                    conv[i](x), self.sinc_max_pool_len[i]))))
            if self.sinc_use_batchnorm[i] == False and self.sinc_use_laynorm[i
                ] == False:
                x = self.drop[i](self.act[i](F.max_pool1d(self.conv[i](x),
                    self.sinc_max_pool_len[i])))
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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=False, groups=1, sample_rate=16000,
        min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel) / self.sample_rate
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size, steps=self.kernel_size)
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2
        self.n_ = torch.arange(-n, n + 1).view(1, -1) / self.sample_rate

    def sinc(self, x):
        x_left = x[:, 0:int((x.shape[1] - 1) / 2)]
        y_left = torch.sin(x_left) / x_left
        y_right = torch.flip(y_left, dims=[1])
        sinc = torch.cat([y_left, torch.ones([x.shape[0], 1]).to(x.device),
            y_right], dim=1)
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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz / self.sample_rate + torch.abs(self.low_hz_)
        high = low + self.min_band_hz / self.sample_rate + torch.abs(self.
            band_hz_)
        f_times_t = torch.matmul(low, self.n_)
        low_pass1 = 2 * low * self.sinc(2 * math.pi * f_times_t * self.
            sample_rate)
        f_times_t = torch.matmul(high, self.n_)
        low_pass2 = 2 * high * self.sinc(2 * math.pi * f_times_t * self.
            sample_rate)
        band_pass = low_pass2 - low_pass1
        max_, _ = torch.max(band_pass, dim=1, keepdim=True)
        band_pass = band_pass / max_
        self.filters = (band_pass * self.window_).view(self.out_channels, 1,
            self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=None, groups=1)


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

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=0, dilation=1, bias=False, groups=1, sample_rate=16000,
        min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()
        if in_channels != 1:
            msg = (
                'SincConv only support one input channel (here, in_channels = {%i})'
                 % in_channels)
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
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.
            out_channels + 1)
        hz = self.to_hz(mel)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))
        n_lin = torch.linspace(0, self.kernel_size / 2 - 1, steps=int(self.
            kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.
            kernel_size)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1
            ) / self.sample_rate

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
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_
            ), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, (0)]
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)
        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)
            ) / (self.n_ / 2) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center,
            band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, (None)])
        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, self.filters, stride=self.stride,
            padding=self.padding, dilation=self.dilation, bias=None, groups=1)


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
        self.is_input_normalized = bool(strtobool(options[
            'sru_is_input_normalized']))
        self.has_skip_term = bool(strtobool(options['sru_has_skip_term']))
        self.rescale = bool(strtobool(options['sru_rescale']))
        self.highway_bias = float(options['sru_highway_bias'])
        self.n_proj = int(options['sru_n_proj'])
        self.sru = sru.SRU(self.input_dim, self.hidden_size, num_layers=
            self.num_layers, dropout=self.dropout, rnn_dropout=self.
            rnn_dropout, bidirectional=self.bidirectional, n_proj=self.
            n_proj, use_tanh=self.use_tanh, use_selu=self.use_selu,
            use_relu=self.use_relu, weight_norm=self.weight_norm,
            layer_norm=self.layer_norm, has_skip_term=self.has_skip_term,
            is_input_normalized=self.is_input_normalized, highway_bias=self
            .highway_bias, rescale=self.rescale)
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


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_santi_pdp_pase(_paritybench_base):
    pass
    def test_000(self):
        self._check(AhoCNNEncoder(*[], **{'input_dim': 4}), [torch.rand([4, 4, 64])], {})

    def test_001(self):
        self._check(AhoCNNHourGlassEncoder(*[], **{'input_dim': 4}), [torch.rand([4, 4, 64])], {})

    def test_002(self):
        self._check(LayerNorm(*[], **{'features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(MelResNet(*[], **{'res_blocks': 1, 'in_dims': 4, 'compute_dims': 4, 'res_out_dims': 4, 'pad': 4}), [torch.rand([4, 4, 64])], {})

    def test_004(self):
        self._check(SimpleResBlock1D(*[], **{'dims': 4}), [torch.rand([4, 4, 64])], {})

    def test_005(self):
        self._check(StatisticalPooling(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(UpsampleNetwork(*[], **{'feat_dims': 4}), [torch.rand([4, 4, 64])], {})

