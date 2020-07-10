import sys
_module = sys.modules[__name__]
del sys
experiments = _module
main = _module
setup = _module
src = _module
algorithms = _module
algorithm_utils = _module
autoencoder = _module
dagmm = _module
donut = _module
lstm_ad = _module
lstm_enc_dec_axl = _module
rnn_ebm = _module
data_loader = _module
datasets = _module
dataset = _module
kdd_cup = _module
multivariate_anomaly_function = _module
real_datasets = _module
synthetic_data_generator = _module
synthetic_dataset = _module
synthetic_multivariate_dataset = _module
evaluation = _module
config = _module
evaluator = _module
plotter = _module
tests = _module
test_initialization = _module

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


import abc


import logging


import random


import numpy as np


import torch


import tensorflow as tf


from tensorflow.python.client import device_lib


from torch.autograd import Variable


import torch.nn as nn


from scipy.stats import multivariate_normal


from torch.utils.data import DataLoader


from torch.utils.data.sampler import SubsetRandomSampler


import torch.nn.functional as F


class PyTorchUtils(metaclass=abc.ABCMeta):

    def __init__(self, seed, gpu):
        self.gpu = gpu
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.manual_seed(self.seed)
        self.framework = 0

    @property
    def device(self):
        return torch.device(f'cuda:{self.gpu}' if torch.cuda.is_available() and self.gpu is not None else 'cpu')

    def to_var(self, t, **kwargs):
        t = t
        return Variable(t, **kwargs)

    def to_device(self, model):
        model


class AutoEncoderModule(nn.Module, PyTorchUtils):

    def __init__(self, n_features: int, sequence_length: int, hidden_size: int, seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        input_length = n_features * sequence_length
        dec_steps = 2 ** np.arange(max(np.ceil(np.log2(hidden_size)), 2), np.log2(input_length))[1:]
        dec_setup = np.concatenate([[hidden_size], dec_steps.repeat(2), [input_length]])
        enc_setup = dec_setup[::-1]
        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in enc_setup.reshape(-1, 2)]).flatten()[:-1]
        self._encoder = nn.Sequential(*layers)
        self.to_device(self._encoder)
        layers = np.array([[nn.Linear(int(a), int(b)), nn.Tanh()] for a, b in dec_setup.reshape(-1, 2)]).flatten()[:-1]
        self._decoder = nn.Sequential(*layers)
        self.to_device(self._decoder)

    def forward(self, ts_batch, return_latent: bool=False):
        flattened_sequence = ts_batch.view(ts_batch.size(0), -1)
        enc = self._encoder(flattened_sequence.float())
        dec = self._decoder(enc)
        reconstructed_sequence = dec.view(ts_batch.size())
        return (reconstructed_sequence, enc) if return_latent else reconstructed_sequence


class DAGMMModule(nn.Module, PyTorchUtils):
    """Residual Block."""

    def __init__(self, autoencoder, n_gmm, latent_dim, seed: int, gpu: int):
        super(DAGMMModule, self).__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.add_module('autoencoder', autoencoder)
        layers = [nn.Linear(latent_dim, 10), nn.Tanh(), nn.Dropout(p=0.5), nn.Linear(10, n_gmm), nn.Softmax(dim=1)]
        self.estimation = nn.Sequential(*layers)
        self.to_device(self.estimation)
        self.register_buffer('phi', self.to_var(torch.zeros(n_gmm)))
        self.register_buffer('mu', self.to_var(torch.zeros(n_gmm, latent_dim)))
        self.register_buffer('cov', self.to_var(torch.zeros(n_gmm, latent_dim, latent_dim)))

    def relative_euclidean_distance(self, a, b, dim=1):
        return (a - b).norm(2, dim=dim) / torch.clamp(a.norm(2, dim=dim), min=1e-10)

    def forward(self, x):
        dec, enc = self.autoencoder(x, return_latent=True)
        rec_cosine = F.cosine_similarity(x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1)
        rec_euclidean = self.relative_euclidean_distance(x.view(x.shape[0], -1), dec.view(dec.shape[0], -1), dim=1)
        z = torch.cat([enc, rec_euclidean.unsqueeze(-1), rec_cosine.unsqueeze(-1)], dim=1)
        gamma = self.estimation(z)
        return enc, dec, z, gamma

    def compute_gmm_params(self, z, gamma):
        N = gamma.size(0)
        sum_gamma = torch.sum(gamma, dim=0)
        phi = sum_gamma / N
        self.phi = phi.data
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / sum_gamma.unsqueeze(-1)
        self.mu = mu.data
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0) / sum_gamma.unsqueeze(-1).unsqueeze(-1)
        self.cov = cov.data
        return phi, mu, cov

    def compute_energy(self, z, phi=None, mu=None, cov=None, size_average=True):
        if phi is None:
            phi = Variable(self.phi)
        if mu is None:
            mu = Variable(self.mu)
        if cov is None:
            cov = Variable(self.cov)
        k, d, _ = cov.size()
        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12
        for i in range(k):
            cov_k = cov[i] + self.to_var(torch.eye(d) * eps)
            pinv = np.linalg.pinv(cov_k.data.numpy())
            cov_inverse.append(Variable(torch.from_numpy(pinv)).unsqueeze(0))
            eigvals = np.linalg.eigvals(cov_k.data.cpu().numpy() * (2 * np.pi))
            if np.min(eigvals) < 0:
                logging.warning(f'Determinant was negative! Clipping Eigenvalues to 0+epsilon from {np.min(eigvals)}')
            determinant = np.prod(np.clip(eigvals, a_min=sys.float_info.epsilon, a_max=None))
            det_cov.append(determinant)
            cov_diag = cov_diag + torch.sum(1 / cov_k.diag())
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = Variable(torch.from_numpy(np.float32(np.array(det_cov))))
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max(exp_term_tmp.clamp(min=0), dim=1, keepdim=True)[0]
        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(torch.sum(self.to_var(phi.unsqueeze(0)) * exp_term / (torch.sqrt(self.to_var(det_cov)) + eps).unsqueeze(0), dim=1) + eps)
        if size_average:
            sample_energy = torch.mean(sample_energy)
        return sample_energy, cov_diag

    def loss_function(self, x, x_hat, z, gamma, lambda_energy, lambda_cov_diag):
        recon_error = torch.mean((x.view(*x_hat.shape) - x_hat) ** 2)
        phi, mu, cov = self.compute_gmm_params(z, gamma)
        sample_energy, cov_diag = self.compute_energy(z, phi, mu, cov)
        loss = recon_error + lambda_energy * sample_energy + lambda_cov_diag * cov_diag
        return loss, sample_energy, recon_error, cov_diag


class LSTMSequence(torch.nn.Module):

    def __init__(self, d, batch_size: int, len_in=1, len_out=10):
        super().__init__()
        self.d = d
        self.batch_size = batch_size
        self.len_in = len_in
        self.len_out = len_out
        self.hidden_size1 = 32
        self.hidden_size2 = 32
        self.lstm1 = torch.nn.LSTMCell(d * len_in, self.hidden_size1)
        self.lstm2 = torch.nn.LSTMCell(self.hidden_size1, self.hidden_size2)
        self.linear = torch.nn.Linear(self.hidden_size2, d * len_out)
        self.register_buffer('h_t', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('c_t', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('h_t2', torch.zeros(self.batch_size, self.hidden_size1))
        self.register_buffer('c_t2', torch.zeros(self.batch_size, self.hidden_size1))

    def forward(self, input):
        outputs = []
        h_t = Variable(self.h_t.double(), requires_grad=False)
        c_t = Variable(self.c_t.double(), requires_grad=False)
        h_t2 = Variable(self.h_t2.double(), requires_grad=False)
        c_t2 = Variable(self.c_t2.double(), requires_grad=False)
        for input_t in input.chunk(input.size(1), dim=1):
            h_t, c_t = self.lstm1(input_t.squeeze(dim=1), (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze()
        return outputs.view(input.size(0), input.size(1), self.d, self.len_out)


class LSTMEDModule(nn.Module, PyTorchUtils):

    def __init__(self, n_features: int, hidden_size: int, n_layers: tuple, use_bias: tuple, dropout: tuple, seed: int, gpu: int):
        super().__init__()
        PyTorchUtils.__init__(self, seed, gpu)
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True, num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.to_device(self.encoder)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True, num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.to_device(self.decoder)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)
        self.to_device(self.hidden2output)

    def _init_hidden(self, batch_size):
        return self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_()), self.to_var(torch.Tensor(self.n_layers[0], batch_size, self.hidden_size).zero_())

    def forward(self, ts_batch, return_latent: bool=False):
        batch_size = ts_batch.shape[0]
        enc_hidden = self._init_hidden(batch_size)
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)
        dec_hidden = enc_hidden
        output = self.to_var(torch.Tensor(ts_batch.size()).zero_())
        for i in reversed(range(ts_batch.shape[1])):
            output[:, (i), :] = self.hidden2output(dec_hidden[0][(0), :])
            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, (i)].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, (i)].unsqueeze(1), dec_hidden)
        return (output, enc_hidden[1][-1]) if return_latent else output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AutoEncoderModule,
     lambda: ([], {'n_features': 4, 'sequence_length': 4, 'hidden_size': 4, 'seed': 4, 'gpu': False}),
     lambda: ([torch.rand([16, 16])], {}),
     False),
]

class Test_KDD_OpenSource_DeepADoTS(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

