import sys
_module = sys.modules[__name__]
del sys
NVLL = _module
analysis = _module
analyze_nvrnn = _module
analyze_samples = _module
analyzer_argparse = _module
cos_loss_bow_code = _module
word_freq = _module
argparser = _module
classification = _module
label_matching = _module
model_export_to_file = _module
train_classifier = _module
data = _module
lm = _module
ng = _module
preprocess_sst_to_ptb = _module
preprocess_yelp13_to_ptb_format = _module
distribution = _module
archived_vmf = _module
empirical_kl = _module
gauss = _module
kl_cost_sheet = _module
try_bessel = _module
vmf_batch = _module
vmf_hypvae = _module
vmf_only = _module
vmf_unif = _module
framework = _module
eval_nvdm = _module
eval_nvrnn = _module
train_eval_nvdm = _module
train_eval_nvrnn = _module
model = _module
nvdm = _module
nvrnn = _module
nvll = _module
util = _module
gpu_flag = _module
hyp_tune_nvdm = _module
hyp_tune_nvrnn = _module
run_on_mav = _module
visual = _module
draw_gauss_ball = _module
draw_vmf_ball = _module
gaussian_scatter = _module
kl_tradeoff = _module
vmf_cos_dispersion = _module
vmf_stat = _module
genut = _module
load_data = _module
preprocess_ptb = _module
preprocess_yelp15 = _module
main = _module
models = _module
lm = _module
lm_vae = _module
seq2seq = _module
seq2seq_vae = _module
modules = _module
attention = _module
dec = _module
copy = _module
decode_greedy = _module
decoder = _module
decoder_beam = _module
decoder_srnn = _module
decoder_step = _module
embedding = _module
enc = _module
encoder = _module
rnn_enc = _module
vae = _module
tmp = _module
beam = _module
eval = _module
eval_lm = _module
feat = _module
helper = _module
logger = _module
struct = _module
train = _module
train_lm = _module
train_s2s = _module
dist = _module
main = _module
miao_nvdm = _module
model = _module
original_pytorch = _module
generate = _module
main = _module
model = _module
vae_proto = _module
lm = _module
main = _module
rnn_model = _module
util = _module
vMF = _module
vae_model = _module

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


import logging


import random


import time


import numpy


import torch


import numpy as np


from scipy import special as sp


import torch.nn as nn


from torch import optim


from torch.autograd import gradcheck


from scipy.special import ive


from scipy.special import iv


import scipy


import scipy.spatial.distance as ds


import math


from torch.autograd import Variable


import torch.nn.functional as F


from torch.autograd import Variable as Var


import torch.optim


from torch.autograd.variable import Variable


from torch import nn


from torch.nn import functional


from scipy.linalg import block_diag


import copy


from collections import namedtuple


from torch.nn import Module


from torch.nn import Parameter


from torch.nn import LSTMCell


from torch.nn import Embedding


from torch.nn import LSTM


from torch.nn import Linear


class Code2Code(torch.nn.Module):

    def __init__(self, inp_dim, tgt_dim):
        super().__init__()
        self.linear = torch.nn.Linear(inp_dim, tgt_dim)
        self.linear2 = torch.nn.Linear(tgt_dim, tgt_dim)
        self.loss_func = torch.nn.CosineEmbeddingLoss()

    def forward(self, inp, tgt):
        pred = self.linear(inp)
        pred = torch.nn.functional.tanh(pred)
        pred = self.linear2(pred)
        loss = 1 - torch.nn.functional.cosine_similarity(pred, tgt)
        loss = torch.mean(loss)
        return loss


class Code2Bit(torch.nn.Module):

    def __init__(self, inp_dim):
        super().__init__()
        self.linear = torch.nn.Linear(inp_dim, inp_dim)
        self.linear2 = torch.nn.Linear(inp_dim, 50)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, inp, tgt):
        pred = self.linear(inp)
        pred = torch.nn.functional.sigmoid(pred)
        pred = self.linear2(pred)
        loss = self.loss_func(pred, tgt)
        return loss, pred


def GVar(x):
    return x.to(device)


class vMF(torch.nn.Module):

    def __init__(self, lat_dim, kappa=0):
        super().__init__()
        self.lat_dim = lat_dim
        self.func_mu = torch.nn.Linear(lat_dim, lat_dim)
        self.kappa = kappa
        self.norm_eps = 1
        self.normclip = torch.nn.Hardtanh(0, 10 - 1)

    def estimate_param(self, latent_code):
        mu = self.mu(latent_code)
        return {'mu': mu}

    def compute_KLD(self):
        kld = GVar(torch.zeros(1))
        return kld

    def vmf_unif_sampler(self, mu):
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            munoise = self.add_norm_noise(munorm, self.norm_eps)
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                w = self._sample_weight(self.kappa, id_dim)
                wtorch = GVar(w * torch.ones(id_dim))
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)
                scale_factr = torch.sqrt(GVar(torch.ones(id_dim)) - torch.
                    pow(wtorch, 2))
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munoise
            else:
                rand_draw = GVar(torch.randn(id_dim))
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(
                    id_dim)
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * GVar(rand_norms)
            result_list.append(sampled_vec)
        return torch.stack(result_list, 0)

    def vmf_sampler(self, mu):
        mu = mu.cpu()
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                w = vMF.sample_vmf_w(self.kappa, id_dim)
                wtorch = GVar(w * torch.ones(id_dim))
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)
                scale_factr = torch.sqrt(GVar(torch.ones(id_dim)) - torch.
                    pow(wtorch, 2))
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munorm
            else:
                rand_draw = GVar(torch.randn(id_dim))
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(
                    id_dim)
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * GVar(rand_norms)
            result_list.append(sampled_vec)
        return torch.stack(result_list, 0)

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        kld = self.compute_KLD()
        vecs = []
        for ns in range(n_sample):
            vec = self.vmf_unif_sampler(tup['mu'])
            vecs.append(vec)
        return tup, kld, vecs

    @staticmethod
    def _sample_weight(kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1
        b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1.0 - b) / (1.0 + b)
        c = kappa * x + dim * np.log(1 - x ** 2)
        while True:
            z = np.random.beta(dim / 2.0, dim / 2.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GVar(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)

    @staticmethod
    def sample_vmf_v(mu):
        import scipy.linalg as la
        mat = np.matrix(mu)
        if mat.shape[1] > mat.shape[0]:
            mat = mat.T
        U, _, _ = la.svd(mat)
        nu = np.matrix(np.random.randn(mat.shape[0])).T
        x = np.dot(U[:, 1:], nu[1:, :])
        return x / la.norm(x)

    @staticmethod
    def sample_vmf_w(kappa, m):
        b = (-2 * kappa + np.sqrt(4.0 * kappa ** 2 + (m - 1) ** 2)) / (m - 1)
        a = (m - 1 + 2 * kappa + np.sqrt(4 * kappa ** 2 + (m - 1) ** 2)) / 4
        d = 4 * a * b / (1 + b) - (m - 1) * np.log(m - 1)
        while True:
            z = np.random.beta(0.5 * (m - 1), 0.5 * (m - 1))
            W = (1 - (1 + b) * z) / (1 + (1 - b) * z)
            T = 2 * a * b / (1 + (1 - b) * z)
            u = np.random.uniform(0, 1)
            if (m - 1) * np.log(T) - T + d >= np.log(u):
                return W

    def add_norm_noise(self, munorm, eps):
        """
        KL loss is - log(maxvalue/eps)
        cut at maxvalue-eps, and add [0,eps] noise.
        """
        trand = torch.rand(1).expand(munorm.size()) * eps
        return self.normclip(munorm) + GVar(trand)


class Gauss(nn.Module):

    def __init__(self, hid_dim, lat_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        self.func_mean = torch.nn.Linear(hid_dim, lat_dim)
        self.func_logvar = torch.nn.Linear(hid_dim, lat_dim)

    def estimate_param(self, latent_code):
        mean = self.func_mean(latent_code)
        logvar = self.func_logvar(latent_code)
        return {'mean': mean, 'logvar': logvar}

    def compute_KLD(self, tup):
        mean = tup['mean']
        logvar = tup['logvar']
        kld = -0.5 * torch.sum(1 - torch.mul(mean, mean) + 2 * logvar -
            torch.exp(2 * logvar), dim=1)
        return kld

    def sample_cell(self, batch_size):
        eps = torch.autograd.Variable(torch.normal(torch.zeros((batch_size,
            self.lat_dim))))
        eps
        return eps.unsqueeze(0)

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        mean = tup['mean']
        logvar = tup['logvar']
        kld = self.compute_KLD(tup)
        if n_sample == 1:
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            return tup, kld, vec
        vecs = []
        for ns in range(n_sample):
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            vecs.append(vec)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

    def get_aux_loss_term(self, tup):
        return torch.from_numpy(np.zeros([1]))


class vMF(torch.nn.Module):

    def __init__(self, hid_dim, lat_dim, kappa=1):
        """
        von Mises-Fisher distribution class with batch support and manual tuning kappa value.
        Implementation follows description of my paper and Guu's.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        self.kappa = kappa
        self.func_mu = torch.nn.Linear(hid_dim, lat_dim)
        self.kld = GVar(torch.from_numpy(vMF._vmf_kld(kappa, lat_dim)).float())
        None

    def estimate_param(self, latent_code):
        ret_dict = {}
        ret_dict['kappa'] = self.kappa
        mu = self.func_mu(latent_code)
        norm = torch.norm(mu, 2, 1, keepdim=True)
        mu_norm_sq_diff_from_one = torch.pow(torch.add(norm, -1), 2)
        redundant_norm = torch.sum(mu_norm_sq_diff_from_one, dim=1, keepdim
            =True)
        ret_dict['norm'] = torch.ones_like(mu)
        ret_dict['redundant_norm'] = redundant_norm
        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        ret_dict['mu'] = mu
        return ret_dict

    def compute_KLD(self, tup, batch_sz):
        return self.kld.expand(batch_sz)

    @staticmethod
    def _vmf_kld(k, d):
        tmp = (k * ((sp.iv(d / 2.0 + 1.0, k) + sp.iv(d / 2.0, k) * d / (2.0 *
            k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) + d * np.log(k) / 2.0 -
            np.log(sp.iv(d / 2.0, k)) - sp.loggamma(d / 2 + 1) - d * np.log
            (2) / 2).real
        if tmp != tmp:
            exit()
        return np.array([tmp])

    @staticmethod
    def _vmf_kld_davidson(k, d):
        """
        This should be the correct KLD.
        Empirically we find that _vmf_kld (as in the Guu paper) only deviates a little (<2%) in most cases we use.
        """
        tmp = k * sp.iv(d / 2, k) / sp.iv(d / 2 - 1, k) + (d / 2 - 1
            ) * torch.log(k) - torch.log(sp.iv(d / 2 - 1, k)) + np.log(np.pi
            ) * d / 2 + np.log(2) - sp.loggamma(d / 2).real - d / 2 * np.log(
            2 * np.pi)
        if tmp != tmp:
            exit()
        return np.array([tmp])

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        mu = tup['mu']
        norm = tup['norm']
        kappa = tup['kappa']
        kld = self.compute_KLD(tup, batch_sz)
        vecs = []
        if n_sample == 1:
            return tup, kld, self.sample_cell(mu, norm, kappa)
        for n in range(n_sample):
            sample = self.sample_cell(mu, norm, kappa)
            vecs.append(sample)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

    def sample_cell(self, mu, norm, kappa):
        batch_sz, lat_dim = mu.size()
        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        w = self._sample_weight_batch(kappa, lat_dim, batch_sz)
        w = w.unsqueeze(1)
        w_var = GVar(w * torch.ones(batch_sz, lat_dim))
        v = self._sample_ortho_batch(mu, lat_dim)
        scale_factr = torch.sqrt(GVar(torch.ones(batch_sz, lat_dim)) -
            torch.pow(w_var, 2))
        orth_term = v * scale_factr
        muscale = mu * w_var
        sampled_vec = orth_term + muscale
        return sampled_vec.unsqueeze(0)

    def _sample_weight_batch(self, kappa, dim, batch_sz=1):
        result = torch.FloatTensor(batch_sz)
        for b in range(batch_sz):
            result[b] = self._sample_weight(kappa, dim)
        return result

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1
        b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1.0 - b) / (1.0 + b)
        c = kappa * x + dim * np.log(1 - x ** 2)
        while True:
            z = np.random.beta(dim / 2.0, dim / 2.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
                return w

    def _sample_ortho_batch(self, mu, dim):
        """

        :param mu: Variable, [batch size, latent dim]
        :param dim: scala. =latent dim
        :return:
        """
        _batch_sz, _lat_dim = mu.size()
        assert _lat_dim == dim
        squeezed_mu = mu.unsqueeze(1)
        v = GVar(torch.randn(_batch_sz, dim, 1))
        rescale_val = torch.bmm(squeezed_mu, v).squeeze(2)
        proj_mu_v = mu * rescale_val
        ortho = v.squeeze() - proj_mu_v
        ortho_norm = torch.norm(ortho, p=2, dim=1, keepdim=True)
        y = ortho / ortho_norm
        return y

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GVar(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)


class BesselIv(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, dim, kappa):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(dim, kappa)
        kappa_copy = kappa.clone()
        m = sp.iv(dim, kappa_copy)
        x = torch.tensor(m).to(device)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        dim, kappa = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * (bessel_iv(dim - 1, kappa) + bessel_iv(dim + 1,
            kappa)) * 0.5
        return None, grad


bessel_iv = BesselIv.apply


class VmfDiff(torch.nn.Module):

    def __init__(self, hid_dim, lat_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        self.func_mu = torch.nn.Linear(hid_dim, lat_dim)
        self.func_kappa = torch.nn.Linear(hid_dim, 1)
        self.nonneg = torch.nn.ReLU()

    def estimate_param(self, latent_code):
        ret_dict = {}
        ret_dict['kappa'] = torch.max(torch.min(self.func_kappa(latent_code
            ) * 10 + 50, torch.tensor(150.0)), torch.tensor(10.0))
        mu = self.func_mu(latent_code)
        norm = torch.norm(mu, 2, 1, keepdim=True)
        mu_norm_sq_diff_from_one = torch.pow(torch.add(norm, -1), 2)
        redundant_norm = torch.sum(mu_norm_sq_diff_from_one, dim=1, keepdim
            =True)
        ret_dict['norm'] = torch.ones_like(mu)
        ret_dict['redundant_norm'] = redundant_norm
        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        ret_dict['mu'] = mu
        return ret_dict

    def compute_KLD(self, tup, batch_sz):
        kappa = tup['kappa']
        d = self.lat_dim
        rt_bag = []
        const = torch.tensor(np.log(np.pi) * d / 2 + np.log(2) - sp.
            loggamma(d / 2).real - d / 2 * np.log(2 * np.pi))
        d = torch.tensor([d], dtype=torch.float)
        batchsz = kappa.size()[0]
        rt_tensor = torch.zeros(batchsz)
        for k_idx in range(batchsz):
            k = kappa[k_idx]
            first = k * bessel_iv(d / 2, k) / bessel_iv(d / 2 - 1, k)
            second = (d / 2 - 1) * torch.log(k) - torch.log(bessel_iv(d / 2 -
                1, k))
            combin = first + second + const
            rt_tensor[k_idx] = combin
        return rt_tensor

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        mu = tup['mu']
        norm = tup['norm']
        kappa = tup['kappa']
        kld = self.compute_KLD(tup, batch_sz)
        vecs = []
        kappa_clone = kappa.detach().cpu().numpy()
        if n_sample == 1:
            return tup, kld, self.sample_cell(mu, norm, kappa_clone)
        for n in range(n_sample):
            sample = self.sample_cell(mu, norm, kappa_clone)
            vecs.append(sample)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

    def sample_cell(self, mu, norm, kappa):
        batch_sz, lat_dim = mu.size()
        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        w = self._sample_weight_batch(kappa, lat_dim, batch_sz)
        w = w.unsqueeze(1)
        w_var = GVar(w * torch.ones(batch_sz, lat_dim))
        v = self._sample_ortho_batch(mu, lat_dim)
        scale_factr = torch.sqrt(GVar(torch.ones(batch_sz, lat_dim)) -
            torch.pow(w_var, 2))
        orth_term = v * scale_factr
        muscale = mu * w_var
        sampled_vec = orth_term + muscale
        return sampled_vec.unsqueeze(0)

    def _sample_weight_batch(self, kappa, dim, batch_sz=1):
        result = np.zeros(batch_sz)
        for b in range(batch_sz):
            result[b] = self._sample_weight(kappa[b], dim)
        return torch.from_numpy(result).float()

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1
        b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1.0 - b) / (1.0 + b)
        c = kappa * x + dim * np.log(1 - x ** 2)
        while True:
            z = np.random.beta(dim / 2.0, dim / 2.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
                return w

    def _sample_ortho_batch(self, mu, dim):
        """

        :param mu: Variable, [batch size, latent dim]
        :param dim: scala. =latent dim
        :return:
        """
        _batch_sz, _lat_dim = mu.size()
        assert _lat_dim == dim
        squeezed_mu = mu.unsqueeze(1)
        v = GVar(torch.randn(_batch_sz, dim, 1))
        rescale_val = torch.bmm(squeezed_mu, v).squeeze(2)
        proj_mu_v = mu * rescale_val
        ortho = v.squeeze() - proj_mu_v
        ortho_norm = torch.norm(ortho, p=2, dim=1, keepdim=True)
        y = ortho / ortho_norm
        return y

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GVar(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)


class vMF(torch.nn.Module):

    def __init__(self, hid_dim, lat_dim, kappa=1):
        super().__init__()
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        self.kappa = kappa
        self.func_mu = torch.nn.Linear(hid_dim, lat_dim)
        self.kld = GVar(torch.from_numpy(vMF._vmf_kld(kappa, lat_dim)).float())
        None

    def estimate_param(self, latent_code):
        ret_dict = {}
        ret_dict['kappa'] = self.kappa
        mu = self.func_mu(latent_code)
        norm = torch.norm(mu, 2, 1, keepdim=True)
        mu_norm_sq_diff_from_one = torch.pow(torch.add(norm, -1), 2)
        redundant_norm = torch.sum(mu_norm_sq_diff_from_one, dim=1, keepdim
            =True)
        ret_dict['norm'] = torch.ones_like(mu)
        ret_dict['redundant_norm'] = redundant_norm
        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        ret_dict['mu'] = mu
        return ret_dict

    def compute_KLD(self, tup, batch_sz):
        return self.kld.expand(batch_sz)

    @staticmethod
    def _vmf_kld(k, d):
        tmp = (k * ((sp.iv(d / 2.0 + 1.0, k) + sp.iv(d / 2.0, k) * d / (2.0 *
            k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) + d * np.log(k) / 2.0 -
            np.log(sp.iv(d / 2.0, k)) - sp.loggamma(d / 2 + 1) - d * np.log
            (2) / 2).real
        if tmp != tmp:
            exit()
        return np.array([tmp])

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        mu = tup['mu']
        norm = tup['norm']
        kappa = tup['kappa']
        kld = self.compute_KLD(tup, batch_sz)
        vecs = []
        if n_sample == 1:
            return tup, kld, self.sample_cell(mu, norm, kappa)
        for n in range(n_sample):
            sample = self.sample_cell(mu, norm, kappa)
            vecs.append(sample)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

    def sample_cell(self, mu, norm, kappa):
        batch_sz, lat_dim = mu.size()
        result = []
        sampled_vecs = GVar(torch.FloatTensor(batch_sz, lat_dim))
        for b in range(batch_sz):
            this_mu = mu[b]
            this_mu = this_mu / torch.norm(this_mu, p=2)
            w = self._sample_weight(kappa, lat_dim)
            w_var = GVar(w * torch.ones(lat_dim))
            v = self._sample_orthonormal_to(this_mu, lat_dim)
            scale_factr = torch.sqrt(GVar(torch.ones(lat_dim)) - torch.pow(
                w_var, 2))
            orth_term = v * scale_factr
            muscale = this_mu * w_var
            sampled_vec = orth_term + muscale
            sampled_vecs[b] = sampled_vec
        return sampled_vecs.unsqueeze(0)

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1
        b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1.0 - b) / (1.0 + b)
        c = kappa * x + dim * np.log(1 - x ** 2)
        while True:
            z = np.random.beta(dim / 2.0, dim / 2.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GVar(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)


class unif_vMF(torch.nn.Module):

    def __init__(self, hid_dim, lat_dim, kappa=1, norm_max=2, norm_func=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.lat_dim = lat_dim
        self.kappa = kappa
        self.func_mu = torch.nn.Linear(hid_dim, lat_dim)
        self.func_norm = torch.nn.Linear(hid_dim, 1)
        self.norm_eps = 1
        self.norm_max = norm_max
        self.norm_clip = torch.nn.Hardtanh(1e-05, self.norm_max - self.norm_eps
            )
        self.norm_func = norm_func
        kld_value = unif_vMF._vmf_kld(kappa, lat_dim) + unif_vMF._uniform_kld(
            0.0, self.norm_eps, 0.0, self.norm_max)
        self.kld = GVar(torch.from_numpy(np.array([kld_value])).float())
        None

    def estimate_param(self, latent_code):
        """
        Compute z_dir and z_norm for vMF.
        norm_func means using another NN to compute the norm (batchsz, 1)
        :param latent_code: batchsz, hidden size
        :return: dict with kappa, mu(batchsz, lat_dim), norm (duplicate in row) (batchsz, lat_dim), (opt)redundant_norm
        """
        ret_dict = {}
        ret_dict['kappa'] = self.kappa
        mu = self.func_mu(latent_code)
        mu = mu / torch.norm(mu, p=2, dim=1, keepdim=True)
        ret_dict['mu'] = mu
        norm = self.func_norm(latent_code)
        clipped_norm = self.norm_clip(norm)
        redundant_norm = torch.max(norm - clipped_norm, torch.zeros_like(norm))
        ret_dict['norm'] = clipped_norm.expand_as(mu)
        ret_dict['redundant_norm'] = redundant_norm
        return ret_dict

    def compute_KLD(self, tup, batch_sz):
        return self.kld.expand(batch_sz)

    @staticmethod
    def _vmf_kld(k, d):
        tmp = (k * ((sp.iv(d / 2.0 + 1.0, k) + sp.iv(d / 2.0, k) * d / (2.0 *
            k)) / sp.iv(d / 2.0, k) - d / (2.0 * k)) + d * np.log(k) / 2.0 -
            np.log(sp.iv(d / 2.0, k)) - sp.loggamma(d / 2 + 1) - d * np.log
            (2) / 2).real
        return tmp

    @staticmethod
    def _uniform_kld(x1, x2, y1, y2):
        if x1 < y1 or x2 > y2:
            raise Exception('KLD is infinite: Unif([' + repr(x1) + ',' +
                repr(x2) + '])||Unif([' + repr(y1) + ',' + repr(y2) + '])')
        return np.log((y2 - y1) / (x2 - x1))

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        mu = tup['mu']
        norm = tup['norm']
        kappa = tup['kappa']
        kld = self.compute_KLD(tup, batch_sz)
        vecs = []
        if n_sample == 1:
            return tup, kld, self.sample_cell(mu, norm, kappa)
        for n in range(n_sample):
            sample = self.sample_cell(mu, norm, kappa)
            vecs.append(sample)
        vecs = torch.cat(vecs, dim=0)
        return tup, kld, vecs

    def sample_cell(self, mu, norm, kappa):
        """

        :param mu: z_dir (batchsz, lat_dim) . ALREADY normed.
        :param norm: z_norm (batchsz, lat_dim).
        :param kappa: scalar
        :return:
        """
        """vMF sampler in pytorch.
        http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
        Args:
            mu (Tensor): of shape (batch_size, 2*word_dim)
            kappa (Float): controls dispersion. kappa of zero is no dispersion.
        """
        batch_sz, lat_dim = mu.size()
        norm_with_noise = self.add_norm_noise_batch(norm, self.norm_eps)
        w = self._sample_weight_batch(kappa, lat_dim, batch_sz)
        w = w.unsqueeze(1)
        w_var = GVar(w * torch.ones(batch_sz, lat_dim))
        v = self._sample_ortho_batch(mu, lat_dim)
        scale_factr = torch.sqrt(GVar(torch.ones(batch_sz, lat_dim)) -
            torch.pow(w_var, 2))
        orth_term = v * scale_factr
        muscale = mu * w_var
        sampled_vec = (orth_term + muscale) * norm_with_noise
        return sampled_vec.unsqueeze(0)

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1
        b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1.0 - b) / (1.0 + b)
        c = kappa * x + dim * np.log(1 - x ** 2)
        while True:
            z = np.random.beta(dim / 2.0, dim / 2.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GVar(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)

    def add_norm_noise(self, munorm, eps):
        """
        KL loss is - log(maxvalue/eps)
        cut at maxvalue-eps, and add [0,eps] noise.
        """
        trand = torch.rand(1).expand(munorm.size()) * eps
        return munorm + GVar(trand)

    def add_norm_noise_batch(self, mu_norm, eps):
        batch_sz, lat_dim = mu_norm.size()
        noise = GVar(torch.FloatTensor(batch_sz, lat_dim).uniform_(0, eps))
        noised_norm = noise + mu_norm
        return noised_norm

    def _sample_weight_batch(self, kappa, dim, batch_sz=1):
        result = torch.FloatTensor(batch_sz)
        for b in range(batch_sz):
            result[b] = self._sample_weight(kappa, dim)
        return result

    def _sample_ortho_batch(self, mu, dim):
        """

        :param mu: Variable, [batch size, latent dim]
        :param dim: scala. =latent dim
        :return:
        """
        _batch_sz, _lat_dim = mu.size()
        assert _lat_dim == dim
        squeezed_mu = mu.unsqueeze(1)
        v = GVar(torch.randn(_batch_sz, dim, 1))
        rescale_val = torch.bmm(squeezed_mu, v).squeeze(2)
        proj_mu_v = mu * rescale_val
        ortho = v.squeeze() - proj_mu_v
        ortho_norm = torch.norm(ortho, p=2, dim=1, keepdim=True)
        y = ortho / ortho_norm
        return y

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GVar(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)


class BowVAE(torch.nn.Module):

    def __init__(self, args, vocab_size, n_hidden, n_lat, n_sample, dist):
        super(BowVAE, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_lat = n_lat
        self.n_sample = n_sample
        self.dist_type = dist
        self.dropout = torch.nn.Dropout(p=args.dropout)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.enc_vec = torch.nn.Linear(self.vocab_size, self.n_hidden)
        self.active = torch.nn.Tanh()
        self.enc_vec_2 = torch.nn.Linear(self.n_hidden, self.n_hidden)
        if self.dist_type == 'nor':
            self.dist = Gauss(n_hidden, n_lat)
        elif self.dist_type == 'vmf':
            self.dist = vMF(n_hidden, n_lat, kappa=self.args.kappa)
        elif self.dist_type == 'unifvmf':
            self.dist = unif_vMF(n_hidden, n_lat, kappa=self.args.kappa,
                norm_func=self.args.norm_func)
        elif self.dist_type == 'sph':
            self.dist = VmfDiff(n_hidden, n_lat)
        else:
            raise NotImplementedError
        self.dec_linear = torch.nn.Linear(self.n_lat, self.n_hidden)
        self.dec_act = torch.nn.Tanh()
        self.out = torch.nn.Linear(self.n_hidden, self.vocab_size)

    def forward(self, x):
        batch_sz = x.size()[0]
        linear_x = self.enc_vec(x)
        linear_x = self.dropout(linear_x)
        active_x = self.active(linear_x)
        linear_x_2 = self.enc_vec_2(active_x)
        tup, kld, vecs = self.dist.build_bow_rep(linear_x_2, self.n_sample)
        if 'redundant_norm' in tup:
            aux_loss = tup['redundant_norm'].view(batch_sz)
        else:
            aux_loss = GVar(torch.zeros(batch_sz))
        avg_cos = BowVAE.check_dispersion(vecs)
        avg_norm = torch.mean(tup['norm'])
        tup['avg_cos'] = avg_cos
        tup['avg_norm'] = avg_norm
        flatten_vecs = vecs.view(self.n_sample * batch_sz, self.n_lat)
        flatten_vecs = self.dec_act(self.dec_linear(flatten_vecs))
        logit = self.dropout(self.out(flatten_vecs))
        logit = torch.nn.functional.log_softmax(logit, dim=1)
        logit = logit.view(self.n_sample, batch_sz, self.vocab_size)
        flatten_x = x.unsqueeze(0).expand(self.n_sample, batch_sz, self.
            vocab_size)
        error = torch.mul(flatten_x, logit)
        error = torch.mean(error, dim=0)
        recon_loss = -torch.sum(error, dim=1, keepdim=False)
        return recon_loss, kld, aux_loss, tup, vecs

    @staticmethod
    def cos(a, b):
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

    @staticmethod
    def check_dispersion(vecs):
        num_sam = 10
        cos_sim = 0
        for i in range(num_sam):
            idx1 = random.randint(0, vecs.size(1) - 1)
            while True:
                idx2 = random.randint(0, vecs.size(1) - 1)
                if idx1 != idx2:
                    break
            cos_sim += BowVAE.cos(vecs[0][idx1], vecs[0][idx2])
        return cos_sim / num_sam


cos = torch.nn.CosineSimilarity()


def check_dispersion(vecs, num_sam=10):
    """
    Check the dispersion of vecs.
    :param vecs:  [n_samples, batch_sz, lat_dim]
    :param num_sam: number of samples to check
    :return:
    """
    if vecs.size(1) <= 2:
        return GVar(torch.zeros(1))
    cos_sim = 0
    for i in range(num_sam):
        idx1 = random.randint(0, vecs.size(1) - 1)
        while True:
            idx2 = random.randint(0, vecs.size(1) - 1)
            if idx1 != idx2:
                break
        cos_sim += cos(vecs[0][idx1], vecs[0][idx2])
    return cos_sim / num_sam


class RNNVAE(nn.Module):
    """Container module with an optional encoder, a prob latent module, and a RNN decoder."""

    def __init__(self, args, enc_type, ntoken, ninp, nhid, lat_dim, nlayers,
        dropout=0.5, tie_weights=False, input_z=False, mix_unk=0, condition
        =False, input_cd_bow=0, input_cd_bit=0):
        assert not condition or condition and (input_cd_bow > 1 or 
            input_cd_bit > 1)
        assert type(input_cd_bit) == int and input_cd_bit >= 0
        assert type(input_cd_bow) == int and input_cd_bow >= 0
        super(RNNVAE, self).__init__()
        self.FLAG_train = True
        self.args = args
        self.enc_type = enc_type
        None
        try:
            self.bi = args.bi
        except AttributeError:
            self.bi = True
        self.input_z = input_z
        self.condition = condition
        self.input_cd_bow = input_cd_bow
        self.input_cd_bit = input_cd_bit
        self.lat_dim = lat_dim
        self.nhid = nhid
        self.nlayers = nlayers
        self.ninp = ninp
        self.ntoken = ntoken
        self.dist_type = args.dist
        self.drop = nn.Dropout(dropout)
        self.emb = nn.Embedding(ntoken, ninp)
        if input_cd_bit > 1:
            self.emb_bit = nn.Embedding(5, input_cd_bit)
        if input_cd_bow > 1:
            self.nn_bow = nn.Linear(ninp, input_cd_bow)
        self.decoder_out = nn.Linear(nhid, ntoken)
        if self.dist_type == 'nor' or 'vmf' or 'sph' or 'unifvmf':
            _factor = 1
            _inp_dim = ninp
            if input_cd_bit > 1:
                _inp_dim += int(input_cd_bit)
            if enc_type == 'lstm' or enc_type == 'gru':
                if enc_type == 'lstm':
                    _factor *= 2
                    self.enc_rnn = nn.LSTM(_inp_dim, nhid, 1, bidirectional
                        =self.bi, dropout=dropout)
                elif enc_type == 'gru':
                    self.enc_rnn = nn.GRU(_inp_dim, nhid, 1, bidirectional=
                        self.bi, dropout=dropout)
                else:
                    raise NotImplementedError
                if self.bi:
                    _factor *= 2
                self.hid4_to_lat = nn.Linear(_factor * nhid, nhid)
                self.enc = self.rnn_funct
            elif enc_type == 'bow':
                self.enc = self.bow_funct
                self.hid4_to_lat = nn.Linear(ninp, nhid)
            else:
                raise NotImplementedError
        elif self.dist_type == 'zero':
            pass
        else:
            raise NotImplementedError
        if args.dist == 'nor':
            self.dist = Gauss(nhid, lat_dim)
        elif args.dist == 'vmf':
            self.dist = vMF(nhid, lat_dim, kappa=self.args.kappa)
        elif args.dist == 'sph':
            self.dist = VmfDiff(nhid, lat_dim)
        elif args.dist == 'zero':
            pass
        elif args.dist == 'unifvmf':
            self.dist = unif_vMF(nhid, lat_dim, kappa=self.args.kappa,
                norm_max=self.args.norm_max)
        else:
            raise NotImplementedError
        self.mix_unk = mix_unk
        self.z_to_h = nn.Linear(lat_dim, nhid * nlayers)
        self.z_to_c = nn.Linear(lat_dim, nhid * nlayers)
        _dec_rnn_inp_dim = ninp
        if input_z:
            _dec_rnn_inp_dim += lat_dim
        if input_cd_bit > 1:
            _dec_rnn_inp_dim += int(input_cd_bit)
        if input_cd_bow > 1:
            _dec_rnn_inp_dim += int(input_cd_bow)
        self.decoder_rnn = nn.LSTM(_dec_rnn_inp_dim, nhid, nlayers, dropout
            =dropout)
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder_out.weight = self.emb.weight
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def bow_funct(self, x):
        y = torch.mean(x, dim=0)
        y = self.hid4_to_lat(y)
        y = torch.nn.functional.tanh(y)
        return y

    def rnn_funct(self, x):
        batch_sz = x.size()[1]
        if self.enc_type == 'lstm':
            output, (h_n, c_n) = self.enc_rnn(x)
            if self.bi:
                concated_h_c = torch.cat((h_n[0], h_n[1], c_n[0], c_n[1]),
                    dim=1)
            else:
                concated_h_c = torch.cat((h_n[0], c_n[0]), dim=1)
        elif self.enc_type == 'gru':
            output, h_n = self.enc_rnn(x)
            if self.bi:
                concated_h_c = torch.cat((h_n[0], h_n[1]), dim=1)
            else:
                concated_h_c = h_n[0]
        else:
            raise NotImplementedError
        return self.hid4_to_lat(concated_h_c)

    def dropword(self, emb, drop_rate=0.3):
        """
        Mix the ground truth word with UNK.
        If drop rate = 1, no ground truth info is used. (Fly mode)
        :param emb:
        :param drop_rate: 0 - no drop; 1 - full drop, all UNK
        :return: mixed embedding
        """
        UNKs = GVar(torch.ones(emb.size()[0], emb.size()[1]).long() * 2)
        UNKs = self.emb(UNKs)
        masks = numpy.random.binomial(1, drop_rate, size=(emb.size()[0],
            emb.size()[1]))
        masks = GVar(torch.FloatTensor(masks)).unsqueeze(2).expand_as(UNKs)
        emb = emb * (1 - masks) + UNKs * masks
        return emb

    def forward(self, inp, target, bit=None):
        """
        Forward with ground truth (maybe mixed with UNK) as input.
        :param inp:  seq_len, batch_sz
        :param target: seq_len, batch_sz
        :param bit: 1, batch_sz
        :return:
        """
        seq_len, batch_sz = inp.size()
        emb = self.drop(self.emb(inp))
        if self.input_cd_bow > 1:
            bow = self.enc_bow(emb)
        else:
            bow = None
        if self.input_cd_bit > 1:
            bit = self.enc_bit(bit)
        else:
            bit = None
        h = self.forward_enc(emb, bit)
        tup, kld, vecs = self.forward_build_lat(h, self.args.nsample)
        if 'redundant_norm' in tup:
            aux_loss = tup['redundant_norm'].view(batch_sz)
        else:
            aux_loss = GVar(torch.zeros(batch_sz))
        if 'norm' not in tup:
            tup['norm'] = GVar(torch.zeros(batch_sz))
        avg_cos = check_dispersion(vecs)
        tup['avg_cos'] = avg_cos
        avg_norm = torch.mean(tup['norm'])
        tup['avg_norm'] = avg_norm
        vec = torch.mean(vecs, dim=0)
        decoded = self.forward_decode_ground(emb, vec, bit, bow)
        flatten_decoded = decoded.view(-1, self.ntoken)
        flatten_target = target.view(-1)
        loss = self.criterion(flatten_decoded, flatten_target)
        return loss, kld, aux_loss, tup, vecs, decoded

    def enc_bit(self, bit):
        if self.input_cd_bit > 1:
            return self.emb_bit(bit)
        else:
            return None

    def enc_bow(self, emb):
        if self.input_cd_bow > 1:
            x = self.nn_bow(torch.mean(emb, dim=0))
            return x
        else:
            return None

    def forward_enc(self, inp, bit=None):
        """
        Given sequence, encode and yield a representation with hid_dim
        :param inp:
        :return:
        """
        seq_len, batch_sz = inp.size()[0:2]
        if self.dist_type == 'zero':
            return torch.zeros(batch_sz)
        if bit is not None:
            bit = bit.unsqueeze(0).expand(seq_len, batch_sz, -1)
            inp = torch.cat([inp, bit], dim=2)
        h = self.enc(inp)
        return h

    def forward_build_lat(self, hidden, nsample=3):
        """

        :param hidden:
        :return: tup, kld [batch_sz], out [nsamples, batch_sz, lat_dim]
        """
        if self.args.dist == 'nor':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'vmf':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'unifvmf':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'vmf_diff':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'sph':
            tup, kld, out = self.dist.build_bow_rep(hidden, nsample)
        elif self.args.dist == 'zero':
            out = GVar(torch.zeros(1, hidden.size()[0], self.lat_dim))
            tup = {}
            kld = GVar(torch.zeros(1))
        else:
            raise NotImplementedError
        return tup, kld, out

    def forward_decode_ground(self, emb, lat_code, bit=None, bow=None):
        """

        :param emb: seq, batch, ninp
        :param lat_code: batch, nlat
        :param bit:
        :param bow:
        :return:
        """
        seq_len, batch_sz, _ = emb.size()
        if self.mix_unk > 0.001:
            emb = self.dropword(emb, self.mix_unk)
        if self.input_z:
            lat_to_cat = lat_code.unsqueeze(0).expand(seq_len, batch_sz, -1)
            emb = torch.cat([emb, lat_to_cat], dim=2)
        if self.input_cd_bow > 1:
            bow = bow.unsqueeze(0).expand(seq_len, batch_sz, -1)
            emb = torch.cat([emb, bow], dim=2)
        if self.input_cd_bit > 1:
            bit = bit.unsqueeze(0).expand(seq_len, batch_sz, -1)
            emb = torch.cat([emb, bit], dim=2)
        init_h, init_c = self.convert_z_to_hidden(lat_code, batch_sz)
        output, hidden = self.decoder_rnn(emb, (init_h, init_c))
        output = self.drop(output)
        decoded = self.decoder_out(output.view(output.size(0) * output.size
            (1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded

    def encode(self, emb):
        """

        :param emb:
        :return: batch_sz, lat_dim
        """
        batch_sz = emb.size()[1]
        _, hidden = self.enc_rnn(emb)
        h = hidden[0]
        c = hidden[1]
        assert h.size()[0] == self.nlayers * 2
        assert h.size()[1] == batch_sz
        x = torch.cat((h, c), dim=0).permute(1, 0, 2).contiguous().view(
            batch_sz, -1)
        if self.dist == 'nor':
            return self.fc_mu(x), self.fc_logvar(x)
        elif self.dist == 'vmf':
            return self.fc(x)
        else:
            raise NotImplementedError

    def convert_z_to_hidden(self, z, batch_sz):
        """

        :param z:   batch, lat_dim
        :param batch_sz:
        :return:
        """
        h = self.z_to_h(z).view(batch_sz, self.nlayers, -1).permute(1, 0, 2
            ).contiguous()
        c = self.z_to_c(z).view(batch_sz, self.nlayers, -1).permute(1, 0, 2
            ).contiguous()
        return h, c

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return GVar(weight.new(self.nlayers, bsz, self.nhid).zero_()
                ), GVar(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return GVar(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.enc_rnn.bias.data.fill_(0)
        self.enc_rnn.weight.data.uniform_(-initrange, initrange)


class RNNLM(nn.Module):

    def __init__(self, opt, pretrain=None):
        super(RNNLM, self).__init__()
        self.opt = opt
        self.hid_dim = opt.hid_dim
        embeds = SingleEmbeddings(opt, pretrain)
        self.emb = embeds
        rnn_dec = SimpleRNNDecoder(opt, rnn_type='lstm', input_size=opt.
            inp_dim, hidden_size=opt.hid_dim, emb=self.emb)
        self.dec = rnn_dec

    def forward(self, inp_var, inp_msk, tgt_var=None, tgt_msk=None, aux=None):
        batch_size = inp_var.size()[0]
        h_t = self.init_hidden(batch_size)
        decoder_outputs_prob, decoder_outputs = self.dec.forward(h_t,
            tgt_var, tgt_msk, aux)
        return decoder_outputs_prob, decoder_outputs

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.opt.dec == 'lstm':
            return Variable(weight.new(bsz, self.hid_dim).zero_()), Variable(
                weight.new(bsz, self.hid_dim).zero_())
        else:
            return Variable(weight.new(bsz, self.hid_dim).zero_())


class RNNLM_VAE(nn.Module):

    def __init__(self, opt, pretrain=None):
        super(RNNLM_VAE, self).__init__()
        self.opt = opt
        embeds = SingleEmbeddings(opt, pretrain)
        self.emb = embeds
        if self.opt.enc == 'lstm':
            enc = RNNEncoder(opt, opt.inp_dim, opt.hid_dim, rnn_type=opt.
                enc.lower())
        else:
            raise NotImplementedError
        self.enc = enc
        rnn_dec = SimpleRNNDecoder(opt, rnn_type='lstm', input_size=opt.
            inp_dim, hidden_size=opt.hid_dim, emb=self.emb)
        self.dec = rnn_dec

    def forward(self, inp_var, inp_msk, tgt_var=None, tgt_msk=None, aux=None):
        emb = self.emb.forward(inp_var)
        h_t = self.enc.forward(emb, inp_msk)
        decoder_outputs_prob, decoder_outputs = self.dec.forward(h_t,
            tgt_var, tgt_msk, aux)
        return decoder_outputs_prob, decoder_outputs


class Seq2seq(nn.Module):

    def __init__(self, opt, pretrain=None):
        super(Seq2seq, self).__init__()
        self.opt = opt
        embeds = SingleEmbeddings(opt, pretrain)
        self.emb = embeds
        if self.opt.enc == 'lstm':
            enc = RNNEncoder(opt, opt.inp_dim + opt.tag_dim * 2, opt.
                hid_dim, rnn_type=opt.enc.lower())
        elif self.opt.enc == 'dconv':
            raise NotImplementedError
        elif self.opt.enc == 'conv':
            raise NotImplementedError
        else:
            raise NotImplementedError
        self.enc = enc
        self.feat = None
        rnn_dec = RNNDecoder(opt, rnn_type='lstm', num_layers=1,
            hidden_size=opt.hid_dim, input_size=opt.inp_dim, attn_type=
            'general', coverage=opt.coverage, copy=opt.copy, dropout=opt.
            dropout, emb=self.emb, full_dict_size=self.opt.full_dict_size,
            word_dict_size=self.opt.word_dict_size, max_len_dec=opt.max_len_dec
            )
        self.dec = rnn_dec

    def forward(self, inp_var, inp_msk, tgt_var=None, tgt_msk=None, aux=None):
        if self.training:
            self.forward_train(inp_var, inp_msk, tgt_var, tgt_msk, aux)
        else:
            self.forward_eval(inp_var, inp_msk, aux)

    def forward_train(self, inp_var, inp_msk, tgt_var, tgt_msk, aux):
        emb = self.emb.forward(inp_var)
        context, h_t = self.enc.forward(emb, inp_msk)
        (decoder_outputs_prob, decoder_outputs, attns, discount, loss_cov,
            p_copys) = (self.dec.forward(context, inp_msk, h_t, tgt_var,
            tgt_msk, inp_var, aux))
        return (decoder_outputs_prob, decoder_outputs, attns, discount,
            loss_cov, p_copys)

    def forward_eval(self, inp_var, inp_mask, aux):
        """

        :param inp_var: (seq len, batch size)
        :param inp_mask: [seq len, ....]
        :return:
        """
        emb = self.emb.forward(inp_var)
        context, h_t = self.enc.forward(emb, inp_mask)
        if self.feat != None:
            feats = self.feat.compute(context, features, feature_msks)
        else:
            feats = None
        contxt_len, batch_size, hdim = context.size()
        context_len__, batch_size__ = inp_var[0].size()
        assert batch_size == 1
        assert context_len__ == contxt_len
        decoder_outputs, attns, p_gens = self.dec.beam_decode(context,
            inp_mask, h_t, inp_var[0], feats, max_oov_len, scatter_mask)
        return decoder_outputs, attns, p_gens


class Attention(nn.Module):

    def __init__(self, opt, dim, attn_type='general', coverage=False,
        nn_feat_dim=None, sp_feat_dim=None):
        super(Attention, self).__init__()
        self.opt = opt
        self.dim = dim
        self.attn_type = attn_type
        if self.attn_type == 'general':
            self.W_h = nn.Linear(dim, dim, bias=True)
            self.W_s = nn.Linear(dim * 2, dim, bias=True)
        else:
            raise NotImplementedError
        if coverage:
            self.W_coverage = nn.Linear(1, dim, bias=True)
        if sp_feat_dim is not None:
            self.sp = True
            _feat_num_activat = sum([(1 if i else 0) for i in [opt.
                feat_word, opt.feat_ent, opt.feat_sent]])
            self.W_sp = nn.Linear(_feat_num_activat * sp_feat_dim, dim,
                bias=True)
        else:
            self.sp = False
        if nn_feat_dim is not None:
            self.nn = True
            self.W_nn = nn.Linear(nn_feat_dim, dim, bias=True)
        else:
            self.nn = False
        self.v = nn.Linear(dim, 1)
        self.mask = None

    def masked_attention(self, e, mask):
        """Take softmax of e then apply enc_padding_mask and re-normalize"""
        max_e = torch.max(e, dim=1, keepdim=True)[0]
        e = e - max_e
        attn_dist = F.softmax(e)
        attn_dist = attn_dist * Var(mask.float(), requires_grad=False)
        masked_sums = torch.sum(attn_dist, dim=1, keepdim=True)
        masked_sums = masked_sums.expand_as(attn_dist)
        return attn_dist / masked_sums

    def score(self, h_i, s_t):
        """
        s_t (FloatTensor): batch x 1 x dim
        h_i (FloatTensor): batch x src_len x dim
        returns scores (FloatTensor): batch x 1 x src_len:
        raw attention scores for each src index
        """
        src_batch, src_len, src_dim = h_i.size()
        tgt_batch, tgt_len, tgt_dim = s_t.size()
        assert src_batch == tgt_batch
        assert src_dim == tgt_dim
        assert tgt_len == 1
        if self.attn_type == 'dot':
            h_s_ = h_i.transpose(1, 2)
            return torch.bmm(s_t, h_s_).squeeze()
        elif self.attn_type == 'general':
            h_s_ = self.linear_in(h_i)
            return torch.bmm(s_t, h_s_.transpose(1, 2)).squeeze()
        elif self.attn_type == 'concat':
            concat = torch.cat((h_i, s_t.expand_as(h_i)), dim=2)
            x = self.linear_context(concat)
            x = self.tanh(x)
            x = self.linear_bottle(x).squeeze()
            return x
        else:
            raise NotImplementedError

    def forward(self, current_state, context, context_mask, last_attn,
        coverage=None, feats=None):
        """

        :param current_state : (FloatTensor): batch x dim: decoder's rnn's output.
        :param context: (FloatTensor): batch x src_len x dim: src hidden states
        :param coverage: (FloatTensor): batch, src_len
        :return: attn_h, batch x dim; attn_vec, batch x context_len
        """
        batch_size, src_len, dim = context.size()
        batch_, dim_ = current_state[0].size()
        cat_current_state = torch.cat(current_state, dim=1)
        assert batch_size == batch_
        assert dim == dim_
        w_cov = 0
        if coverage is not None:
            batch_, src_len_ = coverage.size()
            assert batch_ == batch_size
            assert src_len_ == src_len
            coverage = coverage + 0.001
            cov_sum = torch.sum(coverage, dim=1, keepdim=True)
            coverage = coverage / cov_sum
            coverage = coverage.unsqueeze(2)
            w_cov = self.W_coverage(coverage)
            cov_msk = torch.max(1 - coverage, 0)[0]
        w_state = self.W_s(cat_current_state)
        w_state = w_state.unsqueeze(1)
        w_context = self.W_h(context)
        w_sp = 0
        if self.sp:
            feats = feats.view(batch_size, src_len, -1)
            w_sp = self.W_sp(feats)
            if coverage is not None:
                w_sp = w_sp * cov_msk
        activated = F.tanh(w_state + w_context + w_cov + w_sp)
        e = self.v(activated).squeeze(2)
        max_e = torch.max(e, dim=1, keepdim=True)[0]
        e = e - max_e
        attn_dist = self.masked_attention(e, context_mask)
        exp_attn_dist = attn_dist.unsqueeze(2).expand_as(context)
        attn_h_weighted = torch.sum(exp_attn_dist * context, dim=1)
        return attn_h_weighted, attn_dist


class RNNDecoder(nn.Module):

    def __init__(self, opt, rnn_type='lstm', num_layers=1, hidden_size=100,
        input_size=50, attn_type='dot', coverage=False, copy=False, dropout
        =0.1, emb=None, full_dict_size=None, word_dict_size=None,
        max_len_dec=100, beam=True):
        super(RNNDecoder, self).__init__()
        self.opt = opt
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.word_dict_size = word_dict_size
        self.full_dict_size = full_dict_size
        self.embeddings = emb
        self.max_len_dec = max_len_dec
        self.rnn = self.build_rnn(rnn_type, self.input_size, hidden_size,
            num_layers)
        self.mask = None
        self.attn = Attention(opt, hidden_size, attn_type, coverage, opt.
            feat_nn_dim, opt.feat_sp_dim)
        self.W_out_0 = nn.Linear(hidden_size * 3, word_dict_size, bias=True)
        self.sampling = opt.schedule
        self._coverage = coverage
        self._copy = copy
        if copy:
            self.copy_linear = nn.Linear(hidden_size * 3 + input_size, 1,
                bias=True)
        if beam is True:
            self.beam_size = opt.beam_size
            assert self.beam_size >= 1

    def build_rnn(self, rnn_type, input_size, hidden_size, num_layers):
        if num_layers > 1:
            raise NotImplementedError
        if rnn_type == 'lstm':
            return torch.nn.LSTMCell(input_size, hidden_size, bias=True)
        elif rnn_type == 'gru':
            return torch.nn.GRUCell(input_size, hidden_size, bias=True)
        else:
            raise NotImplementedError

    def run_forward_step(self, input, context, context_mask, feats,
        prev_state, prev_attn, coverage=None, inp_var=None, max_oov_len=
        None, scatter_mask=None):
        """
        :param input: (LongTensor): a sequence of input tokens tensors
                                of size (1 x batch).
        :param context: (FloatTensor): output(tensor sequence) from the enc
                        RNN of size (src_len x batch x hidden_size).
        :param prev_state: tuple (FloatTensor): Maybe a tuple if lstm. (batch x hidden_size) hidden state from the enc RNN for
                                 initializing the decoder.
        :param coverage
        :param inp_var
        :return:
        """
        if isinstance(prev_state, tuple):
            batch_size_, hidden_size_ = prev_state[0].size()
        else:
            batch_size_, hidden_size_ = prev_state.size()
        src_len, batch_size, hidden_size = context.size()
        inp_size, batch_size__ = input.size()
        assert inp_size == 1
        assert batch_size_ == batch_size__ == batch_size
        input = input
        emb = self.embeddings.forward_decoding(input)
        assert emb.dim() == 3
        emb = emb.squeeze(0)
        current_raw_state = self.rnn(emb, prev_state)
        if self.rnn_type == 'lstm':
            assert type(current_raw_state) == tuple
            current_state_h = current_raw_state[0]
            current_state_c = current_raw_state[1]
        elif self.rnn_type == 'gru':
            current_state_h = current_raw_state
            current_state_c = current_raw_state
        attn_h_weighted, a = self.attn.forward(current_raw_state, context.
            clone().transpose(0, 1), context_mask, prev_attn, coverage, feats)
        if self._copy:
            copy_mat = self.copy_linear(torch.cat([attn_h_weighted,
                current_state_h, current_state_c, emb], dim=1))
            p_gen = F.sigmoid(copy_mat)
        hidden_state_vocab = torch.cat([attn_h_weighted, current_state_h,
            current_state_c], 1)
        hidden_state_vocab = self.W_out_0(hidden_state_vocab)
        max_hidden_state_vocab = torch.max(hidden_state_vocab, dim=1,
            keepdim=True)[0]
        hidden_state_vocab = hidden_state_vocab - max_hidden_state_vocab
        prob_vocab = F.softmax(hidden_state_vocab)
        if self._copy:
            prob_vocab = p_gen * prob_vocab
            new_a = a.clone()
            zeros = Var(torch.zeros(batch_size_, self.word_dict_size +
                max_oov_len))
            assert a.size()[1] == src_len
            assert inp_var.size()[0] == src_len
            assert inp_var.size()[1] == batch_size
            new_attn = torch.bmm(scatter_mask, new_a.unsqueeze(2)).squeeze(2)
            prob_copy = zeros.scatter_(1, inp_var.transpose(1, 0).cpu(),
                new_attn.cpu())
            prob_final = Var(torch.zeros(prob_copy.size()))
            prob_final[:, :self.opt.word_dict_size] = prob_vocab
            prob_final = prob_final + (1 - p_gen) * prob_copy
        else:
            p_gen = 1
            zeros = torch.zeros(batch_size_, self.word_dict_size + max_oov_len)
            prob_final = Var(torch.zeros(zeros.size()))
            prob_final[:, :self.opt.word_dict_size] = prob_vocab
        prob_final = torch.log(prob_final + 1e-07)
        return current_raw_state, prob_final, coverage, a, p_gen

    def forward(self, context, inp_msk, h_t, tgt_var, tgt_msk, inp_var, aux):
        """

        :param context:
        :param inp_msk:
        :param h_t:
        :param tgt_var:
        :param tgt_msk:
        :param inp_var:
        :param aux:
        :return:
        """
        tgt_len, batch_size = tgt.size()
        inp_var = inp_var[0]
        src_len, batch_size_, hidden_size = context.size()
        assert batch_size_ == batch_size
        padding_mask = context_mask.transpose(1, 0)
        decoder_outputs = torch.LongTensor(tgt_len, batch_size)
        decoder_outputs_prob = Var(torch.zeros((tgt_len, batch_size, self.
            word_dict_size + max_oov_len)))
        decoder_input = np.ones((1, batch_size), dtype=int) * self.opt.sos
        decoder_input = Var(torch.LongTensor(decoder_input))
        loss_cov = Var(torch.zeros(tgt_len, batch_size))
        if self._copy:
            p_copys = torch.zeros((batch_size, tgt_len))
        if self._coverage:
            coverage = Var(torch.zeros((batch_size, src_len))) + 0.0001
        else:
            coverage = None
        attn = Var(torch.zeros((batch_size, src_len)))
        Attns = Var(torch.zeros((tgt_len, batch_size, src_len)))
        for t in range(tgt_len):
            state, prob_final, coverage, attn, p_gen = self._run_forward_one(
                decoder_input, context, padding_mask, feat, state, attn,
                coverage, inp_var, max_oov_len, scatter_mask)
            if self._copy:
                p_copys[:, (t)] = p_gen.data
            decoder_outputs_prob[t] = prob_final
            topv, topi = prob_final.data.topk(1)
            topi = topi.squeeze()
            decoder_outputs[t] = topi
            Attns[t] = attn
            if self.opt.mul_loss or self.opt.add_loss:
                """
                        This function supports bi-gram pattern matching between txt and abs.
                        eg Input: A B A C A D
                            Out: A B A D
                        :param txt: src_seq, batch
                        :param abs: tgt_seq, batch
                        bigram_msk, repeat_map, window_msk,
                        :return: 1) batch_sz, seq_len, seq_len  records all the bigrams information. the first seq_len means the prev word.
                                                            the second one denotes the location where there are bigrams.
                                                            A[ 0 1 0 0 0 1]  note that C position is 0 since AC doesn't appear in gold summary
                        bigram_msk                         B[ 1 0 0 0 0 0]
                                                            A[ 0 1 0 0 0 1]
                                                            D[ 0 0 0 0 0 0 ]
                                2)  batch_sz, tgt_len           Records all 
                                                        [ 0 , 0 , 1 , 0 ] the first 0 is default zero. 
                             repeat_map                  the second 0 means 'look at B, prev is A, A is a word in txt, 0 is the first time the location it appears'
                                                        if the prefix word doesn't appear in the doc, it is 0. (3) will filter out this
                                3)  batch_sz, tgt_len       accordingly cooperates with (2). val =1 if prefix word appear in the document
                                window_msk
                """
                """
                New Version!
                tgt_msk = current_batch['bigram_msk']
                dict_from_wid_to_pos_in_tgt = current_batch['bigram_dict']

                current_batch['bigram']     batchsz, tgtlen, srclen
                current_batch['bigram_msk'] batchsz, tgtlen
                current_batch['bigram_dict']batchsz,   Not usable now # TODO

                """
                decoder_input = decoder_input.squeeze(0)
                for bidx in range(batch_size_):
                    prev = decoder_input[bidx].data[0]
                    _dict = bigram_dicts[bidx]
                    if _dict.has_key(prev):
                        v = _dict[prev]
                        if v < tgt_len:
                            _bigram = bigram[bidx][v]
                            _sum = torch.sum(_bigram)
                            if (_sum.data > 0).all():
                                tmp = _bigram
                                zeros = Var(torch.zeros(self.word_dict_size +
                                    max_oov_len))
                                x = zeros.scatter_(0, inp_var[:, (bidx)].
                                    cpu(), tmp.cpu())
                            else:
                                x = 0
                    else:
                        x = 0
                    Discount[t, bidx] = x
                """
                window = window_msk[:, t].contiguous()
                repeat = repeat_map[:, t]
                tables = Table[range(batch_size_), repeat, :]
                x = torch.min(attn, 1 - tables)  # accumulative attention matrix for
                y = bigram_msk[range(batch_size_), repeat, :]
                z = x * y
                z = z * window.view(-1, 1)
                zeros = torch.zeros(batch_size, self.word_dict_size + max_oov_len)
                accumulated_z = torch.bmm(scatter_mask, z.unsqueeze(2)).squeeze(2)
                z = zeros.scatter_(1, inp_var.transpose(1, 0).data.cpu(), accumulated_z.data.cpu()).cuda()
                Discount[t] = z
                update = torch.min(Table[range(batch_size_), repeat, :] + attn,
                                   Var(torch.ones((batch_size_, src_len))).cuda())
                Table[range(batch_size_), repeat] = update
                """
            if random.random() >= self.sampling:
                decoder_input = topi
                decoder_input = Var(decoder_input.unsqueeze(0))
            else:
                decoder_input = Var(tgt[t]).unsqueeze(0)
            if self._coverage:
                merged_attn_coverage = torch.cat((attn.unsqueeze(2),
                    coverage.unsqueeze(2)), dim=2)
                merge_min = torch.min(merged_attn_coverage, 2)
                loss_cov[(t), :] = torch.sum(merge_min[0], dim=1)
                coverage = coverage + attn
        return (decoder_outputs_prob, decoder_outputs, Attns, Discount,
            loss_cov, p_copys)


class SimpleRNNDecoder(nn.Module):
    """
    Simple Recurrent Module without attn, copy, coverage.
    """

    def __init__(self, opt, rnn_type, input_size, hidden_size, emb):
        super(SimpleRNNDecoder, self).__init__()
        self.opt = opt
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.full_dict_size = opt.full_dict_size
        self.embeddings = emb
        self.rnn = self.build_rnn(rnn_type, self.input_size, hidden_size)
        self.W_out_0 = nn.Linear(hidden_size * 2, opt.full_dict_size, bias=True
            )

    def build_rnn(self, rnn_type, input_size, hidden_size, num_layers=1):
        if num_layers > 1:
            raise NotImplementedError
        if rnn_type == 'lstm':
            return torch.nn.LSTMCell(input_size, hidden_size, bias=True)
        elif rnn_type == 'gru':
            return torch.nn.GRUCell(input_size, hidden_size, bias=True)
        else:
            raise NotImplementedError

    def run_forward_step(self, input, prev_state):
        """
        :param input: (LongTensor): a sequence of input tokens tensors
                                of size (1 x batch).
        :param context: (FloatTensor): output(tensor sequence) from the enc
                        RNN of size (src_len x batch x hidden_size).
        :param prev_state: tuple (FloatTensor): Maybe a tuple if lstm. (batch x hidden_size) hidden state from the enc RNN for
                                 initializing the decoder.
        :param coverage
        :param inp_var
        :return:
        """
        if isinstance(prev_state, tuple):
            batch_size_, hidden_size_ = prev_state[0].size()
        else:
            batch_size_, hidden_size_ = prev_state.size()
        inp_size, batch_size__ = input.size()
        assert inp_size == 1
        assert batch_size__ == batch_size_
        if self.opt.use_cuda:
            input = input
        emb = self.embeddings.forward_decoding(input)
        assert emb.dim() == 3
        emb = emb.squeeze(0)
        current_raw_state = self.rnn(emb, prev_state)
        if self.rnn_type == 'lstm':
            assert type(current_raw_state) == tuple
            current_state_h = current_raw_state[0]
            current_state_c = current_raw_state[1]
        elif self.rnn_type == 'gru':
            current_state_h = current_raw_state
            current_state_c = current_raw_state
        else:
            raise NotImplementedError
        hidden_state_vocab = torch.cat([current_state_h, current_state_c], 1)
        hidden_state_vocab = self.W_out_0(hidden_state_vocab)
        return current_raw_state, hidden_state_vocab

    def forward(self, state, tgt_var, tgt_msk, aux):
        batch_size, tgt_len = tgt_var.size()
        mode = self.training
        decoder_outputs = torch.LongTensor(tgt_len, batch_size)
        decoder_outputs_prob = Var(torch.zeros((tgt_len, batch_size, self.
            full_dict_size)))
        if self.opt.use_cuda:
            decoder_outputs_prob = decoder_outputs_prob
        decoder_input = np.ones((1, batch_size), dtype=int) * self.opt.unk
        decoder_input = Var(torch.LongTensor(decoder_input))
        for t in range(tgt_len):
            state, prob_final = self.run_forward_step(decoder_input, state)
            decoder_outputs_prob[t] = prob_final
            topv, topi = prob_final.data.topk(1)
            topi = topi.squeeze()
            decoder_outputs[t] = topi
            if mode:
                if random.random() >= self.opt.schedule:
                    decoder_input = topi
                    decoder_input = Var(decoder_input.unsqueeze(0))
                else:
                    decoder_input = Var(tgt_var[:, (t)]).unsqueeze(0)
            else:
                decoder_input = Var(tgt_var[:, (t)]).unsqueeze(0)
        return decoder_outputs_prob, decoder_outputs

    def init_weight(self):
        if self.rnn_type == 'lstm':
            nn.init.xavier_uniform(self.rnn.weight_ih, gain=1)
            nn.init.xavier_uniform(self.rnn.weight_hh, gain=1)
            torch.nn.init.constant(self.rnn.bias_ih, 0)
            nn.init.constant(self.rnn.bias_hh, 0)
        elif self.rnn_type == 'gru':
            nn.init.xavier_uniform(self.rnn.weight.data, gain=1)


class MultiEmbeddings(nn.Module):

    def __init__(self, opt, pretrain=None):
        super(MultiEmbeddings, self).__init__()
        self.opt = opt
        self.word_embedding = nn.Embedding(opt.full_dict_size, opt.inp_dim)
        if pretrain is not None:
            self.word_embedding.weight = nn.Parameter(torch.FloatTensor(
                pretrain))
        self.pos_embedding = nn.Embedding(opt.pos_dict_size, opt.tag_dim)
        self.ner_embedding = nn.Embedding(opt.ner_dict_size, opt.tag_dim)

    def forward(self, inp):
        """

        :param inp: list obj with word, pos, ner.
        :return: Concatenated word embedding. seq_len, batch_sz, all_dim
        """
        seq_word, seq_pos, seq_ner = inp
        embedded_word = self.word_embedding(seq_word)
        embedded_pos = self.pos_embedding(seq_pos)
        embedded_ner = self.ner_embedding(seq_ner)
        final_embedding = torch.cat((embedded_word, embedded_pos,
            embedded_ner), dim=2)
        if self.opt.pe:
            seq_len, batch_sz, dim = final_embedding.size()
            position_enc = np.array([[(pos / np.power(10000, 2.0 * i / dim)
                ) for i in range(dim)] for pos in range(seq_len)])
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])
            position_enc = torch.from_numpy(position_enc).type(torch.
                FloatTensor)
            x = position_enc.unsqueeze(1)
            x = Var(x.expand_as(final_embedding))
            final_embedding = final_embedding + 0.5 * x
        return final_embedding

    def forward_decoding(self, inp):
        embedded_word = self.word_embedding(inp)
        return embedded_word


class SingleEmbeddings(nn.Module):

    def __init__(self, opt, pretrain=None):
        super(SingleEmbeddings, self).__init__()
        self.opt = opt
        self.drop = nn.Dropout(opt.dropout_emb)
        self.word_embedding = nn.Embedding(opt.full_dict_size, opt.inp_dim)
        if pretrain is not None:
            self.word_embedding.weight = nn.Parameter(torch.FloatTensor(
                pretrain))

    def forward(self, inp):
        """

        :param inp:
        :return: seq_len, batch_sz, word_dim
        """
        embedded_word = self.word_embedding(inp)
        emb = self.drop(embedded_word)
        return emb

    def forward_decoding(self, inp):
        embedded_word = self.word_embedding(inp)
        emb = self.drop(embedded_word)
        return emb


class CNNEncoder(nn.Module):

    def __init__(self, inp_dim, hid_dim, kernel_sz, pad, dilat):
        super(CNNEncoder, self).__init__()
        self.encoder = torch.nn.Conv1d(in_channels=inp_dim, out_channels=
            hid_dim, kernel_size=kernel_sz, stride=1, padding=pad, dilation
            =dilat)

    def forward(self, inp, inp_mask):
        inp = inp.permute(1, 2, 0)
        x = torch.nn.functional.relu(self.encoder(inp))
        x = x.permute(2, 0, 1)
        h_t = x[-1], x[-1]
        return x, h_t


class DCNNEncoder(nn.Module):

    def __init__(self, inp_dim, hid_dim=150, kernel_sz=5, pad=2, dilat=1):
        super(DCNNEncoder, self).__init__()
        self.encoder = torch.nn.Conv1d(in_channels=inp_dim, out_channels=
            hid_dim, kernel_size=kernel_sz, stride=1, padding=pad, dilation=1)

    def forward(self, inp, mask):
        inp = inp.permute(1, 2, 0)
        x = torch.nn.functional.relu(self.encoder(inp))
        None


class RNNEncoder(nn.Module):

    def __init__(self, opt, input_size, hidden_size, rnn_type='lstm'):
        super(RNNEncoder, self).__init__()
        self.n_layers = opt.enc_layers
        self.bidirect = True
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rnn_type = rnn_type
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True,
                dropout=opt.dropout, bidirectional=self.bidirect)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1,
                batch_first=True, dropout=opt.dropout, bidirectional=self.
                bidirect)
        else:
            raise NotImplementedError
        self.init_weight()
        self.use_cuda = opt.use_cuda

    def init_weight(self):
        if self.rnn_type == 'lstm':
            nn.init.xavier_uniform(self.rnn.weight_hh_l0, gain=1)
            nn.init.xavier_uniform(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform(self.rnn.weight_ih_l0, gain=1)
            nn.init.xavier_uniform(self.rnn.weight_ih_l0_reverse, gain=1)
            torch.nn.init.constant(self.rnn.bias_hh_l0, 0)
            nn.init.constant(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant(self.rnn.bias_ih_l0, 0)
            nn.init.constant(self.rnn.bias_ih_l0_reverse, 0)
        elif self.rnn_type == 'gru':
            nn.init.xavier_uniform(self.rnn.weight.data, gain=1)

    def init_hidden(self, batch_size):
        if self.bidirect:
            result = Var(torch.zeros(self.n_layers * 2, batch_size, self.
                hidden_size))
        else:
            result = Var(torch.zeros(self.n_layers, batch_size, self.
                hidden_size))
        if self.use_cuda:
            return result
        else:
            return result

    def forward(self, embedded, inp_msk):
        """

        :param input: (seq_len, batch size, inp_dim)
        :param inp_msk: [seq len,....]

        :return: output: PackedSequence (seq_len*batch,  hidden_size * num_directions),
                hidden tupple ((batch, hidden_size*2), ....)
        """
        batch_size = embedded.data.shape[0]
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded,
            inp_msk, batch_first=True)
        if self.rnn_type == 'lstm':
            output, hn = self.rnn(packed_embedding)
        else:
            output, hn = self.rnn(packed_embedding, self.init_hidden(
                batch_size))

        def _fix_hidden(hidden):
            """

            :param hidden: (num_directions, batch, hidden_size)
            :return: batch, hidden_size*2
            """
            hidden = torch.cat((hidden[0], hidden[1]), dim=1)
            return hidden

        def compress_blstm_hidden(whole_context):
            return F.relu(self.reduce_h_W(whole_context))
        output, context_mask_ = nn.utils.rnn.pad_packed_sequence(output)
        if self.bidirect:
            if self.rnn_type == 'lstm':
                output = compress_blstm_hidden(output)
                h, c = hn[0], hn[1]
                h_, c_ = _fix_hidden(h), _fix_hidden(c)
                new_h = self.reduce_h_W(h_)
                new_c = self.reduce_c_W(c_)
                h_t = new_h, new_c
            elif self.rnn_type == 'gru':
                h_t = _fix_hidden(hn)
        else:
            raise NotImplementedError
        return h_t


class VI(nn.Module):

    def __init__(self):
        pass


class Gauss(nn.Module):

    def __init__(self, lat_dim):
        super().__init__()
        self.lat_dim = lat_dim
        self.func_mean = torch.nn.Linear(lat_dim, lat_dim)
        self.func_logvar = torch.nn.Linear(lat_dim, lat_dim)
        self.gate_mean = nn.Parameter(torch.rand(1))
        self.gate_var = nn.Parameter(torch.rand(1))

    def estimate_param(self, latent_code):
        mean = self.func_mean(latent_code)
        mean = self.gate_mean * mean
        logvar = self.func_logvar(latent_code)
        logvar = self.gate_var * logvar + (1 - self.gate_var
            ) * torch.ones_like(logvar)
        return {'mean': mean, 'logvar': logvar}

    def compute_KLD(self, tup):
        mean = tup['mean']
        logvar = tup['logvar']
        kld = -0.5 * torch.sum(1 - torch.mul(mean, mean) + 2 * logvar -
            torch.exp(2 * logvar), dim=1)
        return kld

    def sample_cell(self, batch_size):
        eps = torch.autograd.Variable(torch.normal(torch.zeros((batch_size,
            self.lat_dim))))
        eps = eps
        return eps

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        mean = tup['mean']
        logvar = tup['logvar']
        kld = self.compute_KLD(tup)
        vecs = []
        for ns in range(n_sample):
            eps = self.sample_cell(batch_size=batch_sz)
            vec = torch.mul(torch.exp(logvar), eps) + mean
            vecs.append(vec)
        return tup, kld, vecs


class HighVarGauss(nn.Module):

    def __init__(self, lat_dim):
        super().__init__()
        self.lat_dim = lat_dim
        self.func_mean = torch.nn.Linear(lat_dim, lat_dim)
        self.func_logvar = torch.nn.Linear(lat_dim, lat_dim)
        self.k = 10

    def estimate_param(self, latent_code):
        mean = self.func_mean(latent_code)
        logvar = self.func_logvar(latent_code)
        return {'mean': mean, 'logvar': logvar}

    def compute_KLD(self, tup):
        mean = tup['mean']
        logvar = tup['logvar']
        kld = -0.5 * torch.sum(1 - torch.mul(mean, mean) / self.k + 2 *
            logvar - torch.exp(2 * logvar) / self.k - 2, dim=1)
        return kld

    def sample_cell(self, batch_size):
        eps = torch.autograd.Variable(torch.normal(torch.zeros((batch_size,
            self.lat_dim))))
        eps = eps
        return eps

    def build_bow_rep(self, lat_code):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        kld = self.compute_KLD(tup)
        eps = self.sample_cell(batch_size=batch_sz)
        mean = tup['mean']
        logvar = tup['logvar']
        vec = torch.mul(torch.exp(logvar), eps) + mean
        return tup, kld, vec


class vMF(nn.Module):

    def __init__(self, lat_dim, kappa=0):
        super().__init__()
        self.lat_dim = lat_dim
        self.mu = torch.nn.Linear(lat_dim, lat_dim)
        self.kappa = kappa
        self.norm_eps = 1
        self.normclip = torch.nn.Hardtanh(0, 10 - 1)

    def estimate_param(self, latent_code):
        mu = self.mu(latent_code)
        return {'mu': mu}

    def compute_KLD(self):
        kld = 0
        return kld

    def vmf_unif_sampler(self, mu):
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            munoise = self.add_norm_noise(munorm, self.norm_eps)
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                w = self._sample_weight(self.kappa, id_dim)
                wtorch = torch.autograd.Variable(w * torch.ones(id_dim))
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)
                scale_factr = torch.sqrt(torch.autograd.Variable(torch.ones
                    (id_dim)) - torch.pow(wtorch, 2))
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munoise
            else:
                rand_draw = torch.autograd.Variable(torch.randn(id_dim))
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(
                    id_dim)
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * torch.autograd.Variable(rand_norms)
            result_list.append(sampled_vec)
        return torch.stack(result_list, 0)

    def vmf_sampler(self, mu):
        mu = mu.cpu()
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                w = vMF.sample_vmf_w(self.kappa, id_dim)
                wtorch = torch.autograd.Variable(w * torch.ones(id_dim))
                v = self._sample_orthonormal_to(mu[i] / munorm, id_dim)
                scale_factr = torch.sqrt(Variable(torch.ones(id_dim)) -
                    torch.pow(wtorch, 2))
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale) * munorm
            else:
                rand_draw = Variable(torch.randn(id_dim))
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(
                    id_dim)
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw * Variable(rand_norms)
            result_list.append(sampled_vec)
        return torch.stack(result_list, 0)

    def build_bow_rep(self, lat_code, n_sample):
        batch_sz = lat_code.size()[0]
        tup = self.estimate_param(latent_code=lat_code)
        kld = 0
        vecs = []
        for ns in range(n_sample):
            vec = self.vmf_unif_sampler(tup['mu'])
            vecs.append(vec)
        return tup, kld, vecs

    @staticmethod
    def _sample_weight(kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1
        b = dim / (np.sqrt(4.0 * kappa ** 2 + dim ** 2) + 2 * kappa)
        x = (1.0 - b) / (1.0 + b)
        c = kappa * x + dim * np.log(1 - x ** 2)
        while True:
            z = np.random.beta(dim / 2.0, dim / 2.0)
            w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1.0 - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = Variable(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)

    @staticmethod
    def sample_vmf_v(mu):
        import scipy.linalg as la
        mat = np.matrix(mu)
        if mat.shape[1] > mat.shape[0]:
            mat = mat.T
        U, _, _ = la.svd(mat)
        nu = np.matrix(np.random.randn(mat.shape[0])).T
        x = np.dot(U[:, 1:], nu[1:, :])
        return x / la.norm(x)

    @staticmethod
    def sample_vmf_w(kappa, m):
        b = (-2 * kappa + np.sqrt(4.0 * kappa ** 2 + (m - 1) ** 2)) / (m - 1)
        a = (m - 1 + 2 * kappa + np.sqrt(4 * kappa ** 2 + (m - 1) ** 2)) / 4
        d = 4 * a * b / (1 + b) - (m - 1) * np.log(m - 1)
        while True:
            z = np.random.beta(0.5 * (m - 1), 0.5 * (m - 1))
            W = (1 - (1 + b) * z) / (1 + (1 - b) * z)
            T = 2 * a * b / (1 + (1 - b) * z)
            u = np.random.uniform(0, 1)
            if (m - 1) * np.log(T) - T + d >= np.log(u):
                return W

    def add_norm_noise(self, munorm, eps):
        """
        KL loss is - log(maxvalue/eps)
        cut at maxvalue-eps, and add [0,eps] noise.
        """
        trand = torch.rand(1).expand(munorm.size()) * eps
        return self.normclip(munorm) + torch.autograd.Variable(trand)


class BowVAE(torch.nn.Module):

    def __init__(self, vocab_size, n_hidden, n_lat, n_sample, batch_size,
        non_linearity, dist):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_lat = n_lat
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.batch_size = batch_size
        self.dist_type = dist
        self.dropout = torch.nn.Dropout(p=0.2)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.enc_vec = torch.nn.Linear(self.vocab_size, self.n_hidden)
        self.active = torch.nn.LeakyReLU()
        self.enc_vec_2 = torch.nn.Linear(self.n_hidden, self.n_lat)
        if self.dist_type == 'nor':
            self.dist = Gauss(n_lat)
        elif self.dist_type == 'hnor':
            self.dist = HighVarGauss(n_lat)
        elif self.dist_type == 'vmf':
            self.dist = vMF(n_lat)
        else:
            raise NotImplementedError
        self.out = torch.nn.Linear(self.n_lat, self.vocab_size)

    def forward(self, x, mask):
        batch_sz = x.size()[0]
        linear_x = self.enc_vec(x)
        active_x = self.active(linear_x)
        linear_x_2 = self.enc_vec_2(active_x)
        tup, kld, vecs = self.dist.build_bow_rep(linear_x_2, self.n_sample)
        ys = 0
        for i, v in enumerate(vecs):
            logit = torch.nn.functional.log_softmax(self.out(v))
            logit = self.dropout(logit)
            ys += torch.mul(x, logit)
        y = ys / self.n_sample
        recon_loss = -torch.sum(y, dim=1, keepdim=False)
        kld = kld * mask
        recon_loss = recon_loss * mask
        total_loss = kld + recon_loss
        return recon_loss, kld, total_loss


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
        tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=
                dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[
                    rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                    )
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=
                nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1),
            output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)
            ), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()
                ), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


class LanguageModel(Module):

    def __init__(self, vocab_size, input_dim, hidden_dim, agenda_dim,
        num_layers=1, drop_rate=0.6, tie=False, logger=None):
        super(LanguageModel, self).__init__()
        self.embed = Embedding(vocab_size, input_dim)
        self.decoder_rnn = LSTM(input_dim + agenda_dim, hidden_dim,
            num_layers=num_layers, dropout=drop_rate)
        self.decoder_out = Linear(hidden_dim, vocab_size)
        self.agenda_dim = agenda_dim
        self.logger = logger
        if tie:
            if hidden_dim != input_dim:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.embed.weight = self.encoder.weight
        self.init_weights(input_dim, hidden_dim, agenda_dim)

    def init_weights(self, input_dim, hidden_dim, agenda_dim):
        torch.nn.init.xavier_uniform(self.decoder_rnn.weight_ih_l0.data,
            gain=nn.init.calculate_gain('sigmoid'))
        torch.nn.init.orthogonal(self.decoder_rnn.weight_hh_l0.data, gain=
            nn.init.calculate_gain('sigmoid'))
        self.decoder_rnn.bias.data.fill_(0)
        torch.nn.init.xavier_uniform(self.embed.weight.data, gain=nn.init.
            calculate_gain('linear'))
        torch.nn.init.xavier_uniform(self.decoder_out.weight.data, gain=nn.
            init.calculate_gain('linear'))

    def _encoder_output(self, batch_size):
        return tile_state(self.agenda, batch_size)

    def per_instance_losses(self, examples):
        batch_size = len(examples)
        decoder_input = TrainDecoderInput(examples, self.vocab)
        encoder_output = self._encoder_output(batch_size)
        return self.train_decoder.per_instance_losses(encoder_output,
            decoder_input)

    def loss(self, examples, train_step):
        """Compute training loss.

        Args:
            examples (list[list[unicode]])

        Returns:
            Variable: a scalar
        """
        batch_size = len(examples)
        decoder_input = TrainDecoderInput(examples, self.vocab)
        encoder_output = self._encoder_output(batch_size)
        return self.train_decoder.loss(encoder_output, decoder_input)

    def generate(self, num_samples, decode_method='argmax'):
        examples = range(num_samples)
        prefix_hints = [[]] * num_samples
        encoder_output = self._encoder_output(num_samples)
        if decode_method == 'sample':
            output_beams, decoder_traces = self.sample_decoder.decode(examples,
                encoder_output, beam_size=1, prefix_hints=prefix_hints)
        elif decode_method == 'argmax':
            value_estimators = []
            beam_size = 1
            sibling_penalty = 0.0
            output_beams, decoder_traces = self.beam_decoder.decode(examples,
                encoder_output, weighted_value_estimators=value_estimators,
                beam_size=beam_size, prefix_hints=prefix_hints,
                sibling_penalty=sibling_penalty)
        else:
            raise ValueError(decode_method)
        return [beam[0] for beam in output_beams]


class Encoder(Module):

    def __init__(self, token_embedder, hidden_dim, agenda_dim, num_layers,
        rnn_cell_factory):
        super(Encoder, self).__init__()
        self.token_embedder = token_embedder
        self.word_vocab = token_embedder.vocab
        self.hidden_dim = hidden_dim
        self.agenda_dim = agenda_dim
        self.num_layers = num_layers
        self.source_encoder = MultiLayerSourceEncoder(token_embedder.
            embed_dim, hidden_dim, num_layers, rnn_cell_factory)

    def preprocess(self, examples):
        return SequenceBatch.from_sequences(examples, self.word_vocab)

    def forward(self, examples_seq_batch):
        embeds = self.token_embedder.embed_seq_batch(examples_seq_batch)
        source_encoder_output = self.source_encoder(embeds.split())
        return source_encoder_output

    def make_agenda(self, encoder_output):
        agenda = torch.cat(encoder_output.final_states, 1)
        return agenda


class EncoderNoiser(Module):

    def __init__(self, encoder, kl_weight_steps, kl_weight_rate, kl_weight_cap
        ):
        super(EncoderNoiser, self).__init__()
        self.encoder = encoder
        self.noise_mu = 0
        self.noise_sigma = 1
        self.kl_weight_steps = kl_weight_steps
        self.kl_weight_rate = kl_weight_rate
        self.kl_weight_cap = kl_weight_cap

    def preprocess(self, examples):
        return self.encoder.preprocess(examples)

    def kl_penalty(self, agenda):
        """
        Computes KL penalty given encoder output
        """
        batch_size, agenda_dim = agenda.size()
        return 0.5 * torch.sum(torch.pow(agenda, 2)
            ) * self.noise_sigma / batch_size

    def kl_weight(self, curr_step):
        """
        Compute KL penalty weight
        """
        sigmoid = lambda x, k: 1 / (1 + np.e ** (-k * (2 * x - 1)))
        x = curr_step / float(self.kl_weight_steps)
        return self.kl_weight_cap * sigmoid(x, self.kl_weight_rate)

    def forward(self, examples_seq_batch):
        source_encoder_output = self.encoder(examples_seq_batch)
        agenda = self.encoder.make_agenda(source_encoder_output)
        means = self.noise_mu * torch.ones(agenda.size())
        std = self.noise_sigma * torch.ones(agenda.size())
        noise = GPUVariable(torch.normal(means=means, std=std))
        return agenda, agenda + noise


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, agenda_dim, nlayers=1, dropout=
        0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.embed = nn.Embedding(ntoken, ninp)
        self.decoder_rnn = nn.LSTM(ninp + agenda_dim, nhid, nlayers,
            dropout=dropout)
        self.decoder_out = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder_out.weight = self.embed.weight
        self.init_weights()
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.decoder_out.bias.data.fill_(0)
        self.decoder_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden=None):
        batch_sz = input.size()[1]
        if hidden is None:
            hidden = self.init_hidden(batch_sz)
        emb = self.drop(self.embed(input))
        output, hidden = self.decoder_rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder_out(output.view(output.size(0) * output.size
            (1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)
            ), hidden

    def forward_decode(self, args, input, ntokens):
        seq_len = input.size()[0]
        batch_sz = input.size()[1]
        hidden = None
        outputs_prob = Variable(torch.FloatTensor(seq_len, batch_sz, ntokens))
        if args.cuda:
            outputs_prob = outputs_prob
        outputs = torch.LongTensor(seq_len, batch_sz)
        sos = Variable(torch.ones(batch_sz).long())
        unk = Variable(torch.ones(batch_sz).long()) * 2
        if args.cuda:
            sos = sos
            unk = unk
        emb_0 = self.drop(self.encoder(sos)).unsqueeze(0)
        emb_t = self.drop(self.encoder(unk)).unsqueeze(0)
        for t in range(seq_len):
            if t == 0:
                emb = emb_0
            else:
                emb = emb_t
            output, hidden = self.rnn(emb, hidden)
            output_prob = self.decoder(self.drop(output))
            output_prob = output_prob.squeeze(0)
            outputs_prob[t] = output_prob
            value, ind = torch.topk(output_prob, 1, dim=1)
            outputs[t] = ind.squeeze(1).data
        return outputs_prob, outputs

    def init_hidden(self, bsz):
        return Variable(torch.zeros(self.nlayers, bsz, self.nhid)), Variable(
            torch.zeros(self.nlayers, bsz, self.nhid))


class VAEModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, args, dec_type, ntoken, ninp, nhid, lat_dim, nlayers,
        dropout=0.5, tie_weights=False):
        super(VAEModel, self).__init__()
        self.args = args
        self.lat_dim = lat_dim
        self.nhid = nhid
        self.nlayers = nlayers
        self.ninp = ninp
        self.dist = args.dist
        self.emb = nn.Embedding(ntoken, ninp)
        self.enc_rnn = nn.LSTM(ninp, nhid, nlayers, bidirectional=True,
            dropout=dropout)
        self.drop = nn.Dropout(dropout)
        if args.dist == 'nor':
            self.fc_mu = nn.Linear(2 * nhid * nlayers * 2, lat_dim)
            self.fc_logvar = nn.Linear(2 * nhid * nlayers * 2, lat_dim)
        elif args.dist == 'vmf':
            self.fc = nn.Linear(2 * nhid * nlayers * 2, lat_dim)
            self.vmf = vMF.vmf(1, 10, args.kappa)
        else:
            raise NotImplementedError
        self.z_to_h = nn.Linear(lat_dim, nhid * nlayers)
        self.z_to_c = nn.Linear(lat_dim, nhid * nlayers)
        self.dec_type = dec_type
        self.decoder_out = nn.Linear(nhid, ntoken)
        if dec_type == 'lstm':
            if args.fly:
                self.decoder_rnn = nn.LSTMCell(ninp + nhid, nhid, nlayers)
            else:
                self.decoder_rnn = nn.LSTM(ninp + nhid, nhid, nlayers,
                    dropout=dropout)
        elif dec_type == 'bow':
            self.linear = nn.Linear(nhid + ninp, nhid)
        else:
            raise NotImplementedError
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder_out.weight = self.emb.weight

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def blstm_enc(self, input):
        """
        Encoding the input
        :param input: input sequence
        :return:
        embedding: seq_len, batch_sz, hid_dim
        hidden(from z): (2, batch_sz, 150)
        mu          : batch_sz, hid_dim
        logvar      : batch_sz, hid_dim
        """
        batch_sz = input.size()[1]
        emb = self.drop(self.emb(input))
        if self.dist == 'nor':
            mu, logvar = self.encode(emb)
            z = self.reparameterize(mu, logvar)
            hidden = self.convert_z_to_hidden(z, batch_sz)
            return emb, hidden, mu, logvar
        elif self.dist == 'vmf':
            mu = self.encode(emb)
            mu = mu.cpu()
            z = self.vmf.sample_vMF(mu)
            z = z
            hidden = self.convert_z_to_hidden(z, batch_sz)
            return emb, hidden, mu
        else:
            raise NotImplementedError

    def encode(self, emb):
        """

        :param emb:
        :return: batch_sz, lat_dim
        """
        batch_sz = emb.size()[1]
        _, hidden = self.enc_rnn(emb)
        h = hidden[0]
        c = hidden[1]
        assert h.size()[0] == self.nlayers * 2
        assert h.size()[1] == batch_sz
        x = torch.cat((h, c), dim=0).permute(1, 0, 2).contiguous().view(
            batch_sz, -1)
        if self.dist == 'nor':
            return self.fc_mu(x), self.fc_logvar(x)
        elif self.dist == 'vmf':
            return self.fc(x)
        else:
            raise NotImplementedError

    def forward(self, input):
        """

        :param input: seq_len, batch_sz
        :return:
        """
        batch_sz = input.size()[1]
        seq_len = input.size()[0]
        if self.dist == 'nor':
            emb, hidden, mu, logvar = self.blstm_enc(input)
        elif self.dist == 'vmf':
            emb, hidden, mu = self.blstm_enc(input)
            logvar = None
        if self.dec_type == 'lstm':
            lat_to_cat = hidden[0][0].unsqueeze(0).expand(seq_len, batch_sz, -1
                )
            emb = torch.cat([emb, lat_to_cat], dim=2)
            output, hidden = self.decoder_rnn(emb, hidden)
        elif self.dec_type == 'bow':
            emb = torch.mean(emb, dim=0)
            lat_to_cat = hidden[0][0]
            fusion = torch.cat((emb, lat_to_cat), dim=1)
            output = Variable(torch.FloatTensor(seq_len, batch_sz, self.nhid))
            if self.args.cuda:
                output = output
            for t in range(seq_len):
                noise = 0.1 * Variable(fusion.data.new(fusion.size()).
                    normal_(0, 1))
                if self.args.cuda:
                    noise = noise
                fusion_with_noise = fusion + noise
                fusion_with_noise = self.linear(fusion_with_noise)
                output[t] = fusion_with_noise
        else:
            raise NotImplementedError
        output = self.drop(output)
        decoded = self.decoder_out(output.view(output.size(0) * output.size
            (1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)
            ), mu, logvar

    def forward_decode(self, args, input, ntokens):
        """

        :param args:
        :param input: LongTensor [seq_len, batch_sz]
        :param ntokens:
        :return:
            outputs_prob:   Var         seq_len, batch_sz, ntokens
            outputs:        LongTensor  seq_len, batch_sz
            mu, logvar
        """
        seq_len = input.size()[0]
        batch_sz = input.size()[1]
        emb, lat, mu, logvar = self.blstm_enc(input)
        outputs_prob = Var(torch.FloatTensor(seq_len, batch_sz, ntokens))
        if args.cuda:
            outputs_prob = outputs_prob
        outputs = torch.LongTensor(seq_len, batch_sz)
        sos = Var(torch.ones(batch_sz).long())
        unk = Var(torch.ones(batch_sz).long()) * 2
        if args.cuda:
            sos = sos
            unk = unk
        lat_to_cat = lat[0][0].unsqueeze(0)
        emb_t = self.drop(self.encoder(unk)).unsqueeze(0)
        emb_0 = self.drop(self.encoder(sos)).unsqueeze(0)
        emb_t_comb = torch.cat([emb_t, lat_to_cat], dim=2)
        emt_0_comb = torch.cat([emb_0, lat_to_cat], dim=2)
        hidden = None
        for t in range(seq_len):
            if t == 0:
                emb = emt_0_comb
            else:
                emb = emb_t_comb
            if hidden is None:
                output, hidden = self.rnn(emb, None)
            else:
                output, hidden = self.rnn(emb, hidden)
            output_prob = self.decoder(self.drop(output))
            output_prob = output_prob.squeeze(0)
            outputs_prob[t] = output_prob
            value, ind = torch.topk(output_prob, 1, dim=1)
            outputs[t] = ind.squeeze(1).data
        return outputs_prob, outputs, mu, logvar

    def convert_z_to_hidden(self, z, batch_sz):
        """

        :param z:   batch, lat_dim
        :param batch_sz:
        :return:
        """
        h = self.z_to_h(z).view(batch_sz, self.nlayers, -1).permute(1, 0, 2
            ).contiguous()
        c = self.z_to_c(z).view(batch_sz, self.nlayers, -1).permute(1, 0, 2
            ).contiguous()
        return h, c

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()
                ), Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def init_weights(self):
        initrange = 0.1
        self.emb.weight.data.uniform_(-initrange, initrange)
        self.enc_rnn.bias.data.fill_(0)
        self.enc_rnn.weight.data.uniform_(-initrange, initrange)


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jiacheng_xu_vmf_vae_nlp(_paritybench_base):
    pass
    def test_000(self):
        self._check(Code2Code(*[], **{'inp_dim': 4, 'tgt_dim': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(DCNNEncoder(*[], **{'inp_dim': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})

    def test_002(self):
        self._check(SingleEmbeddings(*[], **{'opt': _mock_config(dropout_emb=0.5, full_dict_size=4, inp_dim=4)}), [torch.zeros([4], dtype=torch.int64)], {})

