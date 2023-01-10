import sys
_module = sys.modules[__name__]
del sys
convert = _module
dataset = _module
encode = _module
evaluate_mse = _module
model = _module
preprocess = _module
train = _module
vision = _module
configs = _module
defaults = _module
main = _module
model = _module
networks = _module
celeba = _module
celebamask_hq = _module
cifar10 = _module
fashion_mnist = _module
mnist = _module
net_28 = _module
net_32 = _module
net_64 = _module
util = _module
quantizer = _module
celebamask_hq = _module
ive = _module
semseg = _module
trainer = _module
trainer_base = _module
util = _module

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


import torch


import numpy as np


from matplotlib import pyplot as plt


from torch.utils.data import Dataset


from random import randint


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions import Categorical


from itertools import chain


import math


import torch.optim as optim


from torch.utils.data import DataLoader


from torch.utils.tensorboard import SummaryWriter


from torch import nn


import torchvision.datasets as dsets


from torchvision import transforms


import scipy.special


from numbers import Number


import time


import random


import matplotlib.pyplot as plt


import matplotlib.gridspec as gridspec


from torchvision import datasets


from torch.utils.data.dataset import Subset


class Jitter(nn.Module):

    def __init__(self, p):
        super().__init__()
        self.p = p
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer('prob', prob)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        else:
            batch_size, sample_size, channels = x.size()
            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)
            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x


class SQEmbedding(nn.Module):

    def __init__(self, param_var_q, n_embeddings, embedding_dim):
        super(SQEmbedding, self).__init__()
        self.param_var_q = param_var_q
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.normal_()
        self.register_parameter('embedding', nn.Parameter(embedding))

    def encode(self, x, log_var_q):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        if self.param_var_q == 'gaussian_1':
            log_var_q_flat = log_var_q.reshape(1, 1)
        elif self.param_var_q == 'gaussian_3':
            log_var_q_flat = log_var_q.reshape(-1, 1)
        elif self.param_var_q == 'gaussian_4':
            log_var_q_flat = log_var_q.reshape(-1, D)
        else:
            raise Exception('Undefined param_var_q')
        x_flat = x_flat.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * (embedding - x_flat) ** 2, dim=1)
        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices

    def forward(self, x, log_var_q, temperature):
        M, D = self.embedding.size()
        batch_size, sample_size, channels = x.size()
        x_flat = x.reshape(-1, D)
        if self.param_var_q == 'gaussian_1':
            log_var_q_flat = log_var_q.reshape(1, 1)
        elif self.param_var_q == 'gaussian_3':
            log_var_q_flat = log_var_q.reshape(-1, 1)
        elif self.param_var_q == 'gaussian_4':
            log_var_q_flat = log_var_q.reshape(-1, D)
        else:
            raise Exception('Undefined param_var_q')
        x_flat = x_flat.unsqueeze(2)
        log_var_flat = log_var_q_flat.unsqueeze(2)
        embedding = self.embedding.t().unsqueeze(0)
        precision_flat = torch.exp(-log_var_flat)
        distances = 0.5 * torch.sum(precision_flat * (embedding - x_flat) ** 2, dim=1)
        indices = torch.argmin(distances.float(), dim=-1)
        logits = -distances
        encodings = self._gumbel_softmax(logits, tau=temperature, dim=-1)
        quantized = torch.matmul(encodings, self.embedding)
        quantized = quantized.view_as(x)
        logits = logits.view(batch_size, sample_size, M)
        probabilities = torch.softmax(logits, dim=-1)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        precision = torch.exp(-log_var_q)
        loss = torch.mean(0.5 * torch.sum(precision * (x - quantized) ** 2, dim=(1, 2)) + torch.sum(probabilities * log_probabilities, dim=(1, 2)))
        encodings = F.one_hot(indices, M).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, loss, perplexity

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, dim=-1):
        eps = torch.finfo(logits.dtype).eps
        gumbels = -(-torch.rand_like(logits).clamp(min=eps, max=1 - eps).log()).log()
        gumbels_new = (logits + gumbels) / tau
        y_soft = gumbels_new.softmax(dim)
        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret


class Encoder(nn.Module):

    def __init__(self, param_var_q, in_channels, channels, n_embeddings, embedding_dim, jitter=0.0):
        super(Encoder, self).__init__()
        self.param_var_q = param_var_q
        self.embedding_dim = embedding_dim
        if self.param_var_q == 'gaussian_1':
            out_channels = embedding_dim
        elif self.param_var_q == 'gaussian_3':
            out_channels = embedding_dim + 1
        elif self.param_var_q == 'gaussian_4':
            out_channels = embedding_dim * 2
        else:
            raise Exception('Undefined param_var_q')
        self.encoder = nn.Sequential(nn.Conv1d(in_channels, channels, 3, 1, 0, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True), nn.Conv1d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True), nn.Conv1d(channels, channels, 4, 2, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True), nn.Conv1d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True), nn.Conv1d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm1d(channels), nn.ReLU(True), nn.Conv1d(channels, out_channels, 1))
        log_var_q_scalar = torch.Tensor(1)
        log_var_q_scalar.fill_(10.0).log_()
        self.register_parameter('log_var_q_scalar', nn.Parameter(log_var_q_scalar))
        self.codebook = SQEmbedding(param_var_q, n_embeddings, embedding_dim)
        self.jitter = Jitter(jitter)

    def forward(self, mels, temperature):
        z = self.encoder(mels)
        z = z.transpose(1, 2)
        if self.param_var_q == 'gaussian_1':
            log_var_q = self.log_var_q_scalar
        elif self.param_var_q == 'gaussian_3' or self.param_var_q == 'gaussian_4':
            log_var_q = z[:, :, self.embedding_dim:] + self.log_var_q_scalar
        else:
            raise Exception('Undefined param_var_q')
        z = z[:, :, :self.embedding_dim]
        z, loss, perplexity = self.codebook(z, log_var_q, temperature)
        z = self.jitter(z)
        return z, loss, perplexity

    def encode(self, mel):
        z = self.encoder(mel)
        z = z.transpose(1, 2)
        if self.param_var_q == 'gaussian_1':
            log_var_q = self.log_var_q_scalar
        elif self.param_var_q == 'gaussian_3' or self.param_var_q == 'gaussian_4':
            log_var_q = z[:, :, self.embedding_dim:] + self.log_var_q_scalar
        else:
            raise Exception('Undefined param_var_q')
        z = z[:, :, :self.embedding_dim]
        z, indices = self.codebook.encode(z, log_var_q)
        return z, indices


class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, n_speakers, speaker_embedding_dim, conditioning_channels, fc_channels):
        super().__init__()
        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
        self.rnn = nn.GRU(in_channels + speaker_embedding_dim, conditioning_channels, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(fc_channels, out_channels)

    def forward(self, z, speakers):
        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)
        speakers = self.speaker_embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, z.size(1), -1)
        z = torch.cat((z, speakers), dim=-1)
        z, _ = self.rnn(z)
        x = self.fc(z)
        return x

    def generate(self, z, speaker):
        output = self.forward(z, speaker)
        return output


class VectorQuantizer(nn.Module):

    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super(VectorQuantizer, self).__init__()
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature

    def forward(self, z_from_encoder, param_q, codebook, flg_train, flg_quant_det=False):
        return self._quantize(z_from_encoder, param_q, codebook, flg_train=flg_train, flg_quant_det=flg_quant_det)

    def _quantize(self):
        raise NotImplementedError()

    def set_temperature(self, value):
        self.temperature = value

    def _calc_distance_bw_enc_codes(self):
        raise NotImplementedError()

    def _calc_distance_bw_enc_dec(self):
        raise NotImplementedError()


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = torch.sum(z_continuous_flat ** 2, dim=1, keepdim=True) + torch.sum(codebook ** 2, dim=1) - 2 * torch.matmul(z_continuous_flat, codebook.t())
    return distances


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape, device='cuda')
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size())
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


class GaussianVectorQuantizer(VectorQuantizer):

    def __init__(self, size_dict, dim_dict, temperature=0.5, param_var_q='gaussian_1'):
        super(GaussianVectorQuantizer, self).__init__(size_dict, dim_dict, temperature)
        self.param_var_q = param_var_q

    def _quantize(self, z_from_encoder, var_q, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        precision_q = 1.0 / torch.clamp(var_q, min=1e-10)
        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook, 0.5 * precision_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device='cuda')
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0, 1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, 0.5 * precision_q).mean()
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-07)))
        return z_to_decoder, loss, perplexity

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight):
        if self.param_var_q == 'gaussian_1':
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == 'gaussian_2':
            weight = weight.tile(1, 1, 8, 8).view(-1, 1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == 'gaussian_3':
            weight = weight.view(-1, 1)
            distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        elif self.param_var_q == 'gaussian_4':
            z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict).unsqueeze(2)
            codebook = codebook.t().unsqueeze(0)
            weight = weight.permute(0, 2, 3, 1).contiguous().view(-1, self.dim_dict).unsqueeze(2)
            distances = torch.sum(weight * (z_from_encoder_flat - codebook) ** 2, dim=1)
        return distances

    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum((x1 - x2) ** 2 * weight, dim=(1, 2, 3))


class VmfVectorQuantizer(VectorQuantizer):

    def __init__(self, size_dict, dim_dict, temperature=0.5):
        super(VmfVectorQuantizer, self).__init__(size_dict, dim_dict, temperature)

    def _quantize(self, z_from_encoder, kappa_q, codebook, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        codebook_norm = F.normalize(codebook, p=2.0, dim=1)
        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook_norm, kappa_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        if flg_train:
            encodings = gumbel_softmax_sample(logit, self.temperature)
            z_quantized = torch.mm(encodings, codebook_norm).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device='cuda')
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook_norm).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0, 1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, kappa_q).mean()
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-07)))
        return z_to_decoder, loss, perplexity

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, kappa_q):
        z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict)
        distances = -kappa_q * torch.matmul(z_from_encoder_flat, codebook.t())
        return distances

    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum(x1 * (x1 - x2) * weight, dim=(1, 2, 3))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SQVAE(nn.Module):

    def __init__(self, cfgs, flgs):
        super(SQVAE, self).__init__()
        dataset = cfgs.dataset.name
        self.dim_x = cfgs.dataset.dim_x
        self.dataset = cfgs.dataset.name
        self.param_var_q = cfgs.model.param_var_q
        self.encoder = eval('net_{}.EncoderVq_{}'.format(dataset.lower(), cfgs.network.name))(cfgs.quantization.dim_dict, cfgs.network, flgs.bn, flgs.var_q)
        self.decoder = eval('net_{}.DecoderVq_{}'.format(dataset.lower(), cfgs.network.name))(cfgs.quantization.dim_dict, cfgs.network, flgs.bn)
        self.apply(weights_init)
        self.size_dict = cfgs.quantization.size_dict
        self.dim_dict = cfgs.quantization.dim_dict
        self.codebook = nn.Parameter(torch.randn(self.size_dict, self.dim_dict))
        self.log_param_q_scalar = nn.Parameter(torch.tensor(cfgs.model.log_param_q_init))
        if self.param_var_q == 'vmf':
            self.quantizer = VmfVectorQuantizer(self.size_dict, self.dim_dict, cfgs.quantization.temperature.init)
        else:
            self.quantizer = GaussianVectorQuantizer(self.size_dict, self.dim_dict, cfgs.quantization.temperature.init, self.param_var_q)

    def forward(self, x, flg_train=False, flg_quant_det=True):
        if self.param_var_q == 'vmf':
            z_from_encoder = F.normalize(self.encoder(x), p=2.0, dim=1)
            self.param_q = self.log_param_q_scalar.exp() + torch.tensor([1.0], device='cuda')
        else:
            if self.param_var_q == 'gaussian_1':
                z_from_encoder = self.encoder(x)
                log_var_q = torch.tensor([0.0], device='cuda')
            else:
                z_from_encoder, log_var = self.encoder(x)
                if self.param_var_q == 'gaussian_2':
                    log_var_q = log_var.mean(dim=(1, 2, 3), keepdim=True)
                elif self.param_var_q == 'gaussian_3':
                    log_var_q = log_var.mean(dim=1, keepdim=True)
                elif self.param_var_q == 'gaussian_4':
                    log_var_q = log_var
                else:
                    raise Exception('Undefined param_var_q')
            self.param_q = log_var_q.exp() + self.log_param_q_scalar.exp()
        z_quantized, loss_latent, perplexity = self.quantizer(z_from_encoder, self.param_q, self.codebook, flg_train, flg_quant_det)
        latents = dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized)
        x_reconst = self.decoder(z_quantized)
        loss = self._calc_loss(x_reconst, x, loss_latent)
        loss['perplexity'] = perplexity
        return x_reconst, latents, loss

    def _calc_loss(self):
        raise NotImplementedError()


class GaussianSQVAE(SQVAE):

    def __init__(self, cfgs, flgs):
        super(GaussianSQVAE, self).__init__(cfgs, flgs)
        self.flg_arelbo = flgs.arelbo
        if not self.flg_arelbo:
            self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))

    def _calc_loss(self, x_reconst, x, loss_latent):
        bs = x.shape[0]
        mse = F.mse_loss(x_reconst, x, reduction='sum') / bs
        if self.flg_arelbo:
            loss_reconst = self.dim_x * torch.log(mse) / 2
        else:
            loss_reconst = mse / (2 * self.logvar_x.exp()) + self.dim_x * self.logvar_x / 2
        loss_all = loss_reconst + loss_latent
        loss = dict(all=loss_all, mse=mse)
        return loss


class IveFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, v, z):
        assert isinstance(v, Number), 'v must be a scalar'
        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()
        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)
        return torch.Tensor(output)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return None, grad_output * (ive(self.v - 1, z) - ive(self.v, z) * (self.v + z) / z)


ive = IveFunction.apply


class VmfSQVAE(SQVAE):

    def __init__(self, cfgs, flgs):
        super(VmfSQVAE, self).__init__(cfgs, flgs)
        self.log_kappa_inv = nn.Parameter(torch.tensor([cfgs.model.log_kappa_inv]))
        self.__m = np.ceil(cfgs.network.num_class / 2)
        self.n_interval = cfgs.network.num_class - 1

    def _calc_loss(self, x_reconst, x, loss_latent):
        x_shape = x.shape
        x = x.view(-1, 1)
        x_reconst_viewed = x_reconst.permute(0, 2, 3, 1).contiguous().view(-1, int(self.__m * 2))
        x_reconst_normed = F.normalize(x_reconst_viewed, p=2.0, dim=-1)
        x_one_hot = F.one_hot(x.long(), num_classes=int(self.__m * 2)).type_as(x)[:, 0, :]
        x_reconst_selected = (x_one_hot * x_reconst_normed).sum(-1).view(x_shape)
        kappa_inv = self.log_kappa_inv.exp().add(1e-09)
        loss_reconst = -1.0 / kappa_inv * x_reconst_selected.sum((1, 2)).mean() - self.dim_x * self._log_normalization(kappa_inv)
        loss_all = loss_reconst + loss_latent
        idx_estimated = torch.argmax(x_reconst_normed, dim=-1, keepdim=True)
        acc = torch.isclose(x, idx_estimated).sum() / idx_estimated.numel()
        loss = dict(all=loss_all, acc=acc)
        return loss

    def _log_normalization(self, kappa_inv):
        coeff = -(self.__m - 1) * kappa_inv.log() - 1.0 / kappa_inv - torch.log(ive(self.__m - 1, 1.0 / kappa_inv))
        return coeff


class ResBlock(nn.Module):

    def __init__(self, dim, act='relu'):
        super().__init__()
        if act == 'relu':
            activation = nn.ReLU()
        elif act == 'elu':
            activation = nn.ELU()
        self.block = nn.Sequential(activation, nn.Conv2d(dim, dim, 3, 1, 1), nn.BatchNorm2d(dim), activation, nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim))

    def forward(self, x):
        return x + self.block(x)


class EncoderVqResnet28(nn.Module):

    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet28, self).__init__()
        self.flg_variance = flg_var_q
        layers_conv = []
        layers_conv.append(nn.Conv2d(1, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU(True))
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        self.conv = nn.Sequential(*layers_conv)
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb - 1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVqResnet28(nn.Module):

    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVqResnet28, self).__init__()
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU(True))
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 1, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)
        return out


class EncoderVqResnet32(nn.Module):

    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet32, self).__init__()
        self.flg_variance = flg_var_q
        layers_conv = []
        layers_conv.append(nn.Conv2d(3, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU(True))
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU(True))
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU(True))
        self.conv = nn.Sequential(*layers_conv)
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb - 1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVqResnet32(nn.Module):

    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVqResnet32, self).__init__()
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU(True))
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU(True))
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)
        return out


class EncoderVqResnet64(nn.Module):

    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet64, self).__init__()
        self.flg_variance = flg_var_q
        layers_conv = []
        layers_conv.append(nn.Sequential(nn.Conv2d(3, dim_z // 2, 4, stride=2, padding=1)))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 3, stride=1, padding=1))
        self.conv = nn.Sequential(*layers_conv)
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb - 1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        out_conv = self.conv(x)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVqResnet64(nn.Module):

    def __init__(self, dim_z, cfgs, flg_bn=True):
        super(DecoderVqResnet64, self).__init__()
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb - 1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 3, stride=1, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, 3, 4, stride=2, padding=1))
        layers_convt.append(nn.Sigmoid())
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)
        return out


class EncoderVqResnet64Label(nn.Module):

    def __init__(self, dim_z, cfgs, flg_bn=True, flg_var_q=False):
        super(EncoderVqResnet64Label, self).__init__()
        self.n_class = int(np.ceil(cfgs.num_class / 2) * 2)
        self.flg_variance = flg_var_q
        layers_conv = []
        layers_conv.append(nn.Conv2d(self.n_class, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z // 2))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z // 2, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_conv.append(nn.BatchNorm2d(dim_z))
        layers_conv.append(nn.ReLU())
        layers_conv.append(nn.Conv2d(dim_z, dim_z, 4, stride=2, padding=1))
        self.conv = nn.Sequential(*layers_conv)
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb - 1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        self.res_m = ResBlock(dim_z)
        if self.flg_variance:
            self.res_v = ResBlock(dim_z)

    def forward(self, x):
        x_one_hot = F.one_hot(x.long(), num_classes=self.n_class).type_as(x).permute(0, 3, 1, 2).contiguous()
        out_conv = self.conv(x_one_hot)
        out_res = self.res(out_conv)
        mu = self.res_m(out_res)
        if self.flg_variance:
            log_var = self.res_v(out_res)
            return mu, log_var
        else:
            return mu


class DecoderVqResnet64Label(nn.Module):

    def __init__(self, dim_z, cfgs, act='linear', flg_bn=True):
        super(DecoderVqResnet64Label, self).__init__()
        self.n_class = int(np.ceil(cfgs.num_class / 2) * 2)
        num_rb = cfgs.num_rb
        layers_resblocks = []
        for i in range(num_rb - 1):
            layers_resblocks.append(ResBlock(dim_z))
        self.res = nn.Sequential(*layers_resblocks)
        layers_convt = []
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z, dim_z // 2, 4, stride=2, padding=1))
        if flg_bn:
            layers_convt.append(nn.BatchNorm2d(dim_z // 2))
        layers_convt.append(nn.ReLU())
        layers_convt.append(nn.ConvTranspose2d(dim_z // 2, self.n_class, 4, stride=2, padding=1))
        if act == 'sigmoid':
            layers_convt.append(nn.Sigmoid())
        elif act == 'exp':
            layers_convt.append(nn.Softplus())
        elif act == 'tanh':
            layers_convt.append(nn.Tanh())
        self.convt = nn.Sequential(*layers_convt)

    def forward(self, z):
        out_res = self.res(z)
        out = self.convt(out_res)
        return out


class Ive(torch.nn.Module):

    def __init__(self, v):
        super(Ive, self).__init__()
        self.v = v

    def forward(self, z):
        return ive(self.v, z)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str(n >> y & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 19:
        cmap = np.array([(0, 0, 0), (204, 0, 0), (76, 153, 0), (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255), (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0), (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153), (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)], dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ np.uint8(str_id[-1]) << 7 - j
                g = g ^ np.uint8(str_id[-2]) << 7 - j
                b = b ^ np.uint8(str_id[-3]) << 7 - j
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):

    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        return color_image


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0
    return label_numpy


def generate_label(inputs, imsize):
    pred_batch = []
    for input in inputs:
        input = input.view(1, 19, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)
    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)
    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, 19))
    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)
    return label_batch


def idx_to_onehot(idx, n_class=19):
    size = idx.size()
    oneHot_size = size[0], n_class, size[2], size[3]
    label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()
    label = label.scatter_(1, idx.data.long(), 1.0)
    return label


def myprint(statement, noflg=False):
    if not noflg:
        None


def plot_images(images, filename, nrows=4, ncols=8, flg_norm=False):
    if images.shape[1] == 1:
        images = np.repeat(images, 3, axis=1)
    fig = plt.figure(figsize=(nrows * 2, ncols))
    gs = gridspec.GridSpec(nrows * 2, ncols)
    gs.update(wspace=0.05, hspace=0.05)
    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if flg_norm:
            plt.imshow(image.transpose((1, 2, 0)) * 0.5 + 0.5)
        else:
            plt.imshow(image.transpose((1, 2, 0)))
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def plot_images_paper(images, filename, nrows=4, ncols=8, flg_norm=False):
    if images.shape[1] == 1:
        images = np.repeat(images, 3, axis=1)
    fig = plt.figure(figsize=(nrows, ncols))
    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.05, hspace=0.05)
    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_aspect('equal')
        if flg_norm:
            plt.imshow(image.transpose((1, 2, 0)) * 0.5 + 0.5)
        else:
            plt.imshow(image.transpose((1, 2, 0)))
    dirname = os.path.dirname(filename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)


class TrainerBase(nn.Module):

    def __init__(self, cfgs, flgs, train_loader, val_loader, test_loader):
        super(TrainerBase, self).__init__()
        self.cfgs = cfgs
        self.flgs = flgs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = eval('nn.DataParallel({}(cfgs, flgs).cuda())'.format(cfgs.model.name))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfgs.train.lr, amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    def load(self, timestamp=''):
        if timestamp != '':
            self.path = os.path.join(self.cfgs.path, timestamp)
        self.model.load_state_dict(torch.load(os.path.join(self.path, 'best.pt')))
        self.plots = np.load(os.path.join(self.path, 'plots.npy'), allow_pickle=True).item()
        None
        self.model.eval()

    def main_loop(self, max_iter=None, timestamp=None):
        if timestamp == None:
            self._make_path()
        else:
            self.path = os.path.join(self.cfgs.path, timestamp)
        BEST_LOSS = 1e+20
        LAST_SAVED = -1
        if max_iter == None:
            max_iter = self.cfgs.train.epoch_max
        for epoch in range(1, max_iter + 1):
            myprint('[Epoch={}]'.format(epoch), self.flgs.noprint)
            res_train = self._train(epoch)
            if self.flgs.save:
                self._writer_train(res_train, epoch)
            res_test = self._test()
            if self.flgs.save:
                self._writer_val(res_test, epoch)
            if self.flgs.save:
                if res_test['loss'] <= BEST_LOSS:
                    BEST_LOSS = res_test['loss']
                    LAST_SAVED = epoch
                    myprint('----Saving model!', self.flgs.noprint)
                    torch.save(self.model.state_dict(), os.path.join(self.path, 'best.pt'))
                    self.generate_reconstructions(os.path.join(self.path, 'reconstrucitons_best'))
                else:
                    myprint('----Not saving model! Last saved: {}'.format(LAST_SAVED), self.flgs.noprint)
                torch.save(self.model.state_dict(), os.path.join(self.path, 'current.pt'))
                self.generate_reconstructions(os.path.join(self.path, 'reconstructions_current'))

    def preprocess(self, x, y):
        if self.cfgs.dataset.name == 'CelebAMask_HQ':
            y[:, 0, :, :] = y[:, 0, :, :] * 255.0
            y = torch.round(y[:, 0, :, :])
        return y

    def test(self, mode='test'):
        result = self._test(mode)
        if mode == 'test':
            self._writer_test(result)
        return result

    def _set_temperature(self, step, param):
        temperature = np.max([param.init * np.exp(-param.decay * step), param.min])
        return temperature

    def _save_config(self):
        tf = open(self.path + '/configs.json', 'w')
        json.dump(self.cfgs, tf)
        tf.close()

    def _train(self):
        raise NotImplementedError()

    def _test(self):
        raise NotImplementedError()

    def print_loss(self):
        raise NotImplementedError()

    def generate_reconstructions(self):
        raise NotImplementedError()

    def generate_reconstructions_paper(self, nrows=1, ncols=10, off_set=0):
        self.model.eval()
        x = self.test_loader.__iter__().next()[0]
        x = x[off_set:off_set + nrows * ncols]
        output = self.model(x, flg_train=False, flg_quant_det=True)
        x_tilde = output[0]
        images_original = x.cpu().data.numpy()
        images_reconst = x_tilde.cpu().data.numpy()
        plot_images_paper(images_original, os.path.join(self.path, 'paper_original'), nrows=nrows, ncols=ncols)
        plot_images_paper(images_reconst, os.path.join(self.path, 'paper_reconst'), nrows=nrows, ncols=ncols)

    def _generate_reconstructions_continuous(self, filename, nrows=4, ncols=8):
        self.model.eval()
        x = self.test_loader.__iter__().next()[0]
        x = x[:nrows * ncols]
        output = self.model(x, flg_train=False, flg_quant_det=True)
        x_tilde = output[0]
        x_cat = torch.cat([x, x_tilde], 0)
        images = x_cat.cpu().data.numpy()
        plot_images(images, filename + '.png', nrows=nrows, ncols=ncols)

    def _generate_reconstructions_discrete(self, filename, nrows=4, ncols=8):
        self.model.eval()
        x, y = self.test_loader.__iter__().next()
        x = x[:nrows * ncols]
        y = y[:nrows * ncols]
        y[:, 0, :, :] = y[:, 0, :, :] * 255.0
        y_long = y
        y = y[:, 0, :, :]
        output = self.model(y, flg_train=False, flg_quant_det=True)
        label_tilde = output[0]
        label_real = idx_to_onehot(y_long)
        label_batch_predict = generate_label(label_tilde[:, :19, :, :], x.shape[-1])
        label_batch_real = generate_label(label_real, x.shape[-1])
        x_cat = torch.cat([label_batch_real, label_batch_predict], 0)
        images = x_cat.cpu().data.numpy()
        plot_images(images, filename + '.png', nrows=nrows, ncols=ncols)

    def _make_path(self):
        dt_now = datetime.datetime.now()
        timestamp = dt_now.strftime('%m%d_%H%M')
        self.path = os.path.join(self.cfgs.path, '{}_seed{}_{}'.format(self.cfgs.network.name, self.cfgs.train.seed, timestamp))
        None
        if self.flgs.save:
            self._makedir(self.path)
            list_dir = self.cfgs.list_dir_for_copy
            files = []
            for dirname in list_dir:
                files.append(glob.glob(dirname + '*.py'))
            target = os.path.join(self.path, 'codes')
            for i, dirname in enumerate(list_dir):
                if not os.path.exists(os.path.join(target, dirname)):
                    os.makedirs(os.path.join(target, dirname))
                for file in files[i]:
                    shutil.copyfile(file, os.path.join(target, file))

    def _makedir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            i = 1
            while True:
                path += '_{}'.format(i)
                if not os.path.exists(path):
                    os.makedirs(path)
                    break
                None
                i += 1
        self._save_config()
        self.path = path

    def _writer_train(self, result, epoch):
        self._append_writer_train(result)
        np.save(os.path.join(self.path, 'plots.npy'), self.plots)

    def _writer_val(self, result, epoch):
        self._append_writer_val(result)
        np.save(os.path.join(self.path, 'plots.npy'), self.plots)

    def _writer_test(self, result):
        self._append_writer_test(result)
        np.save(os.path.join(self.path, 'plots.npy'), self.plots)

    def _append_writer_train(self, result):
        for metric in result:
            self.plots[metric + '_train'].append(result[metric])

    def _append_writer_val(self, result):
        for metric in result:
            self.plots[metric + '_val'].append(result[metric])

    def _append_writer_test(self, result):
        for metric in result:
            self.plots[metric + '_test'].append(result[metric])


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (DecoderVqResnet28,
     lambda: ([], {'dim_z': 4, 'cfgs': _mock_config(num_rb=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderVqResnet32,
     lambda: ([], {'dim_z': 4, 'cfgs': _mock_config(num_rb=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderVqResnet64,
     lambda: ([], {'dim_z': 4, 'cfgs': _mock_config(num_rb=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DecoderVqResnet64Label,
     lambda: ([], {'dim_z': 4, 'cfgs': _mock_config(num_class=4, num_rb=4)}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EncoderVqResnet28,
     lambda: ([], {'dim_z': 4, 'cfgs': _mock_config(num_rb=4)}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     False),
    (EncoderVqResnet32,
     lambda: ([], {'dim_z': 4, 'cfgs': _mock_config(num_rb=4)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (EncoderVqResnet64,
     lambda: ([], {'dim_z': 4, 'cfgs': _mock_config(num_rb=4)}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (Ive,
     lambda: ([], {'v': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (Jitter,
     lambda: ([], {'p': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_sony_sqvae(_paritybench_base):
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

