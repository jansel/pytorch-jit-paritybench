import sys
_module = sys.modules[__name__]
del sys
setup = _module
vq_vae = _module
auto_encoder = _module
main = _module
nearest_embed = _module
test_nearest_embed = _module
util = _module

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


import numpy as np


import logging


import torch


import torch.utils.data


from torch import nn


from torch.nn import functional as F


import time


from torch import optim


import torch.backends.cudnn as cudnn


from torch.utils.tensorboard import SummaryWriter


from torchvision import datasets


from torchvision import transforms


from torchvision.utils import save_image


from torchvision.utils import make_grid


from torch.autograd import Function


from torch.autograd import Variable


import torch.nn.functional as F


from itertools import cycle


import logging.config


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class VAE(nn.Module):
    """Variational AutoEncoder for MNIST
       Taken from pytorch/examples: https://github.com/pytorch/examples/tree/master/vae"""

    def __init__(self, kl_coef=1, **kwargs):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.kl_coef = kl_coef
        self.bce = 0
        self.kl = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = torch.randn(size, 20)
        if self:
            sample = sample
        sample = self.decode(sample).cpu()
        return sample

    def loss_function(self, x, recon_x, mu, logvar):
        self.bce = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
        batch_size = x.size(0)
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return self.bce + self.kl_coef * self.kl

    def latest_losses(self):
        return {'bce': self.bce, 'kl': self.kl}


class NearestEmbedFunc(Function):
    """
    Input:
    ------
    x - (batch_size, emb_dim, *)
        Last dimensions may be arbitrary
    emb - (emb_dim, num_emb)
    """

    @staticmethod
    def forward(ctx, input, emb):
        if input.size(1) != emb.size(0):
            raise RuntimeError('invalid argument: input.size(1) ({}) must be equal to emb.size(0) ({})'.format(input.size(1), emb.size(0)))
        ctx.batch_size = input.size(0)
        ctx.num_latents = int(np.prod(np.array(input.size()[2:])))
        ctx.emb_dim = emb.size(0)
        ctx.num_emb = emb.size(1)
        ctx.input_type = type(input)
        ctx.dims = list(range(len(input.size())))
        x_expanded = input.unsqueeze(-1)
        num_arbitrary_dims = len(ctx.dims) - 2
        if num_arbitrary_dims:
            emb_expanded = emb.view(emb.shape[0], *([1] * num_arbitrary_dims), emb.shape[1])
        else:
            emb_expanded = emb
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [input.shape[0], *list(input.shape[2:]), input.shape[1]]
        result = emb.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, ctx.dims[-1], *ctx.dims[1:-1])
        ctx.save_for_backward(argmin)
        return result.contiguous(), argmin

    @staticmethod
    def backward(ctx, grad_output, argmin=None):
        grad_input = grad_emb = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output
        if ctx.needs_input_grad[1]:
            argmin, = ctx.saved_variables
            latent_indices = torch.arange(ctx.num_emb).type_as(argmin)
            idx_choices = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(grad_output.data)
            n_idx_choice = idx_choices.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            idx_avg_choices = idx_choices / n_idx_choice
            grad_output = grad_output.permute(0, *ctx.dims[2:], 1).contiguous()
            grad_output = grad_output.view(ctx.batch_size * ctx.num_latents, ctx.emb_dim)
            grad_emb = torch.sum(grad_output.data.view(-1, ctx.emb_dim, 1) * idx_avg_choices.view(-1, 1, ctx.num_emb), 0)
        return grad_input, grad_emb, None, None


def nearest_embed(x, emb):
    return NearestEmbedFunc().apply(x, emb)


class NearestEmbed(nn.Module):

    def __init__(self, num_embeddings, embeddings_dim):
        super(NearestEmbed, self).__init__()
        self.weight = nn.Parameter(torch.rand(embeddings_dim, num_embeddings))

    def forward(self, x, weight_sg=False):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        return nearest_embed(x, self.weight.detach() if weight_sg else self.weight)


class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""

    def __init__(self, hidden=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()
        self.emb_size = k
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, hidden)
        self.fc3 = nn.Linear(hidden, 400)
        self.fc4 = nn.Linear(400, 784)
        self.emb = NearestEmbed(k, self.emb_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = hidden
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.fc2(h1)
        return h2.view(-1, self.emb_size, int(self.hidden / self.emb_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, 784))
        z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
        emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.emb_size, int(self.hidden / self.emb_size))
        if self:
            sample = sample
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())
        return self.ce_loss + self.vq_coef * self.vq_loss + self.comit_coef * self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        layers = [nn.ReLU(), nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class CVAE(AbstractAutoEncoder):

    def __init__(self, d, kl_coef=0.1, **kwargs):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, d // 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(d // 2), nn.ReLU(inplace=True), nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True), ResBlock(d, d, bn=True), nn.BatchNorm2d(d), ResBlock(d, d, bn=True))
        self.decoder = nn.Sequential(ResBlock(d, d, bn=True), nn.BatchNorm2d(d), ResBlock(d, d, bn=True), nn.BatchNorm2d(d), nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False), nn.BatchNorm2d(d // 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1, bias=False))
        self.f = 8
        self.d = d
        self.fc11 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.fc12 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.kl_coef = kl_coef
        self.kl_loss = 0
        self.mse = 0

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = torch.randn(size, self.d * self.f ** 2, requires_grad=False)
        if self:
            sample = sample
        return self.decode(sample).cpu()

    def loss_function(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        self.kl_loss /= batch_size * 3 * 1024
        return self.mse + self.kl_coef * self.kl_loss

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl_loss}


class VQ_CVAE(nn.Module):

    def __init__(self, d, k=10, bn=True, vq_coef=1, commit_coef=0.5, num_channels=3, **kwargs):
        super(VQ_CVAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(num_channels, d, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True), nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True), ResBlock(d, d, bn), nn.BatchNorm2d(d), ResBlock(d, d, bn), nn.BatchNorm2d(d))
        self.decoder = nn.Sequential(ResBlock(d, d), nn.BatchNorm2d(d), ResBlock(d, d), nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(d), nn.ReLU(inplace=True), nn.ConvTranspose2d(d, num_channels, kernel_size=4, stride=2, padding=1))
        self.d = d
        self.emb = NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0
        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)
        self.encoder[-1].weight.detach().fill_(1 / 40)
        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f, self.f, requires_grad=False),
        if self:
            sample = sample
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        self.mse = F.mse_loss(recon_x, x)
        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(torch.norm((emb.detach() - z_e) ** 2, 2, 1))
        return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss, 'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):
        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)


class NearestEmbedEMA(nn.Module):

    def __init__(self, n_emb, emb_dim, decay=0.99, eps=1e-05):
        super(NearestEmbedEMA, self).__init__()
        self.decay = decay
        self.eps = eps
        self.embeddings_dim = emb_dim
        self.n_emb = n_emb
        self.emb_dim = emb_dim
        embed = torch.rand(emb_dim, n_emb)
        self.register_buffer('weight', embed)
        self.register_buffer('cluster_size', torch.zeros(n_emb))
        self.register_buffer('embed_avg', embed.clone())

    def forward(self, x):
        """Input:
        ---------
        x - (batch_size, emb_size, *)
        """
        dims = list(range(len(x.size())))
        x_expanded = x.unsqueeze(-1)
        num_arbitrary_dims = len(dims) - 2
        if num_arbitrary_dims:
            emb_expanded = self.weight.view(self.emb_dim, *([1] * num_arbitrary_dims), self.n_emb)
        else:
            emb_expanded = self.weight
        dist = torch.norm(x_expanded - emb_expanded, 2, 1)
        _, argmin = dist.min(-1)
        shifted_shape = [x.shape[0], *list(x.shape[2:]), x.shape[1]]
        result = self.weight.t().index_select(0, argmin.view(-1)).view(shifted_shape).permute(0, dims[-1], *dims[1:-1])
        if self.training:
            latent_indices = torch.arange(self.n_emb).type_as(argmin)
            emb_onehot = (argmin.view(-1, 1) == latent_indices.view(1, -1)).type_as(x.data)
            n_idx_choice = emb_onehot.sum(0)
            n_idx_choice[n_idx_choice == 0] = 1
            flatten = x.permute(1, 0, *dims[-2:]).contiguous().view(x.shape[1], -1)
            self.cluster_size.data.mul_(self.decay).add_(1 - self.decay, n_idx_choice)
            embed_sum = flatten @ emb_onehot
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_emb * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.weight.data.copy_(embed_normalized)
        return result, argmin


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AbstractAutoEncoder,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (CVAE,
     lambda: ([], {'d': 4}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (NearestEmbed,
     lambda: ([], {'num_embeddings': 4, 'embeddings_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (NearestEmbedEMA,
     lambda: ([], {'n_emb': 4, 'emb_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (ResBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_nadavbh12_VQ_VAE(_paritybench_base):
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

