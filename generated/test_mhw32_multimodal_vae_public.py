import sys
_module = sys.modules[__name__]
del sys
datasets = _module
model = _module
sample = _module
train = _module
model = _module
train = _module
model = _module
sample = _module
train = _module
model = _module
sample = _module
train = _module
datasets = _module
model = _module
sample = _module
train = _module
utils = _module
datasets = _module
sample = _module
setup = _module
train = _module

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


import copy


import random


import numpy as np


import numpy.random as npr


from random import shuffle


import torch


from torch.utils.data.dataset import Dataset


import torch.nn as nn


from torch.autograd import Variable


from torch.nn import functional as F


import torch.nn.functional as F


from torchvision import transforms


from torchvision.utils import save_image


import torch.optim as optim


from itertools import combinations


from torch.nn.parameter import Parameter


from torchvision import datasets


from torchvision.datasets import MNIST


import torchvision.datasets as dset


import string


import time


import math


from copy import deepcopy


class Swish(nn.Module):

    def forward(self, x):
        return x * F.sigmoid(x)


class ImageDecoder(nn.Module):
    """Parametrizes p(x|z).

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(nn.Linear(n_latents, 256 * 2 * 2), Swish())
        self.hallucinate = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 0, bias=False), nn.BatchNorm2d(128), Swish(), nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), Swish(), nn.ConvTranspose2d(64, 32, 5, 2, 1, bias=False), nn.BatchNorm2d(32), Swish(), nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False))

    def forward(self, z):
        z = self.upsample(z)
        z = z.view(-1, 256, 2, 2)
        z = self.hallucinate(z)
        return z


class ImageEncoder(nn.Module):
    """Parametrizes q(z|x).

    This task is quite a bit harder than MNIST so we probably need 
    to use an CNN of some form. This will be good to get us ready for
    natural images.

    @param n_latents: integer
                      size of latent vector
    """

    def __init__(self, n_latents):
        super(ImageEncoder, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1, bias=False), Swish(), nn.Conv2d(32, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), Swish(), nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), Swish(), nn.Conv2d(128, 256, 4, 2, 0, bias=False), nn.BatchNorm2d(256), Swish())
        self.classifier = nn.Sequential(nn.Linear(256 * 2 * 2, 512), Swish(), nn.Dropout(p=0.1), nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.features(x)
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return x[:, :n_latents], x[:, n_latents:]


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, mu, logvar, eps=1e-08):
        var = torch.exp(logvar) + eps
        T = 1.0 / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)
        return pd_mu, pd_logvar


all_characters = '0123456789'


n_characters = len(all_characters)


SOS = n_characters


max_length = 4


def swish(x):
    return x * F.sigmoid(x)


class TextDecoder(nn.Module):
    """Parametrizes p(y|z).

    GRU for text decoding. Given a start token, sample a character
    via an RNN and repeat for a fixed length.

    @param n_latents: integer
                      size of latent vector
    @param n_characters: integer
                         size of characters (10 for MNIST)
    @param n_hiddens: integer [default: 200]
                      number of hidden units in GRU
    """

    def __init__(self, n_latents, n_characters, n_hiddens=200):
        super(TextDecoder, self).__init__()
        self.embed = nn.Embedding(n_characters, n_hiddens)
        self.z2h = nn.Linear(n_latents, n_hiddens)
        self.gru = nn.GRU(n_hiddens + n_latents, n_hiddens, 2, dropout=0.1)
        self.h2o = nn.Linear(n_hiddens + n_latents, n_characters)
        self.n_latents = n_latents
        self.n_characters = n_characters

    def forward(self, z):
        n_latents = self.n_latents
        n_characters = self.n_characters
        batch_size = z.size(0)
        c_in = Variable(torch.LongTensor([SOS]).repeat(batch_size))
        words = Variable(torch.zeros(batch_size, max_length, n_characters))
        if z.is_cuda:
            c_in = c_in
            words = words
        h = self.z2h(z).unsqueeze(0).repeat(2, 1, 1)
        for i in xrange(max_length):
            c_out, h = self.step(i, z, c_in, h)
            sample = torch.max(F.log_softmax(c_out, dim=1), dim=1)[1]
            words[:, i] = c_out
            c_in = sample
        return words

    def step(self, ix, z, c_in, h):
        c_in = swish(self.embed(c_in))
        c_in = torch.cat((c_in, z), dim=1)
        c_in = c_in.unsqueeze(0)
        c_out, h = self.gru(c_in, h)
        c_out = c_out.squeeze(0)
        c_out = torch.cat((c_out, z), dim=1)
        c_out = self.h2o(c_out)
        return c_out, h


class TextEncoder(nn.Module):
    """Parametrizes q(z|y).

    We train an embedding layer from the 10 digit space
    to move to a continuous domain. The GRU is optionally 
    bidirectional.

    @param n_latents: integer
                      size of latent vector
    @param n_characters: integer
                         number of possible characters (10 for MNIST)
    @param n_hiddens: integer [default: 200]
                      number of hidden units in GRU
    @param bidirectional: boolean [default: True]
                          hyperparameter for GRU.
    """

    def __init__(self, n_latents, n_characters, n_hiddens=200, bidirectional=True):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(n_characters, n_hiddens)
        self.gru = nn.GRU(n_hiddens, n_hiddens, 1, dropout=0.1, bidirectional=bidirectional)
        self.h2p = nn.Linear(n_hiddens, n_latents * 2)
        self.n_latents = n_latents
        self.n_hiddens = n_hiddens
        self.bidirectional = bidirectional

    def forward(self, x):
        n_hiddens = self.n_hiddens
        n_latents = self.n_latents
        x = self.embed(x)
        x = x.transpose(0, 1)
        x, h = self.gru(x, None)
        x = x[-1]
        if self.bidirectional:
            x = x[:, :n_hiddens] + x[:, n_hiddens:]
        x = self.h2p(x)
        return x[:, :n_latents], x[:, n_latents:]


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.log(torch.ones(size)))
    if use_cuda:
        mu, logvar = mu, logvar
    return mu, logvar


class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.

    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, n_latents):
        super(MVAE, self).__init__()
        self.image_encoder = ImageEncoder(n_latents)
        self.image_decoder = ImageDecoder(n_latents)
        self.text_encoder = TextEncoder(n_latents, n_characters, n_hiddens=200, bidirectional=True)
        self.text_decoder = TextDecoder(n_latents, n_characters, n_hiddens=200)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, image=None, text=None):
        mu, logvar = self.infer(image, text)
        z = self.reparametrize(mu, logvar)
        img_recon = self.image_decoder(z)
        txt_recon = self.text_decoder(z)
        return img_recon, txt_recon, mu, logvar

    def infer(self, image=None, text=None):
        batch_size = image.size(0) if image is not None else text.size(0)
        use_cuda = next(self.parameters()).is_cuda
        mu, logvar = prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)
        if image is not None:
            img_mu, img_logvar = self.image_encoder(image)
            mu = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)
        if text is not None:
            txt_mu, txt_logvar = self.text_encoder(text)
            mu = torch.cat((mu, txt_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, txt_logvar.unsqueeze(0)), dim=0)
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class AttributeEncoder(nn.Module):
    """Parametrizes q(z|y). 

    We use a single inference network that encodes 
    a single attribute.

    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents):
        super(AttributeEncoder, self).__init__()
        self.net = nn.Sequential(nn.Embedding(2, 512), Swish(), nn.Linear(512, 512), Swish(), nn.Linear(512, n_latents * 2))
        self.n_latents = n_latents

    def forward(self, x):
        n_latents = self.n_latents
        x = self.net(x.long())
        return x[:, :n_latents], x[:, n_latents:]


class AttributeDecoder(nn.Module):
    """Parametrizes p(y|z).

    We use a single generative network that decodes 
    a single attribute.

    @param n_latents: integer
                      number of latent variable dimensions.
    """

    def __init__(self, n_latents):
        super(AttributeDecoder, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_latents, 512), Swish(), nn.Linear(512, 512), Swish(), nn.Linear(512, 512), Swish(), nn.Linear(512, 1))

    def forward(self, z):
        z = self.net(z)
        return z


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AttributeDecoder,
     lambda: ([], {'n_latents': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (AttributeEncoder,
     lambda: ([], {'n_latents': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageDecoder,
     lambda: ([], {'n_latents': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ImageEncoder,
     lambda: ([], {'n_latents': 4}),
     lambda: ([torch.rand([4, 1, 64, 64])], {}),
     True),
    (ProductOfExperts,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (Swish,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_mhw32_multimodal_vae_public(_paritybench_base):
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

