import sys
_module = sys.modules[__name__]
del sys
data = _module
carracing = _module
generation_script = _module
loaders = _module
envs = _module
simulated_carracing = _module
examine_data = _module
models = _module
controller = _module
mdrnn = _module
vae = _module
test_controller = _module
test_data = _module
test_envs = _module
test_gmm = _module
traincontroller = _module
trainmdrnn = _module
trainvae = _module
utils = _module
learning = _module
misc = _module

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


import torch


import torch.utils.data


import numpy as np


from torch.distributions.categorical import Categorical


import torch.nn as nn


import torch.nn.functional as f


from torch.distributions.normal import Normal


import torch.nn.functional as F


import time


from torchvision import transforms


from time import sleep


from torch.multiprocessing import Process


from torch.multiprocessing import Queue


from functools import partial


from torch.utils.data import DataLoader


from torch import optim


from torch.nn import functional as F


from torchvision.utils import save_image


from torch.optim import Optimizer


import math


class Controller(nn.Module):
    """ Controller """

    def __init__(self, latents, recurrents, actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, actions)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)


class _MDRNNBase(nn.Module):

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__()
        self.latents = latents
        self.actions = actions
        self.hiddens = hiddens
        self.gaussians = gaussians
        self.gmm_linear = nn.Linear(hiddens, (2 * latents + 1) * gaussians + 2)

    def forward(self, *inputs):
        pass


class MDRNN(_MDRNNBase):
    """ MDRNN model for multi steps forward """

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTM(latents + actions, hiddens)

    def forward(self, actions, latents):
        """ MULTI STEPS forward.

        :args actions: (SEQ_LEN, BSIZE, ASIZE) torch tensor
        :args latents: (SEQ_LEN, BSIZE, LSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, rs, ds, parameters of the GMM
        prediction for the next latent, gaussian prediction of the reward and
        logit prediction of terminality.
            - mu_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (SEQ_LEN, BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (SEQ_LEN, BSIZE, N_GAUSS) torch tensor
            - rs: (SEQ_LEN, BSIZE) torch tensor
            - ds: (SEQ_LEN, BSIZE) torch tensor
        """
        seq_len, bs = actions.size(0), actions.size(1)
        ins = torch.cat([actions, latents], dim=-1)
        outs, _ = self.rnn(ins)
        gmm_outs = self.gmm_linear(outs)
        stride = self.gaussians * self.latents
        mus = gmm_outs[:, :, :stride]
        mus = mus.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = gmm_outs[:, :, stride:2 * stride]
        sigmas = sigmas.view(seq_len, bs, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)
        pi = gmm_outs[:, :, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(seq_len, bs, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)
        rs = gmm_outs[:, :, (-2)]
        ds = gmm_outs[:, :, (-1)]
        return mus, sigmas, logpi, rs, ds


class MDRNNCell(_MDRNNBase):
    """ MDRNN model for one step forward """

    def __init__(self, latents, actions, hiddens, gaussians):
        super().__init__(latents, actions, hiddens, gaussians)
        self.rnn = nn.LSTMCell(latents + actions, hiddens)

    def forward(self, action, latent, hidden):
        """ ONE STEP forward.

        :args actions: (BSIZE, ASIZE) torch tensor
        :args latents: (BSIZE, LSIZE) torch tensor
        :args hidden: (BSIZE, RSIZE) torch tensor

        :returns: mu_nlat, sig_nlat, pi_nlat, r, d, next_hidden, parameters of
        the GMM prediction for the next latent, gaussian prediction of the
        reward, logit prediction of terminality and next hidden state.
            - mu_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - sigma_nlat: (BSIZE, N_GAUSS, LSIZE) torch tensor
            - logpi_nlat: (BSIZE, N_GAUSS) torch tensor
            - rs: (BSIZE) torch tensor
            - ds: (BSIZE) torch tensor
        """
        in_al = torch.cat([action, latent], dim=1)
        next_hidden = self.rnn(in_al, hidden)
        out_rnn = next_hidden[0]
        out_full = self.gmm_linear(out_rnn)
        stride = self.gaussians * self.latents
        mus = out_full[:, :stride]
        mus = mus.view(-1, self.gaussians, self.latents)
        sigmas = out_full[:, stride:2 * stride]
        sigmas = sigmas.view(-1, self.gaussians, self.latents)
        sigmas = torch.exp(sigmas)
        pi = out_full[:, 2 * stride:2 * stride + self.gaussians]
        pi = pi.view(-1, self.gaussians)
        logpi = f.log_softmax(pi, dim=-1)
        r = out_full[:, (-2)]
        d = out_full[:, (-1)]
        return mus, sigmas, logpi, r, d, next_hidden


class Decoder(nn.Module):
    """ VAE decoder """

    def __init__(self, img_channels, latent_size):
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction


class Encoder(nn.Module):
    """ VAE encoder """

    def __init__(self, img_channels, latent_size):
        super(Encoder, self).__init__()
        self.latent_size = latent_size
        self.img_channels = img_channels
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_mu = nn.Linear(2 * 2 * 256, latent_size)
        self.fc_logsigma = nn.Linear(2 * 2 * 256, latent_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        return mu, logsigma


class VAE(nn.Module):
    """ Variational Autoencoder """

    def __init__(self, img_channels, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_channels, latent_size)
        self.decoder = Decoder(img_channels, latent_size)

    def forward(self, x):
        mu, logsigma = self.encoder(x)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Decoder,
     lambda: ([], {'img_channels': 4, 'latent_size': 4}),
     lambda: ([torch.rand([4, 4])], {}),
     True),
    (Encoder,
     lambda: ([], {'img_channels': 4, 'latent_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (MDRNN,
     lambda: ([], {'latents': 4, 'actions': 4, 'hiddens': 4, 'gaussians': 4}),
     lambda: ([torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {}),
     True),
    (VAE,
     lambda: ([], {'img_channels': 4, 'latent_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (_MDRNNBase,
     lambda: ([], {'latents': 4, 'actions': 4, 'hiddens': 4, 'gaussians': 4}),
     lambda: ([], {}),
     False),
]

class Test_ctallec_world_models(_paritybench_base):
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

