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


import torch.nn as nn


import torch.nn.functional as f


from torch.distributions.normal import Normal


import torch.nn.functional as F


from torch.distributions.categorical import Categorical


from functools import partial


from torch.utils.data import DataLoader


import numpy as np


import torch.utils.data


from torch import optim


from torch.nn import functional as F


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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ctallec_world_models(_paritybench_base):
    pass
    def test_000(self):
        self._check(Decoder(*[], **{'img_channels': 4, 'latent_size': 4}), [torch.rand([4, 4])], {})

    def test_001(self):
        self._check(Encoder(*[], **{'img_channels': 4, 'latent_size': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_002(self):
        self._check(VAE(*[], **{'img_channels': 4, 'latent_size': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(_MDRNNBase(*[], **{'latents': 4, 'actions': 4, 'hiddens': 4, 'gaussians': 4}), [], {})

