import sys
_module = sys.modules[__name__]
del sys
experiment = _module
models = _module
base = _module
beta_vae = _module
betatc_vae = _module
cat_vae = _module
cvae = _module
dfcvae = _module
dip_vae = _module
fvae = _module
gamma_vae = _module
hvae = _module
info_vae = _module
iwae = _module
joint_vae = _module
logcosh_vae = _module
lvae = _module
miwae = _module
mssim_vae = _module
swae = _module
twostage_vae = _module
types_ = _module
vampvae = _module
vanilla_vae = _module
vq_vae = _module
wae_mmd = _module
run = _module
bvae = _module
test_betatcvae = _module
test_cat_vae = _module
test_dfc = _module
test_dipvae = _module
test_fvae = _module
test_gvae = _module
test_hvae = _module
test_iwae = _module
test_joint_Vae = _module
test_logcosh = _module
test_lvae = _module
test_miwae = _module
test_mssimvae = _module
test_swae = _module
test_vae = _module
test_vq_vae = _module
test_wae = _module
text_cvae = _module
text_vamp = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, sklearn, string, tensorflow, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
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


import math


import torch


from torch import optim


from torchvision import transforms


import torchvision.utils as vutils


from torchvision.datasets import CelebA


from torch.utils.data import DataLoader


from torch import nn


from abc import abstractmethod


from torch.nn import functional as F


import numpy as np


from torchvision.models import vgg19_bn


from torch.distributions import Gamma


import torch.nn.init as init


import torch.nn.functional as F


from math import floor


from math import pi


from math import log


from torch.distributions import Normal


from math import exp


from torch import distributions as dist


from typing import List


from typing import Callable


from typing import Union


from typing import Any


from typing import TypeVar


from typing import Tuple


import torch.backends.cudnn as cudnn


class BaseVAE(nn.Module):

    def __init__(self) ->None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) ->List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) ->Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) ->Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) ->Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) ->Tensor:
        pass


class BetaVAE(BaseVAE):
    num_iter = 0

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, beta: int=4, gamma: float=1000.0, max_capacity: int=25, Capacity_max_iter: int=100000.0, loss_type: str='B', **kwargs) ->None:
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) ->dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B':
            self.C_max = self.C_max
            C = torch.clamp(self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class BetaTCVAE(BaseVAE):
    num_iter = 0

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, anneal_steps: int=200, alpha: float=1.0, beta: float=6.0, gamma: float=1.0, **kwargs) ->None:
        super(BetaTCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.anneal_steps = anneal_steps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 32]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1] * 16, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, 256 * 2)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 32, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]

    def log_density_gaussian(self, x: Tensor, mu: Tensor, logvar: Tensor):
        """
        Computes the log pdf of the Gaussian with parameters mu and logvar at x
        :param x: (Tensor) Point at whichGaussian PDF is to be evaluated
        :param mu: (Tensor) Mean of the Gaussian distribution
        :param logvar: (Tensor) Log variance of the Gaussian distribution
        :return:
        """
        norm = -0.5 * (math.log(2 * math.pi) + logvar)
        log_density = norm - 0.5 * ((x - mu) ** 2 * torch.exp(-logvar))
        return log_density

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        weight = 1
        recons_loss = F.mse_loss(recons, input, reduction='sum')
        log_q_zx = self.log_density_gaussian(z, mu, log_var).sum(dim=1)
        zeros = torch.zeros_like(z)
        log_p_z = self.log_density_gaussian(z, zeros, zeros).sum(dim=1)
        batch_size, latent_dim = z.shape
        mat_log_q_z = self.log_density_gaussian(z.view(batch_size, 1, latent_dim), mu.view(1, batch_size, latent_dim), log_var.view(1, batch_size, latent_dim))
        dataset_size = 1 / kwargs['M_N'] * batch_size
        strat_weight = (dataset_size - batch_size + 1) / (dataset_size * (batch_size - 1))
        importance_weights = torch.Tensor(batch_size, batch_size).fill_(1 / (batch_size - 1))
        importance_weights.view(-1)[::batch_size] = 1 / dataset_size
        importance_weights.view(-1)[1::batch_size] = strat_weight
        importance_weights[batch_size - 2, 0] = strat_weight
        log_importance_weights = importance_weights.log()
        mat_log_q_z += log_importance_weights.view(batch_size, batch_size, 1)
        log_q_z = torch.logsumexp(mat_log_q_z.sum(2), dim=1, keepdim=False)
        log_prod_q_z = torch.logsumexp(mat_log_q_z, dim=1, keepdim=False).sum(1)
        mi_loss = (log_q_zx - log_q_z).mean()
        tc_loss = (log_q_z - log_prod_q_z).mean()
        kld_loss = (log_prod_q_z - log_p_z).mean()
        if self.training:
            self.num_iter += 1
            anneal_rate = min(0 + 1 * self.num_iter / self.anneal_steps, 1)
        else:
            anneal_rate = 1.0
        loss = recons_loss / batch_size + self.alpha * mi_loss + weight * (self.beta * tc_loss + anneal_rate * self.gamma * kld_loss)
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss, 'TC_Loss': tc_loss, 'MI_Loss': mi_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class CategoricalVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, categorical_dim: int=40, hidden_dims: List=None, temperature: float=0.5, anneal_rate: float=3e-05, anneal_interval: int=100, alpha: float=30.0, **kwargs) ->None:
        super(CategoricalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1] * 4, self.latent_dim * self.categorical_dim)
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim * self.categorical_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())
        self.sampling_dist = torch.distributions.OneHotCategorical(1.0 / categorical_dim * torch.ones((self.categorical_dim, 1)))

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) Latent code [B x D x Q]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        z = z.view(-1, self.latent_dim, self.categorical_dim)
        return [z]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x Q]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, z: Tensor, eps: float=1e-07) ->Tensor:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param z: (Tensor) Latent Codes [B x D x Q]
        :return: (Tensor) [B x D]
        """
        u = torch.rand_like(z)
        g = -torch.log(-torch.log(u + eps) + eps)
        s = F.softmax((z + g) / self.temp, dim=-1)
        s = s.view(-1, self.latent_dim * self.categorical_dim)
        return s

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        q = self.encode(input)[0]
        z = self.reparameterize(q)
        return [self.decode(z), input, q]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        q = args[2]
        q_p = F.softmax(q, dim=-1)
        kld_weight = kwargs['M_N']
        batch_idx = kwargs['batch_idx']
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate * batch_idx), self.min_temp)
        recons_loss = F.mse_loss(recons, input, reduction='mean')
        eps = 1e-07
        h1 = q_p * torch.log(q_p + eps)
        h2 = q_p * np.log(1.0 / self.categorical_dim + eps)
        kld_loss = torch.mean(torch.sum(h1 - h2, dim=(1, 2)), dim=0)
        loss = self.alpha * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        M = num_samples * self.latent_dim
        np_y = np.zeros((M, self.categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(self.categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M // self.latent_dim, self.latent_dim, self.categorical_dim])
        z = torch.from_numpy(np_y)
        z = z.view(num_samples, self.latent_dim * self.categorical_dim)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class ConditionalVAE(BaseVAE):

    def __init__(self, in_channels: int, num_classes: int, latent_dim: int, hidden_dims: List=None, img_size: int=64, **kwargs) ->None:
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        in_channels += 1
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        y = kwargs['labels'].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)
        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        z = torch.cat([z, y], dim=1)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) ->dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        y = kwargs['labels'].float()
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x, **kwargs)[0]


class DFCVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, alpha: float=1, beta: float=0.5, **kwargs) ->None:
        super(DFCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())
        self.feature_network = vgg19_bn(pretrained=True)
        for param in self.feature_network.parameters():
            param.requires_grad = False
        self.feature_network.eval()

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)
        recons_features = self.extract_features(recons)
        input_features = self.extract_features(input)
        return [recons, input, recons_features, input_features, mu, log_var]

    def extract_features(self, input: Tensor, feature_layers: List=None) ->List[Tensor]:
        """
        Extracts the features from the pretrained model
        at the layers indicated by feature_layers.
        :param input: (Tensor) [B x C x H x W]
        :param feature_layers: List of string of IDs
        :return: List of the extracted features
        """
        if feature_layers is None:
            feature_layers = ['14', '24', '34', '43']
        features = []
        result = input
        for key, module in self.feature_network.features._modules.items():
            result = module(result)
            if key in feature_layers:
                features.append(result)
        return features

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        recons_features = args[2]
        input_features = args[3]
        mu = args[4]
        log_var = args[5]
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        feature_loss = 0.0
        for r, i in zip(recons_features, input_features):
            feature_loss += F.mse_loss(r, i)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = self.beta * (recons_loss + feature_loss) + self.alpha * kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class DIPVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, lambda_diag: float=10.0, lambda_offdiag: float=5.0, **kwargs) ->None:
        super(DIPVAE, self).__init__()
        self.latent_dim = latent_dim
        self.lambda_diag = lambda_diag
        self.lambda_offdiag = lambda_offdiag
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = 1
        recons_loss = F.mse_loss(recons, input, reduction='sum')
        kld_loss = torch.sum(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        centered_mu = mu - mu.mean(dim=1, keepdim=True)
        cov_mu = centered_mu.t().matmul(centered_mu).squeeze()
        cov_z = cov_mu + torch.mean(torch.diagonal((2.0 * log_var).exp(), dim1=0), dim=0)
        cov_diag = torch.diag(cov_z)
        cov_offdiag = cov_z - torch.diag(cov_diag)
        dip_loss = self.lambda_offdiag * torch.sum(cov_offdiag ** 2) + self.lambda_diag * torch.sum((cov_diag - 1) ** 2)
        loss = recons_loss + kld_weight * kld_loss + dip_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss, 'DIP_Loss': dip_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class FactorVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, gamma: float=40.0, **kwargs) ->None:
        super(FactorVAE, self).__init__()
        self.latent_dim = latent_dim
        self.gamma = gamma
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())
        self.discriminator = nn.Sequential(nn.Linear(self.latent_dim, 1000), nn.BatchNorm1d(1000), nn.LeakyReLU(0.2), nn.Linear(1000, 1000), nn.BatchNorm1d(1000), nn.LeakyReLU(0.2), nn.Linear(1000, 1000), nn.BatchNorm1d(1000), nn.LeakyReLU(0.2), nn.Linear(1000, 2))
        self.D_z_reserve = None

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]

    def permute_latent(self, z: Tensor) ->Tensor:
        """
        Permutes each of the latent codes in the batch
        :param z: [B x D]
        :return: [B x D]
        """
        B, D = z.size()
        inds = torch.cat([(D * i + torch.randperm(D)) for i in range(B)])
        return z.view(-1)[inds].view(B, D)

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        kld_weight = kwargs['M_N']
        optimizer_idx = kwargs['optimizer_idx']
        if optimizer_idx == 0:
            recons_loss = F.mse_loss(recons, input)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            self.D_z_reserve = self.discriminator(z)
            vae_tc_loss = (self.D_z_reserve[:, (0)] - self.D_z_reserve[:, (1)]).mean()
            loss = recons_loss + kld_weight * kld_loss + self.gamma * vae_tc_loss
            return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss, 'VAE_TC_Loss': vae_tc_loss}
        elif optimizer_idx == 1:
            device = input.device
            true_labels = torch.ones(input.size(0), dtype=torch.long, requires_grad=False)
            false_labels = torch.zeros(input.size(0), dtype=torch.long, requires_grad=False)
            z = z.detach()
            z_perm = self.permute_latent(z)
            D_z_perm = self.discriminator(z_perm)
            D_tc_loss = 0.5 * (F.cross_entropy(self.D_z_reserve, false_labels) + F.cross_entropy(D_z_perm, true_labels))
            return {'loss': D_tc_loss, 'D_TC_Loss': D_tc_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


def init_(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.orthogonal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class GammaVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, gamma_shape: float=8.0, prior_shape: float=2.0, prior_rate: float=1.0, **kwargs) ->None:
        super(GammaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.B = gamma_shape
        self.prior_alpha = torch.tensor([prior_shape])
        self.prior_beta = torch.tensor([prior_rate])
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(nn.Linear(hidden_dims[-1] * 4, latent_dim), nn.Softmax())
        self.fc_var = nn.Sequential(nn.Linear(hidden_dims[-1] * 4, latent_dim), nn.Softmax())
        modules = []
        self.decoder_input = nn.Sequential(nn.Linear(latent_dim, hidden_dims[-1] * 4))
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Sigmoid())
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                init_(m)

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        alpha = self.fc_mu(result)
        beta = self.fc_var(result)
        return [alpha, beta]

    def decode(self, z: Tensor) ->Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, alpha: Tensor, beta: Tensor) ->Tensor:
        """
        Reparameterize the Gamma distribution by the shape augmentation trick.
        Reference:
        [1] https://arxiv.org/pdf/1610.05683.pdf

        :param alpha: (Tensor) Shape parameter of the latent Gamma
        :param beta: (Tensor) Rate parameter of the latent Gamma
        :return:
        """
        alpha_ = alpha.clone().detach()
        z_hat = Gamma(alpha_ + self.B, torch.ones_like(alpha_)).sample()
        eps = self.inv_h_func(alpha + self.B, z_hat)
        z = self.h_func(alpha + self.B, eps)
        return z / beta

    def h_func(self, alpha: Tensor, eps: Tensor) ->Tensor:
        """
        Reparameterize a sample eps ~ N(0, 1) so that h(z) ~ Gamma(alpha, 1)
        :param alpha: (Tensor) Shape parameter
        :param eps: (Tensor) Random sample to reparameterize
        :return: (Tensor)
        """
        z = (alpha - 1.0 / 3.0) * (1 + eps / torch.sqrt(9.0 * alpha - 3.0)) ** 3
        return z

    def inv_h_func(self, alpha: Tensor, z: Tensor) ->Tensor:
        """
        Inverse reparameterize the given z into eps.
        :param alpha: (Tensor)
        :param z: (Tensor)
        :return: (Tensor)
        """
        eps = torch.sqrt(9.0 * alpha - 3.0) * ((z / (alpha - 1.0 / 3.0)) ** (1.0 / 3.0) - 1.0)
        return eps

    def forward(self, input: Tensor, **kwargs) ->Tensor:
        alpha, beta = self.encode(input)
        z = self.reparameterize(alpha, beta)
        return [self.decode(z), input, alpha, beta]

    def I_function(self, a, b, c, d):
        return -c * d / a - b * torch.log(a) - torch.lgamma(b) + (b - 1) * (torch.digamma(d) + torch.log(c))

    def vae_gamma_kl_loss(self, a, b, c, d):
        """
        https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
        b and d are Gamma shape parameters and
        a and c are scale parameters.
        (All, therefore, must be positive.)
        """
        a = 1 / a
        c = 1 / c
        losses = self.I_function(c, d, c, d) - self.I_function(a, b, c, d)
        return torch.sum(losses, dim=1)

    def loss_function(self, *args, **kwargs) ->dict:
        recons = args[0]
        input = args[1]
        alpha = args[2]
        beta = args[3]
        curr_device = input.device
        kld_weight = kwargs['M_N']
        recons_loss = torch.mean(F.mse_loss(recons, input, reduction='none'), dim=(1, 2, 3))
        self.prior_alpha = self.prior_alpha
        self.prior_beta = self.prior_beta
        kld_loss = self.vae_gamma_kl_loss(alpha, beta, self.prior_alpha, self.prior_beta)
        loss = recons_loss + kld_loss
        loss = torch.mean(loss, dim=0)
        return {'loss': loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the modelSay
        :return: (Tensor)
        """
        z = Gamma(self.prior_alpha, self.prior_beta).sample((num_samples, self.latent_dim))
        z = z.squeeze()
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class HVAE(BaseVAE):

    def __init__(self, in_channels: int, latent1_dim: int, latent2_dim: int, hidden_dims: List=None, img_size: int=64, pseudo_input_size: int=128, **kwargs) ->None:
        super(HVAE, self).__init__()
        self.latent1_dim = latent1_dim
        self.latent2_dim = latent2_dim
        self.img_size = img_size
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        channels = in_channels
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            channels = h_dim
        self.encoder_z2_layers = nn.Sequential(*modules)
        self.fc_z2_mu = nn.Linear(hidden_dims[-1] * 4, latent2_dim)
        self.fc_z2_var = nn.Linear(hidden_dims[-1] * 4, latent2_dim)
        self.embed_z2_code = nn.Linear(latent2_dim, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        modules = []
        channels = in_channels + 1
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            channels = h_dim
        self.encoder_z1_layers = nn.Sequential(*modules)
        self.fc_z1_mu = nn.Linear(hidden_dims[-1] * 4, latent1_dim)
        self.fc_z1_var = nn.Linear(hidden_dims[-1] * 4, latent1_dim)
        self.recons_z1_mu = nn.Linear(latent2_dim, latent1_dim)
        self.recons_z1_log_var = nn.Linear(latent2_dim, latent1_dim)
        self.debed_z1_code = nn.Linear(latent1_dim, 1024)
        self.debed_z2_code = nn.Linear(latent2_dim, 1024)
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode_z2(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder_z2_layers(input)
        result = torch.flatten(result, start_dim=1)
        z2_mu = self.fc_z2_mu(result)
        z2_log_var = self.fc_z2_var(result)
        return [z2_mu, z2_log_var]

    def encode_z1(self, input: Tensor, z2: Tensor) ->List[Tensor]:
        x = self.embed_data(input)
        z2 = self.embed_z2_code(z2)
        z2 = z2.view(-1, self.img_size, self.img_size).unsqueeze(1)
        result = torch.cat([x, z2], dim=1)
        result = self.encoder_z1_layers(result)
        result = torch.flatten(result, start_dim=1)
        z1_mu = self.fc_z1_mu(result)
        z1_log_var = self.fc_z1_var(result)
        return [z1_mu, z1_log_var]

    def encode(self, input: Tensor) ->List[Tensor]:
        z2_mu, z2_log_var = self.encode_z2(input)
        z2 = self.reparameterize(z2_mu, z2_log_var)
        z1_mu, z1_log_var = self.encode_z1(input, z2)
        return [z1_mu, z1_log_var, z2_mu, z2_log_var, z2]

    def decode(self, input: Tensor) ->Tensor:
        result = self.decoder(input)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        z1_mu, z1_log_var, z2_mu, z2_log_var, z2 = self.encode(input)
        z1 = self.reparameterize(z1_mu, z1_log_var)
        debedded_z1 = self.debed_z1_code(z1)
        debedded_z2 = self.debed_z2_code(z2)
        result = torch.cat([debedded_z1, debedded_z2], dim=1)
        result = result.view(-1, 512, 2, 2)
        recons = self.decode(result)
        return [recons, input, z1_mu, z1_log_var, z2_mu, z2_log_var, z1, z2]

    def loss_function(self, *args, **kwargs) ->dict:
        recons = args[0]
        input = args[1]
        z1_mu = args[2]
        z1_log_var = args[3]
        z2_mu = args[4]
        z2_log_var = args[5]
        z1 = args[6]
        z2 = args[7]
        z1_p_mu = self.recons_z1_mu(z2)
        z1_p_log_var = self.recons_z1_log_var(z2)
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        z1_kld = torch.mean(-0.5 * torch.sum(1 + z1_log_var - z1_mu ** 2 - z1_log_var.exp(), dim=1), dim=0)
        z2_kld = torch.mean(-0.5 * torch.sum(1 + z2_log_var - z2_mu ** 2 - z2_log_var.exp(), dim=1), dim=0)
        z1_p_kld = torch.mean(-0.5 * torch.sum(1 + z1_p_log_var - (z1 - z1_p_mu) ** 2 - z1_p_log_var.exp(), dim=1), dim=0)
        z2_p_kld = torch.mean(-0.5 * z2 ** 2, dim=0)
        kld_loss = -(z1_p_kld - z1_kld - z2_kld)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, batch_size: int, current_device: int, **kwargs) ->Tensor:
        z2 = torch.randn(batch_size, self.latent2_dim)
        z2 = z2
        z1_mu = self.recons_z1_mu(z2)
        z1_log_var = self.recons_z1_log_var(z2)
        z1 = self.reparameterize(z1_mu, z1_log_var)
        debedded_z1 = self.debed_z1_code(z1)
        debedded_z2 = self.debed_z2_code(z2)
        result = torch.cat([debedded_z1, debedded_z2], dim=1)
        result = result.view(-1, 512, 2, 2)
        samples = self.decode(result)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class InfoVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, alpha: float=-0.5, beta: float=5.0, reg_weight: int=100, kernel_type: str='imq', latent_var: float=2.0, **kwargs) ->None:
        super(InfoVAE, self).__init__()
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var
        assert alpha <= 0, 'alpha must be negative or zero.'
        self.alpha = alpha
        self.beta = beta
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, z, mu, log_var]

    def loss_function(self, *args, **kwargs) ->dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        mu = args[3]
        log_var = args[4]
        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = self.beta * recons_loss + (1.0 - self.alpha) * kld_weight * kld_loss + (self.alpha + self.reg_weight - 1.0) / bias_corr * mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'MMD': mmd_loss, 'KLD': -kld_loss}

    def compute_kernel(self, x1: Tensor, x2: Tensor) ->Tensor:
        D = x1.size(1)
        N = x1.size(0)
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-3)
        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)
        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')
        return result

    def compute_rbf(self, x1: Tensor, x2: Tensor, eps: float=1e-07) ->Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * self.z_var
        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1: Tensor, x2: Tensor, eps: float=1e-07) ->Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \\sum rac{C}{C + \\|x_1 - x_2 \\|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))
        result = kernel.sum() - kernel.diag().sum()
        return result

    def compute_mmd(self, z: Tensor) ->Tensor:
        prior_z = torch.randn_like(z)
        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)
        mmd = prior_z__kernel.mean() + z__kernel.mean() - 2 * priorz_z__kernel.mean()
        return mmd

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class IWAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, num_samples: int=5, **kwargs) ->None:
        super(IWAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes of S samples
        onto the image space.
        :param z: (Tensor) [B x S x D]
        :return: (Tensor) [B x S x C x H x W]
        """
        B, _, _ = z.size()
        z = z.view(-1, self.latent_dim)
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = result.view([B, -1, result.size(1), result.size(2), result.size(3)])
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2)
        log_var = log_var.repeat(self.num_samples, 1, 1).permute(1, 0, 2)
        z = self.reparameterize(mu, log_var)
        eps = (z - mu) / log_var
        return [self.decode(z), input, mu, log_var, z, eps]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        eps = args[5]
        input = input.repeat(self.num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        kld_weight = kwargs['M_N']
        log_p_x_z = ((recons - input) ** 2).flatten(2).mean(-1)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2)
        log_weight = log_p_x_z + kld_weight * kld_loss
        weight = F.softmax(log_weight, dim=-1)
        loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim=0)
        return {'loss': loss, 'Reconstruction_Loss': log_p_x_z.mean(), 'KLD': -kld_loss.mean()}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 1, self.latent_dim)
        z = z
        samples = self.decode(z).squeeze()
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image.
        Returns only the first reconstructed sample
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0][:, (0), :]


class JointVAE(BaseVAE):
    num_iter = 1

    def __init__(self, in_channels: int, latent_dim: int, categorical_dim: int, latent_min_capacity: float=0.0, latent_max_capacity: float=25.0, latent_gamma: float=30.0, latent_num_iter: int=25000, categorical_min_capacity: float=0.0, categorical_max_capacity: float=25.0, categorical_gamma: float=30.0, categorical_num_iter: int=25000, hidden_dims: List=None, temperature: float=0.5, anneal_rate: float=3e-05, anneal_interval: int=100, alpha: float=30.0, **kwargs) ->None:
        super(JointVAE, self).__init__()
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim
        self.temp = temperature
        self.min_temp = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        self.cont_min = latent_min_capacity
        self.cont_max = latent_max_capacity
        self.disc_min = categorical_min_capacity
        self.disc_max = categorical_max_capacity
        self.cont_gamma = latent_gamma
        self.disc_gamma = categorical_gamma
        self.cont_iter = latent_num_iter
        self.disc_iter = categorical_num_iter
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)
        self.fc_z = nn.Linear(hidden_dims[-1] * 4, self.categorical_dim)
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim + self.categorical_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())
        self.sampling_dist = torch.distributions.OneHotCategorical(1.0 / categorical_dim * torch.ones((self.categorical_dim, 1)))

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x H x W]
        :return: (Tensor) Latent code [B x D x Q]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        z = self.fc_z(result)
        z = z.view(-1, self.categorical_dim)
        return [mu, log_var, z]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x Q]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor, q: Tensor, eps: float=1e-07) ->Tensor:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param mu: (Tensor) mean of the latent Gaussian  [B x D]
        :param log_var: (Tensor) Log variance of the latent Gaussian [B x D]
        :param q: (Tensor) Categorical latent Codes [B x Q]
        :return: (Tensor) [B x (D + Q)]
        """
        std = torch.exp(0.5 * log_var)
        e = torch.randn_like(std)
        z = e * std + mu
        u = torch.rand_like(q)
        g = -torch.log(-torch.log(u + eps) + eps)
        s = F.softmax((q + g) / self.temp, dim=-1)
        s = s.view(-1, self.categorical_dim)
        return torch.cat([z, s], dim=1)

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var, q = self.encode(input)
        z = self.reparameterize(mu, log_var, q)
        return [self.decode(z), input, q, mu, log_var]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        q = args[2]
        mu = args[3]
        log_var = args[4]
        q_p = F.softmax(q, dim=-1)
        kld_weight = kwargs['M_N']
        batch_idx = kwargs['batch_idx']
        if batch_idx % self.anneal_interval == 0 and self.training:
            self.temp = np.maximum(self.temp * np.exp(-self.anneal_rate * batch_idx), self.min_temp)
        recons_loss = F.mse_loss(recons, input, reduction='mean')
        disc_curr = (self.disc_max - self.disc_min) * self.num_iter / float(self.disc_iter) + self.disc_min
        disc_curr = min(disc_curr, np.log(self.categorical_dim))
        eps = 1e-07
        h1 = q_p * torch.log(q_p + eps)
        h2 = q_p * np.log(1.0 / self.categorical_dim + eps)
        kld_disc_loss = torch.mean(torch.sum(h1 - h2, dim=1), dim=0)
        cont_curr = (self.cont_max - self.cont_min) * self.num_iter / float(self.cont_iter) + self.cont_min
        cont_curr = min(cont_curr, self.cont_max)
        kld_cont_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        capacity_loss = self.disc_gamma * torch.abs(disc_curr - kld_disc_loss) + self.cont_gamma * torch.abs(cont_curr - kld_cont_loss)
        loss = self.alpha * recons_loss + kld_weight * capacity_loss
        if self.training:
            self.num_iter += 1
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'Capacity_Loss': capacity_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        M = num_samples
        np_y = np.zeros((M, self.categorical_dim), dtype=np.float32)
        np_y[range(M), np.random.choice(self.categorical_dim, M)] = 1
        np_y = np.reshape(np_y, [M, self.categorical_dim])
        q = torch.from_numpy(np_y)
        z = torch.cat([z, q], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class LogCoshVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, alpha: float=100.0, beta: float=10.0, **kwargs) ->None:
        super(LogCoshVAE, self).__init__()
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        t = recons - input
        recons_loss = self.alpha * t + torch.log(1.0 + torch.exp(-2 * self.alpha * t)) - torch.log(torch.tensor(2.0))
        recons_loss = 1.0 / self.alpha * recons_loss.mean()
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.beta * kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


def conv_out_shape(img_size):
    return floor((img_size + 2 - 3) / 2.0) + 1


class EncoderBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, latent_dim: int, img_size: int):
        super(EncoderBlock, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(out_channels), nn.LeakyReLU())
        out_size = conv_out_shape(img_size)
        self.encoder_mu = nn.Linear(out_channels * out_size ** 2, latent_dim)
        self.encoder_var = nn.Linear(out_channels * out_size ** 2, latent_dim)

    def forward(self, input: Tensor) ->Tensor:
        result = self.encoder(input)
        h = torch.flatten(result, start_dim=1)
        mu = self.encoder_mu(h)
        log_var = self.encoder_var(h)
        return [result, mu, log_var]


class LadderBlock(nn.Module):

    def __init__(self, in_channels: int, latent_dim: int):
        super(LadderBlock, self).__init__()
        self.decode = nn.Sequential(nn.Linear(in_channels, latent_dim), nn.BatchNorm1d(latent_dim))
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: Tensor) ->Tensor:
        z = self.decode(z)
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        return [mu, log_var]


class LVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dims: List, hidden_dims: List, **kwargs) ->None:
        super(LVAE, self).__init__()
        self.latent_dims = latent_dims
        self.hidden_dims = hidden_dims
        self.num_rungs = len(latent_dims)
        assert len(latent_dims) == len(hidden_dims), 'Length of the latentand hidden dims must be the same'
        modules = []
        img_size = 64
        for i, h_dim in enumerate(hidden_dims):
            modules.append(EncoderBlock(in_channels, h_dim, latent_dims[i], img_size))
            img_size = conv_out_shape(img_size)
            in_channels = h_dim
        self.encoders = nn.Sequential(*modules)
        modules = []
        for i in range(self.num_rungs - 1, 0, -1):
            modules.append(LadderBlock(latent_dims[i], latent_dims[i - 1]))
        self.ladders = nn.Sequential(*modules)
        self.decoder_input = nn.Linear(latent_dims[0], hidden_dims[-1] * 4)
        hidden_dims.reverse()
        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())
        hidden_dims.reverse()

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        h = input
        post_params = []
        for encoder_block in self.encoders:
            h, mu, log_var = encoder_block(h)
            post_params.append((mu, log_var))
        return post_params

    def decode(self, z: Tensor, post_params: List) ->Tuple:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        kl_div = 0
        post_params.reverse()
        for i, ladder_block in enumerate(self.ladders):
            mu_e, log_var_e = post_params[i]
            mu_t, log_var_t = ladder_block(z)
            mu, log_var = self.merge_gauss(mu_e, mu_t, log_var_e, log_var_t)
            z = self.reparameterize(mu, log_var)
            kl_div += self.compute_kl_divergence(z, (mu, log_var), (mu_e, log_var_e))
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        return self.final_layer(result), kl_div

    def merge_gauss(self, mu_1: Tensor, mu_2: Tensor, log_var_1: Tensor, log_var_2: Tensor) ->List:
        p_1 = 1.0 / (log_var_1.exp() + 1e-07)
        p_2 = 1.0 / (log_var_2.exp() + 1e-07)
        mu = (mu_1 * p_1 + mu_2 * p_2) / (p_1 + p_2)
        log_var = torch.log(1.0 / (p_1 + p_2))
        return [mu, log_var]

    def compute_kl_divergence(self, z: Tensor, q_params: Tuple, p_params: Tuple):
        mu_q, log_var_q = q_params
        mu_p, log_var_p = p_params
        kl = log_var_p - log_var_q + (log_var_q.exp() + (mu_q - mu_p) ** 2) / (2 * log_var_p.exp()) - 0.5
        kl = torch.sum(kl, dim=-1)
        return kl

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        post_params = self.encode(input)
        mu, log_var = post_params.pop()
        z = self.reparameterize(mu, log_var)
        recons, kl_div = self.decode(z, post_params)
        return [recons, input, kl_div]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        kl_div = args[2]
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(kl_div, dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dims[-1])
        z = z
        for ladder_block in self.ladders:
            mu, log_var = ladder_block(z)
            z = self.reparameterize(mu, log_var)
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        samples = self.final_layer(result)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class MIWAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, num_samples: int=5, num_estimates: int=5, **kwargs) ->None:
        super(MIWAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.num_estimates = num_estimates
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes of S samples
        onto the image space.
        :param z: (Tensor) [B x S x D]
        :return: (Tensor) [B x S x C x H x W]
        """
        B, M, S, D = z.size()
        z = z.view(-1, self.latent_dim)
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = result.view([B, M, S, result.size(-3), result.size(-2), result.size(-1)])
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        mu = mu.repeat(self.num_estimates, self.num_samples, 1, 1).permute(2, 0, 1, 3)
        log_var = log_var.repeat(self.num_estimates, self.num_samples, 1, 1).permute(2, 0, 1, 3)
        z = self.reparameterize(mu, log_var)
        eps = (z - mu) / log_var
        return [self.decode(z), input, mu, log_var, z, eps]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        eps = args[5]
        input = input.repeat(self.num_estimates, self.num_samples, 1, 1, 1, 1).permute(2, 0, 1, 3, 4, 5)
        kld_weight = kwargs['M_N']
        log_p_x_z = ((recons - input) ** 2).flatten(3).mean(-1)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=3)
        log_weight = log_p_x_z + kld_weight * kld_loss
        weight = F.softmax(log_weight, dim=-1)
        loss = torch.mean(torch.mean(torch.sum(weight * log_weight, dim=-1), dim=-2), dim=0)
        return {'loss': loss, 'Reconstruction_Loss': log_p_x_z.mean(), 'KLD': -kld_loss.mean()}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, 1, 1, self.latent_dim)
        z = z
        samples = self.decode(z).squeeze()
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image.
        Returns only the first reconstructed sample
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0][:, (0), (0), :]


class MSSIM(nn.Module):

    def __init__(self, in_channels: int=3, window_size: int=11, size_average: bool=True) ->None:
        """
        Computes the differentiable MS-SSIM loss
        Reference:
        [1] https://github.com/jorge-pessoa/pytorch-msssim/blob/dev/pytorch_msssim/__init__.py
            (MIT License)

        :param in_channels: (Int)
        :param window_size: (Int)
        :param size_average: (Bool)
        """
        super(MSSIM, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.size_average = size_average

    def gaussian_window(self, window_size: int, sigma: float) ->Tensor:
        kernel = torch.tensor([exp((x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
        return kernel / kernel.sum()

    def create_window(self, window_size, in_channels):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(in_channels, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1: Tensor, img2: Tensor, window_size: int, in_channel: int, size_average: bool) ->Tensor:
        device = img1.device
        window = self.create_window(window_size, in_channel)
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=in_channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=in_channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=in_channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=in_channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=in_channel) - mu1_mu2
        img_range = img1.max() - img1.min()
        C1 = (0.01 * img_range) ** 2
        C2 = (0.03 * img_range) ** 2
        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)
        ssim_map = (2 * mu1_mu2 + C1) * v1 / ((mu1_sq + mu2_sq + C1) * v2)
        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)
        return ret, cs

    def forward(self, img1: Tensor, img2: Tensor) ->Tensor:
        device = img1.device
        weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        levels = weights.size()[0]
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.ssim(img1, img2, self.window_size, self.in_channels, self.size_average)
            mssim.append(sim)
            mcs.append(cs)
            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))
        mssim = torch.stack(mssim)
        mcs = torch.stack(mcs)
        pow1 = mcs ** weights
        pow2 = mssim ** weights
        output = torch.prod(pow1[:-1] * pow2[-1])
        return 1 - output


class MSSIMVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, window_size: int=11, size_average: bool=True, **kwargs) ->None:
        super(MSSIMVAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())
        self.mssim_loss = MSSIM(self.in_channels, window_size, size_average)

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args: Any, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        recons_loss = self.mssim_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class SWAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, reg_weight: int=100, wasserstein_deg: float=2.0, num_projections: int=50, projection_dist: str='normal', **kwargs) ->None:
        super(SWAE, self).__init__()
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.p = wasserstein_deg
        self.num_projections = num_projections
        self.proj_dist = projection_dist
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z

    def decode(self, z: Tensor) ->Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        z = self.encode(input)
        return [self.decode(z), input, z]

    def loss_function(self, *args, **kwargs) ->dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr
        recons_loss_l2 = F.mse_loss(recons, input)
        recons_loss_l1 = F.l1_loss(recons, input)
        swd_loss = self.compute_swd(z, self.p, reg_weight)
        loss = recons_loss_l2 + recons_loss_l1 + swd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss_l2 + recons_loss_l1, 'SWD': swd_loss}

    def get_random_projections(self, latent_dim: int, num_samples: int) ->Tensor:
        """
        Returns random samples from latent distribution's (Gaussian)
        unit sphere for projecting the encoded samples and the
        distribution samples.

        :param latent_dim: (Int) Dimensionality of the latent space (D)
        :param num_samples: (Int) Number of samples required (S)
        :return: Random projections from the latent unit sphere
        """
        if self.proj_dist == 'normal':
            rand_samples = torch.randn(num_samples, latent_dim)
        elif self.proj_dist == 'cauchy':
            rand_samples = dist.Cauchy(torch.tensor([0.0]), torch.tensor([1.0])).sample((num_samples, latent_dim)).squeeze()
        else:
            raise ValueError('Unknown projection distribution.')
        rand_proj = rand_samples / rand_samples.norm(dim=1).view(-1, 1)
        return rand_proj

    def compute_swd(self, z: Tensor, p: float, reg_weight: float) ->Tensor:
        """
        Computes the Sliced Wasserstein Distance (SWD) - which consists of
        randomly projecting the encoded and prior vectors and computing
        their Wasserstein distance along those projections.

        :param z: Latent samples # [N  x D]
        :param p: Value for the p^th Wasserstein distance
        :param reg_weight:
        :return:
        """
        prior_z = torch.randn_like(z)
        device = z.device
        proj_matrix = self.get_random_projections(self.latent_dim, num_samples=self.num_projections).transpose(0, 1)
        latent_projections = z.matmul(proj_matrix)
        prior_projections = prior_z.matmul(proj_matrix)
        w_dist = torch.sort(latent_projections.t(), dim=1)[0] - torch.sort(prior_projections.t(), dim=1)[0]
        w_dist = w_dist.pow(p)
        return reg_weight * w_dist.mean()

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class TwoStageVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, hidden_dims2: List=None, **kwargs) ->None:
        super(TwoStageVAE, self).__init__()
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        if hidden_dims2 is None:
            hidden_dims2 = [1024, 1024]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())
        encoder2 = []
        in_channels = self.latent_dim
        for h_dim in hidden_dims2:
            encoder2.append(nn.Sequential(nn.Linear(in_channels, h_dim), nn.BatchNorm1d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder2 = nn.Sequential(*encoder2)
        self.fc_mu2 = nn.Linear(hidden_dims2[-1], self.latent_dim)
        self.fc_var2 = nn.Linear(hidden_dims2[-1], self.latent_dim)
        decoder2 = []
        hidden_dims2.reverse()
        in_channels = self.latent_dim
        for h_dim in hidden_dims2:
            decoder2.append(nn.Sequential(nn.Linear(in_channels, h_dim), nn.BatchNorm1d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.decoder2 = nn.Sequential(*decoder2)

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class VampVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, num_components: int=50, **kwargs) ->None:
        super(VampVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_components = num_components
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())
        self.pseudo_input = torch.eye(self.num_components, requires_grad=False)
        self.embed_pseudo = nn.Sequential(nn.Linear(self.num_components, 12288), nn.Hardtanh(0.0, 1.0))

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var, z]

    def loss_function(self, *args, **kwargs) ->dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        z = args[4]
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        E_log_q_z = torch.mean(torch.sum(-0.5 * (log_var + (z - mu) ** 2) / log_var.exp(), dim=1), dim=0)
        M, C, H, W = input.size()
        curr_device = input.device
        self.pseudo_input = self.pseudo_input
        x = self.embed_pseudo(self.pseudo_input)
        x = x.view(-1, C, H, W)
        prior_mu, prior_log_var = self.encode(x)
        z_expand = z.unsqueeze(1)
        prior_mu = prior_mu.unsqueeze(0)
        prior_log_var = prior_log_var.unsqueeze(0)
        E_log_p_z = torch.sum(-0.5 * (prior_log_var + (z_expand - prior_mu) ** 2) / prior_log_var.exp(), dim=2) - torch.log(torch.tensor(self.num_components).float())
        E_log_p_z = torch.logsumexp(E_log_p_z, dim=1)
        E_log_p_z = torch.mean(E_log_p_z, dim=0)
        kld_loss = -(E_log_p_z - E_log_q_z)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class VanillaVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, **kwargs) ->None:
        super(VanillaVAE, self).__init__()
        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) ->Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        Computes the VAE loss function.
        KL(N(\\mu, \\sigma), N(0, 1)) = \\log rac{1}{\\sigma} + rac{\\sigma^2 + \\mu^2}{2} - rac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class VectorQuantizer(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float=0.25):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) ->Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1) - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view(latents_shape)
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())
        vq_loss = commitment_loss * self.beta + embedding_loss
        quantized_latents = latents + (quantized_latents - latents).detach()
        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss


class ResidualLayer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.ReLU(True), nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, input: Tensor) ->Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self, in_channels: int, embedding_dim: int, num_embeddings: int, hidden_dims: List=None, beta: float=0.25, img_size: int=64, **kwargs) ->None:
        super(VQVAE, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=4, stride=2, padding=1), nn.LeakyReLU()))
            in_channels = h_dim
        modules.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()))
        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Sequential(nn.Conv2d(in_channels, embedding_dim, kernel_size=1, stride=1), nn.LeakyReLU()))
        self.encoder = nn.Sequential(*modules)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, self.beta)
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(embedding_dim, hidden_dims[-1], kernel_size=3, stride=1, padding=1), nn.LeakyReLU()))
        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.LeakyReLU())
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=4, stride=2, padding=1), nn.LeakyReLU()))
        modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], out_channels=3, kernel_size=4, stride=2, padding=1), nn.Tanh()))
        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) ->List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) ->Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder(z)
        return result

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self, *args, **kwargs) ->dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]
        recons_loss = F.mse_loss(recons, input)
        loss = recons_loss + vq_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'VQ_Loss': vq_loss}

    def sample(self, num_samples: int, current_device: Union[int, str], **kwargs) ->Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


class WAE_MMD(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List=None, reg_weight: int=100, kernel_type: str='imq', latent_var: float=2.0, **kwargs) ->None:
        super(WAE_MMD, self).__init__()
        self.latent_dim = latent_dim
        self.reg_weight = reg_weight
        self.kernel_type = kernel_type
        self.z_var = latent_var
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.fc_z = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Sequential(nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[i + 1]), nn.LeakyReLU()))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hidden_dims[-1]), nn.LeakyReLU(), nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1), nn.Tanh())

    def encode(self, input: Tensor) ->Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        z = self.fc_z(result)
        return z

    def decode(self, z: Tensor) ->Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input: Tensor, **kwargs) ->List[Tensor]:
        z = self.encode(input)
        return [self.decode(z), input, z]

    def loss_function(self, *args, **kwargs) ->dict:
        recons = args[0]
        input = args[1]
        z = args[2]
        batch_size = input.size(0)
        bias_corr = batch_size * (batch_size - 1)
        reg_weight = self.reg_weight / bias_corr
        recons_loss = F.mse_loss(recons, input)
        mmd_loss = self.compute_mmd(z, reg_weight)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'MMD': mmd_loss}

    def compute_kernel(self, x1: Tensor, x2: Tensor) ->Tensor:
        D = x1.size(1)
        N = x1.size(0)
        x1 = x1.unsqueeze(-2)
        x2 = x2.unsqueeze(-3)
        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)
        if self.kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')
        return result

    def compute_rbf(self, x1: Tensor, x2: Tensor, eps: float=1e-07) ->Tensor:
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2.0 * z_dim * self.z_var
        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1: Tensor, x2: Tensor, eps: float=1e-07) ->Tensor:
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \\sum rac{C}{C + \\|x_1 - x_2 \\|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim=-1))
        result = kernel.sum() - kernel.diag().sum()
        return result

    def compute_mmd(self, z: Tensor, reg_weight: float) ->Tensor:
        prior_z = torch.randn_like(z)
        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)
        mmd = reg_weight * prior_z__kernel.mean() + reg_weight * z__kernel.mean() - 2 * reg_weight * priorz_z__kernel.mean()
        return mmd

    def sample(self, num_samples: int, current_device: int, **kwargs) ->Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        z = z
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) ->Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (BaseVAE,
     lambda: ([], {}),
     lambda: ([], {}),
     False),
    (BetaTCVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (BetaVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (CategoricalVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (DIPVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (EncoderBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'latent_dim': 4, 'img_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (FactorVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (GammaVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (HVAE,
     lambda: ([], {'in_channels': 4, 'latent1_dim': 4, 'latent2_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (InfoVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (JointVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4, 'categorical_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (LVAE,
     lambda: ([], {'in_channels': 4, 'latent_dims': [4, 4], 'hidden_dims': [4, 4]}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (LadderBlock,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (LogCoshVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (MSSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (MSSIMVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (ResidualLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (SWAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (TwoStageVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (VQVAE,
     lambda: ([], {'in_channels': 4, 'embedding_dim': 4, 'num_embeddings': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (VampVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (VanillaVAE,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
    (VectorQuantizer,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (WAE_MMD,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     False),
]

class Test_AntixK_PyTorch_VAE(_paritybench_base):
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

