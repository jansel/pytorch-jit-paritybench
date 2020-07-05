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
import abc, collections, copy, enum, functools, inspect, itertools, logging, math, numbers, numpy, random, re, scipy, string, time, torch, torchaudio, torchtext, torchvision, types, typing, uuid, warnings
import numpy as np
from torch import Tensor
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


from torch import nn


from abc import abstractmethod


import torch


from torch.nn import functional as F


import math


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (EncoderBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'latent_dim': 4, 'img_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (LadderBlock,
     lambda: ([], {'in_channels': 4, 'latent_dim': 4}),
     lambda: ([torch.rand([4, 4, 4])], {}),
     False),
    (MSSIM,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64]), torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResidualLayer,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VectorQuantizer,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
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

