import sys
_module = sys.modules[__name__]
del sys
datasets = _module
functions = _module
modules = _module
pixelcnn_baseline = _module
pixelcnn_prior = _module
test_functions = _module
vae = _module
vqvae = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.distributions.normal import Normal


from torch.distributions import kl_divergence


from torchvision import datasets


from torchvision import transforms


import numpy as np


from torchvision.utils import save_image


import time


from torchvision.utils import make_grid


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print('Skipping initialization of ', classname)


class VAE(nn.Module):

    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1), nn
            .BatchNorm2d(dim), nn.ReLU(True), nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim), nn.ReLU(True), nn.Conv2d(dim, dim, 5, 1, 0
            ), nn.BatchNorm2d(dim), nn.ReLU(True), nn.Conv2d(dim, z_dim * 2,
            3, 1, 0), nn.BatchNorm2d(z_dim * 2))
        self.decoder = nn.Sequential(nn.ConvTranspose2d(z_dim, dim, 3, 1, 0
            ), nn.BatchNorm2d(dim), nn.ReLU(True), nn.ConvTranspose2d(dim,
            dim, 5, 1, 0), nn.BatchNorm2d(dim), nn.ReLU(True), nn.
            ConvTranspose2d(dim, dim, 4, 2, 1), nn.BatchNorm2d(dim), nn.
            ReLU(True), nn.ConvTranspose2d(dim, input_dim, 4, 2, 1), nn.Tanh())
        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        q_z_x = Normal(mu, logvar.mul(0.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


class VectorQuantization(Function):

    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)
            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)
            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            'Trying to call `.grad()` on graph containing `VectorQuantization`. The function `VectorQuantization` is not differentiable. Use `VectorQuantizationStraightThrough` if you want a straight-through estimator of the gradient.'
            )


vq = VectorQuantization.apply


class VectorQuantizationStraightThrough(Function):

    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)
        codes_flatten = torch.index_select(codebook, dim=0, index=
            indices_flatten)
        codes = codes_flatten.view_as(inputs)
        return codes, indices_flatten

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)
            grad_output_flatten = grad_output.contiguous().view(-1,
                embedding_size)
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)
        return grad_inputs, grad_codebook


vq_st = VectorQuantizationStraightThrough.apply


class VQEmbedding(nn.Module):

    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1.0 / K, 1.0 / K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()
        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0,
            index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()
        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.ReLU(True), nn.Conv2d(dim, dim, 3, 1,
            1), nn.BatchNorm2d(dim), nn.ReLU(True), nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim))

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):

    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1), nn
            .BatchNorm2d(dim), nn.ReLU(True), nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim), ResBlock(dim))
        self.codebook = VQEmbedding(K, dim)
        self.decoder = nn.Sequential(ResBlock(dim), ResBlock(dim), nn.ReLU(
            True), nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.BatchNorm2d(
            dim), nn.ReLU(True), nn.ConvTranspose2d(dim, input_dim, 4, 2, 1
            ), nn.Tanh())
        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


class GatedActivation(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedPixelCNN(nn.Module):

    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim
        self.embedding = nn.Embedding(input_dim, dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True
            self.layers.append(GatedMaskedConv2d(mask_type, dim, kernel,
                residual, n_classes))
        self.output_conv = nn.Sequential(nn.Conv2d(dim, 512, 1), nn.ReLU(
            True), nn.Conv2d(512, input_dim, 1))
        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2)
        x_v, x_h = x, x
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)
        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros((batch_size, *shape), dtype=torch.int64, device=
            param.device)
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, (i), (j)], -1)
                x.data[:, (i), (j)].copy_(probs.multinomial(1).squeeze().data)
        return x


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ritheshkumar95_pytorch_vqvae(_paritybench_base):
    pass
    def test_000(self):
        self._check(GatedActivation(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ResBlock(*[], **{'dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(VAE(*[], **{'input_dim': 4, 'dim': 4, 'z_dim': 4}), [torch.rand([4, 4, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(VQEmbedding(*[], **{'K': 4, 'D': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(VectorQuantizedVAE(*[], **{'input_dim': 4, 'dim': 4}), [torch.rand([4, 4, 4, 4])], {})

