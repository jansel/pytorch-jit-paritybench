import sys
_module = sys.modules[__name__]
del sys
conf = _module
eval_pretrained = _module
sngan_example = _module
ssgan_tutorial = _module
setup = _module
test_data_utils = _module
test_image_loader = _module
test_fid = _module
test_inception_utils = _module
test_inception_score = _module
test_kid = _module
test_compute_fid = _module
test_compute_is = _module
test_compute_kid = _module
test_compute_metrics = _module
test_layers = _module
test_losses = _module
test_resblocks = _module
test_spectral_norm = _module
test_basemodel = _module
test_cgan_pd_128 = _module
test_cgan_pd_32 = _module
test_dcgan_128 = _module
test_dcgan_32 = _module
test_dcgan_48 = _module
test_dcgan_64 = _module
test_dcgan_cifar = _module
test_cgan = _module
test_gan = _module
test_infomax_gan_128 = _module
test_infomax_gan_32 = _module
test_infomax_gan_48 = _module
test_infomax_gan_64 = _module
test_infomax_gan_base = _module
test_sagan_128 = _module
test_sagan_32 = _module
test_sngan_128 = _module
test_sngan_32 = _module
test_sngan_48 = _module
test_sngan_64 = _module
test_ssgan_128 = _module
test_ssgan_32 = _module
test_ssgan_48 = _module
test_ssgan_64 = _module
test_ssgan_base = _module
test_wgan_gp_128 = _module
test_wgan_gp_32 = _module
test_wgan_gp_48 = _module
test_wgan_gp_64 = _module
test_wgan_gp_resblocks = _module
test_logger = _module
test_metric_log = _module
test_scheduler = _module
test_trainer = _module
test_common = _module
torch_mimicry = _module
datasets = _module
data_utils = _module
image_loader = _module
imagenet = _module
imagenet = _module
metrics = _module
compute_fid = _module
compute_is = _module
compute_kid = _module
compute_metrics = _module
fid = _module
fid_utils = _module
inception_model = _module
inception_utils = _module
inception_score = _module
inception_score_utils = _module
kid = _module
kid_utils = _module
modules = _module
layers = _module
losses = _module
resblocks = _module
spectral_norm = _module
nets = _module
basemodel = _module
basemodel = _module
cgan_pd = _module
cgan_pd_128 = _module
cgan_pd_32 = _module
cgan_pd_base = _module
dcgan = _module
dcgan_128 = _module
dcgan_32 = _module
dcgan_48 = _module
dcgan_64 = _module
dcgan_base = _module
dcgan_cifar = _module
gan = _module
cgan = _module
gan = _module
infomax_gan = _module
infomax_gan_128 = _module
infomax_gan_32 = _module
infomax_gan_48 = _module
infomax_gan_64 = _module
infomax_gan_base = _module
sagan = _module
sagan_128 = _module
sagan_32 = _module
sagan_base = _module
sngan = _module
sngan_128 = _module
sngan_32 = _module
sngan_48 = _module
sngan_64 = _module
sngan_base = _module
ssgan = _module
ssgan_128 = _module
ssgan_32 = _module
ssgan_48 = _module
ssgan_64 = _module
ssgan_base = _module
wgan_gp = _module
wgan_gp_128 = _module
wgan_gp_32 = _module
wgan_gp_48 = _module
wgan_gp_64 = _module
wgan_gp_base = _module
wgan_gp_resblocks = _module
training = _module
logger = _module
metric_log = _module
scheduler = _module
trainer = _module
utils = _module
common = _module

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


import torch.optim as optim


import torch.nn as nn


import torch.nn.functional as F


import numpy as np


import tensorflow as tf


from itertools import product


import math


from torch.utils.data import Dataset


import torchvision


from torchvision import transforms


import random


import torchvision.transforms as transforms


from torchvision.datasets import ImageFolder


from torchvision.datasets.utils import check_integrity


from torchvision.datasets.utils import download_and_extract_archive


from torchvision.datasets.utils import extract_archive


from torchvision.datasets.utils import verify_str_arg


import time


from abc import ABC


from abc import abstractmethod


from torch import autograd


import torch.functional as F


from torch.utils.tensorboard import SummaryWriter


from torchvision import utils as vutils


import re


class SpectralNorm(object):
    """
    Spectral Normalization for GANs (Miyato 2018).

    Inheritable class for performing spectral normalization of weights,
    as approximated using power iteration.

    Details: See Algorithm 1 of Appendix A (Miyato 2018).

    Attributes:
        n_dim (int): Number of dimensions.
        num_iters (int): Number of iterations for power iter.
        eps (float): Epsilon for zero division tolerance when normalizing.
    """

    def __init__(self, n_dim, num_iters=1, eps=1e-12):
        self.num_iters = num_iters
        self.eps = eps
        self.register_buffer('sn_u', torch.randn(1, n_dim))
        self.register_buffer('sn_sigma', torch.ones(1))

    @property
    def u(self):
        return getattr(self, 'sn_u')

    @property
    def sigma(self):
        return getattr(self, 'sn_sigma')

    def _power_iteration(self, W, u, num_iters, eps=1e-12):
        with torch.no_grad():
            for _ in range(num_iters):
                v = F.normalize(torch.matmul(u, W), eps=eps)
                u = F.normalize(torch.matmul(v, W.t()), eps=eps)
        sigma = torch.mm(u, torch.mm(W, v.t()))
        return sigma, u, v

    def sn_weights(self):
        """
        Spectrally normalize current weights of the layer.
        """
        W = self.weight.view(self.weight.shape[0], -1)
        sigma, u, v = self._power_iteration(W=W, u=self.u, num_iters=self.num_iters, eps=self.eps)
        if self.training:
            with torch.no_grad():
                self.sigma[:] = sigma
                self.u[:] = u
        return self.weight / sigma


class SNConv2d(nn.Conv2d, SpectralNorm):
    """
    Spectrally normalized layer for Conv2d.

    Attributes:
        in_channels (int): Input channel dimension.
        out_channels (int): Output channel dimensions.
    """

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        nn.Conv2d.__init__(self, in_channels, out_channels, *args, **kwargs)
        SpectralNorm.__init__(self, n_dim=out_channels, num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.conv2d(input=x, weight=self.sn_weights(), bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)


class SelfAttention(nn.Module):
    """
    Self-attention layer based on version used in BigGAN code:
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
    """

    def __init__(self, num_feat, spectral_norm=True):
        super().__init__()
        self.num_feat = num_feat
        self.spectral_norm = spectral_norm
        if self.spectral_norm:
            self.theta = SNConv2d(self.num_feat, self.num_feat >> 3, 1, 1, padding=0, bias=False)
            self.phi = SNConv2d(self.num_feat, self.num_feat >> 3, 1, 1, padding=0, bias=False)
            self.g = SNConv2d(self.num_feat, self.num_feat >> 1, 1, 1, padding=0, bias=False)
            self.o = SNConv2d(self.num_feat >> 1, self.num_feat, 1, 1, padding=0, bias=False)
        else:
            self.theta = nn.Conv2d(self.num_feat, self.num_feat >> 3, 1, 1, padding=0, bias=False)
            self.phi = nn.Conv2d(self.num_feat, self.num_feat >> 3, 1, 1, padding=0, bias=False)
            self.g = nn.Conv2d(self.num_feat, self.num_feat >> 1, 1, 1, padding=0, bias=False)
            self.o = nn.Conv2d(self.num_feat >> 1, self.num_feat, 1, 1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        """
        Feedforward function. Implementation differs from actual SAGAN paper,
        see note from BigGAN:
        https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py#L142

        See official TF Implementation:
        https://github.com/brain-research/self-attention-gan/blob/master/non_local.py

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Feature map weighed with attention map.
        """
        N, C, H, W = x.shape
        location_num = H * W
        downsampled_num = location_num >> 2
        theta = self.theta(x)
        theta = theta.view(N, C >> 3, location_num)
        phi = self.phi(x)
        phi = F.max_pool2d(phi, [2, 2], stride=2)
        phi = phi.view(N, C >> 3, downsampled_num)
        attn = torch.bmm(theta.transpose(1, 2), phi)
        attn = F.softmax(attn, -1)
        g = self.g(x)
        g = F.max_pool2d(g, [2, 2], stride=2)
        g = g.view(N, C >> 1, downsampled_num)
        attn_g = torch.bmm(g, attn.transpose(1, 2))
        attn_g = attn_g.view(N, C >> 1, H, W)
        attn_g = self.o(attn_g)
        output = x + self.gamma * attn_g
        return output


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Norm as implemented in
    https://github.com/pytorch/pytorch/issues/8985

    Attributes:
        num_features (int): Size of feature map for batch norm.
        num_classes (int): Determines size of embedding layer to condition BN.
    """

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        """
        Feedforwards for conditional batch norm.

        Args:
            x (Tensor): Input feature map.
            y (Tensor): Input class labels for embedding.

        Returns:
            Tensor: Output feature map.
        """
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


class SNLinear(nn.Linear, SpectralNorm):
    """
    Spectrally normalized layer for Linear.

    Attributes:
        in_features (int): Input feature dimensions.
        out_features (int): Output feature dimensions.
    """

    def __init__(self, in_features, out_features, *args, **kwargs):
        nn.Linear.__init__(self, in_features, out_features, *args, **kwargs)
        SpectralNorm.__init__(self, n_dim=out_features, num_iters=kwargs.get('num_iters', 1))

    def forward(self, x):
        return F.linear(input=x, weight=self.sn_weights(), bias=self.bias)


class SNEmbedding(nn.Embedding, SpectralNorm):
    """
    Spectrally normalized layer for Embedding.

    Attributes:
        num_embeddings (int): Number of embeddings.
        embedding_dim (int): Dimensions of each embedding vector
    """

    def __init__(self, num_embeddings, embedding_dim, *args, **kwargs):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, *args, **kwargs)
        SpectralNorm.__init__(self, n_dim=num_embeddings)

    def forward(self, x):
        return F.embedding(input=x, weight=self.sn_weights())


class BaseModel(nn.Module, ABC):
    """
    BaseModel with basic functionalities for checkpointing and restoration.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @property
    def device(self):
        return next(self.parameters()).device

    def restore_checkpoint(self, ckpt_file, optimizer=None):
        """
        Restores checkpoint from a pth file and restores optimizer state.

        Args:
            ckpt_file (str): A PyTorch pth file containing model weights.
            optimizer (Optimizer): A vanilla optimizer to have its state restored from.

        Returns:
            int: Global step variable where the model was last checkpointed.
        """
        if not ckpt_file:
            raise ValueError('No checkpoint file to be restored.')
        try:
            ckpt_dict = torch.load(ckpt_file)
        except RuntimeError:
            ckpt_dict = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        self.load_state_dict(ckpt_dict['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
        return ckpt_dict['global_step']

    def save_checkpoint(self, directory, global_step, optimizer=None, name=None):
        """
        Saves checkpoint at a certain global step during training. Optimizer state
        is also saved together.

        Args:
            directory (str): Path to save checkpoint to.
            global_step (int): The global step variable during training.
            optimizer (Optimizer): Optimizer state to be saved concurrently.
            name (str): The name to save the checkpoint file as.

        Returns:
            None
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        ckpt_dict = {'model_state_dict': self.state_dict(), 'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None, 'global_step': global_step}
        if name is None:
            name = '{}_{}_steps.pth'.format(os.path.basename(directory), global_step)
        torch.save(ckpt_dict, os.path.join(directory, name))

    def count_params(self):
        """
        Computes the number of parameters in this model.

        Args: None

        Returns:
            int: Total number of weight parameters for this model.
            int: Total number of trainable parameters for this model.

        """
        num_total_params = sum(p.numel() for p in self.parameters())
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num_total_params, num_trainable_params


class BaseGenerator(basemodel.BaseModel):
    """
    Base class for a generic unconditional generator model.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.
    """

    def __init__(self, nz, ngf, bottom_width, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.nz = nz
        self.ngf = ngf
        self.bottom_width = bottom_width
        self.loss_type = loss_type

    def generate_images(self, num_images, device=None):
        """
        Generates num_images randomly.

        Args:
            num_images (int): Number of images to generate
            device (torch.device): Device to send images to.

        Returns:
            Tensor: A batch of generated images.
        """
        if device is None:
            device = self.device
        noise = torch.randn((num_images, self.nz), device=device)
        fake_images = self.forward(noise)
        return fake_images

    def compute_gan_loss(self, output):
        """
        Computes GAN loss for generator.

        Args:
            output (Tensor): A batch of output logits from the discriminator of shape (N, 1).

        Returns:
            Tensor: A batch of GAN losses for the generator.
        """
        if self.loss_type == 'gan':
            errG = losses.minimax_loss_gen(output)
        elif self.loss_type == 'ns':
            errG = losses.ns_loss_gen(output)
        elif self.loss_type == 'hinge':
            errG = losses.hinge_loss_gen(output)
        elif self.loss_type == 'wasserstein':
            errG = losses.wasserstein_loss_gen(output)
        else:
            raise ValueError('Invalid loss_type {} selected.'.format(self.loss_type))
        return errG

    def train_step(self, real_batch, netD, optG, log_data, device=None, global_step=None, **kwargs):
        """
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (dict): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()
        batch_size = real_batch[0].shape[0]
        fake_images = self.generate_images(num_images=batch_size, device=device)
        output = netD(fake_images)
        errG = self.compute_gan_loss(output=output)
        errG.backward()
        optG.step()
        log_data.add_metric('errG', errG, group='loss')
        return log_data


class InfoMaxGANBaseGenerator(gan.BaseGenerator):
    """
    ResNet backbone generator for InfoMax-GAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
        infomax_loss_scale (float): The alpha parameter used for scaling the generator infomax loss.
    """

    def __init__(self, nz, ngf, bottom_width, loss_type='hinge', infomax_loss_scale=0.2, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, loss_type=loss_type, **kwargs)
        self.infomax_loss_scale = infomax_loss_scale

    def train_step(self, real_batch, netD, optG, log_data, device=None, global_step=None, **kwargs):
        """
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()
        real_images, _ = real_batch
        batch_size = real_images.shape[0]
        fake_images = self.generate_images(num_images=batch_size, device=device)
        output_fake, local_feat_fake, global_feat_fake = netD(fake_images)
        local_feat_fake, global_feat_fake = netD.project_features(local_feat=local_feat_fake, global_feat=global_feat_fake)
        errG = self.compute_gan_loss(output_fake)
        errG_IM = netD.compute_infomax_loss(local_feat=local_feat_fake, global_feat=global_feat_fake, scale=self.infomax_loss_scale)
        errG_total = errG + errG_IM
        errG_total.backward()
        optG.step()
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_IM', errG_IM, group='loss_IM')
        return log_data


class SSGANBaseGenerator(gan.BaseGenerator):
    """
    ResNet backbone generator for SSGAN.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
        ss_loss_scale (float): Self-supervised loss scale for generator.
    """

    def __init__(self, nz, ngf, bottom_width, loss_type='hinge', ss_loss_scale=0.2, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, loss_type=loss_type, **kwargs)
        self.ss_loss_scale = ss_loss_scale

    def train_step(self, real_batch, netD, optG, log_data, device=None, global_step=None, **kwargs):
        """
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()
        batch_size = real_batch[0].shape[0]
        fake_images = self.generate_images(num_images=batch_size, device=device)
        output, _ = netD(fake_images)
        errG = self.compute_gan_loss(output)
        errG_SS, _ = netD.compute_ss_loss(images=fake_images, scale=self.ss_loss_scale)
        errG_total = errG + errG_SS
        errG_total.backward()
        optG.step()
        log_data.add_metric('errG', errG, group='loss')
        log_data.add_metric('errG_SS', errG_SS, group='loss_SS')
        return log_data


class WGANGPBaseGenerator(gan.BaseGenerator):
    """
    ResNet backbone generator for WGAN-GP.

    Attributes:
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """

    def __init__(self, nz, ngf, bottom_width, loss_type='wasserstein', **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, loss_type=loss_type, **kwargs)

    def train_step(self, real_batch, netD, optG, log_data, device=None, global_step=None, **kwargs):
        """
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): An object to add custom metrics for visualisations.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()
        batch_size = real_batch[0].shape[0]
        fake_images = self.generate_images(num_images=batch_size, device=device)
        output = netD(fake_images)
        errG = self.compute_gan_loss(output)
        errG.backward()
        optG.step()
        log_data.add_metric('errG', errG, group='loss')
        return log_data


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (SNConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SNLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
]

class Test_kwotsin_mimicry(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

