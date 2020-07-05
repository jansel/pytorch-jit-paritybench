import sys
_module = sys.modules[__name__]
del sys
conf = _module
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
imagenet_utils = _module
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


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


import numpy as np


from torch.utils.data import Dataset


import math


from abc import ABC


from abc import abstractmethod


from torch import autograd


import torch.functional as F


def SNConv2d(*args, **kwargs):
    """
    Wrapper for applying spectral norm on conv2d layer.
    """
    if kwargs.get('default', True):
        return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
    else:
        return spectral_norm.SNConv2d(*args, **kwargs)


class SelfAttention(nn.Module):
    """
    Self-attention layer based on version used in BigGAN code:
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
    """

    def __init__(self, num_feat, spectral_norm=True):
        super().__init__()
        self.num_feat = num_feat
        if spectral_norm:
            self.f = SNConv2d(self.num_feat, self.num_feat >> 3, 1, 1, padding=0, bias=False)
            self.g = SNConv2d(self.num_feat, self.num_feat >> 3, 1, 1, padding=0, bias=False)
            self.h = SNConv2d(self.num_feat, self.num_feat >> 1, 1, 1, padding=0, bias=False)
            self.o = SNConv2d(self.num_feat >> 1, self.num_feat, 1, 1, padding=0, bias=False)
        else:
            self.f = nn.Conv2d(self.num_feat, self.num_feat >> 3, 1, 1, padding=0, bias=False)
            self.g = nn.Conv2d(self.num_feat, self.num_feat >> 3, 1, 1, padding=0, bias=False)
            self.h = nn.Conv2d(self.num_feat, self.num_feat >> 1, 1, 1, padding=0, bias=False)
            self.o = nn.Conv2d(self.num_feat >> 1, self.num_feat, 1, 1, padding=0, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        """
        Feedforward function. Implementation differs from actual SAGAN paper,
        see note from BigGAN:
        https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py#L142
        """
        f = self.f(x)
        g = F.max_pool2d(self.g(x), [2, 2])
        h = F.max_pool2d(self.h(x), [2, 2])
        f = f.view(-1, self.num_feat >> 3, x.shape[2] * x.shape[3])
        g = g.view(-1, self.num_feat >> 3, x.shape[2] * x.shape[3] >> 2)
        h = h.view(-1, self.num_feat >> 1, x.shape[2] * x.shape[3] >> 2)
        beta = F.softmax(torch.bmm(f.transpose(1, 2), g), -1)
        o = self.o(torch.bmm(h, beta.transpose(1, 2)).view(-1, self.num_feat >> 1, x.shape[2], x.shape[3]))
        return self.gamma * o + x


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


class GBlock(nn.Module):
    """
    Residual block for generator.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        num_classes (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False, num_classes=0, spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample
        self.num_classes = num_classes
        self.spectral_norm = spectral_norm
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, padding=1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, padding=1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, padding=1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, padding=1)
        if self.num_classes == 0:
            self.b1 = nn.BatchNorm2d(self.in_channels)
            self.b2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.b1 = ConditionalBatchNorm2d(self.in_channels, self.num_classes)
            self.b2 = ConditionalBatchNorm2d(self.hidden_channels, self.num_classes)
        self.activation = nn.ReLU(True)
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, padding=0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _upsample_conv(self, x, conv):
        """
        Helper function for performing convolution after upsampling.
        """
        return conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def _residual_conditional(self, x, y):
        """
        Helper function for feedforwarding through main layers, including conditional BN.
        """
        h = x
        h = self.b1(h, y)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y)
        h = self.activation(h)
        h = self.c2(h)
        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self._upsample_conv(x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None):
        """
        Residual block feedforward function.
        """
        if y is None:
            return self._residual(x) + self._shortcut(x)
        else:
            return self._residual_conditional(x, y) + self._shortcut(x)


class DBlock(nn.Module):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """

    def __init__(self, in_channels, out_channels, hidden_channels=None, downsample=False, spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = in_channels != out_channels or downsample
        self.spectral_norm = spectral_norm
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, 1)
        self.activation = nn.ReLU(True)
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool2d(x, 2) if self.downsample else x
        else:
            return x

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)


class DBlockOptimized(nn.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """

    def __init__(self, in_channels, out_channels, spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm
        if self.spectral_norm:
            self.c1 = SNConv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.c2 = SNConv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.c_sc = SNConv2d(self.in_channels, self.out_channels, 1, 1, 0)
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1)
            self.c2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
            self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0)
        self.activation = nn.ReLU(True)
        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)
        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)


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


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConditionalBatchNorm2d,
     lambda: ([], {'num_features': 4, 'num_classes': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {}),
     True),
    (DBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (DBlockOptimized,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (GBlock,
     lambda: ([], {'in_channels': 4, 'out_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SNConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SNEmbedding,
     lambda: ([], {'num_embeddings': 4, 'embedding_dim': 4}),
     lambda: ([torch.zeros([4], dtype=torch.int64)], {}),
     False),
    (SNLinear,
     lambda: ([], {'in_features': 4, 'out_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (SelfAttention,
     lambda: ([], {'num_feat': 64}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     False),
]

class Test_kwotsin_mimicry(_paritybench_base):
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

