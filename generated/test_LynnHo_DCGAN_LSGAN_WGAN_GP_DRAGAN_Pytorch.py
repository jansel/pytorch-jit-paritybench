import sys
_module = sys.modules[__name__]
del sys
data = _module
imlib = _module
basic = _module
dtype = _module
transform = _module
make_gif = _module
module = _module
pylib = _module
argument = _module
path = _module
processing = _module
serialization = _module
timer = _module
torchlib = _module
dataset = _module
layers = _module
layers = _module
utils = _module
torchprob = _module
gan = _module
gradient_penalty = _module
loss = _module
train = _module
models_64x64 = _module
train_celeba_dcgan = _module
train_celeba_dragan = _module
train_celeba_lsgan = _module
train_celeba_wgan_gp = _module

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


import functools


from torch import nn


import torch


import torch.nn as nn


from torch.autograd import Variable


from torch.autograd import grad


def _get_norm_layer_2d(norm):
    if norm == 'none':
        return torchlib.Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError


class ConvGenerator(nn.Module):

    def __init__(self, input_dim=128, output_channels=3, dim=64,
        n_upsamplings=4, norm='batch_norm'):
        super().__init__()
        Norm = _get_norm_layer_2d(norm)

        def dconv_norm_relu(in_dim, out_dim, kernel_size=4, stride=2, padding=1
            ):
            return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim,
                kernel_size, stride=stride, padding=padding, bias=False or 
                Norm == torchlib.Identity), Norm(out_dim), nn.ReLU())
        layers = []
        d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
        layers.append(dconv_norm_relu(input_dim, d, kernel_size=4, stride=1,
            padding=0))
        for i in range(n_upsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
            layers.append(dconv_norm_relu(d_last, d, kernel_size=4, stride=
                2, padding=1))
        layers.append(nn.ConvTranspose2d(d, output_channels, kernel_size=4,
            stride=2, padding=1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        x = self.net(z)
        return x


class ConvDiscriminator(nn.Module):

    def __init__(self, input_channels=3, dim=64, n_downsamplings=4, norm=
        'batch_norm'):
        super().__init__()
        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1
            ):
            return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size,
                stride=stride, padding=padding, bias=False or Norm ==
                torchlib.Identity), Norm(out_dim), nn.LeakyReLU(0.2))
        layers = []
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2,
            padding=1))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=
                2, padding=1))
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y


class Identity(torch.nn.Module):

    def __init__(self, *args, **keyword_args):
        super().__init__()

    def forward(self, x):
        return x


class Reshape(torch.nn.Module):

    def __init__(self, *new_shape):
        super().__init__()
        self._new_shape = new_shape

    def forward(self, x):
        new_shape = (x.size(i) if self._new_shape[i] == 0 else self.
            _new_shape[i] for i in range(len(self._new_shape)))
        return x.view(*new_shape)


class DepthToSpace(torch.nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // self.bs ** 2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(N, C // self.bs ** 2, H * self.bs, W * self.bs)
        return x


class SpaceToDepth(torch.nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * self.bs ** 2, H // self.bs, W // self.bs)
        return x


class ColorTransform(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X, Y, eps=1e-05):
        N, C, H, W = X.size()
        X = X.view(N, C, -1)
        Y = Y.view(N, C, -1)
        O = torch.ones(N, 1, H * W, dtype=X.dtype, device=X.device)
        X_ = torch.cat((X, O), dim=1)
        X__T = X_.permute(0, 2, 1)
        I = torch.eye(C + 1, dtype=X.dtype, device=X.device).view(-1, C + 1,
            C + 1).repeat([N, 1, 1])
        A = Y.matmul(X__T).matmul((X_.matmul(X__T) + eps * I).inverse())
        return A.matmul(X_).view(N, C, H, W)


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-05, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class Generator(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(Generator, self).__init__()

        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                padding=2, output_padding=1, bias=False), nn.BatchNorm2d(
                out_dim), nn.ReLU())
        self.l1 = nn.Sequential(nn.Linear(in_dim, dim * 8 * 4 * 4, bias=
            False), nn.BatchNorm1d(dim * 8 * 4 * 4), nn.ReLU())
        self.l2_5 = nn.Sequential(dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2), dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Tanh())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(Discriminator, self).__init__()

        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(nn.Conv2d(in_dim, out_dim, 5, 2, 2), nn.
                BatchNorm2d(out_dim), nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.
            LeakyReLU(0.2), conv_bn_lrelu(dim, dim * 2), conv_bn_lrelu(dim *
            2, dim * 4), conv_bn_lrelu(dim * 4, dim * 8), nn.Conv2d(dim * 8,
            1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


class DiscriminatorWGANGP(nn.Module):

    def __init__(self, in_dim, dim=64):
        super(DiscriminatorWGANGP, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(nn.Conv2d(in_dim, out_dim, 5, 2, 2), nn.
                InstanceNorm2d(out_dim, affine=True), nn.LeakyReLU(0.2))
        self.ls = nn.Sequential(nn.Conv2d(in_dim, dim, 5, 2, 2), nn.
            LeakyReLU(0.2), conv_ln_lrelu(dim, dim * 2), conv_ln_lrelu(dim *
            2, dim * 4), conv_ln_lrelu(dim * 4, dim * 8), nn.Conv2d(dim * 8,
            1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_LynnHo_DCGAN_LSGAN_WGAN_GP_DRAGAN_Pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ColorTransform(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ConvDiscriminator(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_002(self):
        self._check(ConvGenerator(*[], **{}), [torch.rand([4, 128, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(DepthToSpace(*[], **{'block_size': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Discriminator(*[], **{'in_dim': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_005(self):
        self._check(DiscriminatorWGANGP(*[], **{'in_dim': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_006(self):
        self._check(Generator(*[], **{'in_dim': 4}), [torch.rand([4, 4])], {})

    def test_007(self):
        self._check(Identity(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(LayerNorm(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_009(self):
        self._check(SpaceToDepth(*[], **{'block_size': 1}), [torch.rand([4, 4, 4, 4])], {})

