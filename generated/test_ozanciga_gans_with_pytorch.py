import sys
_module = sys.modules[__name__]
del sys
cgan = _module
ct_gan = _module
models = _module
dcgan = _module
gan = _module
infogan = _module
lsgan = _module
datasets = _module
models = _module
pix2pix = _module
preprocess_cat_dataset = _module
ralsgan = _module
datasets = _module
models = _module
srgan = _module
models = _module
wgan_gp = _module
dataset_loader = _module
wgan = _module

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


from torch import nn


from torch import optim


from torch.autograd.variable import Variable


from torchvision import transforms


from torchvision import datasets


from torch.utils.data import DataLoader


import torch.nn.functional as F


from torchvision.utils import save_image


from torch.utils import data


import torchvision.transforms as transforms


import numpy


import numpy as np


from torchvision.models import vgg19


from torch.autograd import Variable


parser = argparse.ArgumentParser()


opt = parser.parse_args()


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def convblock(n_input, n_output, k_size=4, stride=2, padding=0, normalize=True):
            block = [nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.ReLU(inplace=True))
            return block
        self.project = nn.Sequential(nn.Linear(opt.latent_dim, 256 * 4 * 4), nn.BatchNorm1d(256 * 4 * 4), nn.ReLU(inplace=True))
        self.model = nn.Sequential(*convblock(opt.latent_dim, 256, 4, 1, 0), *convblock(256, 128, 4, 2, 1), *convblock(128, 64, 4, 2, 1), nn.ConvTranspose2d(64, opt.channels, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), *img_dims)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def convblock(n_input, n_output, kernel_size=4, stride=2, padding=1, normalize=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size, stride, padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        self.model = nn.Sequential(*convblock(opt.channels, 64, 4, 2, 1, normalize=False), *convblock(64, 128, 4, 2, 1), *convblock(128, 256, 4, 2, 1), nn.Conv2d(256, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, img):
        prob = self.model(img)
        return prob


class MeanPoolConv(nn.Module):

    def __init__(self, n_input, n_output, k_size):
        super(MeanPoolConv, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size - 1) // 2, bias=True)
        self.model = nn.Sequential(conv1)

    def forward(self, x):
        out = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] + x[:, :, 1::2, 1::2]) / 4.0
        out = self.model(out)
        return out


class ConvMeanPool(nn.Module):

    def __init__(self, n_input, n_output, k_size):
        super(ConvMeanPool, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size - 1) // 2, bias=True)
        self.model = nn.Sequential(conv1)

    def forward(self, x):
        out = self.model(x)
        out = (out[:, :, ::2, ::2] + out[:, :, 1::2, ::2] + out[:, :, ::2, 1::2] + out[:, :, 1::2, 1::2]) / 4.0
        return out


class UpsampleConv(nn.Module):

    def __init__(self, n_input, n_output, k_size):
        super(UpsampleConv, self).__init__()
        self.model = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(k_size - 1) // 2, bias=True))

    def forward(self, x):
        x = x.repeat((1, 4, 1, 1))
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, n_input, n_output, k_size, resample='up', bn=True, spatial_dim=None):
        super(ResidualBlock, self).__init__()
        self.resample = resample
        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(k_size - 1) // 2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size - 1) // 2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim]
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size - 1) // 2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(k_size - 1) // 2)
            self.conv_shortcut = None
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]
        self.model = nn.Sequential(nn.BatchNorm2d(n_input) if bn else nn.LayerNorm(self.ln_dims), nn.ReLU(inplace=True), self.conv1, nn.BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims), nn.ReLU(inplace=True), self.conv2)

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)


class DiscBlock1(nn.Module):

    def __init__(self, n_output):
        super(DiscBlock1, self).__init__()
        self.conv1 = nn.Conv2d(3, n_output, 3, padding=(3 - 1) // 2)
        self.conv2 = ConvMeanPool(n_output, n_output, 1)
        self.conv_shortcut = MeanPoolConv(3, n_output, 1)
        self.model = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.conv2)

    def forward(self, x):
        return self.conv_shortcut(x) + self.model(x)


class ShuffleBlock(nn.Module):

    def __init__(self, n_input, n_output, k_size=3, stride=1, padding=1):
        super(ShuffleBlock, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(n_input, n_output, k_size, stride, padding), nn.PixelShuffle(2), nn.PReLU())
        """
        Input: :math:`(N, C * upscale_factor^2, H, W)`
        Output: :math:`(N, C, H * upscale_factor, W * upscale_factor)`
        """

    def forward(self, x):
        return self.model(x)


class VGGFeatures(nn.Module):

    def __init__(self):
        super(VGGFeatures, self).__init__()
        model = vgg19(pretrained=True)
        children = list(model.features.children())
        max_pool_indices = [index for index, m in enumerate(children) if isinstance(m, nn.MaxPool2d)]
        target_features = children[:max_pool_indices[4]]
        """
          We use vgg-5,4 which is the layer output after 5th conv 
          and right before the 4th max pool.
        """
        self.features = nn.Sequential(*target_features)
        for p in self.features.parameters():
            p.requires_grad = False
        """
        # VGG means and stdevs on pretrained imagenet
        mean = -1 + Variable(torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        std = 2*Variable(torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # This is for cuda compatibility.
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        """

    def forward(self, input):
        output = self.features(input)
        return output


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (ConvMeanPool,
     lambda: ([], {'n_input': 4, 'n_output': 4, 'k_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (DiscBlock1,
     lambda: ([], {'n_output': 4}),
     lambda: ([torch.rand([4, 3, 4, 4])], {}),
     True),
    (MeanPoolConv,
     lambda: ([], {'n_input': 4, 'n_output': 4, 'k_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ShuffleBlock,
     lambda: ([], {'n_input': 4, 'n_output': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UpsampleConv,
     lambda: ([], {'n_input': 4, 'n_output': 4, 'k_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (VGGFeatures,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
]

class Test_ozanciga_gans_with_pytorch(_paritybench_base):
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

