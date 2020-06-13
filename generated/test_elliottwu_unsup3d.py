import sys
_module = sys.modules[__name__]
del sys
celeba_crop = _module
demo = _module
utils = _module
run = _module
unsup3d = _module
dataloaders = _module
meters = _module
model = _module
networks = _module
renderer = _module
trainer = _module

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


import numpy as np


import torch


import torch.nn as nn


import math


class Encoder(nn.Module):

    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1,
            bias=False), nn.ReLU(inplace=True), nn.Conv2d(nf, nf * 2,
            kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(
            inplace=True), nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=
            2, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(nf *
            4, nf * 8, kernel_size=4, stride=2, padding=1, bias=False), nn.
            ReLU(inplace=True), nn.Conv2d(nf * 8, nf * 8, kernel_size=4,
            stride=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.
            Conv2d(nf * 8, cout, kernel_size=1, stride=1, padding=0, bias=
            False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class EDDeconv(nn.Module):

    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDDeconv, self).__init__()
        network = [nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1,
            bias=False), nn.GroupNorm(16, nf), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1,
            bias=False), nn.GroupNorm(16 * 2, nf * 2), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=
            2, padding=1, bias=False), nn.GroupNorm(16 * 4, nf * 4), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(nf * 4, nf * 8,
            kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(
            0.2, inplace=True), nn.Conv2d(nf * 8, zdim, kernel_size=4,
            stride=1, padding=0, bias=False), nn.ReLU(inplace=True)]
        network += [nn.ConvTranspose2d(zdim, nf * 8, kernel_size=4, stride=
            1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(nf *
            8, nf * 8, kernel_size=3, stride=1, padding=1, bias=False), nn.
            ReLU(inplace=True), nn.ConvTranspose2d(nf * 8, nf * 4,
            kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(
            16 * 4, nf * 4), nn.ReLU(inplace=True), nn.Conv2d(nf * 4, nf * 
            4, kernel_size=3, stride=1, padding=1, bias=False), nn.
            GroupNorm(16 * 4, nf * 4), nn.ReLU(inplace=True), nn.
            ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2,
            padding=1, bias=False), nn.GroupNorm(16 * 2, nf * 2), nn.ReLU(
            inplace=True), nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=
            1, padding=1, bias=False), nn.GroupNorm(16 * 2, nf * 2), nn.
            ReLU(inplace=True), nn.ConvTranspose2d(nf * 2, nf, kernel_size=
            4, stride=2, padding=1, bias=False), nn.GroupNorm(16, nf), nn.
            ReLU(inplace=True), nn.Conv2d(nf, nf, kernel_size=3, stride=1,
            padding=1, bias=False), nn.GroupNorm(16, nf), nn.ReLU(inplace=
            True), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(
            nf, nf, kernel_size=3, stride=1, padding=1, bias=False), nn.
            GroupNorm(16, nf), nn.ReLU(inplace=True), nn.Conv2d(nf, nf,
            kernel_size=5, stride=1, padding=2, bias=False), nn.GroupNorm(
            16, nf), nn.ReLU(inplace=True), nn.Conv2d(nf, cout, kernel_size
            =5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class Encoder(nn.Module):

    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1,
            bias=False), nn.ReLU(inplace=True), nn.Conv2d(nf, nf * 2,
            kernel_size=4, stride=2, padding=1, bias=False), nn.ReLU(
            inplace=True), nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=
            2, padding=1, bias=False), nn.ReLU(inplace=True), nn.Conv2d(nf *
            4, nf * 8, kernel_size=4, stride=2, padding=1, bias=False), nn.
            ReLU(inplace=True), nn.Conv2d(nf * 8, nf * 8, kernel_size=4,
            stride=1, padding=0, bias=False), nn.ReLU(inplace=True), nn.
            Conv2d(nf * 8, cout, kernel_size=1, stride=1, padding=0, bias=
            False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


class EDDeconv(nn.Module):

    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDDeconv, self).__init__()
        network = [nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1,
            bias=False), nn.GroupNorm(16, nf), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1,
            bias=False), nn.GroupNorm(16 * 2, nf * 2), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=
            2, padding=1, bias=False), nn.GroupNorm(16 * 4, nf * 4), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(nf * 4, nf * 8,
            kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(
            0.2, inplace=True), nn.Conv2d(nf * 8, zdim, kernel_size=4,
            stride=1, padding=0, bias=False), nn.ReLU(inplace=True)]
        network += [nn.ConvTranspose2d(zdim, nf * 8, kernel_size=4, stride=
            1, padding=0, bias=False), nn.ReLU(inplace=True), nn.Conv2d(nf *
            8, nf * 8, kernel_size=3, stride=1, padding=1, bias=False), nn.
            ReLU(inplace=True), nn.ConvTranspose2d(nf * 8, nf * 4,
            kernel_size=4, stride=2, padding=1, bias=False), nn.GroupNorm(
            16 * 4, nf * 4), nn.ReLU(inplace=True), nn.Conv2d(nf * 4, nf * 
            4, kernel_size=3, stride=1, padding=1, bias=False), nn.
            GroupNorm(16 * 4, nf * 4), nn.ReLU(inplace=True), nn.
            ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2,
            padding=1, bias=False), nn.GroupNorm(16 * 2, nf * 2), nn.ReLU(
            inplace=True), nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=
            1, padding=1, bias=False), nn.GroupNorm(16 * 2, nf * 2), nn.
            ReLU(inplace=True), nn.ConvTranspose2d(nf * 2, nf, kernel_size=
            4, stride=2, padding=1, bias=False), nn.GroupNorm(16, nf), nn.
            ReLU(inplace=True), nn.Conv2d(nf, nf, kernel_size=3, stride=1,
            padding=1, bias=False), nn.GroupNorm(16, nf), nn.ReLU(inplace=
            True), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(
            nf, nf, kernel_size=3, stride=1, padding=1, bias=False), nn.
            GroupNorm(16, nf), nn.ReLU(inplace=True), nn.Conv2d(nf, nf,
            kernel_size=5, stride=1, padding=2, bias=False), nn.GroupNorm(
            16, nf), nn.ReLU(inplace=True), nn.Conv2d(nf, cout, kernel_size
            =5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class ConfNet(nn.Module):

    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        network = [nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1,
            bias=False), nn.GroupNorm(16, nf), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1,
            bias=False), nn.GroupNorm(16 * 2, nf * 2), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=
            2, padding=1, bias=False), nn.GroupNorm(16 * 4, nf * 4), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(nf * 4, nf * 8,
            kernel_size=4, stride=2, padding=1, bias=False), nn.LeakyReLU(
            0.2, inplace=True), nn.Conv2d(nf * 8, zdim, kernel_size=4,
            stride=1, padding=0, bias=False), nn.ReLU(inplace=True)]
        network += [nn.ConvTranspose2d(zdim, nf * 8, kernel_size=4, padding
            =0, bias=False), nn.ReLU(inplace=True), nn.ConvTranspose2d(nf *
            8, nf * 4, kernel_size=4, stride=2, padding=1, bias=False), nn.
            GroupNorm(16 * 4, nf * 4), nn.ReLU(inplace=True), nn.
            ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2,
            padding=1, bias=False), nn.GroupNorm(16 * 2, nf * 2), nn.ReLU(
            inplace=True)]
        self.network = nn.Sequential(*network)
        out_net1 = [nn.ConvTranspose2d(nf * 2, nf, kernel_size=4, stride=2,
            padding=1, bias=False), nn.GroupNorm(16, nf), nn.ReLU(inplace=
            True), nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2,
            padding=1, bias=False), nn.GroupNorm(16, nf), nn.ReLU(inplace=
            True), nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2,
            bias=False), nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)
        out_net2 = [nn.Conv2d(nf * 2, 2, kernel_size=3, stride=1, padding=1,
            bias=False), nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        out = self.network(input)
        return self.out_net1(out), self.out_net2(out)


EPS = 1e-07


class PerceptualLoss(nn.Module):

    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True
            ).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x / 2 + 0.5
        out = (out - self.mean_rgb.view(1, 3, 1, 1)) / self.std_rgb.view(1,
            3, 1, 1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1, im2], 0)
        im = self.normalize(im)
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]
        losses = []
        for f1, f2 in feats[2:3]:
            loss = (f1 - f2) ** 2
            if conf_sigma is not None:
                loss = loss / (2 * conf_sigma ** 2 + EPS) + (conf_sigma + EPS
                    ).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm // h, wm // w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh, sw),
                    stride=(sh, sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_elliottwu_unsup3d(_paritybench_base):
    pass
    def test_000(self):
        self._check(Encoder(*[], **{'cin': 4, 'cout': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_001(self):
        self._check(EDDeconv(*[], **{'cin': 4, 'cout': 4}), [torch.rand([4, 4, 64, 64])], {})

    def test_002(self):
        self._check(ConfNet(*[], **{'cin': 4, 'cout': 4}), [torch.rand([4, 4, 64, 64])], {})

