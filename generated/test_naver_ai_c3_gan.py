import sys
_module = sys.modules[__name__]
del sys
base_layer = _module
config = _module
datasets = _module
evals = _module
inception = _module
model = _module
train = _module
utils = _module

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


import torch.nn as nn


import torch.nn.functional as F


from torch.nn import Upsample


import torch.utils.data as data


import torchvision.transforms as transforms


import torchvision


import random


import numpy as np


from torch.nn import functional as F


import torchvision.utils as vutils


from torch.utils.tensorboard import SummaryWriter


import math


import torch.optim as optim


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class ResBlock(nn.Module):

    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel_num, channel_num * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(channel_num * 2), GLU(), nn.Conv2d(channel_num, channel_num, 3, 1, 1, bias=False), nn.BatchNorm2d(channel_num))

    def forward(self, x):
        return x + self.block(x)


def _inception_v3(*args, **kwargs):
    """Wraps `torchvision.models.inception_v3`
    Skips default weight inititialization if supported by torchvision version.
    See https://github.com/mseitzer/pytorch-fid/issues/28.
    """
    try:
        version = tuple(map(int, torchvision.__version__.split('.')[:2]))
    except ValueError:
        version = 0,
    if version >= (0, 6):
        kwargs['init_weights'] = False
    return torchvision.models.inception_v3(*args, **kwargs)


class FIDInceptionA(torchvision.models.inception.InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(torchvision.models.inception.InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_1(torchvision.models.inception.InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE_2(torchvision.models.inception.InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3), self.branch3x3_2b(branch3x3)]
        branch3x3 = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


def fid_inception_v3():
    """Build pretrained Inception model for FID computation
    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.
    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008, aux_logits=False, pretrained=False)
    inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
    inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
    inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
    inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
    inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
    inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
    inception.Mixed_7b = FIDInceptionE_1(1280)
    inception.Mixed_7c = FIDInceptionE_2(2048)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)
    return inception


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""
    DEFAULT_BLOCK_INDEX = 3
    BLOCK_INDEX_BY_DIM = {(64): 0, (192): 1, (768): 2, (2048): 3}

    def __init__(self, output_blocks=(DEFAULT_BLOCK_INDEX,), resize_input=True, normalize_input=True, requires_grad=False, use_fid_inception=True):
        """Build pretrained InceptionV3
        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)
        assert self.last_needed_block <= 3, 'Last possible output block index is 3'
        self.blocks = nn.ModuleList()
        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(pretrained=True)
        block0 = [inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
        self.blocks.append(nn.Sequential(*block0))
        if self.last_needed_block >= 1:
            block1 = [inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2)]
            self.blocks.append(nn.Sequential(*block1))
        if self.last_needed_block >= 2:
            block2 = [inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e]
            self.blocks.append(nn.Sequential(*block2))
        if self.last_needed_block >= 3:
            block3 = [inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1))]
            self.blocks.append(nn.Sequential(*block3))
        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp
        if self.resize_input:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = 2 * x - 1
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)
            if idx == self.last_needed_block:
                break
        return outp


def multi_ResBlock(num_residual, ngf):
    layers = []
    for _ in range(num_residual):
        layers.append(ResBlock(ngf))
    return nn.Sequential(*layers)


def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(out_planes * 2), GLU())
    return block


class bg_generator(nn.Module):

    def __init__(self, ngf=512):
        super(bg_generator, self).__init__()
        self.z_dim = cfg.GAN.Z_DIM
        self.ngf = ngf
        self.fc = nn.Sequential(nn.Linear(self.z_dim, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())
        self.layers = nn.Sequential(upBlock(ngf, ngf // 2), upBlock(ngf // 2, ngf // 4), upBlock(ngf // 4, ngf // 8), upBlock(ngf // 8, ngf // 32), upBlock(ngf // 32, ngf // 32), multi_ResBlock(3, ngf // 32), nn.Conv2d(ngf // 32, 3, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, z):
        out = self.fc(z).view(-1, self.ngf, 4, 4)
        out = self.layers(out)
        return out


def sameBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Conv2d(in_planes, out_planes * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(out_planes * 2), GLU())
    return block


class fg_generator(nn.Module):

    def __init__(self, c_dim, ngf=512):
        super(fg_generator, self).__init__()
        self.z_dim = cfg.GAN.Z_DIM
        self.cz_dim = cfg.GAN.CZ_DIM
        self.c_dim = c_dim
        self.ngf = ngf
        self.fc = nn.Sequential(nn.Linear(self.z_dim, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())
        self.emb_c = nn.Sequential(nn.Linear(self.c_dim, self.cz_dim * 2 * 2), nn.BatchNorm1d(self.cz_dim * 2 * 2), GLU())
        self.base = nn.Sequential(upBlock(ngf + self.cz_dim, ngf // 2), upBlock(ngf // 2, ngf // 4), upBlock(ngf // 4, ngf // 8), upBlock(ngf // 8, ngf // 32), upBlock(ngf // 32, ngf // 32), multi_ResBlock(3, ngf // 32))
        self.to_mask = nn.Sequential(sameBlock(ngf // 32, ngf // 32), nn.Conv2d(ngf // 32, 1, 3, 1, 1, bias=False))
        self.to_img = nn.Sequential(sameBlock(self.c_dim + ngf // 32, ngf // 32), multi_ResBlock(2, ngf // 32), sameBlock(ngf // 32, ngf // 32), nn.Conv2d(ngf // 32, 3, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, z, c, cz):
        c_ = self.emb_c(c)
        c_mu = c_[:, :self.cz_dim]
        c_std = c_[:, self.cz_dim:]
        cz_ = c_mu + c_std.exp() * cz
        cz_ = cz_.view(-1, self.cz_dim, 1, 1).repeat(1, 1, 4, 4)
        out = self.fc(z).view(-1, self.ngf, 4, 4)
        out = self.base(torch.cat((out, cz_), 1))
        out_mask = torch.sigmoid(self.to_mask(out))
        h, w = out.size(2), out.size(3)
        c = c.view(-1, self.c_dim, 1, 1).repeat(1, 1, h, w)
        out = torch.cat((out, c), 1)
        out_img = self.to_img(out)
        return out_mask, out_img


class Generator(nn.Module):

    def __init__(self, c_dim):
        super(Generator, self).__init__()
        self.bg_gen = bg_generator()
        self.fg_gen = fg_generator(c_dim)

    def forward(self, z, cz, c, grid=None):
        bg_img = self.bg_gen(z)
        fg_mask, fg_img = self.fg_gen(z, c, cz)
        if grid != None:
            fg_mask = F.grid_sample(fg_mask, grid, align_corners=True)
            fg_img = F.grid_sample(fg_img, grid, align_corners=True)
        final_img = bg_img * (1 - fg_mask) + fg_img * fg_mask
        return bg_img, fg_mask, fg_img, final_img


def encode_img(ndf=64, in_c=3):
    layers = nn.Sequential(nn.Conv2d(in_c, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):

    def __init__(self, c_dim, ndf=64):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.c_dim = c_dim
        self.base = encode_img()
        self.info_head = nn.Sequential(nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=4))
        self.rf_head = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4))
        self.centroids = nn.Linear(self.c_dim, ndf * 8)

    def forward(self, x, eye, masked_x=None):
        out = self.base(x)
        info = self.info_head(out).view(-1, self.ndf * 8)
        rf = self.rf_head(out).view(-1, 1)
        class_emb = self.centroids(eye)
        if masked_x != None:
            fg_out = self.base(masked_x)
            fg_info = self.info_head(fg_out).view(-1, self.ndf * 8)
            return info, rf, class_emb, fg_info
        else:
            return info, rf, class_emb


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Discriminator,
     lambda: ([], {'c_dim': 4}),
     lambda: ([torch.rand([4, 3, 128, 128]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (FIDInceptionA,
     lambda: ([], {'in_channels': 4, 'pool_features': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionC,
     lambda: ([], {'in_channels': 4, 'channels_7x7': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_1,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (FIDInceptionE_2,
     lambda: ([], {'in_channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (GLU,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (ResBlock,
     lambda: ([], {'channel_num': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
]

class Test_naver_ai_c3_gan(_paritybench_base):
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

