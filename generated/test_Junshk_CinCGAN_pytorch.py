import sys
_module = sys.modules[__name__]
del sys
code = _module
DIV2K = _module
DIV2K_jpeg = _module
MyImage = _module
data = _module
benchmark = _module
common = _module
demo = _module
div2k = _module
myimage = _module
srdata = _module
dataloader = _module
loss = _module
adversarial = _module
discriminator = _module
tvloss = _module
vgg = _module
main = _module
EDSR = _module
MDSR = _module
model = _module
common = _module
ddbpn = _module
edsr = _module
mask = _module
mdsr = _module
option = _module
srresnet = _module
template = _module
jpeg2binary = _module
png2binary = _module
trainer = _module
utility = _module

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


import random


import math


import numpy as np


import scipy.misc as misc


import torch


import torch.utils.data as data


from torchvision import transforms


from torch.utils.data.dataloader import default_collate


import queue


import collections


import torch.multiprocessing as multiprocessing


from torch._C import _set_worker_signal_handlers


from torch._C import _remove_worker_pids


from torch._C import _error_if_any_worker_fails


from torch.utils.data.dataloader import DataLoader


from torch.utils.data.dataloader import ExceptionWrapper


import torch.nn as nn


import torch.nn.functional as F


import torch.optim as optim


from torch.autograd import Variable


import torchvision.models as models


import torch.backends.cudnn as cudnn


from torch.utils.data import DataLoader


import torchvision.utils as utils


import copy


import scipy.misc


import matplotlib.pyplot as plt


from math import sqrt


from functools import reduce


import torchvision.utils as tu


import time


import matplotlib


import torch.optim.lr_scheduler as lrs


class Adversarial(nn.Module):

    def __init__(self, args, gan_type):
        super(Adversarial, self).__init__()
        self.gan_type = gan_type
        self.gan_k = args.gan_k
        self.discriminator = discriminator.Discriminator(args, gan_type)
        if gan_type != 'WGAN_GP':
            self.optimizer = utility.make_optimizer(args, self.discriminator)
        else:
            self.optimizer = optim.Adam(self.discriminator.parameters(), betas=(0, 0.9), eps=1e-08, lr=1e-05)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

    def forward(self, fake, real):
        fake_detach = fake.detach()
        self.loss = 0
        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if self.gan_type == 'GAN':
                label_fake = Variable(d_fake.data.new(d_fake.size()).fill_(0))
                label_real = Variable(d_real.data.new(d_real.size()).fill_(1))
                loss_d = F.binary_cross_entropy_with_logits(d_fake, label_fake) + F.binary_cross_entropy_with_logits(d_real, label_real)
            elif self.gan_type.find('WGAN') >= 0:
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = Variable(fake_detach.data.new(fake.size(0), 1, 1, 1).uniform_(), requires_grad=False)
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(outputs=d_hat.sum(), inputs=hat, retain_graph=True, create_graph=True, only_inputs=True)[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty
            self.loss += loss_d.data[0]
            loss_d.backward()
            self.optimizer.step()
            if self.gan_type == 'WGAN':
                for p in self.discriminator.parameters():
                    p.data.clamp_(-1, 1)
        self.loss /= self.gan_k
        d_fake_for_g = self.discriminator(fake)
        if self.gan_type == 'GAN':
            loss_g = F.binary_cross_entropy_with_logits(d_fake_for_g, label_real)
        elif self.gan_type.find('WGAN') >= 0:
            loss_g = -d_fake_for_g.mean()
        return loss_g

    def state_dict(self, *args, **kwargs):
        state_discriminator = self.discriminator.state_dict(*args, **kwargs)
        state_optimizer = self.optimizer.state_dict()
        return dict(**state_discriminator, **state_optimizer)


class Discriminator(nn.Module):

    def __init__(self, args, gan_type='GAN'):
        super(Discriminator, self).__init__()
        in_channels = 3
        out_channels = 64
        depth = 7
        bn = True
        act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        m_features = [common.BasicBlock(args.n_colors, out_channels, 3, bn=bn, act=act)]
        for i in range(depth):
            in_channels = out_channels
            if i % 2 == 1:
                stride = 1
                out_channels *= 2
            else:
                stride = 2
            m_features.append(common.BasicBlock(in_channels, out_channels, 3, stride=stride, bn=bn, act=act))
        self.features = nn.Sequential(*m_features)
        patch_size = args.patch_size // 2 ** ((depth + 1) // 2)
        m_classifier = [nn.Linear(out_channels * patch_size ** 2, 1024), act, nn.Linear(1024, 1)]
        self.classifier = nn.Sequential(*m_classifier)

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features.view(features.size(0), -1))
        return output


class TVLoss(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class VGG(nn.Module):

    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35])
        vgg_mean = 0.485, 0.456, 0.406
        vgg_std = 0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        self.vgg.requires_grad = False

    def forward(self, sr, hr):

        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
        hr_detach = hr.detach()
        hr_detach.volatile = True
        hr_detach.requires_grad = False
        vgg_sr, vgg_hr = [_forward(v) for v in (sr, hr_detach)]
        loss = F.mse_loss(vgg_sr, vgg_hr)
        return loss


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean) * rgb_range
        for params in self.parameters():
            params.requires_grad = False


class ResBlock(nn.Module):

    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):
        modules = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feat, 4 * n_feat, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if bn:
                    modules.append(nn.BatchNorm2d(n_feat))
                if act:
                    modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feat, 9 * n_feat, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if act:
                modules.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*modules)


def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {(2): (6, 2, 2), (4): (8, 4, 2), (8): (12, 8, 2)}[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d
    return conv_f(in_channels, out_channels, kernel_size, stride=stride, padding=padding)


class DenseProjection(nn.Module):

    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        if bottleneck:
            self.bottleneck = nn.Sequential(*[nn.Conv2d(in_channels, nr, 1), nn.PReLU(nr)])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels
        self.conv_1 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])
        self.conv_2 = nn.Sequential(*[projection_conv(nr, inter_channels, scale, not up), nn.PReLU(inter_channels)])
        self.conv_3 = nn.Sequential(*[projection_conv(inter_channels, nr, scale, up), nn.PReLU(nr)])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)
        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)
        out = a_0.add(a_1)
        return out


class DDBPN(nn.Module):

    def __init__(self, args):
        super(DDBPN, self).__init__()
        scale = args.scale[0]
        n0 = 128
        nr = 32
        self.depth = 6
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        initial = [nn.Conv2d(args.n_colors, n0, 3, padding=1), nn.PReLU(n0), nn.Conv2d(n0, nr, 1), nn.PReLU(nr)]
        self.initial = nn.Sequential(*initial)
        self.upmodules = nn.ModuleList()
        self.downmodules = nn.ModuleList()
        channels = nr
        for i in range(self.depth):
            self.upmodules.append(DenseProjection(channels, nr, scale, True, i > 1))
            if i != 0:
                channels += nr
        channels = nr
        for i in range(self.depth - 1):
            self.downmodules.append(DenseProjection(channels, nr, scale, False, i != 0))
            channels += nr
        reconstruction = [nn.Conv2d(self.depth * nr, args.n_colors, 3, padding=1)]
        self.reconstruction = nn.Sequential(*reconstruction)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.initial(x)
        h_list = []
        l_list = []
        for i in range(self.depth - 1):
            if i == 0:
                l = x
            else:
                l = torch.cat(l_list, dim=1)
            h_list.append(self.upmodules[i](l))
            l_list.append(self.downmodules[i](torch.cat(h_list, dim=1)))
        h_list.append(self.upmodules[-1](torch.cat(l_list, dim=1)))
        out = self.reconstruction(torch.cat(h_list, dim=1))
        out = self.add_mean(out)
        return out


class Conv_ReLU_Block(nn.Module):

    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv_Res_Block(nn.Module):

    def __init__(self):
        super(Conv_Res_Block, self).__init__()
        self.convs = nn.Sequential(Conv_ReLU_Block(), Conv_ReLU_Block())

    def forward(self, x):
        return self.convs(x) + x


class dense(nn.Module):

    def __init__(self):
        super(dense, self).__init__()
        self.res1 = Conv_Res_Block()
        self.res2 = Conv_Res_Block()
        self.res3 = Conv_Res_Block()
        self.res4 = Conv_Res_Block()

    def forward(self, x):
        x1 = x
        x2 = self.res1(x1)
        x3 = self.res2(x2)
        x4 = self.res3(x3)
        x5 = self.res4(x4)
        return x5


class MASK(nn.Module):

    def __init__(self, scale=2):
        super(MASK, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 10)
        self.input = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.ReLU())
        self.output = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False))
        self.dense_down_1 = dense()
        self.dense_down_2 = dense()
        self.dense_up_1 = dense()
        self.dense_up_2 = dense()
        self.dense_bottom = dense()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out1 = self.input(x)
        out2 = self.dense_down_1(F.max_pool2d(out1, 2))
        out3 = self.dense_down_2(F.avg_pool2d(out2, 2))
        out3_ = out3 + self.dense_bottom(out3)
        out2_ = out2 + F.upsample(self.dense_up_2(out3_), scale_factor=2)
        out1_ = out1 + F.upsample(self.dense_up_1(out2_), scale_factor=2)
        out = self.output(out1_)
        return F.sigmoid(out)


class _Conv_Block(nn.Module):

    def __init__(self):
        super(_Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        return output


class _Residual_Block(nn.Module):

    def __init__(self):
        super(_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.relu(self.conv2(output))
        output = torch.add(output, identity_data)
        return output


class _NetG_DOWN(nn.Module):

    def __init__(self, stride=2):
        super(_NetG_DOWN, self).__init__()
        self.conv_input = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=stride + 2, stride=stride, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=stride + 2, stride=stride, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.residual = self.make_layer(_Residual_Block, 6)
        self.conv_output = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        out = self.residual(out)
        out = self.conv_output(out)
        return out


class _NetD(nn.Module):

    def __init__(self, stride=1):
        super(_NetD, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=stride, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=stride, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=stride, padding=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1))

    def forward(self, input):
        out = self.features(input)
        return out


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Conv_ReLU_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (Conv_Res_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (MASK,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
    (ResBlock,
     lambda: ([], {'conv': _mock_layer, 'n_feat': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (TVLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (_Conv_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (_NetD,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_NetG_DOWN,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     True),
    (_Residual_Block,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
    (dense,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 64, 64, 64])], {}),
     True),
]

class Test_Junshk_CinCGAN_pytorch(_paritybench_base):
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

