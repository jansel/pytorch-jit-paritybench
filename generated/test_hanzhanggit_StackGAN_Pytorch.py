import sys
_module = sys.modules[__name__]
del sys
main = _module
miscc = _module
config = _module
datasets = _module
utils = _module
model = _module
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


from copy import deepcopy


from torch.nn import init


import torch


import torch.nn as nn


import torch.nn.parallel


from torch.autograd import Variable


import torch.backends.cudnn as cudnn


import torch.optim as optim


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class ResBlock(nn.Module):

    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(conv3x3(channel_num, channel_num), nn.
            BatchNorm2d(channel_num), nn.ReLU(True), conv3x3(channel_num,
            channel_num), nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


_global_config['GAN'] = 4


_global_config['TEXT'] = 4


_global_config['CUDA'] = 4


class CA_NET(nn.Module):

    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):

    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(conv3x3(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(nn.Conv2d(ndf * 8, 1,
                kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code
        output = self.outlogits(h_c_code)
        return output.view(-1)


def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes), nn.BatchNorm2d(out_planes), nn.ReLU
        (True))
    return block


_global_config['Z_DIM'] = 4


class STAGE1_G(nn.Module):

    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.define_module()

    def define_module(self):
        ninput = self.z_dim + self.ef_dim
        ngf = self.gf_dim
        self.ca_net = CA_NET()
        self.fc = nn.Sequential(nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4), nn.ReLU(True))
        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        self.img = nn.Sequential(conv3x3(ngf // 16, 3), nn.Tanh())

    def forward(self, text_embedding, noise):
        c_code, mu, logvar = self.ca_net(text_embedding)
        z_c_code = torch.cat((noise, c_code), 1)
        h_code = self.fc(z_c_code)
        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar


class STAGE1_D(nn.Module):

    def __init__(self):
        super(STAGE1_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=
            False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2,
            4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(
            ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None

    def forward(self, image):
        img_embedding = self.encode_img(image)
        return img_embedding


class STAGE2_G(nn.Module):

    def __init__(self, STAGE1_G):
        super(STAGE2_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.STAGE1_G = STAGE1_G
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        self.ca_net = CA_NET()
        self.encoder = nn.Sequential(conv3x3(3, ngf), nn.ReLU(True), nn.
            Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf *
            2), nn.ReLU(True), nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=
            False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.hr_joint = nn.Sequential(conv3x3(self.ef_dim + ngf * 4, ngf * 
            4), nn.BatchNorm2d(ngf * 4), nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        self.upsample2 = upBlock(ngf * 2, ngf)
        self.upsample3 = upBlock(ngf, ngf // 2)
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        self.img = nn.Sequential(conv3x3(ngf // 4, 3), nn.Tanh())

    def forward(self, text_embedding, noise):
        _, stage1_img, _, _ = self.STAGE1_G(text_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)
        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        fake_img = self.img(h_code)
        return stage1_img, fake_img, mu, logvar


class STAGE2_D(nn.Module):

    def __init__(self):
        super(STAGE2_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=
            False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2,
            4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(
            ndf * 8), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 8, 
            ndf * 16, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 16), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 16, ndf * 32, 4, 
            2, 1, bias=False), nn.BatchNorm2d(ndf * 32), nn.LeakyReLU(0.2,
            inplace=True), conv3x3(ndf * 32, ndf * 16), nn.BatchNorm2d(ndf *
            16), nn.LeakyReLU(0.2, inplace=True), conv3x3(ndf * 16, ndf * 8
            ), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, image):
        img_embedding = self.encode_img(image)
        return img_embedding


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hanzhanggit_StackGAN_Pytorch(_paritybench_base):
    pass

    def test_000(self):
        self._check(ResBlock(*[], **{'channel_num': 4}), [torch.rand([4, 4, 4, 4])], {})
    @_fails_compile()

    def test_001(self):
        self._check(D_GET_LOGITS(*[], **{'ndf': 4, 'nef': 4}), [torch.rand([4, 36, 64, 64])], {})
