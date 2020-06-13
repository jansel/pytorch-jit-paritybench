import sys
_module = sys.modules[__name__]
del sys
convert_cub_to_hd5_script = _module
convert_flowers_to_hd5_script = _module
loss_estimator = _module
gan = _module
gan_cls = _module
gan_factory = _module
wgan = _module
wgan_cls = _module
runtime = _module
trainer = _module
txt2image_dataset = _module
utils = _module
visualize = _module

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


import torch


from torch import nn


import numpy as np


from torch.autograd import Variable


import torch.nn.functional as F


import torch.nn as nn


from torch.utils.data import DataLoader


from torch.utils.data import Dataset


from torch import autograd


class generator_loss(torch.nn.Module):

    def __init__(self):
        super(generator_loss, self).__init__()
        self.estimator = nn.BCELoss()

    def forward(self, fake):
        batch_size = fake.size()[0]
        self.labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        return self.estimator(fake, self.labels)


class discriminator_loss(torch.nn.Module):

    def __init__(self):
        super(discriminator_loss, self).__init__()
        self.estimator = nn.BCELoss()

    def forward(self, real, wrong, fake):
        batch_size = real.size()[0]
        self.real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        self.fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        return self.estimator(real, self.real_labels) + 0.5 * (self.
            estimator(wrong, self.fake_labels) + self.estimator(fake, self.
            fake_labels))


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.ngf = 64
        self.netG = nn.Sequential(nn.ConvTranspose2d(self.noise_dim, self.
            ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(self.ngf * 8), nn
            .ReLU(True), nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 
            2, 1, bias=False), nn.BatchNorm2d(self.ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=
            False), nn.BatchNorm2d(self.ngf * 2), nn.ReLU(True), nn.
            ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf), nn.ReLU(True), nn.ConvTranspose2d(
            self.ngf, self.num_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z):
        return self.netG(z)


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16
        self.netD_1 = nn.Sequential(nn.Conv2d(self.num_channels, self.ndf, 
            4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False), nn.
            BatchNorm2d(self.ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False), nn.
            BatchNorm2d(self.ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False), nn.
            BatchNorm2d(self.ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        self.netD_2 = nn.Sequential(nn.Conv2d(self.ndf * 8, 1, 4, 1, 0,
            bias=False), nn.Sigmoid())

    def forward(self, inp):
        x_intermediate = self.netD_1(inp)
        output = self.netD_2(x_intermediate)
        return output.view(-1, 1).squeeze(1), x_intermediate


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64
        self.projection = nn.Sequential(nn.Linear(in_features=self.
            embed_dim, out_features=self.projected_embed_dim), nn.
            BatchNorm1d(num_features=self.projected_embed_dim), nn.
            LeakyReLU(negative_slope=0.2, inplace=True))
        self.netG = nn.Sequential(nn.ConvTranspose2d(self.latent_dim, self.
            ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(self.ngf * 8), nn
            .ReLU(True), nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 
            2, 1, bias=False), nn.BatchNorm2d(self.ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=
            False), nn.BatchNorm2d(self.ngf * 2), nn.ReLU(True), nn.
            ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf), nn.ReLU(True), nn.ConvTranspose2d(
            self.ngf, self.num_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, embed_vector, z):
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(
            3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.netG(latent_vector)
        return output


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.ndf = 64
        self.B_dim = 128
        self.C_dim = 16
        self.netD_1 = nn.Sequential(nn.Conv2d(self.num_channels, self.ndf, 
            4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False), nn.
            BatchNorm2d(self.ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False), nn.
            BatchNorm2d(self.ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.
            Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False), nn.
            BatchNorm2d(self.ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)
        self.netD_2 = nn.Sequential(nn.Conv2d(self.ndf * 8 + self.
            projected_embed_dim, 1, 4, 1, 0, bias=False), nn.Sigmoid())

    def forward(self, inp, embed):
        x_intermediate = self.netD_1(inp)
        x = self.projector(x_intermediate, embed)
        x = self.netD_2(x)
        return x.view(-1, 1).squeeze(1), x_intermediate


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.ngf = 64
        self.netG = nn.Sequential(nn.ConvTranspose2d(self.noise_dim, self.
            ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(self.ngf * 8), nn
            .ReLU(True), nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 
            2, 1, bias=False), nn.BatchNorm2d(self.ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=
            False), nn.BatchNorm2d(self.ngf * 2), nn.ReLU(True), nn.
            ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf), nn.ReLU(True), nn.ConvTranspose2d(
            self.ngf, self.num_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z):
        output = self.netG(z)
        return output


class discriminator(nn.Module):

    def __init__(self, improved=False):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.ndf = 64
        if improved:
            self.netD_1 = nn.Sequential(nn.Conv2d(self.num_channels, self.
                ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False), nn.
                LeakyReLU(0.2, inplace=True), nn.Conv2d(self.ndf * 2, self.
                ndf * 4, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=
                True), nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=
                False), nn.LeakyReLU(0.2, inplace=True))
        else:
            self.netD_1 = nn.Sequential(nn.Conv2d(self.num_channels, self.
                ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False), nn.
                BatchNorm2d(self.ndf * 2), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 4), nn.LeakyReLU(0.2, inplace=
                True), nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=
                False), nn.BatchNorm2d(self.ndf * 8), nn.LeakyReLU(0.2,
                inplace=True))
        self.netD_2 = nn.Sequential(nn.Conv2d(self.ndf * 8, 1, 4, 1, 0,
            bias=False))

    def forward(self, inp):
        x_intermediate = self.netD_1(inp)
        x = self.netD_2(x_intermediate)
        x = x.mean(0)
        return x.view(1), x_intermediate


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.latent_dim = self.noise_dim + self.projected_embed_dim
        self.ngf = 64
        self.projection = nn.Sequential(nn.Linear(in_features=self.
            embed_dim, out_features=self.projected_embed_dim), nn.
            BatchNorm1d(num_features=self.projected_embed_dim), nn.
            LeakyReLU(negative_slope=0.2, inplace=True))
        self.netG = nn.Sequential(nn.ConvTranspose2d(self.latent_dim, self.
            ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(self.ngf * 8), nn
            .ReLU(True), nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 
            2, 1, bias=False), nn.BatchNorm2d(self.ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=
            False), nn.BatchNorm2d(self.ngf * 2), nn.ReLU(True), nn.
            ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf), nn.ReLU(True), nn.ConvTranspose2d(
            self.ngf, self.num_channels, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, embed_vector, z):
        projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(
            3)
        latent_vector = torch.cat([projected_embed, z], 1)
        output = self.netG(latent_vector)
        return output


class discriminator(nn.Module):

    def __init__(self, improved=False):
        super(discriminator, self).__init__()
        self.image_size = 64
        self.num_channels = 3
        self.embed_dim = 1024
        self.projected_embed_dim = 128
        self.ndf = 64
        if improved:
            self.netD_1 = nn.Sequential(nn.Conv2d(self.num_channels, self.
                ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False), nn.
                LeakyReLU(0.2, inplace=True), nn.Conv2d(self.ndf * 2, self.
                ndf * 4, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=
                True), nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=
                False), nn.LeakyReLU(0.2, inplace=True))
        else:
            self.netD_1 = nn.Sequential(nn.Conv2d(self.num_channels, self.
                ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False), nn.
                BatchNorm2d(self.ndf * 2), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 4), nn.LeakyReLU(0.2, inplace=
                True), nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=
                False), nn.BatchNorm2d(self.ndf * 8), nn.LeakyReLU(0.2,
                inplace=True))
        self.projector = Concat_embed(self.embed_dim, self.projected_embed_dim)
        self.netD_2 = nn.Sequential(nn.Conv2d(self.ndf * 8 + self.
            projected_embed_dim, 1, 4, 1, 0, bias=False))

    def forward(self, inp, embed):
        x_intermediate = self.netD_1(inp)
        x = self.projector(x_intermediate, embed)
        x = self.netD_2(x)
        x = x.mean(0)
        return x.view(1), x_intermediate


class Concat_embed(nn.Module):

    def __init__(self, embed_dim, projected_embed_dim):
        super(Concat_embed, self).__init__()
        self.projection = nn.Sequential(nn.Linear(in_features=embed_dim,
            out_features=projected_embed_dim), nn.BatchNorm1d(num_features=
            projected_embed_dim), nn.LeakyReLU(negative_slope=0.2, inplace=
            True))

    def forward(self, inp, embed):
        projected_embed = self.projection(embed)
        replicated_embed = projected_embed.repeat(4, 4, 1, 1).permute(2, 3,
            0, 1)
        hidden_concat = torch.cat([inp, replicated_embed], 1)
        return hidden_concat


class minibatch_discriminator(nn.Module):

    def __init__(self, num_channels, B_dim, C_dim):
        super(minibatch_discriminator, self).__init__()
        self.B_dim = B_dim
        self.C_dim = C_dim
        self.num_channels = num_channels
        T_init = torch.randn(num_channels * 4 * 4, B_dim * C_dim) * 0.1
        self.T_tensor = nn.Parameter(T_init, requires_grad=True)

    def forward(self, inp):
        inp = inp.view(-1, self.num_channels * 4 * 4)
        M = inp.mm(self.T_tensor)
        M = M.view(-1, self.B_dim, self.C_dim)
        op1 = M.unsqueeze(3)
        op2 = M.permute(1, 2, 0).unsqueeze(0)
        output = torch.sum(torch.abs(op1 - op2), 2)
        output = torch.sum(torch.exp(-output), 2)
        output = output.view(M.size(0), -1)
        output = torch.cat((inp, output), 1)
        return output


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_aelnouby_Text_to_Image_Synthesis(_paritybench_base):
    pass
    def test_000(self):
        self._check(discriminator(*[], **{}), [torch.rand([4, 3, 64, 64]), torch.rand([4, 1024])], {})

    def test_001(self):
        self._check(Concat_embed(*[], **{'embed_dim': 4, 'projected_embed_dim': 4}), [torch.rand([4, 4, 4, 16]), torch.rand([4, 4, 4])], {})

    def test_002(self):
        self._check(minibatch_discriminator(*[], **{'num_channels': 4, 'B_dim': 4, 'C_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

