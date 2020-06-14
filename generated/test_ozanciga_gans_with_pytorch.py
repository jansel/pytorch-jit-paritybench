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
models = _module
srgan = _module
models = _module
wgan_gp = _module
dataset_loader = _module
wgan = _module

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


from torch import optim


from torch.autograd.variable import Variable


from torch.utils.data import DataLoader


import torch.nn.functional as F


import numpy


from torch.autograd import Variable


parser = argparse.ArgumentParser()


opt = parser.parse_args()


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(240, 1), nn.Sigmoid())
        self.x_map = nn.Sequential(nn.Linear(n_features, 240 * 5))
        self.y_map = nn.Sequential(nn.Linear(opt.n_classes, 50 * 5))
        self.j_map = nn.Sequential(nn.Linear(240 + 50, 240 * 4))

    def forward(self, x, y):
        x = x.view(-1, n_features)
        x = self.x_map(x)
        x, _ = x.view(-1, 240, 5).max(dim=2)
        y = y.view(-1, opt.n_classes)
        y = self.y_map(y)
        y, _ = y.view(-1, 50, 5).max(dim=2)
        jmx = torch.cat((x, y), dim=1)
        jmx = self.j_map(jmx)
        jmx, _ = jmx.view(-1, 240, 4).max(dim=2)
        prob = self.model(jmx)
        return prob


class MeanPoolConv(nn.Module):

    def __init__(self, n_input, n_output, k_size, kaiming_init=True):
        super(MeanPoolConv, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(
            k_size - 1) // 2, bias=True)
        if kaiming_init:
            nn.init.kaiming_uniform_(conv1.weight, mode='fan_in',
                nonlinearity='relu')
        self.model = nn.Sequential(conv1)

    def forward(self, x):
        out = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] +
            x[:, :, 1::2, 1::2]) / 4.0
        out = self.model(out)
        return out


class ConvMeanPool(nn.Module):

    def __init__(self, n_input, n_output, k_size, kaiming_init=True):
        super(ConvMeanPool, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(
            k_size - 1) // 2, bias=True)
        if kaiming_init:
            nn.init.kaiming_uniform_(conv1.weight, mode='fan_in',
                nonlinearity='relu')
        self.model = nn.Sequential(conv1)

    def forward(self, x):
        out = self.model(x)
        out = (out[:, :, ::2, ::2] + out[:, :, 1::2, ::2] + out[:, :, ::2, 
            1::2] + out[:, :, 1::2, 1::2]) / 4.0
        return out


class UpsampleConv(nn.Module):

    def __init__(self, n_input, n_output, k_size, kaiming_init=True):
        super(UpsampleConv, self).__init__()
        conv_layer = nn.Conv2d(n_input, n_output, k_size, stride=1, padding
            =(k_size - 1) // 2, bias=True)
        if kaiming_init:
            nn.init.kaiming_uniform_(conv_layer.weight, mode='fan_in',
                nonlinearity='relu')
        self.model = nn.Sequential(nn.PixelShuffle(2), conv_layer)

    def forward(self, x):
        x = x.repeat((1, 4, 1, 1))
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, n_input, n_output, k_size, resample='up', bn=True,
        spatial_dim=None):
        super(ResidualBlock, self).__init__()
        self.resample = resample
        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size,
                kaiming_init=True)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(
                k_size - 1) // 2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size,
                kaiming_init=True)
            self.out_dim = n_output
            nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in',
                nonlinearity='relu')
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(
                k_size - 1) // 2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size,
                kaiming_init=True)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size,
                kaiming_init=True)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim]
            nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in',
                nonlinearity='relu')
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(
                k_size - 1) // 2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(
                k_size - 1) // 2)
            self.conv_shortcut = None
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]
            nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in',
                nonlinearity='relu')
            nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in',
                nonlinearity='relu')
        self.model = nn.Sequential(nn.BatchNorm2d(n_input) if bn else nn.
            LayerNorm(self.ln_dims), nn.ReLU(inplace=True), self.conv1, nn.
            BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True), self.conv2)

    def forward(self, x):
        if self.conv_shortcut is None:
            return x + self.model(x)
        else:
            return self.conv_shortcut(x) + self.model(x)


class DiscBlock1(nn.Module):

    def __init__(self, n_output):
        super(DiscBlock1, self).__init__()
        self.conv1 = nn.Conv2d(3, n_output, 3, padding=(3 - 1) // 2)
        self.conv2 = ConvMeanPool(n_output, n_output, 1, kaiming_init=True)
        self.conv_shortcut = MeanPoolConv(3, n_output, 1, kaiming_init=False)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in',
            nonlinearity='relu')
        self.model = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.
            conv2)

    def forward(self, x):
        return self.conv_shortcut(x) + self.model(x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 1, 0),
            ResidualBlock(128, 128, 3, resample='up'), ResidualBlock(128, 
            128, 3, resample='up'), ResidualBlock(128, 128, 3, resample=
            'up'), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(
            128, 3, 3, padding=(3 - 1) // 2), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        n_output = 128
        """
        This is a parameter but since we experiment with a single size
        of 3 x 32 x 32 images, it is hardcoded here.
        """
        self.DiscBlock1 = DiscBlock1(n_output)
        self.block1 = nn.Sequential(ResidualBlock(n_output, n_output, 3,
            resample='down', bn=False, spatial_dim=16))
        self.block2 = nn.Sequential(ResidualBlock(n_output, n_output, 3,
            resample=None, bn=False, spatial_dim=8))
        self.block3 = nn.Sequential(ResidualBlock(n_output, n_output, 3,
            resample=None, bn=False, spatial_dim=8))
        self.l1 = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x, dropout=0.0, intermediate_output=False):
        y = self.DiscBlock1(x)
        y = self.block1(y)
        y = F.dropout(y, training=True, p=dropout)
        y = self.block2(y)
        y = F.dropout(y, training=True, p=dropout)
        y = self.block3(y)
        y = F.dropout(y, training=True, p=dropout)
        y = F.relu(y)
        y = y.view(x.size(0), 128, -1)
        y = y.mean(dim=2)
        critic_value = self.l1(y).unsqueeze_(1).unsqueeze_(2)
        if intermediate_output:
            return critic_value, y
        return critic_value


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(opt.channels, 64, 4, 2, 1,
            bias=False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128,
            4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True), nn.
            BatchNorm2d(128), nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.
            LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(256), nn.Conv2d(
            256, 1, 4, 1, 0, bias=False), nn.Sigmoid())
        semantic_dim = opt.img_size // 2 ** 3
        self.l1 = nn.Sequential(nn.Linear(256 * semantic_dim ** 2, 1), nn.
            Sigmoid())

    def forward(self, img):
        prob = self.model(img)
        return prob


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def ganlayer(n_input, n_output, dropout=True):
            pipeline = [nn.Linear(n_input, n_output)]
            pipeline.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                pipeline.append(nn.Dropout(0.25))
            return pipeline
        self.model = nn.Sequential(*ganlayer(opt.latent_dim, 128, dropout=
            False), *ganlayer(128, 256), *ganlayer(256, 512), *ganlayer(512,
            1024), nn.Linear(1024, n_features), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_dims)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(n_features, 512), nn.LeakyReLU
            (0.2, inplace=True), nn.Linear(512, 256), nn.LeakyReLU(0.2,
            inplace=True), nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, img):
        flatimg = img.view(img.size(0), -1)
        prob = self.model(flatimg)
        return prob


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def convlayer(n_input, n_output, k_size=5, stride=2, padding=0,
            output_padding=0):
            block = [nn.ConvTranspose2d(n_input, n_output, kernel_size=
                k_size, stride=stride, padding=padding, bias=False,
                output_padding=output_padding), nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True)]
            return block
        self.conv_block = nn.Sequential(*convlayer(opt.latent_dim, 1024, 1,
            1), *convlayer(1024, 128, 7, 1, 0), *convlayer(128, 64, 4, 2, 1
            ), nn.ConvTranspose2d(64, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, z):
        z = z.view(-1, opt.latent_dim, 1, 1)
        img = self.conv_block(z)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0,
            normalize=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size,
                stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.1, inplace=True))
            return block
        self.model = nn.Sequential(*convlayer(opt.channels, 64, 4, 2, 1,
            normalize=False), *convlayer(64, 128, 4, 2, 1), *convlayer(128,
            1024, 7, 1, 0))
        self.d_head = nn.Sequential(nn.Linear(1024, 1), nn.Sigmoid())
        self.q_head_C = nn.Sequential(nn.Linear(1024, 128), nn.BatchNorm1d(
            128), nn.LeakyReLU(0.1, inplace=True), nn.Linear(128, 2))
        self.q_head_D = nn.Sequential(nn.Linear(1024, 128), nn.BatchNorm1d(
            128), nn.LeakyReLU(0.1, inplace=True), nn.Linear(128, 10), nn.
            Softmax(dim=1))
        """self.q_head_C_mu = nn.Sequential(
            nn.Linear(128, 2)
        )
        self.q_head_C_std = nn.Sequential(
            nn.Linear(128, 2)
        )"""

    def forward(self, img):
        conv_out = self.model(img)
        conv_out = conv_out.squeeze(dim=3).squeeze(dim=2)
        prob = self.d_head(conv_out)
        q = self.q_head_C(conv_out)
        digit_probs = self.q_head_D(conv_out)
        return prob, digit_probs, q


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def convlayer(n_input, n_output, k_size=5, stride=2, padding=0,
            output_padding=0):
            block = [nn.ConvTranspose2d(n_input, n_output, kernel_size=
                k_size, stride=stride, padding=padding, bias=False,
                output_padding=output_padding), nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True)]
            return block
        self.model = nn.Sequential(*convlayer(opt.latent_dim, 128, 7, 1, 0),
            *convlayer(128, 64, 5, 2, 2, output_padding=1), *convlayer(64, 
            32, 5, 2, 2, output_padding=1), nn.ConvTranspose2d(32, opt.
            channels, 5, 1, 2), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0,
            normalize=True, dilation=1):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size,
                stride=stride, padding=padding, bias=False, dilation=dilation)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        self.conv_block = nn.Sequential(*convlayer(opt.channels, 32, 5, 2, 
            2, normalize=False), *convlayer(32, 64, 5, 2, 2))
        self.fc_block = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.
            BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True), nn.Linear(
            512, 1))

    def forward(self, img):
        conv_out = self.conv_block(img)
        conv_out = conv_out.view(img.size(0), 64 * 7 * 7)
        l2_value = self.fc_block(conv_out)
        l2_value = l2_value.unsqueeze_(dim=2).unsqueeze_(dim=3)
        return l2_value


class Generator(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        def convblock(n_input, n_output, k_size=4, stride=2, padding=1,
            normalize=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size,
                stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        g_layers = [64, 128, 256, 512, 512, 512, 512, 512]
        self.encoder = nn.ModuleList(convblock(in_channels, g_layers[0],
            normalize=False))
        for iter in range(1, len(g_layers)):
            self.encoder += convblock(g_layers[iter - 1], g_layers[iter])

        def convdblock(n_input, n_output, k_size=4, stride=2, padding=1,
            dropout=0.0):
            block = [nn.ConvTranspose2d(n_input, n_output, kernel_size=
                k_size, stride=stride, padding=padding, bias=False), nn.
                BatchNorm2d(n_output)]
            if dropout:
                block.append(nn.Dropout(dropout))
            block.append(nn.ReLU(inplace=True))
            return block
        d_layers = [512, 1024, 1024, 1024, 1024, 512, 256, 128]
        self.decoder = nn.ModuleList(convdblock(g_layers[-1], d_layers[0]))
        for iter in range(1, len(d_layers) - 1):
            self.decoder += convdblock(d_layers[iter], d_layers[iter + 1] // 2)
        self.model = nn.Sequential(nn.ConvTranspose2d(d_layers[-1],
            out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        e = [self.encoder[1](self.encoder[0](z))]
        module_iter = 2
        for iter in range(1, 8):
            result = self.encoder[module_iter + 0](e[iter - 1])
            result = self.encoder[module_iter + 1](result)
            result = self.encoder[module_iter + 2](result)
            e.append(result)
            module_iter += 3
        d1 = self.decoder[2](self.decoder[1](self.decoder[0](e[-1])))
        d1 = torch.cat((d1, e[-2]), dim=1)
        d = [d1]
        module_iter = 3
        for iter in range(1, 7):
            result = self.decoder[module_iter + 0](d[iter - 1])
            result = self.decoder[module_iter + 1](result)
            result = self.decoder[module_iter + 2](result)
            result = torch.cat((result, e[-(iter + 2)]), dim=1)
            d.append(result)
            module_iter += 3
        img = self.model(d[-1])
        return img


class Discriminator(nn.Module):

    def __init__(self, a_channels=3, b_channels=3):
        super(Discriminator, self).__init__()

        def convblock(n_input, n_output, k_size=4, stride=2, padding=1,
            normalize=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size,
                stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        self.model = nn.Sequential(*convblock(a_channels + b_channels, 64,
            normalize=False), *convblock(64, 128), *convblock(128, 256), *
            convblock(256, 512))
        self.l1 = nn.Linear(512 * 16 * 16, 1)

    def forward(self, img_A, img_B):
        img = torch.cat((img_A, img_B), dim=1)
        conv_out = self.model(img)
        conv_out = conv_out.view(img_A.size(0), -1)
        prob = self.l1(conv_out)
        prob = F.sigmoid(prob)
        return prob


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [nn.ConvTranspose2d(n_input, n_output, kernel_size=
                k_size, stride=stride, padding=padding, bias=False), nn.
                BatchNorm2d(n_output), nn.ReLU(inplace=True)]
            return block
        self.model = nn.Sequential(*convlayer(opt.latent_dim, 1024, 4, 1, 0
            ), *convlayer(1024, 512, 4, 2, 1), *convlayer(512, 256, 4, 2, 1
            ), *convlayer(256, 128, 4, 2, 1), *convlayer(128, 64, 4, 2, 1),
            *convlayer(64, 32, 4, 2, 1), nn.ConvTranspose2d(32, opt.
            channels, 4, 2, 1), nn.Tanh())
        """
        There is a slight error in v2 of the relativistic gan paper, where
        the architecture goes from 128>64>32 but then 64>3.
        """

    def forward(self, z):
        z = z.view(-1, opt.latent_dim, 1, 1)
        img = self.model(z)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=
            False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size,
                stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        self.model = nn.Sequential(*convlayer(opt.channels * 2, 32, 4, 2, 1
            ), *convlayer(32, 64, 4, 2, 1), *convlayer(64, 128, 4, 2, 1, bn
            =True), *convlayer(128, 256, 4, 2, 1, bn=True), *convlayer(256,
            512, 4, 2, 1, bn=True), *convlayer(512, 1024, 4, 2, 1, bn=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False))

    def forward(self, imgs):
        critic_value = self.model(imgs)
        critic_value = critic_value.view(imgs.size(0), -1)
        return critic_value


class ResidualBlock(nn.Module):

    def __init__(self, n_output=64, k_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(n_output, n_output, k_size,
            stride, padding), nn.BatchNorm2d(n_output), nn.PReLU(), nn.
            Conv2d(n_output, n_output, k_size, stride, padding), nn.
            BatchNorm2d(n_output))

    def forward(self, x):
        return x + self.model(x)


class ShuffleBlock(nn.Module):

    def __init__(self, n_input, n_output, k_size=3, stride=1, padding=1):
        super(ShuffleBlock, self).__init__()
        self.model = nn.Sequential(nn.Conv2d(n_input, n_output, k_size,
            stride, padding), nn.PixelShuffle(2), nn.PReLU())
        """
        Input: :math:`(N, C * upscale_factor^2, H, W)`
        Output: :math:`(N, C, H * upscale_factor, W * upscale_factor)`
        """

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):

    def __init__(self, n_input=3, n_output=3, n_fmap=64, B=16):
        super(Generator, self).__init__()
        self.l1 = nn.Sequential(nn.Conv2d(n_input, n_fmap, 9, 1, 4), nn.PReLU()
            )
        self.R = []
        for _ in range(B):
            self.R.append(ResidualBlock(n_fmap))
        self.R = nn.Sequential(*self.R)
        self.l2 = nn.Sequential(nn.Conv2d(n_fmap, n_fmap, 3, 1, 1), nn.
            BatchNorm2d(n_fmap))
        self.px = nn.Sequential(ShuffleBlock(64, 256), ShuffleBlock(64, 256))
        self.conv_final = nn.Sequential(nn.Conv2d(64, n_output, 9, 1, 4),
            nn.Tanh())

    def forward(self, img_in):
        out_1 = self.l1(img_in)
        out_2 = self.R(out_1)
        out_3 = out_1 + self.l2(out_2)
        out_4 = self.px(out_3)
        return self.conv_final(out_4)


class Discriminator(nn.Module):

    def __init__(self, lr_channels=3):
        super(Discriminator, self).__init__()

        def convblock(n_input, n_output, k_size=3, stride=1, padding=1, bn=True
            ):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size,
                stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        self.conv = nn.Sequential(*convblock(lr_channels, 64, 3, 1, 1, bn=
            False), *convblock(64, 64, 3, 2, 1), *convblock(64, 128, 3, 1, 
            1), *convblock(128, 128, 3, 2, 1), *convblock(128, 256, 3, 1, 1
            ), *convblock(256, 256, 3, 2, 1), *convblock(256, 512, 3, 1, 1),
            *convblock(512, 512, 3, 2, 1))
        self.fc = nn.Sequential(nn.Linear(512 * 16 * 16, 1024), nn.
            LeakyReLU(0.2, inplace=True), nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, img):
        out_1 = self.conv(img)
        out_1 = out_1.view(img.size(0), -1)
        out_2 = self.fc(out_1)
        return out_2


class VGGFeatures(nn.Module):

    def __init__(self):
        super(VGGFeatures, self).__init__()
        model = vgg19(pretrained=True)
        children = list(model.features.children())
        max_pool_indices = [index for index, m in enumerate(children) if
            isinstance(m, nn.MaxPool2d)]
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


class MeanPoolConv(nn.Module):

    def __init__(self, n_input, n_output, k_size):
        super(MeanPoolConv, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(
            k_size - 1) // 2, bias=True)
        self.model = nn.Sequential(conv1)

    def forward(self, x):
        out = (x[:, :, ::2, ::2] + x[:, :, 1::2, ::2] + x[:, :, ::2, 1::2] +
            x[:, :, 1::2, 1::2]) / 4.0
        out = self.model(out)
        return out


class ConvMeanPool(nn.Module):

    def __init__(self, n_input, n_output, k_size):
        super(ConvMeanPool, self).__init__()
        conv1 = nn.Conv2d(n_input, n_output, k_size, stride=1, padding=(
            k_size - 1) // 2, bias=True)
        self.model = nn.Sequential(conv1)

    def forward(self, x):
        out = self.model(x)
        out = (out[:, :, ::2, ::2] + out[:, :, 1::2, ::2] + out[:, :, ::2, 
            1::2] + out[:, :, 1::2, 1::2]) / 4.0
        return out


class UpsampleConv(nn.Module):

    def __init__(self, n_input, n_output, k_size):
        super(UpsampleConv, self).__init__()
        self.model = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(n_input,
            n_output, k_size, stride=1, padding=(k_size - 1) // 2, bias=True))

    def forward(self, x):
        x = x.repeat((1, 4, 1, 1))
        out = self.model(x)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, n_input, n_output, k_size, resample='up', bn=True,
        spatial_dim=None):
        super(ResidualBlock, self).__init__()
        self.resample = resample
        if resample == 'up':
            self.conv1 = UpsampleConv(n_input, n_output, k_size)
            self.conv2 = nn.Conv2d(n_output, n_output, k_size, padding=(
                k_size - 1) // 2)
            self.conv_shortcut = UpsampleConv(n_input, n_output, k_size)
            self.out_dim = n_output
        elif resample == 'down':
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(
                k_size - 1) // 2)
            self.conv2 = ConvMeanPool(n_input, n_output, k_size)
            self.conv_shortcut = ConvMeanPool(n_input, n_output, k_size)
            self.out_dim = n_output
            self.ln_dims = [n_input, spatial_dim, spatial_dim]
        else:
            self.conv1 = nn.Conv2d(n_input, n_input, k_size, padding=(
                k_size - 1) // 2)
            self.conv2 = nn.Conv2d(n_input, n_input, k_size, padding=(
                k_size - 1) // 2)
            self.conv_shortcut = None
            self.out_dim = n_input
            self.ln_dims = [n_input, spatial_dim, spatial_dim]
        self.model = nn.Sequential(nn.BatchNorm2d(n_input) if bn else nn.
            LayerNorm(self.ln_dims), nn.ReLU(inplace=True), self.conv1, nn.
            BatchNorm2d(self.out_dim) if bn else nn.LayerNorm(self.ln_dims),
            nn.ReLU(inplace=True), self.conv2)

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
        self.model = nn.Sequential(self.conv1, nn.ReLU(inplace=True), self.
            conv2)

    def forward(self, x):
        return self.conv_shortcut(x) + self.model(x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 1, 0),
            ResidualBlock(128, 128, 3, resample='up'), ResidualBlock(128, 
            128, 3, resample='up'), ResidualBlock(128, 128, 3, resample=
            'up'), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Conv2d(
            128, 3, 3, padding=(3 - 1) // 2), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        n_output = 128
        """
        This is a parameter but since we experiment with a single size
        of 3 x 32 x 32 images, it is hardcoded here.
        """
        self.DiscBlock1 = DiscBlock1(n_output)
        self.model = nn.Sequential(ResidualBlock(n_output, n_output, 3,
            resample='down', bn=False, spatial_dim=16), ResidualBlock(
            n_output, n_output, 3, resample=None, bn=False, spatial_dim=8),
            ResidualBlock(n_output, n_output, 3, resample=None, bn=False,
            spatial_dim=8), nn.ReLU(inplace=True))
        self.l1 = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x):
        y = self.DiscBlock1(x)
        y = self.model(y)
        y = y.view(x.size(0), 128, -1)
        y = y.mean(dim=2)
        out = self.l1(y).unsqueeze_(1).unsqueeze_(2)
        return out


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        def convblock(n_input, n_output, k_size=4, stride=2, padding=0,
            normalize=True):
            block = [nn.ConvTranspose2d(n_input, n_output, kernel_size=
                k_size, stride=stride, padding=padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.ReLU(inplace=True))
            return block
        self.project = nn.Sequential(nn.Linear(opt.latent_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4), nn.ReLU(inplace=True))
        self.model = nn.Sequential(*convblock(opt.latent_dim, 256, 4, 1, 0),
            *convblock(256, 128, 4, 2, 1), *convblock(128, 64, 4, 2, 1), nn
            .ConvTranspose2d(64, opt.channels, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), *img_dims)
        return img


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        def convblock(n_input, n_output, kernel_size=4, stride=2, padding=1,
            normalize=True):
            block = [nn.Conv2d(n_input, n_output, kernel_size, stride,
                padding, bias=False)]
            if normalize:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        self.model = nn.Sequential(*convblock(opt.channels, 64, 4, 2, 1,
            normalize=False), *convblock(64, 128, 4, 2, 1), *convblock(128,
            256, 4, 2, 1), nn.Conv2d(256, 1, 4, 1, 0, bias=False), nn.Sigmoid()
            )

    def forward(self, img):
        prob = self.model(img)
        return prob


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ozanciga_gans_with_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(ConvMeanPool(*[], **{'n_input': 4, 'n_output': 4, 'k_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(DiscBlock1(*[], **{'n_output': 4}), [torch.rand([4, 3, 4, 4])], {})

    def test_002(self):
        self._check(MeanPoolConv(*[], **{'n_input': 4, 'n_output': 4, 'k_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(ShuffleBlock(*[], **{'n_input': 4, 'n_output': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(UpsampleConv(*[], **{'n_input': 4, 'n_output': 4, 'k_size': 4}), [torch.rand([4, 4, 4, 4])], {})

