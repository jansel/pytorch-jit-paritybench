import sys
_module = sys.modules[__name__]
del sys
AE = _module
AE_test = _module
CGAN = _module
CGAN_TEST = _module
DAE = _module
DAE_test = _module
DCGAN = _module
GAN = _module
VAE = _module
VAE_test = _module
WDCGAN = _module
WGAN = _module

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


import torch.autograd


import torch.nn as nn


import numpy as np


import random


import torch


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


import math


import torch.nn.functional as F


z_dimension = 100


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.
            BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.
            BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 16, 3, 1, 1), nn.
            BatchNorm2d(16), nn.ReLU())
        self.encoder_fc = nn.Linear(16 * 7 * 7, z_dimension)
        self.decoder_fc = nn.Linear(z_dimension, 16 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16), nn.Tanh(), nn.ConvTranspose2d(16, 1, 4, 2, 
            1), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        code = self.encoder_fc(x)
        x = self.decoder_fc(code)
        x = x.view(x.shape[0], 16, 7, 7)
        decode = self.decoder(x)
        return code, decode


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.
            BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.
            BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 16, 3, 1, 1), nn.
            BatchNorm2d(16), nn.ReLU())
        self.encoder_fc = nn.Linear(16 * 7 * 7, z_dimension)
        self.decoder_fc = nn.Linear(z_dimension, 16 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16), nn.Tanh(), nn.ConvTranspose2d(16, 1, 4, 2, 
            1), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        code = self.encoder_fc(x)
        x = self.decoder_fc(code)
        x = x.view(x.shape[0], 16, 7, 7)
        decode = self.decoder(x)
        return code, decode


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn
            .Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 10), nn.
            Softmax())

    def forward(self, x):
        x = self.dis(x)
        return x


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(z_dimension + 10, 256), nn.ReLU(
            True), nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784),
            nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(z_dimension + 10, 256), nn.ReLU(
            True), nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784),
            nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder_conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(16, 32, kernel_size=3, stride=2,
            padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.
            BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True))
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, nz)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz + 10, 32 * 7 * 7)
        self.decoder_deconv = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2,
            1), nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid())

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output

    def encoder(self, x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        return z, mean, logstd

    def decoder(self, z):
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder_deconv(out3)
        return out3


class Discriminator(nn.Module):

    def __init__(self, outputn=1):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True), nn.MaxPool2d((2, 2)), nn.Conv2d(32, 64,
            3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(
            (2, 2)))
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 1024), nn.LeakyReLU(
            0.2, True), nn.Linear(1024, outputn), nn.Sigmoid())

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder_conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3,
            stride=2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(16, 32, kernel_size=3, stride=2,
            padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.
            BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True))
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, nz)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz + 10, 32 * 7 * 7)
        self.decoder_deconv = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2,
            1), nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid())

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output

    def encoder(self, x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        return z, mean, logstd

    def decoder(self, z):
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder_deconv(out3)
        return out3


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.
            BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.
            BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 16, 3, 1, 1), nn.
            BatchNorm2d(16), nn.ReLU())
        self.encoder_fc = nn.Linear(16 * 7 * 7, z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(z_dimension, 16 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16), nn.Tanh(), nn.ConvTranspose2d(16, 1, 4, 2, 
            1), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        code = self.encoder_fc(x)
        code = self.Sigmoid(code)
        x = self.decoder_fc(code)
        x = x.view(x.shape[0], 16, 7, 7)
        decode = self.decoder(x)
        return code, decode


class autoencoder(nn.Module):

    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.
            BatchNorm2d(16), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.
            BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 16, 3, 1, 1), nn.
            BatchNorm2d(16), nn.ReLU())
        self.encoder_fc = nn.Linear(16 * 7 * 7, z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(z_dimension, 16 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16), nn.Tanh(), nn.ConvTranspose2d(16, 1, 4, 2, 
            1), nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        code = self.encoder_fc(x)
        code = self.Sigmoid(code)
        x = self.decoder_fc(code)
        x = x.view(x.shape[0], 16, 7, 7)
        decode = self.decoder(x)
        return code, decode


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True), nn.MaxPool2d((2, 2)), nn.Conv2d(32, 64,
            3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(
            (2, 2)))
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 1024), nn.LeakyReLU(
            0.2, True), nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class generator(nn.Module):

    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU(True))
        self.gen = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 32, 3, stride=
            1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Conv2d(32,
            1, 3, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 1, 56, 56)
        x = self.br(x)
        x = self.gen(x)
        return x


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn
            .Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.
            Sigmoid())

    def forward(self, x):
        x = self.dis(x)
        return x


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(100, 256), nn.ReLU(True), nn.
            Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride
            =2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, nz)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz, 32 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.
            Sigmoid())

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder(out3)
        return out3, mean, logstd


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True), nn.MaxPool2d((2, 2)), nn.Conv2d(32, 64,
            3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(
            (2, 2)))
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 1024), nn.LeakyReLU(
            0.2, True), nn.Linear(1024, 1), nn.Sigmoid())

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride
            =2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, nz)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz, 32 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.
            Sigmoid())

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder(out3)
        return out3, mean, logstd


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride
            =2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, nz)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz, 32 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.
            Sigmoid())

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder(out3)
        return out3, mean, logstd


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True), nn.MaxPool2d((2, 2)), nn.Conv2d(32, 64,
            3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(
            (2, 2)))
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 1024), nn.LeakyReLU(
            0.2, True), nn.Linear(1024, 1))

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride
            =2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, nz)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz, 32 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.
            Sigmoid())

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder(out3)
        return out3, mean, logstd


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride
            =2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, z_dimension)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, z_dimension)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(z_dimension, 32 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.
            Sigmoid())

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder(out3)
        return out3, mean, logstd


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, stride
            =2, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.2, inplace=
            True), nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(
            32, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True))
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, z_dimension)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, z_dimension)
        self.decoder_fc = nn.Linear(z_dimension, 32 * 7 * 7)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True), nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.
            Sigmoid())

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        out3 = self.decoder(out3)
        return out3, mean, logstd


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True), nn.MaxPool2d((2, 2)), nn.Conv2d(32, 64,
            3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(
            (2, 2)))
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 1024), nn.LeakyReLU(
            0.2, True), nn.Linear(1024, 1))

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class generator(nn.Module):

    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU(True))
        self.gen = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 32, 3, stride=
            1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Conv2d(32,
            1, 3, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 1, 56, 56)
        x = self.br(x)
        x = self.gen(x)
        return x


class generator(nn.Module):

    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU(True))
        self.gen = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 32, 3, stride=
            1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Conv2d(32,
            1, 3, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 1, 56, 56)
        x = self.br(x)
        x = self.gen(x)
        return x


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn
            .Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, x):
        x = self.dis(x)
        return x


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(100, 256), nn.ReLU(True), nn.
            Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True), nn.MaxPool2d((2, 2)), nn.Conv2d(32, 64,
            3, stride=1, padding=1), nn.LeakyReLU(0.2, True), nn.MaxPool2d(
            (2, 2)))
        self.fc = nn.Sequential(nn.Linear(7 * 7 * 64, 1024), nn.LeakyReLU(
            0.2, True), nn.Linear(1024, 1))

    def forward(self, x):
        x = self.dis(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class generator(nn.Module):

    def __init__(self, input_size, num_feature):
        super(generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)
        self.br = nn.Sequential(nn.BatchNorm2d(1), nn.ReLU(True))
        self.gen = nn.Sequential(nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.Conv2d(64, 32, 3, stride=
            1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Conv2d(32,
            1, 3, stride=2, padding=1), nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 1, 56, 56)
        x = self.br(x)
        x = self.gen(x)
        return x


class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn
            .Linear(256, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, x):
        x = self.dis(x)
        return x


class generator(nn.Module):

    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(nn.Linear(100, 256), nn.ReLU(True), nn.
            Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_YixinChen_AI_CVAE_GAN_zoos_PyTorch_Beginner(_paritybench_base):
    pass
    def test_000(self):
        self._check(discriminator(*[], **{}), [torch.rand([784, 784])], {})

    def test_001(self):
        self._check(generator(*[], **{}), [torch.rand([100, 100])], {})

