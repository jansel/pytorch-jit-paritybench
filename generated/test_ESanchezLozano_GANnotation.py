import sys
_module = sys.modules[__name__]
del sys
GANnotation = _module
demo_gannotation = _module
model = _module
GANnotation = _module
Train_options = _module
databases = _module
model_GAN = _module
train = _module
utils = _module

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


import torch.nn as nn


import torch.nn.init as weight_init


import math


import torch.nn.functional as F


import functools


import collections


import numpy as np


import itertools


from torch.autograd import Variable


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import ConcatDataset


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.reflection1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
            padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.reflection2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
            padding=0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.reflection1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.reflection2(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = x + residual
        return out


class ResidualBlock(nn.Module):

    def __init__(self, channels, norm_layer=nn.InstanceNorm2d):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = norm_layer(channels)
        self.prelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = norm_layer(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        return x + residual


def init_weights(net, init_type='normal', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or 
            classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in'
                    )
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type
                    )
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


class Generator(nn.Module):

    def __init__(self, conv_dim=64, c_dim=66, repeat_num=6):
        super(Generator, self).__init__()
        initial_layer = [nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7,
            stride=1, padding=3, bias=False)]
        initial_layer += [nn.InstanceNorm2d(conv_dim, affine=True)]
        initial_layer += [nn.LeakyReLU(0.2, inplace=True)]
        curr_dim = conv_dim
        for i in range(2):
            initial_layer += [nn.Conv2d(curr_dim, curr_dim * 2, kernel_size
                =4, stride=2, padding=1, bias=False)]
            initial_layer += [nn.InstanceNorm2d(curr_dim * 2, affine=True)]
            initial_layer += [nn.LeakyReLU(0.2, inplace=True)]
            curr_dim = curr_dim * 2
        self.down_conv = nn.Sequential(*initial_layer)
        bottleneck = []
        for i in range(repeat_num):
            bottleneck += [ResidualBlock(curr_dim)]
        self.bottleneck = nn.Sequential(*bottleneck)
        features = []
        for i in range(2):
            features += [nn.ConvTranspose2d(curr_dim, curr_dim // 2,
                kernel_size=4, stride=2, padding=1, bias=False)]
            features += [nn.InstanceNorm2d(curr_dim // 2, affine=True)]
            features += [nn.LeakyReLU(0.2, inplace=True)]
            curr_dim = curr_dim // 2
        self.feature_layer = nn.Sequential(*features)
        colour = [nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3,
            bias=False)]
        colour += [nn.Tanh()]
        self.colour_layer = nn.Sequential(*colour)
        mask = [nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3,
            bias=False)]
        mask += [nn.Sigmoid()]
        self.mask_layer = nn.Sequential(*mask)
        init_weights(self)

    def forward(self, x):
        down = self.down_conv(x)
        bottle = self.bottleneck(down)
        features = self.feature_layer(bottle)
        col = self.colour_layer(features)
        mask = self.mask_layer(features)
        output = mask * (x[:, 0:3, :, :] - col) + col
        return output


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(nn.Conv2d(dim_in, dim_out, kernel_size=3,
            stride=1, padding=1, bias=False), nn.InstanceNorm2d(dim_out,
            affine=True, track_running_stats=True), nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1,
            bias=False), nn.InstanceNorm2d(dim_out, affine=True,
            track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=3, repeat_num=6):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=
            1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True,
            track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4,
                stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True,
                track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2,
                kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True,
                track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1,
            padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, maps):
        x = torch.cat((x, maps), 1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, ndim=66, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3 + ndim, conv_dim, kernel_size=4, stride=2,
            padding=1))
        layers.append(nn.LeakyReLU(0.01))
        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4,
                stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2
        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1,
            padding=1, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        return out_src


class LossNetwork(torch.nn.Module):

    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.LossOutput = collections.namedtuple('LossOutput', ['relu1_2',
            'relu2_2', 'relu3_3', 'relu4_3'])
        self.vgg_layers = vgg_model.features if hasattr(vgg_model, 'features'
            ) else vgg_model.module.features
        self.layer_name_mapping = {'3': 'relu1_2', '8': 'relu2_2', '15':
            'relu3_3', '22': 'relu4_3'}

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return self.LossOutput(**output)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_ESanchezLozano_GANnotation(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ResidualBlock(*[], **{'dim_in': 4, 'dim_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Generator(*[], **{}), [torch.rand([4, 2, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Discriminator(*[], **{}), [torch.rand([4, 69, 64, 64])], {})

