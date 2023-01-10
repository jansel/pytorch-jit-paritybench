import sys
_module = sys.modules[__name__]
del sys
demo = _module
demo_8bitart = _module
demo_nst = _module
demo_prog = _module
imitator = _module
loss = _module
morphology = _module
networks = _module
painter = _module
predict = _module
pytorch_batch_sinkhorn = _module
renderer = _module
runway_model = _module
train_imitator = _module
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


import torch.optim as optim


import numpy as np


import matplotlib.pyplot as plt


from torch.optim import lr_scheduler


import torch.nn as nn


import torchvision


import random


import math


import torch.nn.functional as F


from torch.nn import init


import functools


from torchvision import models


from torch.autograd import Variable


import torchvision.transforms.functional as TF


from torch.utils.data import Dataset


from torch.utils.data import DataLoader


from torch.utils.data import Subset


from torchvision import transforms


from torchvision import utils


class PixelLoss(nn.Module):

    def __init__(self, p=1):
        super(PixelLoss, self).__init__()
        self.p = p

    def forward(self, canvas, gt, ignore_color=False):
        if ignore_color:
            canvas = torch.mean(canvas, dim=1)
            gt = torch.mean(gt, dim=1)
        loss = torch.mean(torch.abs(canvas - gt) ** self.p)
        return loss


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class VGGPerceptualLoss(torch.nn.Module):

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.resize = resize

    def forward(self, input, target, ignore_color=False):
        self.mean = self.mean.type_as(input)
        self.std = self.std.type_as(input)
        if ignore_color:
            input = torch.mean(input, dim=1, keepdim=True)
            target = torch.mean(target, dim=1, keepdim=True)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


class VGGStyleLoss(torch.nn.Module):

    def __init__(self, transfer_mode, resize=True):
        super(VGGStyleLoss, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        blocks = []
        if transfer_mode == 0:
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
        else:
            blocks.append(vgg.features[:4].eval())
            blocks.append(vgg.features[4:9].eval())
            blocks.append(vgg.features[9:16].eval())
            blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.resize = resize

    def gram_matrix(self, y):
        b, ch, h, w = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * w * h)
        return gram

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            gm_x = self.gram_matrix(x)
            gm_y = self.gram_matrix(y)
            loss += torch.sum((gm_x - gm_y) ** 2)
        return loss


class SinkhornLoss(nn.Module):

    def __init__(self, epsilon=0.01, niter=5, normalize=False):
        super(SinkhornLoss, self).__init__()
        self.epsilon = epsilon
        self.niter = niter
        self.normalize = normalize

    def _mesh_grids(self, batch_size, h, w):
        a = torch.linspace(0.0, h - 1.0, h)
        b = torch.linspace(0.0, w - 1.0, w)
        y_grid = a.view(-1, 1).repeat(batch_size, 1, w) / h
        x_grid = b.view(1, -1).repeat(batch_size, h, 1) / w
        grids = torch.cat([y_grid.view(batch_size, -1, 1), x_grid.view(batch_size, -1, 1)], dim=-1)
        return grids

    def forward(self, canvas, gt):
        batch_size, c, h, w = gt.shape
        if h > 24:
            canvas = nn.functional.interpolate(canvas, [24, 24], mode='area')
            gt = nn.functional.interpolate(gt, [24, 24], mode='area')
            batch_size, c, h, w = gt.shape
        canvas_grids = self._mesh_grids(batch_size, h, w)
        gt_grids = torch.clone(canvas_grids)
        i = random.randint(0, 2)
        img_1 = canvas[:, [i], :, :]
        img_2 = gt[:, [i], :, :]
        mass_x = img_1.reshape(batch_size, -1)
        mass_y = img_2.reshape(batch_size, -1)
        if self.normalize:
            loss = spc.sinkhorn_normalized(canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter, mass_x=mass_x, mass_y=mass_y)
        else:
            loss = spc.sinkhorn_loss(canvas_grids, gt_grids, epsilon=self.epsilon, niter=self.niter, mass_x=mass_x, mass_y=mass_y)
        return loss


class Erosion2d(nn.Module):

    def __init__(self, m=1):
        super(Erosion2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=1000000000.0)
        for i in range(c):
            channel = self.unfold(x_pad[:, [i], :, :])
            channel = torch.min(channel, dim=1, keepdim=True)[0]
            channel = channel.view([batch_size, 1, h, w])
            x[:, [i], :, :] = channel
        return x


class Dilation2d(nn.Module):

    def __init__(self, m=1):
        super(Dilation2d, self).__init__()
        self.m = m
        self.pad = [m, m, m, m]
        self.unfold = nn.Unfold(2 * m + 1, padding=0, stride=1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x_pad = F.pad(x, pad=self.pad, mode='constant', value=-1000000000.0)
        for i in range(c):
            channel = self.unfold(x_pad[:, [i], :, :])
            channel = torch.max(channel, dim=1, keepdim=True)[0]
            channel = channel.view([batch_size, 1, h, w])
            x[:, [i], :, :] = channel
        return x


class Identity(nn.Module):

    def forward(self, x):
        return x


class DCGAN(nn.Module):

    def __init__(self, rdrr, ngf=64):
        super(DCGAN, self).__init__()
        input_nc = rdrr.d
        self.out_size = 128
        self.main = nn.Sequential(nn.ConvTranspose2d(input_nc, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True), nn.ConvTranspose2d(ngf, 6, 4, 2, 1, bias=False))

    def forward(self, input):
        output_tensor = self.main(input)
        return output_tensor[:, 0:3, :, :], output_tensor[:, 3:6, :, :]


class DCGAN_32(nn.Module):

    def __init__(self, rdrr, ngf=64):
        super(DCGAN_32, self).__init__()
        input_nc = rdrr.d
        self.out_size = 32
        self.main = nn.Sequential(nn.ConvTranspose2d(input_nc, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True), nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True), nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(ngf * 2, 6, 4, 2, 1, bias=False))

    def forward(self, input):
        output_tensor = self.main(input)
        return output_tensor[:, 0:3, :, :], output_tensor[:, 3:6, :, :]


class PixelShuffleNet(nn.Module):

    def __init__(self, input_nc):
        super(PixelShuffleNet, self).__init__()
        self.fc1 = nn.Linear(input_nc, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(8, 4 * 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = x.view(-1, 3, 128, 128)
        return x


class PixelShuffleNet_32(nn.Module):

    def __init__(self, input_nc):
        super(PixelShuffleNet_32, self).__init__()
        self.fc1 = nn.Linear(input_nc, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.conv1 = nn.Conv2d(8, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 4 * 3, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 8, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = x.view(-1, 3, 32, 32)
        return x


class HuangNet(nn.Module):

    def __init__(self, rdrr):
        super(HuangNet, self).__init__()
        self.rdrr = rdrr
        self.out_size = 128
        self.fc1 = nn.Linear(rdrr.d, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(8, 4 * 6, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        output_tensor = x.view(-1, 6, 128, 128)
        return output_tensor[:, 0:3, :, :], output_tensor[:, 3:6, :, :]


class ZouFCNFusion(nn.Module):

    def __init__(self, rdrr):
        super(ZouFCNFusion, self).__init__()
        self.rdrr = rdrr
        self.out_size = 128
        self.huangnet = PixelShuffleNet(rdrr.d_shape)
        self.dcgan = DCGAN(rdrr)

    def forward(self, x):
        x_shape = x[:, 0:self.rdrr.d_shape, :, :]
        x_alpha = x[:, [-1], :, :]
        if self.rdrr.renderer in ['oilpaintbrush', 'airbrush']:
            x_alpha = torch.tensor(1.0)
        mask = self.huangnet(x_shape)
        color, _ = self.dcgan(x)
        return color * mask, x_alpha * mask


class ZouFCNFusionLight(nn.Module):

    def __init__(self, rdrr):
        super(ZouFCNFusionLight, self).__init__()
        self.rdrr = rdrr
        self.out_size = 32
        self.huangnet = PixelShuffleNet_32(rdrr.d_shape)
        self.dcgan = DCGAN_32(rdrr)

    def forward(self, x):
        x_shape = x[:, 0:self.rdrr.d_shape, :, :]
        x_alpha = x[:, [-1], :, :]
        if self.rdrr.renderer in ['oilpaintbrush', 'airbrush']:
            x_alpha = torch.tensor(1.0)
        mask = self.huangnet(x_shape)
        color, _ = self.dcgan(x)
        return color * mask, x_alpha * mask


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class UNet(torch.nn.Module):

    def __init__(self, rdrr):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(UNet, self).__init__()
        norm_layer = get_norm_layer(norm_type='batch')
        self.unet = UnetGenerator(rdrr.d, 6, 7, norm_layer=norm_layer, use_dropout=False)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.repeat(1, 1, 128, 128)
        output_tensor = self.unet(x)
        return output_tensor[:, 0:3, :, :], output_tensor[:, 3:6, :, :]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (Dilation2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Erosion2d,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (Identity,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {}),
     False),
    (PixelShuffleNet,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelShuffleNet_32,
     lambda: ([], {'input_nc': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (UnetGenerator,
     lambda: ([], {'input_nc': 4, 'output_nc': 4, 'num_downs': 4}),
     lambda: ([torch.rand([4, 4, 64, 64])], {}),
     True),
    (VGGPerceptualLoss,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
    (VGGStyleLoss,
     lambda: ([], {'transfer_mode': 4}),
     lambda: ([torch.rand([4, 4]), torch.rand([4, 4])], {}),
     False),
]

class Test_jiupinjia_stylized_neural_painting(_paritybench_base):
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

