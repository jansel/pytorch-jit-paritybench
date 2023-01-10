import sys
_module = sys.modules[__name__]
del sys
data = _module
kpuv_dataset = _module
layered_video_dataset = _module
iuv_crop2full = _module
models = _module
kp2uv_model = _module
lnr_model = _module
networks = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
run_kp2uv = _module
test = _module
third_party = _module
data = _module
base_dataset = _module
fast_data_loader = _module
image_folder = _module
base_model = _module
networks = _module
util = _module
html = _module
util = _module
visualizer = _module
train = _module

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


import torchvision.transforms as transforms


import torch.nn.functional as F


import torch


import numpy as np


import torch.nn as nn


import torch.utils.data


import random


import torch.utils.data as data


from abc import ABC


from abc import abstractmethod


from collections import OrderedDict


from torch.optim import lr_scheduler


import time


class MaskLoss(nn.Module):
    """Define the loss which encourages the predicted alpha matte to match the mask (trimap)."""

    def __init__(self):
        super(MaskLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='none')

    def __call__(self, prediction, target):
        """Calculate loss given predicted alpha matte and trimap.

        Balance positive and negative regions. Exclude 'unknown' region from loss.

        Parameters:
            prediction (tensor) - - predicted alpha
            target (tensor) - - trimap

        Returns: the computed loss
        """
        mask_err = self.loss(prediction, target)
        pos_mask = F.relu(target)
        neg_mask = F.relu(-target)
        pos_mask_loss = (pos_mask * mask_err).sum() / (1 + pos_mask.sum())
        neg_mask_loss = (neg_mask * mask_err).sum() / (1 + neg_mask.sum())
        loss = 0.5 * (pos_mask_loss + neg_mask_loss)
        return loss


class ConvBlock(nn.Module):
    """Helper module consisting of a convolution, optional normalization and activation, with padding='same'."""

    def __init__(self, conv, in_channels, out_channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Create a conv block.

        Parameters:
            conv (convolutional layer) - - the type of conv layer, e.g. Conv2d, ConvTranspose2d
            in_channels (int) - - the number of input channels
            in_channels (int) - - the number of output channels
            ksize (int) - - the kernel size
            stride (int) - - stride
            dil (int) - - dilation
            norm (norm layer) - - the type of normalization layer, e.g. BatchNorm2d, InstanceNorm2d
            activation (str)  -- the type of activation: relu | leaky | tanh | none
        """
        super(ConvBlock, self).__init__()
        self.k = ksize
        self.s = stride
        self.d = dil
        self.conv = conv(in_channels, out_channels, ksize, stride=stride, dilation=dil)
        if norm is not None:
            self.norm = norm(out_channels)
        else:
            self.norm = None
        if activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None

    def forward(self, x):
        """Forward pass. Compute necessary padding and cropping because pytorch doesn't have pad=same."""
        height, width = x.shape[-2:]
        if isinstance(self.conv, nn.modules.ConvTranspose2d):
            desired_height = height * self.s
            desired_width = width * self.s
            pady = 0
            padx = 0
        else:
            desired_height = height // self.s
            desired_width = width // self.s
            pady = 0.5 * (self.s * (desired_height - 1) + (self.k - 1) * (self.d - 1) + self.k - height)
            padx = 0.5 * (self.s * (desired_width - 1) + (self.k - 1) * (self.d - 1) + self.k - width)
        x = F.pad(x, [int(np.floor(padx)), int(np.ceil(padx)), int(np.floor(pady)), int(np.ceil(pady))])
        x = self.conv(x)
        if x.shape[-2] != desired_height or x.shape[-1] != desired_width:
            cropy = x.shape[-2] - desired_height
            cropx = x.shape[-1] - desired_width
            x = x[:, :, int(np.floor(cropy / 2.0)):-int(np.ceil(cropy / 2.0)), int(np.floor(cropx / 2.0)):-int(np.ceil(cropx / 2.0))]
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    """Define a residual block."""

    def __init__(self, channels, ksize=4, stride=1, dil=1, norm=None, activation='relu'):
        """Initialize the residual block, which consists of 2 conv blocks with a skip connection."""
        super(ResBlock, self).__init__()
        self.convblock1 = ConvBlock(nn.Conv2d, channels, channels, ksize=ksize, stride=stride, dil=dil, norm=norm, activation=activation)
        self.convblock2 = ConvBlock(nn.Conv2d, channels, channels, ksize=ksize, stride=stride, dil=dil, norm=norm, activation=None)

    def forward(self, x):
        identity = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x += identity
        return x


class kp2uv(nn.Module):
    """UNet architecture for converting keypoint image to UV map.

    Same person UV map format as described in https://arxiv.org/pdf/1802.00434.pdf.
    """

    def __init__(self, nf=64):
        super(kp2uv, self).__init__(),
        self.encoder = nn.ModuleList([ConvBlock(nn.Conv2d, 3, nf, ksize=4, stride=2), ConvBlock(nn.Conv2d, nf, nf * 2, ksize=4, stride=2, norm=nn.InstanceNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 2, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=3, stride=1, norm=nn.InstanceNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=3, stride=1, norm=nn.InstanceNorm2d, activation='leaky')])
        self.decoder = nn.ModuleList([ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d), ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.InstanceNorm2d), ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 2, ksize=4, stride=2, norm=nn.InstanceNorm2d), ConvBlock(nn.ConvTranspose2d, nf * 2 * 2, nf, ksize=4, stride=2, norm=nn.InstanceNorm2d), ConvBlock(nn.ConvTranspose2d, nf * 2, nf, ksize=4, stride=2, norm=nn.InstanceNorm2d)])
        self.id_pred = ConvBlock(nn.Conv2d, nf + 3, 25, ksize=3, stride=1, activation='none')
        self.uv_pred = ConvBlock(nn.Conv2d, nf + 3, 2 * 24, ksize=3, stride=1, activation='tanh')

    def forward(self, x):
        """Forward pass through UNet, handling skip connections.
        Parameters:
            x (tensor) - - rendered keypoint image, shape [B, 3, H, W]

        Returns:
            x_id (tensor): part id class probabilities
            x_uv (tensor): uv coordinates for each part id
        """
        skips = [x]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 5:
                skips.append(x)
        for layer in self.decoder:
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)
        x = torch.cat((x, skips.pop()), 1)
        x_id = self.id_pred(x)
        x_uv = self.uv_pred(x)
        return x_id, x_uv


class LayeredNeuralRenderer(nn.Module):
    """Layered Neural Rendering model for video decomposition.

    Consists of neural texture, UNet, upsampling module.
    """

    def __init__(self, nf=64, texture_channels=16, texture_res=16, n_textures=25):
        super(LayeredNeuralRenderer, self).__init__(),
        """Initialize layered neural renderer.

        Parameters:
            nf (int) -- the number of channels in the first/last conv layers
            texture_channels (int) -- the number of channels in the neural texture
            texture_res (int) -- the size of each individual texture map
            n_textures (int) -- the number of individual texture maps
        """
        self.texture = nn.Parameter(torch.randn(1, texture_channels, texture_res, n_textures * texture_res))
        self.encoder = nn.ModuleList([ConvBlock(nn.Conv2d, texture_channels + 1, nf, ksize=4, stride=2), ConvBlock(nn.Conv2d, nf, nf * 2, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky'), ConvBlock(nn.Conv2d, nf * 4, nf * 4, ksize=4, stride=1, dil=2, norm=nn.BatchNorm2d, activation='leaky')])
        self.decoder = nn.ModuleList([ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d), ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 4, ksize=4, stride=2, norm=nn.BatchNorm2d), ConvBlock(nn.ConvTranspose2d, nf * 4 * 2, nf * 2, ksize=4, stride=2, norm=nn.BatchNorm2d), ConvBlock(nn.ConvTranspose2d, nf * 2 * 2, nf, ksize=4, stride=2, norm=nn.BatchNorm2d), ConvBlock(nn.ConvTranspose2d, nf * 2, nf, ksize=4, stride=2, norm=nn.BatchNorm2d)])
        self.final_rgba = ConvBlock(nn.Conv2d, nf, 4, ksize=4, stride=1, activation='tanh')
        upsampling_ic = texture_channels + 5 + nf
        self.upsample_block = nn.Sequential(ConvBlock(nn.Conv2d, upsampling_ic, nf, ksize=3, stride=1, norm=nn.InstanceNorm2d), ResBlock(nf, ksize=3, stride=1, norm=nn.InstanceNorm2d), ResBlock(nf, ksize=3, stride=1, norm=nn.InstanceNorm2d), ResBlock(nf, ksize=3, stride=1, norm=nn.InstanceNorm2d), ConvBlock(nn.Conv2d, nf, 4, ksize=3, stride=1, activation='none'))

    def render(self, x):
        """Pass inputs for a single layer through UNet.

        Parameters:
            x (tensor) - - sampled texture concatenated with person IDs

        Returns RGBA for the input layer and the final feature maps.
        """
        skips = [x]
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < 5:
                skips.append(x)
        for layer in self.decoder:
            x = torch.cat((x, skips.pop()), 1)
            x = layer(x)
        rgba = self.final_rgba(x)
        return rgba, x

    def forward(self, uv_map, id_layers, uv_map_upsampled=None, crop_params=None):
        """Forward pass through layered neural renderer.

        Steps:
        1. Sample from the neural texture using uv_map
        2. Input uv_map and id_layers into UNet
            2a. If doing upsampling, then pass upsampled inputs and results through upsampling module
        3. Composite RGBA outputs.

        Parameters:
            uv_map (tensor) - - UV maps for all layers, with shape [B, (2*L), H, W]
            id_layers (tensor) - - person ID for all layers, with shape [B, L, H, W]
            uv_map_upsampled (tensor) - - upsampled UV maps to input to upsampling module (if None, skip upsampling)
            crop_params
        """
        b_sz = uv_map.shape[0]
        n_layers = uv_map.shape[1] // 2
        texture = self.texture.repeat(b_sz, 1, 1, 1)
        composite = None
        layers = []
        sampled_textures = []
        for i in range(n_layers):
            uv_map_i = uv_map[:, i * 2:(i + 1) * 2, ...]
            uv_map_i = uv_map_i.permute(0, 2, 3, 1)
            sampled_texture = F.grid_sample(texture, uv_map_i, mode='bilinear', padding_mode='zeros')
            inputs = torch.cat([sampled_texture, id_layers[:, i:i + 1]], 1)
            rgba, last_feat = self.render(inputs)
            if uv_map_upsampled is not None:
                uv_map_up_i = uv_map_upsampled[:, i * 2:(i + 1) * 2, ...]
                uv_map_up_i = uv_map_up_i.permute(0, 2, 3, 1)
                sampled_texture_up = F.grid_sample(texture, uv_map_up_i, mode='bilinear', padding_mode='zeros')
                id_layers_up = F.interpolate(id_layers[:, i:i + 1], size=sampled_texture_up.shape[-2:], mode='bilinear')
                inputs_up = torch.cat([sampled_texture_up, id_layers_up], 1)
                upsampled_size = inputs_up.shape[-2:]
                rgba = F.interpolate(rgba, size=upsampled_size, mode='bilinear')
                last_feat = F.interpolate(last_feat, size=upsampled_size, mode='bilinear')
                if crop_params is not None:
                    starty, endy, startx, endx = crop_params
                    rgba = rgba[:, :, starty:endy, startx:endx]
                    last_feat = last_feat[:, :, starty:endy, startx:endx]
                    inputs_up = inputs_up[:, :, starty:endy, startx:endx]
                rgba_residual = self.upsample_block(torch.cat((rgba, inputs_up, last_feat), 1))
                rgba += 0.01 * rgba_residual
                rgba = torch.clamp(rgba, -1, 1)
                sampled_texture = sampled_texture_up
            if composite is None:
                composite = rgba
            else:
                alpha = rgba[:, 3:4] * 0.5 + 0.5
                composite = rgba * alpha + composite * (1.0 - alpha)
            layers.append(rgba)
            sampled_textures.append(sampled_texture)
        outputs = {'reconstruction': composite, 'layers': torch.stack(layers, 1), 'sampled texture': sampled_textures}
        return outputs


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (MaskLoss,
     lambda: ([], {}),
     lambda: ([], {'prediction': torch.rand([4, 4]), 'target': torch.rand([4, 4])}),
     True),
    (ResBlock,
     lambda: ([], {'channels': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     False),
    (kp2uv,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 3, 64, 64])], {}),
     False),
]

class Test_google_retiming(_paritybench_base):
    def test_000(self):
        self._check(*TESTCASES[0])

    def test_001(self):
        self._check(*TESTCASES[1])

    def test_002(self):
        self._check(*TESTCASES[2])

