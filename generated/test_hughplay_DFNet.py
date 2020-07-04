import sys
_module = sys.modules[__name__]
del sys
loss = _module
model = _module
test = _module
utils = _module

from _paritybench_helpers import _mock_config, patch_functional
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
import re, math, string, numpy, torch, torchtext, torchaudio, logging, itertools, numbers, inspect, functools, copy, scipy, types, time, torchvision, enum, random, typing, warnings, abc, collections, uuid
import numpy as np
patch_functional()
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import torch


from torch import nn


from torchvision import models


import torch.nn.functional as F


import numpy as np


class ReconstructionLoss(nn.L1Loss):

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, results, targets):
        loss = 0.0
        for i, (res, target) in enumerate(zip(results, targets)):
            loss += self.l1(res, target)
        return loss / len(results)


class VGGFeature(nn.Module):

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        for para in vgg16.parameters():
            para.requires_grad = False
        self.vgg16_pool_1 = nn.Sequential(*vgg16.features[0:5])
        self.vgg16_pool_2 = nn.Sequential(*vgg16.features[5:10])
        self.vgg16_pool_3 = nn.Sequential(*vgg16.features[10:17])

    def forward(self, x):
        pool_1 = self.vgg16_pool_1(x)
        pool_2 = self.vgg16_pool_2(pool_1)
        pool_3 = self.vgg16_pool_3(pool_2)
        return [pool_1, pool_2, pool_3]


class PerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def forward(self, vgg_results, vgg_targets):
        loss = 0.0
        for i, (vgg_res, vgg_target) in enumerate(zip(vgg_results, vgg_targets)
            ):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(feat_res, feat_target)
        return loss / len(vgg_results)


class StyleLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()

    def gram(self, feature):
        n, c, h, w = feature.shape
        feature = feature.view(n, c, h * w)
        gram_mat = torch.bmm(feature, torch.transpose(feature, 1, 2))
        return gram_mat / (c * h * w)

    def forward(self, vgg_results, vgg_targets):
        loss = 0.0
        for i, (vgg_res, vgg_target) in enumerate(zip(vgg_results, vgg_targets)
            ):
            for feat_res, feat_target in zip(vgg_res, vgg_target):
                loss += self.l1loss(self.gram(feat_res), self.gram(feat_target)
                    )
        return loss / len(vgg_results)


def resize_like(x, target, mode='bilinear'):
    return F.interpolate(x, target.shape[-2:], mode=mode, align_corners=False)


class TotalVariationLoss(nn.Module):

    def __init__(self, c_img=3):
        super().__init__()
        self.c_img = c_img
        kernel = torch.FloatTensor([[0, 1, 0], [1, -2, 0], [0, 0, 0]]).view(
            1, 1, 3, 3)
        kernel = torch.cat([kernel] * c_img, dim=0)
        self.register_buffer('kernel', kernel)

    def gradient(self, x):
        return nn.functional.conv2d(x, self.kernel, stride=1, padding=1,
            groups=self.c_img)

    def forward(self, results, mask):
        loss = 0.0
        for i, res in enumerate(results):
            grad = self.gradient(res) * resize_like(mask, res)
            loss += torch.mean(torch.abs(grad))
        return loss / len(results)


class InpaintLoss(nn.Module):

    def __init__(self, c_img=3, w_l1=6.0, w_percep=0.1, w_style=240.0, w_tv
        =0.1, structure_layers=[0, 1, 2, 3, 4, 5], texture_layers=[0, 1, 2]):
        super().__init__()
        self.l_struct = structure_layers
        self.l_text = texture_layers
        self.w_l1 = w_l1
        self.w_percep = w_percep
        self.w_style = w_style
        self.w_tv = w_tv
        self.reconstruction_loss = ReconstructionLoss()
        self.vgg_feature = VGGFeature()
        self.style_loss = StyleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.tv_loss = TotalVariationLoss(c_img)

    def forward(self, results, target, mask):
        targets = [resize_like(target, res) for res in results]
        loss_struct = 0.0
        loss_text = 0.0
        loss_list = {}
        if len(self.l_struct) > 0:
            struct_r = [results[i] for i in self.l_struct]
            struct_t = [targets[i] for i in self.l_struct]
            loss_struct = self.reconstruction_loss(struct_r, struct_t
                ) * self.w_l1
            loss_list['reconstruction_loss'] = loss_struct.item()
        if len(self.l_text) > 0:
            text_r = [targets[i] for i in self.l_text]
            text_t = [results[i] for i in self.l_text]
            vgg_r = [self.vgg_feature(f) for f in text_r]
            vgg_t = [self.vgg_feature(t) for t in text_t]
            loss_style = self.style_loss(vgg_r, vgg_t) * self.w_style
            loss_percep = self.perceptual_loss(vgg_r, vgg_t) * self.w_percep
            loss_tv = self.tv_loss(text_r, mask) * self.w_tv
            loss_text = loss_style + loss_percep + loss_tv
            loss_list.update({'perceptual_loss': loss_percep.item(),
                'style_loss': loss_style.item(), 'total_variation_loss':
                loss_tv.item()})
        loss_total = loss_struct + loss_text
        return loss_total, loss_list


class Conv2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = self.conv_same_pad(kernel_size, stride)
        if type(padding) is not tuple:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride, padding)
        else:
            self.conv = nn.Sequential(nn.ConstantPad2d(padding * 2, 0), nn.
                Conv2d(in_channels, out_channels, kernel_size, stride, 0))

    def conv_same_pad(self, ksize, stride):
        if (ksize - stride) % 2 == 0:
            return (ksize - stride) // 2
        else:
            left = (ksize - stride) // 2
            right = left + 1
            return left, right

    def forward(self, x):
        return self.conv(x)


class ConvTranspose2dSame(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding, output_padding = self.deconv_same_pad(kernel_size, stride)
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels,
            kernel_size, stride, padding, output_padding)

    def deconv_same_pad(self, ksize, stride):
        pad = (ksize - stride + 1) // 2
        outpad = 2 * pad + stride - ksize
        return pad, outpad

    def forward(self, x):
        return self.trans_conv(x)


class UpBlock(nn.Module):

    def __init__(self, mode='nearest', scale=2, channel=None, kernel_size=4):
        super().__init__()
        self.mode = mode
        if mode == 'deconv':
            self.up = ConvTranspose2dSame(channel, channel, kernel_size,
                stride=scale)
        else:

            def upsample(x):
                return F.interpolate(x, scale_factor=scale, mode=mode)
            self.up = upsample

    def forward(self, x):
        return self.up(x)


def get_activation(name):
    if name == 'relu':
        activation = nn.ReLU()
    elif name == 'elu':
        activation == nn.ELU()
    elif name == 'leaky_relu':
        activation = nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        activation = nn.Tanh()
    elif name == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None
    return activation


def get_norm(name, out_channels):
    if name == 'batch':
        norm = nn.BatchNorm2d(out_channels)
    elif name == 'instance':
        norm = nn.InstanceNorm2d(out_channels)
    else:
        norm = None
    return norm


class EncodeBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
        normalization=None, activation=None):
        super().__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        layers = []
        layers.append(Conv2dSame(self.c_in, self.c_out, kernel_size, stride))
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(get_activation(activation))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class DecodeBlock(nn.Module):

    def __init__(self, c_from_up, c_from_down, c_out, mode='nearest',
        kernel_size=4, scale=2, normalization='batch', activation='relu'):
        super().__init__()
        self.c_from_up = c_from_up
        self.c_from_down = c_from_down
        self.c_in = c_from_up + c_from_down
        self.c_out = c_out
        self.up = UpBlock(mode, scale, c_from_up, kernel_size=scale)
        layers = []
        layers.append(Conv2dSame(self.c_in, self.c_out, kernel_size, stride=1))
        if normalization:
            layers.append(get_norm(normalization, self.c_out))
        if activation:
            layers.append(get_activation(activation))
        self.decode = nn.Sequential(*layers)

    def forward(self, x, concat=None):
        out = self.up(x)
        if self.c_from_down > 0:
            out = torch.cat([out, concat], dim=1)
        out = self.decode(out)
        return out


class BlendBlock(nn.Module):

    def __init__(self, c_in, c_out, ksize_mid=3, norm='batch', act='leaky_relu'
        ):
        super().__init__()
        c_mid = max(c_in // 2, 32)
        self.blend = nn.Sequential(Conv2dSame(c_in, c_mid, 1, 1), get_norm(
            norm, c_mid), get_activation(act), Conv2dSame(c_mid, c_out,
            ksize_mid, 1), get_norm(norm, c_out), get_activation(act),
            Conv2dSame(c_out, c_out, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return self.blend(x)


class FusionBlock(nn.Module):

    def __init__(self, c_feat, c_alpha=1):
        super().__init__()
        c_img = 3
        self.map2img = nn.Sequential(Conv2dSame(c_feat, c_img, 1, 1), nn.
            Sigmoid())
        self.blend = BlendBlock(c_img * 2, c_alpha)

    def forward(self, img_miss, feat_de):
        img_miss = resize_like(img_miss, feat_de)
        raw = self.map2img(feat_de)
        alpha = self.blend(torch.cat([img_miss, raw], dim=1))
        result = alpha * raw + (1 - alpha) * img_miss
        return result, alpha, raw


class DFNet(nn.Module):

    def __init__(self, c_img=3, c_mask=1, c_alpha=3, mode='nearest', norm=
        'batch', act_en='relu', act_de='leaky_relu', en_ksize=[7, 5, 5, 3, 
        3, 3, 3, 3], de_ksize=[3] * 8, blend_layers=[0, 1, 2, 3, 4, 5]):
        super().__init__()
        c_init = c_img + c_mask
        self.n_en = len(en_ksize)
        self.n_de = len(de_ksize)
        assert self.n_en == self.n_de, 'The number layer of Encoder and Decoder must be equal.'
        assert self.n_en >= 1, 'The number layer of Encoder and Decoder must be greater than 1.'
        assert 0 in blend_layers, 'Layer 0 must be blended.'
        self.en = []
        c_in = c_init
        self.en.append(EncodeBlock(c_in, 64, en_ksize[0], 2, None, None))
        for k_en in en_ksize[1:]:
            c_in = self.en[-1].c_out
            c_out = min(c_in * 2, 512)
            self.en.append(EncodeBlock(c_in, c_out, k_en, stride=2,
                normalization=norm, activation=act_en))
        for i, en in enumerate(self.en):
            self.__setattr__('en_{}'.format(i), en)
        self.de = []
        self.fuse = []
        for i, k_de in enumerate(de_ksize):
            c_from_up = self.en[-1].c_out if i == 0 else self.de[-1].c_out
            c_out = c_from_down = self.en[-i - 1].c_in
            layer_idx = self.n_de - i - 1
            self.de.append(DecodeBlock(c_from_up, c_from_down, c_out, mode,
                k_de, scale=2, normalization=norm, activation=act_de))
            if layer_idx in blend_layers:
                self.fuse.append(FusionBlock(c_out, c_alpha))
            else:
                self.fuse.append(None)
        for i, de in enumerate(self.de[::-1]):
            self.__setattr__('de_{}'.format(i), de)
        for i, fuse in enumerate(self.fuse[::-1]):
            if fuse:
                self.__setattr__('fuse_{}'.format(i), fuse)

    def forward(self, img_miss, mask):
        out = torch.cat([img_miss, mask], dim=1)
        out_en = [out]
        for encode in self.en:
            out = encode(out)
            out_en.append(out)
        results = []
        alphas = []
        raws = []
        for i, (decode, fuse) in enumerate(zip(self.de, self.fuse)):
            out = decode(out, out_en[-i - 2])
            if fuse:
                result, alpha, raw = fuse(img_miss, out)
                results.append(result)
                alphas.append(alpha)
                raws.append(raw)
        return results[::-1], alphas[::-1], raws[::-1]


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hughplay_DFNet(_paritybench_base):
    pass
    def test_000(self):
        self._check(BlendBlock(*[], **{'c_in': 4, 'c_out': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(Conv2dSame(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(EncodeBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(FusionBlock(*[], **{'c_feat': 4}), [torch.rand([4, 3, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(PerceptualLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(ReconstructionLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_006(self):
        self._check(UpBlock(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(VGGFeature(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

