import sys
_module = sys.modules[__name__]
del sys
Composition_code = _module
dataloader = _module
data_generator = _module
image_file = _module
prefetcher = _module
demo = _module
main = _module
networks = _module
decoders = _module
res_gca_dec = _module
res_shortcut_dec = _module
resnet_dec = _module
encoders = _module
res_gca_enc = _module
res_shortcut_enc = _module
resnet_enc = _module
generators = _module
ops = _module
tester = _module
trainer = _module
utils = _module
config = _module
evaluate = _module
logger = _module
util = _module

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


import math


import random


import logging


import numpy as np


import torch


from torch.utils.data import Dataset


from torch.nn import functional as F


import torch.nn as nn


import torch.nn.functional as F


from torch import nn


from torch.nn import Parameter


from torch.autograd import Variable


import torch.nn.utils as nn_utils


import torch.backends.cudnn as cudnn


from torch.nn import SyncBatchNorm


import torch.optim.lr_scheduler as lr_scheduler


from torch.nn.parallel import DistributedDataParallel


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
        padding=2, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None,
        norm_layer=None, large_kernel=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride
        conv = conv5x5 if large_kernel else conv3x3
        if self.stride > 1:
            self.conv1 = SpectralNorm(nn.ConvTranspose2d(inplanes, inplanes,
                kernel_size=4, stride=2, padding=1, bias=False))
        else:
            self.conv1 = SpectralNorm(conv(inplanes, inplanes))
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = SpectralNorm(conv(inplanes, planes))
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.activation(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class ResNet_D_Dec(nn.Module):

    def __init__(self, block, layers, norm_layer=None, large_kernel=False,
        late_downsample=False):
        super(ResNet_D_Dec, self).__init__()
        self.logger = logging.getLogger('Logger')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3
        self.inplanes = 512 if layers[0] > 0 else 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32
        self.conv1 = SpectralNorm(nn.ConvTranspose2d(self.midplanes, 32,
            kernel_size=4, stride=2, padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=self.kernel_size, stride=
            1, padding=self.kernel_size // 2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3],
            stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight_bar'):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
        self.logger.debug(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                SpectralNorm(conv1x1(self.inplanes, planes * block.
                expansion)), norm_layer(planes * block.expansion))
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(SpectralNorm(conv1x1(self.inplanes, 
                planes * block.expansion)), norm_layer(planes * block.
                expansion))
        layers = [block(self.inplanes, planes, stride, upsample, norm_layer,
            self.large_kernel)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=
                norm_layer, large_kernel=self.large_kernel))
        return nn.Sequential(*layers)

    def forward(self, x, mid_fea):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        alpha = (self.tanh(x) + 1.0) / 2.0
        return alpha, None


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = SpectralNorm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = SpectralNorm(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.activation(out)
        return out


class Generator(nn.Module):

    def __init__(self, encoder, decoder):
        super(Generator, self).__init__()
        if encoder not in encoders.__all__:
            raise NotImplementedError('Unknown Encoder {}'.format(encoder))
        self.encoder = encoders.__dict__[encoder]()
        if decoder not in decoders.__all__:
            raise NotImplementedError('Unknown Decoder {}'.format(decoder))
        self.decoder = decoders.__dict__[decoder]()

    def forward(self, image, trimap):
        inp = torch.cat((image, trimap), dim=1)
        embedding, mid_fea = self.encoder(inp)
        alpha, info_dict = self.decoder(embedding, mid_fea)
        return alpha, info_dict


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    and add _noupdate_u_v() for evaluation
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data),
                u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w = getattr(self.module, self.name + '_bar')
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + '_u', u)
        self.module.register_parameter(self.name + '_v', v)
        self.module.register_parameter(self.name + '_bar', w_bar)

    def forward(self, *args):
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)


class GuidedCxtAtten(nn.Module):

    def __init__(self, out_channels, guidance_channels, rate=2):
        super(GuidedCxtAtten, self).__init__()
        self.rate = rate
        self.padding = nn.ReflectionPad2d(1)
        self.up_sample = nn.Upsample(scale_factor=self.rate, mode='nearest')
        self.guidance_conv = nn.Conv2d(in_channels=guidance_channels,
            out_channels=guidance_channels // 2, kernel_size=1, stride=1,
            padding=0)
        self.W = nn.Sequential(nn.Conv2d(in_channels=out_channels,
            out_channels=out_channels, kernel_size=1, stride=1, padding=0,
            bias=False), nn.BatchNorm2d(out_channels))
        nn.init.xavier_uniform_(self.guidance_conv.weight)
        nn.init.constant_(self.guidance_conv.bias, 0)
        nn.init.xavier_uniform_(self.W[0].weight)
        nn.init.constant_(self.W[1].weight, 0.001)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, f, alpha, unknown=None, ksize=3, stride=1, fuse_k=3,
        softmax_scale=1.0, training=True):
        f = self.guidance_conv(f)
        raw_int_fs = list(f.size())
        raw_int_alpha = list(alpha.size())
        kernel = 2 * self.rate
        alpha_w = self.extract_patches(alpha, kernel=kernel, stride=self.rate)
        alpha_w = alpha_w.permute(0, 2, 3, 4, 5, 1)
        alpha_w = alpha_w.contiguous().view(raw_int_alpha[0], raw_int_alpha
            [2] // self.rate, raw_int_alpha[3] // self.rate, -1)
        alpha_w = alpha_w.contiguous().view(raw_int_alpha[0], -1, kernel,
            kernel, raw_int_alpha[1])
        alpha_w = alpha_w.permute(0, 1, 4, 2, 3)
        f = F.interpolate(f, scale_factor=1 / self.rate, mode='nearest')
        fs = f.size()
        f_groups = torch.split(f, 1, dim=0)
        int_fs = list(fs)
        w = self.extract_patches(f)
        w = w.permute(0, 2, 3, 4, 5, 1)
        w = w.contiguous().view(raw_int_fs[0], raw_int_fs[2] // self.rate, 
            raw_int_fs[3] // self.rate, -1)
        w = w.contiguous().view(raw_int_fs[0], -1, ksize, ksize, raw_int_fs[1])
        w = w.permute(0, 1, 4, 2, 3)
        if unknown is not None:
            unknown = unknown.clone()
            unknown = F.interpolate(unknown, scale_factor=1 / self.rate,
                mode='nearest')
            assert unknown.size(2) == f.size(2
                ), 'mask should have same size as f at dim 2,3'
            unknown_mean = unknown.mean(dim=[2, 3])
            known_mean = 1 - unknown_mean
            unknown_scale = torch.clamp(torch.sqrt(unknown_mean /
                known_mean), 0.1, 10)
            known_scale = torch.clamp(torch.sqrt(known_mean / unknown_mean),
                0.1, 10)
            softmax_scale = torch.cat([unknown_scale, known_scale], dim=1)
        else:
            unknown = torch.ones([fs[0], 1, fs[2], fs[3]])
            softmax_scale = torch.FloatTensor([softmax_scale, softmax_scale]
                ).view(1, 2).repeat(fs[0], 1)
        m = self.extract_patches(unknown)
        m = m.permute(0, 2, 3, 4, 5, 1)
        m = m.contiguous().view(raw_int_fs[0], raw_int_fs[2] // self.rate, 
            raw_int_fs[3] // self.rate, -1)
        m = m.contiguous().view(raw_int_fs[0], -1, ksize, ksize)
        m = self.reduce_mean(m)
        mm = m.gt(0.0).float()
        self_mask = F.one_hot(torch.arange(fs[2] * fs[3]).view(fs[2], fs[3]
            ).contiguous().long(), num_classes=int_fs[2] * int_fs[3])
        self_mask = self_mask.permute(2, 0, 1).view(1, fs[2] * fs[3], fs[2],
            fs[3]).float() * -10000.0
        w_groups = torch.split(w, 1, dim=0)
        alpha_w_groups = torch.split(alpha_w, 1, dim=0)
        mm_groups = torch.split(mm, 1, dim=0)
        scale_group = torch.split(softmax_scale, 1, dim=0)
        y = []
        offsets = []
        k = fuse_k
        y_test = []
        for xi, wi, alpha_wi, mmi, scale in zip(f_groups, w_groups,
            alpha_w_groups, mm_groups, scale_group):
            wi = wi[0]
            escape_NaN = Variable(torch.FloatTensor([0.0001]))
            wi_normed = wi / torch.max(self.l2_norm(wi), escape_NaN)
            xi = F.pad(xi, (1, 1, 1, 1), mode='reflect')
            yi = F.conv2d(xi, wi_normed, stride=1, padding=0)
            y_test.append(yi)
            yi = yi.permute(0, 2, 3, 1)
            yi = yi.contiguous().view(1, fs[2], fs[3], fs[2] * fs[3])
            yi = yi.permute(0, 3, 1, 2)
            yi = yi * (scale[0, 0] * mmi.gt(0.0).float() + scale[0, 1] *
                mmi.le(0.0).float())
            yi = yi + self_mask * mmi
            yi = F.softmax(yi, dim=1)
            _, offset = torch.max(yi, dim=1)
            offset = torch.stack([offset // fs[3], offset % fs[3]], dim=1)
            wi_center = alpha_wi[0]
            if self.rate == 1:
                left = kernel // 2
                right = (kernel - 1) // 2
                yi = F.pad(yi, (left, right, left, right), mode='reflect')
                wi_center = wi_center.permute(1, 0, 2, 3)
                yi = F.conv2d(yi, wi_center, padding=0) / 4.0
            else:
                yi = F.conv_transpose2d(yi, wi_center, stride=self.rate,
                    padding=1) / 4.0
            y.append(yi)
            offsets.append(offset)
        y = torch.cat(y, dim=0)
        y.contiguous().view(raw_int_alpha)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view([int_fs[0]] + [2] + int_fs[2:])
        offsets = offsets - torch.Tensor([fs[2] // 2, fs[3] // 2]).view(1, 
            2, 1, 1).long()
        y = self.W(y) + alpha
        return y, (offsets, softmax_scale)

    @staticmethod
    def extract_patches(x, kernel=3, stride=1):
        left = (kernel - stride + 1) // 2
        right = (kernel - stride) // 2
        x = F.pad(x, (left, right, left, right), mode='reflect')
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        return all_patches

    @staticmethod
    def reduce_mean(x):
        for i in range(4):
            if i <= 1:
                continue
            x = torch.mean(x, dim=i, keepdim=True)
        return x

    @staticmethod
    def l2_norm(x):

        def reduce_sum(x):
            for i in range(4):
                if i == 0:
                    continue
                x = torch.sum(x, dim=i, keepdim=True)
            return x
        x = x ** 2
        x = reduce_sum(x)
        return torch.sqrt(x)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Yaoyi_Li_GCA_Matting(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(GuidedCxtAtten(*[], **{'out_channels': 4, 'guidance_channels': 4}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

