import sys
_module = sys.modules[__name__]
del sys
data = _module
dataset_dncnn = _module
dataset_dnpatch = _module
dataset_dpsr = _module
dataset_fdncnn = _module
dataset_ffdnet = _module
dataset_l = _module
dataset_plain = _module
dataset_plainpatch = _module
dataset_sr = _module
dataset_srmd = _module
select_dataset = _module
main_test_dncnn = _module
main_test_dncnn3_deblocking = _module
main_test_dpsr = _module
main_test_fdncnn = _module
main_test_ffdnet = _module
main_test_imdn = _module
main_test_msrresnet = _module
main_test_rrdb = _module
main_test_srmd = _module
main_train_dncnn = _module
main_train_dpsr = _module
main_train_fdncnn = _module
main_train_ffdnet = _module
main_train_imdn = _module
main_train_msrresnet_gan = _module
main_train_msrresnet_psnr = _module
main_train_rrdb_psnr = _module
main_train_srmd = _module
basicblock = _module
loss = _module
loss_ssim = _module
model_base = _module
model_gan = _module
model_plain = _module
model_plain2 = _module
network_discriminator = _module
network_dncnn = _module
network_dpsr = _module
network_feature = _module
network_ffdnet = _module
network_imdn = _module
network_msrresnet = _module
network_rrdb = _module
network_srmd = _module
network_usrnet = _module
select_model = _module
select_network = _module
utils_bnorm = _module
utils_deblur = _module
utils_image = _module
utils_logger = _module
utils_matconvnet = _module
utils_model = _module
utils_option = _module
utils_params = _module
utils_regularizers = _module
utils_sisr = _module

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


from collections import OrderedDict


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import numpy as np


from math import exp


from torch.optim import lr_scheduler


from torch.optim import Adam


from torch.nn.parallel import DataParallel


from torch.nn.utils import spectral_norm


import math


import functools


import torch.nn.init as init


from torch.nn import init


import re


def pixel_unshuffle(input, upscale_factor):
    """Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    batch_size, channels, in_height, in_width = input.size()
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    input_view = input.contiguous().view(batch_size, channels, out_height,
        upscale_factor, out_width, upscale_factor)
    channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return unshuffle_out.view(batch_size, channels, out_height, out_width)


class PixelUnShuffle(nn.Module):
    """Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """

    def __init__(self, upscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)


class ConditionalBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1,
            self.num_features, 1, 1)
        return out


class ConcatBlock(nn.Module):

    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), dim=1)
        return output

    def __repr__(self):
        return self.sub.__repr__() + 'concat'


class ShortcutBlock(nn.Module):

    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=
    1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=
                out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                out_channels=out_channels, kernel_size=kernel_size, stride=
                stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=0.0001,
                affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
                )
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)


class ResBlock(nn.Module):

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3,
        stride=1, padding=1, bias=True, mode='CRC', negative_slope=0.2):
        super(ResBlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride,
            padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


class IMDBlock(nn.Module):
    """
    @inproceedings{hui2019lightweight,
      title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
      author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
      booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
      pages={2024--2032},
      year={2019}
    }
    @inproceedings{zhang2019aim,
      title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
      author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
      booktitle={IEEE International Conference on Computer Vision Workshops},
      year={2019}
    }
    """

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3,
        stride=1, padding=1, bias=True, mode='CL', d_rate=0.25,
        negative_slope=0.05):
        super(IMDBlock, self).__init__()
        self.d_nc = int(in_channels * d_rate)
        self.r_nc = int(in_channels - self.d_nc)
        assert mode[0] == 'C', 'convolutional layer first'
        self.conv1 = conv(in_channels, in_channels, kernel_size, stride,
            padding, bias, mode, negative_slope)
        self.conv2 = conv(self.r_nc, in_channels, kernel_size, stride,
            padding, bias, mode, negative_slope)
        self.conv3 = conv(self.r_nc, in_channels, kernel_size, stride,
            padding, bias, mode, negative_slope)
        self.conv4 = conv(self.r_nc, self.d_nc, kernel_size, stride,
            padding, bias, mode[0], negative_slope)
        self.conv1x1 = conv(self.d_nc * 4, out_channels, kernel_size=1,
            stride=1, padding=0, bias=bias, mode=mode[0], negative_slope=
            negative_slope)

    def forward(self, x):
        d1, r1 = torch.split(self.conv1(x), (self.d_nc, self.r_nc), dim=1)
        d2, r2 = torch.split(self.conv2(r1), (self.d_nc, self.r_nc), dim=1)
        d3, r3 = torch.split(self.conv3(r2), (self.d_nc, self.r_nc), dim=1)
        d4 = self.conv4(r3)
        res = self.conv1x1(torch.cat((d1, d2, d3, d4), dim=1))
        return x + res


class CALayer(nn.Module):

    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc = nn.Sequential(nn.Conv2d(channel, channel //
            reduction, 1, padding=0, bias=True), nn.ReLU(inplace=True), nn.
            Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_fc(y)
        return x * y


class RCABlock(nn.Module):

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3,
        stride=1, padding=1, bias=True, mode='CRC', reduction=16,
        negative_slope=0.2):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride,
            padding, bias, mode, negative_slope)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


class RCAGroup(nn.Module):

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3,
        stride=1, padding=1, bias=True, mode='CRC', reduction=16, nb=12,
        negative_slope=0.2):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        RG = [RCABlock(in_channels, out_channels, kernel_size, stride,
            padding, bias, mode, reduction, negative_slope) for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)

    def forward(self, x):
        res = self.rg(x)
        return res + x


class ResidualDenseBlock_5C(nn.Module):

    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1,
        bias=True, mode='CR', negative_slope=0.2):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode,
            negative_slope)
        self.conv2 = conv(nc + gc, gc, kernel_size, stride, padding, bias,
            mode, negative_slope)
        self.conv3 = conv(nc + 2 * gc, gc, kernel_size, stride, padding,
            bias, mode, negative_slope)
        self.conv4 = conv(nc + 3 * gc, gc, kernel_size, stride, padding,
            bias, mode, negative_slope)
        self.conv5 = conv(nc + 4 * gc, nc, kernel_size, stride, padding,
            bias, mode[:-1], negative_slope)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


class RRDB(nn.Module):

    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1,
        bias=True, mode='CR', negative_slope=0.2):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
            padding, bias, mode, negative_slope)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
            padding, bias, mode, negative_slope)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
            padding, bias, mode, negative_slope)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3,
    stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'
        ], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode
        [0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride,
        padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3,
    stride=1, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3'
        ], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode
        [0], negative_slope=negative_slope)
    pool_tail = conv(in_channels, out_channels, kernel_size, stride,
        padding, bias, mode=mode[1:], negative_slope=negative_slope)
    return sequential(pool, pool_tail)


def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2,
    stride=2, padding=0, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'
        ], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding,
        bias, mode, negative_slope)
    return down1


class NonLocalBlock2D(nn.Module):

    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True,
        act_mode='B', downsample=False, downsample_mode='maxpool',
        negative_slope=0.2):
        super(NonLocalBlock2D, self).__init__()
        inter_nc = nc // 2
        self.inter_nc = inter_nc
        self.W = conv(inter_nc, nc, kernel_size, stride, padding, bias,
            mode='C' + act_mode)
        self.theta = conv(nc, inter_nc, kernel_size, stride, padding, bias,
            mode='C')
        if downsample:
            if downsample_mode == 'avgpool':
                downsample_block = downsample_avgpool
            elif downsample_mode == 'maxpool':
                downsample_block = downsample_maxpool
            elif downsample_mode == 'strideconv':
                downsample_block = downsample_strideconv
            else:
                raise NotImplementedError('downsample mode [{:s}] is not found'
                    .format(downsample_mode))
            self.phi = downsample_block(nc, inter_nc, kernel_size, stride,
                padding, bias, mode='2')
            self.g = downsample_block(nc, inter_nc, kernel_size, stride,
                padding, bias, mode='2')
        else:
            self.phi = conv(nc, inter_nc, kernel_size, stride, padding,
                bias, mode='C')
            self.g = conv(nc, inter_nc, kernel_size, stride, padding, bias,
                mode='C')

    def forward(self, x):
        """
        :param x: (b, c, t, h, w)
        :return:
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_nc, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_nc, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_nc, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_nc, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class GANLoss(nn.Module):

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                return -1 * input.mean() if target else input.mean()
            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format
                (self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class TVLoss(nn.Module):

    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w
            ) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GradientPenaltyLoss(nn.Module):

    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=
            interp, grad_outputs=grad_outputs, create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)
        loss = ((grad_interp_norm - 1) ** 2).mean()
        return loss


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2,
        groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2,
        groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2,
        groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq +
        C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * 
        sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0
        )
    window = Variable(_2D_window.expand(channel, 1, window_size,
        window_size).contiguous())
    return window


class SSIMLoss(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type(
            ) == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.
            size_average)


class Discriminator_VGG_96(nn.Module):

    def __init__(self, in_nc=3, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_96, self).__init__()
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C' +
            ac_type)
        conv2 = B.conv(base_nc, base_nc * 2, kernel_size=3, stride=1, mode=
            'C' + ac_type)
        conv3 = B.conv(base_nc * 2, base_nc * 2, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv4 = B.conv(base_nc * 2, base_nc * 4, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv5 = B.conv(base_nc * 4, base_nc * 4, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv6 = B.conv(base_nc * 4, base_nc * 8, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv7 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv8 = B.conv(base_nc * 8, base_nc * 8, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv9 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2,
            mode='C' + ac_type)
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
            conv5, conv6, conv7, conv8, conv9)
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100), nn.
            LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_128(nn.Module):

    def __init__(self, in_nc=3, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_128, self).__init__()
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C' +
            ac_type)
        conv2 = B.conv(base_nc, base_nc * 2, kernel_size=3, stride=1, mode=
            'C' + ac_type)
        conv3 = B.conv(base_nc * 2, base_nc * 2, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv4 = B.conv(base_nc * 2, base_nc * 4, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv5 = B.conv(base_nc * 4, base_nc * 4, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv6 = B.conv(base_nc * 4, base_nc * 8, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv7 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv8 = B.conv(base_nc * 8, base_nc * 8, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv9 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2,
            mode='C' + ac_type)
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
            conv5, conv6, conv7, conv8, conv9)
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100), nn.
            LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):

    def __init__(self, in_nc=3, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_192, self).__init__()
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C' +
            ac_type)
        conv2 = B.conv(base_nc, base_nc * 2, kernel_size=3, stride=1, mode=
            'C' + ac_type)
        conv3 = B.conv(base_nc * 2, base_nc * 2, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv4 = B.conv(base_nc * 2, base_nc * 4, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv5 = B.conv(base_nc * 4, base_nc * 4, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv6 = B.conv(base_nc * 4, base_nc * 8, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv7 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv8 = B.conv(base_nc * 8, base_nc * 8, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv9 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2,
            mode='C' + ac_type)
        conv10 = B.conv(base_nc * 8, base_nc * 8, kernel_size=3, stride=1,
            mode='C' + ac_type)
        conv11 = B.conv(base_nc * 8, base_nc * 8, kernel_size=4, stride=2,
            mode='C' + ac_type)
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
            conv5, conv6, conv7, conv8, conv9, conv10, conv11)
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100), nn.
            LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_128_SN(nn.Module):

    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.conv0 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        self.conv8 = spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        self.linear0 = spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class DnCNN(nn.Module):

    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        m_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in
            range(nb - 2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x - n


class FDnCNN(nn.Module):

    def __init__(self, in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        """
        super(FDnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        m_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in
            range(nb - 2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)
        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


class MSRResNet_prior(nn.Module):

    def __init__(self, in_nc=4, out_nc=3, nc=96, nb=16, upscale=4, act_mode
        ='R', upsample_mode='upconv'):
        super(MSRResNet_prior, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in
            range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.
                format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in
                range(n_upscale)]
        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*
            m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


class SRResNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode
        ='R', upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in
            range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.
                format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in
                range(n_upscale)]
        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*
            m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


class VGGFeatureExtractor(nn.Module):

    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
        device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(
                device)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(
                device)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:
            feature_layer + 1])
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class FFDNet(nn.Module):

    def __init__(self, in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        # ------------------------------------
        """
        super(FFDNet, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        sf = 2
        self.m_down = B.PixelUnShuffle(upscale_factor=sf)
        m_head = B.conv(in_nc * sf * sf + 1, nc, mode='C' + act_mode[-1],
            bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in
            range(nb - 2)]
        m_tail = B.conv(nc, out_nc * sf * sf, mode='C', bias=bias)
        self.model = B.sequential(m_head, *m_body, m_tail)
        self.m_up = nn.PixelShuffle(upscale_factor=sf)

    def forward(self, x, sigma):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 2) * 2 - h)
        paddingRight = int(np.ceil(w / 2) * 2 - w)
        x = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        x = self.m_down(x)
        m = sigma.repeat(1, 1, x.size()[-2], x.size()[-1])
        x = torch.cat((x, m), 1)
        x = self.model(x)
        x = self.m_up(x)
        x = x[(...), :h, :w]
        return x


class IMDN(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode=
        'L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock(nc, nc, mode='C' + act_mode, negative_slope=
            negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.
                format(upsample_mode))
        m_uper = upsample_block(nc, out_nc, mode=str(upscale))
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*
            m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x


class MSRResNet0(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode
        ='R', upsample_mode='upconv'):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(MSRResNet0, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in
            range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.
                format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in
                range(n_upscale)]
        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*
            m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class MSRResNet1(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=4, act_mode
        ='R', upsample_mode='upconv'):
        super(MSRResNet1, self).__init__()
        self.upscale = upscale
        self.conv_first = nn.Conv2d(in_nc, nc, 3, 1, 1, bias=True)
        basic_block = functools.partial(ResidualBlock_noBN, nc=nc)
        self.recon_trunk = make_layer(basic_block, nb)
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nc, nc * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nc, nc * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nc, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        initialize_weights([self.conv_first, self.upconv1, self.HRconv,
            self.conv_last], 0.1)
        if self.upscale == 4:
            initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)
        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear',
            align_corners=False)
        out += base
        return out


class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nc=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


class RRDB(nn.Module):
    """
    gc: number of growth channels
    nb: number of RRDB
    """

    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=23, gc=32, upscale=4,
        act_mode='L', upsample_mode='upconv'):
        super(RRDB, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.RRDB(nc, gc=32, mode='C' + act_mode) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.
                format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in
                range(n_upscale)]
        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)
        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*
            m_body)), *m_uper, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


class SRMD(nn.Module):

    def __init__(self, in_nc=19, out_nc=3, nc=128, nb=12, upscale=4,
        act_mode='R', upsample_mode='pixelshuffle'):
        """
        # ------------------------------------
        in_nc: channel number of input, default: 3+15
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        upscale: scale factor
        act_mode: batch norm + activation function; 'BR' means BN+ReLU
        upsample_mode: default 'pixelshuffle' = conv + pixelshuffle
        # ------------------------------------
        """
        super(SRMD, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.
                format(upsample_mode))
        m_head = B.conv(in_nc, nc, mode='C' + act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C' + act_mode, bias=bias) for _ in
            range(nb - 2)]
        m_tail = upsample_block(nc, out_nc, mode=str(upscale), bias=bias)
        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        x = self.model(x)
        return x


class ResUNet(nn.Module):

    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2,
        act_mode='R', downsample_mode='strideconv', upsample_mode=
        'convtranspose'):
        super(ResUNet, self).__init__()
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'
                .format(downsample_mode))
        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False,
            mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False,
            mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False,
            mode='C' + act_mode + 'C') for _ in range(nb)],
            downsample_block(nc[2], nc[3], bias=False, mode='2'))
        self.m_body = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False,
            mode='C' + act_mode + 'C') for _ in range(nb)])
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.
                format(upsample_mode))
        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False,
            mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C' +
            act_mode + 'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False,
            mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C' +
            act_mode + 'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False,
            mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C' +
            act_mode + 'C') for _ in range(nb)])
        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        x = x[(...), :h, :w]
        return x


def csum(x, y):
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cmul(t1, t2):
    """complex multiplication

    Args:
        t1: NxCxHxWx2, complex tensor
        t2: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    """
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + 
        imag1 * real2], dim=-1)


def cdiv(x, y):
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c ** 2 + d ** 2
    return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)


def splits(a, sf):
    """split a into sfxsf distinct blocks

    Args:
        a: NxCxWxHx2
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x2x(sf^2)
    """
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


class DataNet(nn.Module):

    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):
        FR = FBFy + torch.rfft(alpha * x, 2, onesided=False)
        x1 = cmul(FB, FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = cdiv(FBR, csum(invW, alpha))
        FCBinvWBR = cmul(FBC, invWBR.repeat(1, 1, sf, sf, 1))
        FX = (FR - FCBinvWBR) / alpha.unsqueeze(-1)
        Xest = torch.irfft(FX, 2, onesided=False)
        return Xest


class HyPaNet(nn.Module):

    def __init__(self, in_nc=2, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_nc, channel, 1, padding=0,
            bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel, channel, 
            1, padding=0, bias=True), nn.ReLU(inplace=True), nn.Conv2d(
            channel, out_nc, 1, padding=0, bias=True), nn.Softplus())

    def forward(self, x):
        x = self.mlp(x) + 1e-06
        return x


def p2o(psf, shape):
    """
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[(...), :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(
        torch.tensor(psf.shape).type_as(psf)))
    otf[..., 1][torch.abs(otf[..., 1]) < n_ops * 2.22e-16] = torch.tensor(0
        ).type_as(psf)
    return otf


def upsample(x, sf=3):
    """s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    """
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)
        ).type_as(x)
    z[(...), st::sf, st::sf].copy_(x)
    return z


def cconj(t, inplace=False):
    """complex's conjugation

    Args:
        t: NxCxHxWx2

    Returns:
        output: NxCxHxWx2
    """
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def r2c(x):
    return torch.stack([x, torch.zeros_like(x)], -1)


def cabs2(x):
    return x[..., 0] ** 2 + x[..., 1] ** 2


class USRNet(nn.Module):

    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 
        256, 512], nb=2, act_mode='R', downsample_mode='strideconv',
        upsample_mode='convtranspose'):
        super(USRNet, self).__init__()
        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode
            =act_mode, downsample_mode=downsample_mode, upsample_mode=
            upsample_mode)
        self.h = HyPaNet(in_nc=2, out_nc=n_iter * 2, channel=h_nc)
        self.n = n_iter

    def forward(self, x, k, sf, sigma):
        """
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        """
        w, h = x.shape[-2:]
        FB = p2o(k, (w * sf, h * sf))
        FBC = cconj(FB, inplace=False)
        F2B = r2c(cabs2(FB))
        STy = upsample(x, sf=sf)
        FBFy = cmul(FBC, torch.rfft(STy, 2, onesided=False))
        x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')
        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).
            expand_as(sigma)), dim=1))
        for i in range(self.n):
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i + 1, (...)], sf)
            x = self.p(torch.cat((x, ab[:, i + self.n:i + self.n + 1, (...)
                ].repeat(1, 1, x.size(2), x.size(3))), dim=1))
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_cszn_KAIR(_paritybench_base):
    pass
    def test_000(self):
        self._check(CALayer(*[], **{}), [torch.rand([4, 64, 4, 4])], {})

    def test_001(self):
        self._check(ConditionalBatchNorm2d(*[], **{'num_features': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_002(self):
        self._check(HyPaNet(*[], **{}), [torch.rand([4, 2, 64, 64])], {})

    def test_003(self):
        self._check(IMDBlock(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(MSRResNet1(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_005(self):
        self._check(NonLocalBlock2D(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_006(self):
        self._check(RCABlock(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_007(self):
        self._check(RCAGroup(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_008(self):
        self._check(ResBlock(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_009(self):
        self._check(ResidualBlock_noBN(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_010(self):
        self._check(ResidualDenseBlock_5C(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    @_fails_compile()
    def test_011(self):
        self._check(SSIMLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(TVLoss(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

