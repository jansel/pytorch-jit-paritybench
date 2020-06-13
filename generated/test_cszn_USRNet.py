import sys
_module = sys.modules[__name__]
del sys
main_test_bicubic = _module
main_test_realapplication = _module
main_test_table1 = _module
basicblock = _module
network_usrnet = _module
utils_deblur = _module
utils_image = _module
utils_logger = _module
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


import numpy as np


import scipy


import scipy.stats as ss


import scipy.io as io


from scipy import ndimage


from scipy.interpolate import interp2d


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


class FFTBlock(nn.Module):

    def __init__(self, channel=64):
        super(FFTBlock, self).__init__()
        self.conv_fc = nn.Sequential(nn.Conv2d(1, channel, 1, padding=0,
            bias=True), nn.ReLU(inplace=True), nn.Conv2d(channel, 1, 1,
            padding=0, bias=True), nn.Softplus())

    def forward(self, x, u, d, sigma):
        rho = self.conv_fc(sigma)
        x = torch.irfft(self.divcomplex(u + rho.unsqueeze(-1) * torch.rfft(
            x, 2, onesided=False), d + self.real2complex(rho)), 2, onesided
            =False)
        return x

    def divcomplex(self, x, y):
        a = x[..., 0]
        b = x[..., 1]
        c = y[..., 0]
        d = y[..., 1]
        cd2 = c ** 2 + d ** 2
        return torch.stack([(a * c + b * d) / cd2, (b * c - a * d) / cd2], -1)

    def real2complex(self, x):
        return torch.stack([x, torch.zeros(x.shape).type_as(x)], -1)


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
    1, bias=True, mode='CBR'):
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
            L.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
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
        stride=1, padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride,
            padding, bias, mode)

    def forward(self, x):
        res = self.res(x)
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
        stride=1, padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(in_channels, out_channels, kernel_size, stride,
            padding, bias, mode)
        self.ca = CALayer(out_channels, reduction)

    def forward(self, x):
        res = self.res(x)
        res = self.ca(res)
        return res + x


class RCAGroup(nn.Module):

    def __init__(self, in_channels=64, out_channels=64, kernel_size=3,
        stride=1, padding=1, bias=True, mode='CRC', reduction=16, nb=12):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]
        RG = [RCABlock(in_channels, out_channels, kernel_size, stride,
            padding, bias, mode, reduction) for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))
        self.rg = nn.Sequential(*RG)

    def forward(self, x):
        res = self.rg(x)
        return res + x


class ResidualDenseBlock_5C(nn.Module):

    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1,
        bias=True, mode='CR'):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = conv(nc, gc, kernel_size, stride, padding, bias, mode)
        self.conv2 = conv(nc + gc, gc, kernel_size, stride, padding, bias, mode
            )
        self.conv3 = conv(nc + 2 * gc, gc, kernel_size, stride, padding,
            bias, mode)
        self.conv4 = conv(nc + 3 * gc, gc, kernel_size, stride, padding,
            bias, mode)
        self.conv5 = conv(nc + 4 * gc, nc, kernel_size, stride, padding,
            bias, mode[:-1])

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul_(0.2) + x


class RRDB(nn.Module):

    def __init__(self, nc=64, gc=32, kernel_size=3, stride=1, padding=1,
        bias=True, mode='CR'):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
            padding, bias, mode)
        self.RDB2 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
            padding, bias, mode)
        self.RDB3 = ResidualDenseBlock_5C(nc, gc, kernel_size, stride,
            padding, bias, mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul_(0.2) + x


def downsample_avgpool(in_channels=64, out_channels=64, kernel_size=3,
    stride=1, padding=1, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3'
        ], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'AC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride,
        padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


def downsample_maxpool(in_channels=64, out_channels=64, kernel_size=3,
    stride=1, padding=0, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3'
        ], 'mode examples: 2, 2R, 2BR, 3, ..., 3BR.'
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], 'MC')
    pool = conv(kernel_size=kernel_size_pool, stride=stride_pool, mode=mode[0])
    pool_tail = conv(in_channels, out_channels, kernel_size, stride,
        padding, bias, mode=mode[1:])
    return sequential(pool, pool_tail)


def downsample_strideconv(in_channels=64, out_channels=64, kernel_size=2,
    stride=2, padding=0, bias=True, mode='2R'):
    assert len(mode) < 4 and mode[0] in ['2', '3', '4'
        ], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], 'C')
    down1 = conv(in_channels, out_channels, kernel_size, stride, padding,
        bias, mode)
    return down1


class NonLocalBlock2D(nn.Module):

    def __init__(self, nc=64, kernel_size=1, stride=1, padding=0, bias=True,
        act_mode='B', downsample=False, downsample_mode='maxpool'):
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
    """
    complex multiplication
    t1: NxCxHxWx2
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
    """
    a: tensor NxCxWxHx2
    sf: scale factor
    out: tensor NxCx(W/sf)x(H/sf)x2x(sf^2)
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
    Args:
        psf: NxCxhxw
        shape: [H,W]

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


def upsample(x, sf=3, center=False):
    """
    x: tensor image, NxCxWxH
    """
    st = (sf - 1) // 2 if center else 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)
        ).type_as(x)
    z[(...), st::sf, st::sf].copy_(x)
    return z


def cconj(t, inplace=False):
    """
    # complex's conjugation
    t: NxCxHxWx2
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

class Test_cszn_USRNet(_paritybench_base):
    pass
    def test_000(self):
        self._check(CALayer(*[], **{}), [torch.rand([4, 64, 4, 4])], {})

    def test_001(self):
        self._check(ConditionalBatchNorm2d(*[], **{'num_features': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4]), torch.zeros([4], dtype=torch.int64)], {})

    def test_002(self):
        self._check(HyPaNet(*[], **{}), [torch.rand([4, 2, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(NonLocalBlock2D(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_004(self):
        self._check(RCABlock(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_005(self):
        self._check(RCAGroup(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_006(self):
        self._check(RRDB(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_007(self):
        self._check(ResBlock(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

    def test_008(self):
        self._check(ResidualDenseBlock_5C(*[], **{}), [torch.rand([4, 64, 64, 64])], {})

