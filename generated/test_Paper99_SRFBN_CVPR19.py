import sys
_module = sys.modules[__name__]
del sys
LRHR_dataset = _module
LR_dataset = _module
data = _module
blocks = _module
dbpn_arch = _module
edsr_arch = _module
rdn_arch = _module
srfbn_arch = _module
options = _module
Prepare_TrainData_HR_LR = _module
flags = _module
plot_log = _module
solvers = _module
base_solver = _module
test = _module
utils = _module
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


import torch


import torch.nn as nn


from collections import OrderedDict


import math


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * 255.0 * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        for p in self.parameters():
            p.requires_grad = False


def activation(act_type='relu', inplace=True, slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    layer = None
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    else:
        raise NotImplementedError(
            '[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    layer = None
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError(
            '[ERROR] Normalization layer [%s] is not implemented!' % norm_type)
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    layer = None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError(
            '[ERROR] Padding layer [%s] is not implemented!' % pad_type)
    return layer


def get_valid_padding(kernel_size, dilation):
    """
    Padding value to remain feature size.
    """
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                '[ERROR] %s.sequential() does not support OrderedDict' %
                sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1,
    bias=True, valid_padding=True, padding=0, act_type='relu', norm_type=
    'bn', pad_type='zero', mode='CNA'):
    assert mode in ['CNA', 'NAC'], '[ERROR] Wrong mode in [%s]!' % sys.modules[
        __name__]
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
        padding=padding, dilation=dilation, bias=bias)
    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, conv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, conv)


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channle, mid_channel, kernel_size,
        stride=1, valid_padding=True, padding=0, dilation=1, bias=True,
        pad_type='zero', norm_type='bn', act_type='relu', mode='CNA',
        res_scale=1):
        super(ResBlock, self).__init__()
        conv0 = ConvBlock(in_channel, mid_channel, kernel_size, stride,
            dilation, bias, valid_padding, padding, act_type, norm_type,
            pad_type, mode)
        act_type = None
        norm_type = None
        conv1 = ConvBlock(mid_channel, out_channle, kernel_size, stride,
            dilation, bias, valid_padding, padding, act_type, norm_type,
            pad_type, mode)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res


def DeconvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=
    1, bias=True, padding=0, act_type='relu', norm_type='bn', pad_type=
    'zero', mode='CNA'):
    assert mode in ['CNA', 'NAC'], '[ERROR] Wrong mode in [%s]!' % sys.modules[
        __name__]
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride, padding, dilation=dilation, bias=bias)
    if mode == 'CNA':
        act = activation(act_type) if act_type else None
        n = norm(out_channels, norm_type) if norm_type else None
        return sequential(p, deconv, n, act)
    elif mode == 'NAC':
        act = activation(act_type, inplace=False) if act_type else None
        n = norm(in_channels, norm_type) if norm_type else None
        return sequential(n, act, p, deconv)


class UpprojBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        valid_padding=False, padding=0, bias=True, pad_type='zero',
        norm_type=None, act_type='prelu'):
        super(UpprojBlock, self).__init__()
        self.deconv_1 = DeconvBlock(in_channel, out_channel, kernel_size,
            stride=stride, padding=padding, norm_type=norm_type, act_type=
            act_type)
        self.conv_1 = ConvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, valid_padding=valid_padding,
            norm_type=norm_type, act_type=act_type)
        self.deconv_2 = DeconvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, norm_type=norm_type, act_type=
            act_type)

    def forward(self, x):
        H_0_t = self.deconv_1(x)
        L_0_t = self.conv_1(H_0_t)
        H_1_t = self.deconv_2(L_0_t - x)
        return H_0_t + H_1_t


class D_UpprojBlock(torch.nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        valid_padding=False, padding=0, bias=True, pad_type='zero',
        norm_type=None, act_type='prelu'):
        super(D_UpprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size=1,
            norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, norm_type=norm_type, act_type=
            act_type)
        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, valid_padding=valid_padding,
            norm_type=norm_type, act_type=act_type)
        self.deconv_2 = DeconvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, norm_type=norm_type, act_type=
            act_type)

    def forward(self, x):
        x = self.conv_1(x)
        H_0_t = self.deconv_1(x)
        L_0_t = self.conv_2(H_0_t)
        H_1_t = self.deconv_2(L_0_t - x)
        return H_1_t + H_0_t


class DownprojBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        valid_padding=True, padding=0, dilation=1, bias=True, pad_type=
        'zero', norm_type=None, act_type='prelu', mode='CNA', res_scale=1):
        super(DownprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size,
            stride=stride, padding=padding, valid_padding=valid_padding,
            norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, norm_type=norm_type, act_type=
            act_type)
        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, valid_padding=valid_padding,
            norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        L_0_t = self.conv_1(x)
        H_0_t = self.deconv_1(L_0_t)
        L_1_t = self.conv_2(H_0_t - x)
        return L_0_t + L_1_t


class D_DownprojBlock(torch.nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
        valid_padding=False, padding=0, bias=True, pad_type='zero',
        norm_type=None, act_type='prelu'):
        super(D_DownprojBlock, self).__init__()
        self.conv_1 = ConvBlock(in_channel, out_channel, kernel_size=1,
            norm_type=norm_type, act_type=act_type)
        self.conv_2 = ConvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, valid_padding=valid_padding,
            norm_type=norm_type, act_type=act_type)
        self.deconv_1 = DeconvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, norm_type=norm_type, act_type=
            act_type)
        self.conv_3 = ConvBlock(out_channel, out_channel, kernel_size,
            stride=stride, padding=padding, valid_padding=valid_padding,
            norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x = self.conv_1(x)
        L_0_t = self.conv_2(x)
        H_0_t = self.deconv_1(L_0_t)
        L_1_t = self.conv_3(H_0_t - x)
        return L_1_t + L_0_t


class DensebackprojBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, bp_stages,
        stride=1, valid_padding=True, padding=0, dilation=1, bias=True,
        pad_type='zero', norm_type=None, act_type='prelu', mode='CNA',
        res_scale=1):
        super(DensebackprojBlock, self).__init__()
        self.upproj = nn.ModuleList()
        self.downproj = nn.ModuleList()
        self.bp_stages = bp_stages
        self.upproj.append(UpprojBlock(in_channel, out_channel, kernel_size,
            stride=stride, valid_padding=False, padding=padding, norm_type=
            norm_type, act_type=act_type))
        for index in range(self.bp_stages - 1):
            if index < 1:
                self.upproj.append(UpprojBlock(out_channel, out_channel,
                    kernel_size, stride=stride, valid_padding=False,
                    padding=padding, norm_type=norm_type, act_type=act_type))
            else:
                uc = ConvBlock(out_channel * (index + 1), out_channel,
                    kernel_size=1, norm_type=norm_type, act_type=act_type)
                u = UpprojBlock(out_channel, out_channel, kernel_size,
                    stride=stride, valid_padding=False, padding=padding,
                    norm_type=norm_type, act_type=act_type)
                self.upproj.append(sequential(uc, u))
            if index < 1:
                self.downproj.append(DownprojBlock(out_channel, out_channel,
                    kernel_size, stride=stride, valid_padding=False,
                    padding=padding, norm_type=norm_type, act_type=act_type))
            else:
                dc = ConvBlock(out_channel * (index + 1), out_channel,
                    kernel_size=1, norm_type=norm_type, act_type=act_type)
                d = DownprojBlock(out_channel, out_channel, kernel_size,
                    stride=stride, valid_padding=False, padding=padding,
                    norm_type=norm_type, act_type=act_type)
                self.downproj.append(sequential(dc, d))

    def forward(self, x):
        low_features = []
        high_features = []
        H = self.upproj[0](x)
        high_features.append(H)
        for index in range(self.bp_stages - 1):
            if index < 1:
                L = self.downproj[index](H)
                low_features.append(L)
                H = self.upproj[index + 1](L)
                high_features.append(H)
            else:
                H_concat = torch.cat(tuple(high_features), 1)
                L = self.downproj[index](H_concat)
                low_features.append(L)
                L_concat = torch.cat(tuple(low_features), 1)
                H = self.upproj[index + 1](L_concat)
                high_features.append(H)
        output = torch.cat(tuple(high_features), 1)
        return output


class ResidualDenseBlock_8C(nn.Module):
    """
    Residual Dense Block
    style: 8 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True,
        pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
        super(ResidualDenseBlock_8C, self).__init__()
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias,
            pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode
            =mode)
        self.conv2 = ConvBlock(nc + gc, gc, kernel_size, stride, bias=bias,
            pad_type=pad_type, norm_type=norm_type, act_type=act_type, mode
            =mode)
        self.conv3 = ConvBlock(nc + 2 * gc, gc, kernel_size, stride, bias=
            bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type,
            mode=mode)
        self.conv4 = ConvBlock(nc + 3 * gc, gc, kernel_size, stride, bias=
            bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type,
            mode=mode)
        self.conv5 = ConvBlock(nc + 4 * gc, gc, kernel_size, stride, bias=
            bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type,
            mode=mode)
        self.conv6 = ConvBlock(nc + 5 * gc, gc, kernel_size, stride, bias=
            bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type,
            mode=mode)
        self.conv7 = ConvBlock(nc + 6 * gc, gc, kernel_size, stride, bias=
            bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type,
            mode=mode)
        self.conv8 = ConvBlock(nc + 7 * gc, gc, kernel_size, stride, bias=
            bias, pad_type=pad_type, norm_type=norm_type, act_type=act_type,
            mode=mode)
        if mode == 'CNA':
            last_act = None
        else:
            last_act = act_type
        self.conv9 = ConvBlock(nc + 8 * gc, nc, 1, stride, bias=bias,
            pad_type=pad_type, norm_type=norm_type, act_type=last_act, mode
            =mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.conv6(torch.cat((x, x1, x2, x3, x4, x5), 1))
        x7 = self.conv7(torch.cat((x, x1, x2, x3, x4, x5, x6), 1))
        x8 = self.conv8(torch.cat((x, x1, x2, x3, x4, x5, x6, x7), 1))
        x9 = self.conv9(torch.cat((x, x1, x2, x3, x4, x5, x6, x7, x8), 1))
        return x9.mul(0.2) + x


class ShortcutBlock(nn.Module):

    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


class ConcatBlock(nn.Module):

    def __init__(self, submodule):
        super(ConcatBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = torch.cat((x, self.sub(x)), 1)
        return output


class DBPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_features, bp_stages,
        upscale_factor=4, norm_type=None, act_type='prelu'):
        super(DBPN, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            projection_filter = 12
        feature_extract_1 = B.ConvBlock(in_channels, 128, kernel_size=3,
            norm_type=norm_type, act_type=act_type)
        feature_extract_2 = B.ConvBlock(128, num_features, kernel_size=1,
            norm_type=norm_type, act_type=act_type)
        bp_units = []
        for _ in range(bp_stages - 1):
            bp_units.extend([B.UpprojBlock(num_features, num_features,
                projection_filter, stride=stride, valid_padding=False,
                padding=padding, norm_type=norm_type, act_type=act_type), B
                .DownprojBlock(num_features, num_features,
                projection_filter, stride=stride, valid_padding=False,
                padding=padding, norm_type=norm_type, act_type=act_type)])
        last_bp_unit = B.UpprojBlock(num_features, num_features,
            projection_filter, stride=stride, valid_padding=False, padding=
            padding, norm_type=norm_type, act_type=act_type)
        conv_hr = B.ConvBlock(num_features, out_channels, kernel_size=1,
            norm_type=None, act_type=None)
        self.network = B.sequential(feature_extract_1, feature_extract_2, *
            bp_units, last_bp_unit, conv_hr)

    def forward(self, x):
        return self.network(x)


class D_DBPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_features, bp_stages,
        upscale_factor=4, norm_type=None, act_type='prelu'):
        super(D_DBPN, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            projection_filter = 12
        feature_extract_1 = B.ConvBlock(in_channels, 256, kernel_size=3,
            norm_type=norm_type, act_type=act_type)
        feature_extract_2 = B.ConvBlock(256, num_features, kernel_size=1,
            norm_type=norm_type, act_type=act_type)
        bp_units = B.DensebackprojBlock(num_features, num_features,
            projection_filter, bp_stages, stride=stride, valid_padding=
            False, padding=padding, norm_type=norm_type, act_type=act_type)
        conv_hr = B.ConvBlock(num_features * bp_stages, out_channels,
            kernel_size=3, norm_type=None, act_type=None)
        self.network = B.sequential(feature_extract_1, feature_extract_2,
            bp_units, conv_hr)

    def forward(self, x):
        return self.network(x)


class MeanShift(nn.Conv2d):

    def __init__(self, rgb_range=255, rgb_mean=(0.4488, 0.4371, 0.404),
        rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):

    def __init__(self, conv, in_channels, out_channels, kernel_size, stride
        =1, bias=False, bn=True, act=nn.ReLU(True)):
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):

    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act
        =nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):

    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=
        kernel_size // 2, bias=bias)


class EDSR(nn.Module):

    def __init__(self, in_channels, out_channels, num_features, num_blocks,
        res_scale, upscale_factor, conv=default_conv):
        super(EDSR, self).__init__()
        n_resblocks = num_blocks
        n_feats = num_features
        kernel_size = 3
        scale = upscale_factor
        act = nn.ReLU(True)
        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)
        m_head = [conv(in_channels, n_feats, kernel_size)]
        m_body = [ResBlock(conv, n_feats, kernel_size, act=act, res_scale=
            res_scale) for _ in range(n_resblocks)]
        m_body.append(conv(n_feats, n_feats, kernel_size))
        m_tail = [Upsampler(conv, scale, n_feats, act=False), conv(n_feats,
            out_channels, kernel_size)]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError(
                            'While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'
                            .format(name, own_state[name].size(), param.size())
                            )
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.
                        format(name))


class RDB_Conv(nn.Module):

    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[nn.Conv2d(Cin, G, kSize, padding=(kSize -
            1) // 2, stride=1), nn.ReLU()])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):

    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):

    def __init__(self, in_channels, out_channels, num_features, num_blocks,
        num_layers, upscale_factor):
        super(RDN, self).__init__()
        r = upscale_factor
        G0 = num_features
        kSize = 3
        self.D, C, G = [num_blocks, num_layers, num_features]
        self.SFENet1 = nn.Conv2d(in_channels, G0, kSize, padding=(kSize - 1
            ) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(RDB(growRate0=G0, growRate=G, nConvLayers=C))
        self.GFF = nn.Sequential(*[nn.Conv2d(self.D * G0, G0, 1, padding=0,
            stride=1), nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2,
            stride=1)])
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * r * r, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(r), nn
                .Conv2d(G, out_channels, kSize, padding=(kSize - 1) // 2,
                stride=1)])
        elif r == 4:
            self.UPNet = nn.Sequential(*[nn.Conv2d(G0, G * 4, kSize,
                padding=(kSize - 1) // 2, stride=1), nn.PixelShuffle(2), nn
                .Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1
                ), nn.PixelShuffle(2), nn.Conv2d(G, out_channels, kSize,
                padding=(kSize - 1) // 2, stride=1)])
        else:
            raise ValueError('scale must be 2 or 3 or 4.')

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)
        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1
        return self.UPNet(x)


class FeedbackBlock(nn.Module):

    def __init__(self, num_features, num_groups, upscale_factor, act_type,
        norm_type):
        super(FeedbackBlock, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12
        self.num_groups = num_groups
        self.compress_in = ConvBlock(2 * num_features, num_features,
            kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.upBlocks = nn.ModuleList()
        self.downBlocks = nn.ModuleList()
        self.uptranBlocks = nn.ModuleList()
        self.downtranBlocks = nn.ModuleList()
        for idx in range(self.num_groups):
            self.upBlocks.append(DeconvBlock(num_features, num_features,
                kernel_size=kernel_size, stride=stride, padding=padding,
                act_type=act_type, norm_type=norm_type))
            self.downBlocks.append(ConvBlock(num_features, num_features,
                kernel_size=kernel_size, stride=stride, padding=padding,
                act_type=act_type, norm_type=norm_type, valid_padding=False))
            if idx > 0:
                self.uptranBlocks.append(ConvBlock(num_features * (idx + 1),
                    num_features, kernel_size=1, stride=1, act_type=
                    act_type, norm_type=norm_type))
                self.downtranBlocks.append(ConvBlock(num_features * (idx + 
                    1), num_features, kernel_size=1, stride=1, act_type=
                    act_type, norm_type=norm_type))
        self.compress_out = ConvBlock(num_groups * num_features,
            num_features, kernel_size=1, act_type=act_type, norm_type=norm_type
            )
        self.should_reset = True
        self.last_hidden = None

    def forward(self, x):
        if self.should_reset:
            self.last_hidden = torch.zeros(x.size())
            self.last_hidden.copy_(x)
            self.should_reset = False
        x = torch.cat((x, self.last_hidden), dim=1)
        x = self.compress_in(x)
        lr_features = []
        hr_features = []
        lr_features.append(x)
        for idx in range(self.num_groups):
            LD_L = torch.cat(tuple(lr_features), 1)
            if idx > 0:
                LD_L = self.uptranBlocks[idx - 1](LD_L)
            LD_H = self.upBlocks[idx](LD_L)
            hr_features.append(LD_H)
            LD_H = torch.cat(tuple(hr_features), 1)
            if idx > 0:
                LD_H = self.downtranBlocks[idx - 1](LD_H)
            LD_L = self.downBlocks[idx](LD_H)
            lr_features.append(LD_L)
        del hr_features
        output = torch.cat(tuple(lr_features[1:]), 1)
        output = self.compress_out(output)
        self.last_hidden = output
        return output

    def reset_state(self):
        self.should_reset = True


class SRFBN(nn.Module):

    def __init__(self, in_channels, out_channels, num_features, num_steps,
        num_groups, upscale_factor, act_type='prelu', norm_type=None):
        super(SRFBN, self).__init__()
        if upscale_factor == 2:
            stride = 2
            padding = 2
            kernel_size = 6
        elif upscale_factor == 3:
            stride = 3
            padding = 2
            kernel_size = 7
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            kernel_size = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            kernel_size = 12
        self.num_steps = num_steps
        self.num_features = num_features
        self.upscale_factor = upscale_factor
        rgb_mean = 0.4488, 0.4371, 0.404
        rgb_std = 1.0, 1.0, 1.0
        self.sub_mean = MeanShift(rgb_mean, rgb_std)
        self.conv_in = ConvBlock(in_channels, 4 * num_features, kernel_size
            =3, act_type=act_type, norm_type=norm_type)
        self.feat_in = ConvBlock(4 * num_features, num_features,
            kernel_size=1, act_type=act_type, norm_type=norm_type)
        self.block = FeedbackBlock(num_features, num_groups, upscale_factor,
            act_type, norm_type)
        self.out = DeconvBlock(num_features, num_features, kernel_size=
            kernel_size, stride=stride, padding=padding, act_type='prelu',
            norm_type=norm_type)
        self.conv_out = ConvBlock(num_features, out_channels, kernel_size=3,
            act_type=None, norm_type=norm_type)
        self.add_mean = MeanShift(rgb_mean, rgb_std, 1)

    def forward(self, x):
        self._reset_state()
        x = self.sub_mean(x)
        inter_res = nn.functional.interpolate(x, scale_factor=self.
            upscale_factor, mode='bilinear', align_corners=False)
        x = self.conv_in(x)
        x = self.feat_in(x)
        outs = []
        for _ in range(self.num_steps):
            h = self.block(x)
            h = torch.add(inter_res, self.conv_out(self.out(h)))
            h = self.add_mean(h)
            outs.append(h)
        return outs

    def _reset_state(self):
        self.block.reset_state()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_Paper99_SRFBN_CVPR19(_paritybench_base):
    pass
    def test_000(self):
        self._check(MeanShift(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    def test_001(self):
        self._check(UpprojBlock(*[], **{'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(D_UpprojBlock(*[], **{'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(D_DownprojBlock(*[], **{'in_channel': 4, 'out_channel': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(DensebackprojBlock(*[], **{'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'bp_stages': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(ResidualDenseBlock_8C(*[], **{'nc': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(RDB_Conv(*[], **{'inChannels': 4, 'growRate': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(RDB(*[], **{'growRate0': 4, 'growRate': 4, 'nConvLayers': 4}), [torch.rand([4, 4, 4, 4])], {})

