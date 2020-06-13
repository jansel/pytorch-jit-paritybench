import sys
_module = sys.modules[__name__]
del sys
ArgsTools = _module
base_args = _module
base_args_jupyter = _module
test_args = _module
FileTools = _module
Prepro = _module
DataTools = _module
metrics = _module
LogTools = _module
logger_tensorboard = _module
VGG = _module
LossTools = _module
loss = _module
TorchTools = _module
progress_bar = _module
crop_imgs = _module
crop_mats = _module
generate_train_df2k = _module
generate_train_mat = _module
load_dataset = _module
model = _module
common = _module
ddsr = _module
demo = _module
denoraw = _module
denorgb = _module
ruofan = _module
srraw = _module
srrgb = _module
tenet1 = _module
tenet2 = _module
rgb2raw = _module
test = _module
train = _module
train_mat = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo


import math


import torch.nn


from torch.autograd import Variable


import torch.nn.init as init


import torch.utils.data as data


import random


from scipy.io import loadmat


from torch.utils.data import DataLoader


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512 * 7 * 7, 4096), nn.
            ReLU(True), nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self, epsilon=1e-06):
        super(Charbonnier_loss, self).__init__()
        self.eps = epsilon ** 2

    def forward(self, X, Y):
        diff = X - Y
        loss = torch.mean(torch.sqrt(diff ** 2 + self.eps))
        return loss


class L1_TVLoss(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(L1_TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return h_tv + w_tv


class L1_TVLoss_Charbonnier(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(L1_TVLoss_Charbonnier, self).__init__()
        self.e = 1e-06 ** 2

    def forward(self, x):
        batch_size = x.size()[0]
        h_tv = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))
        w_tv = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))
        return h_tv + w_tv


class TV_L1LOSS(nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(TV_L1LOSS, self).__init__()

    def forward(self, x, y):
        size = x.size()
        h_tv_diff = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :] - (y[:, :, 1
            :, :] - y[:, :, :-1, :])).sum()
        w_tv_diff = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1] - (y[:, :, :,
            1:] - y[:, :, :, :-1])).sum()
        return (h_tv_diff + w_tv_diff) / size[0] / size[1] / size[2] / size[3]


class MSEloss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self, epsilon=0.001):
        super(MSEloss, self).__init__()
        self.eps = epsilon ** 2

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        sum_square_err = torch.sum(diff * diff)
        loss = sum_square_err / X.data.shape[0] / 2.0
        return loss


class Perceptual_loss(nn.Module):

    def __init__(self, content_layer):
        super(Perceptual_loss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        model = nn.Sequential()
        i = 1
        j = 1
        temp = list(vgg)
        for layer in list(vgg):
            if isinstance(layer, nn.Conv2d):
                name = 'conv_' + str(i)
                model.add_module(name, layer)
                if name == content_layer:
                    break
            if isinstance(layer, nn.ReLU):
                name = 'relu_' + str(i)
                model.add_module(name, layer)
                if name == content_layer:
                    break
                i += 1
            if isinstance(layer, nn.MaxPool2d):
                name = 'pool_' + str(j)
                model.add_module(name, layer)
                j += 1
        self.model = model
        self.criteria = nn.MSELoss()

    def forward(self, X, Y):
        X_content = self.model(X)
        Y_content = self.model(Y)
        loss = ((X_content - Y_content) ** 2).sum()
        loss /= X_content.size()[0] * X_content.size()[1] * X_content.size()[2
            ] * X_content.size()[3]
        return loss


def l1_loss(input, output):
    return torch.mean(torch.abs(input - output))


class VGGLoss(nn.Module):
    """
    VGG(
    (features): Sequential(
    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (19): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)
    (27): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    )

    """

    def __init__(self, vgg_path, layers='45', input='RGB', loss='l1'):
        super(VGGLoss, self).__init__()
        self.input = input
        vgg = models.vgg19(pretrained=False)
        if vgg_path is not '':
            vgg.load_state_dict(torch.load(vgg_path))
        self.layers = [int(l) for l in layers]
        layers_dict = [0, 4, 9, 18, 27, 36]
        self.vgg = []
        if loss == 'l1':
            self.loss_func = l1_loss
        elif loss == 'l2':
            self.loss_func = nn.functional.mse_loss
        else:
            raise Exception('Do not support this loss.')
        i = 0
        for j in self.layers:
            self.vgg.append(nn.Sequential(*list(vgg.features.children())[
                layers_dict[i]:layers_dict[j]]))
            i = j

    def cuda(self, device=None):
        for Seq in self.vgg:
            Seq

    def forward(self, input, target):
        if self.input == 'RGB':
            input_R, input_G, input_B = torch.split(input, 1, dim=1)
            target_R, target_G, target_B = torch.split(target, 1, dim=1)
            input_BGR = torch.cat([input_B, input_G, input_R], dim=1)
            target_BGR = torch.cat([target_B, target_G, target_R], dim=1)
        else:
            input_BGR = input
            target_BGR = target
        input_list = [input_BGR]
        target_list = [target_BGR]
        for Sequential in self.vgg:
            input_list.append(Sequential(input_list[-1]))
            target_list.append(Sequential(target_list[-1]))
        loss = []
        for i in range(len(self.layers)):
            loss.append(self.loss_func(input_list[i + 1], target_list[i + 1]))
        return sum(loss)


def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' %
            act_type)
    return layer


def default_conv(in_channelss, out_channels, kernel_size, stride=1, bias=False
    ):
    return nn.Conv2d(in_channelss, out_channels, kernel_size, padding=
        kernel_size // 2, stride=stride, bias=bias)


def norm_layer(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' %
            norm_type)
    return layer


class ConvBlock(nn.Sequential):

    def __init__(self, in_channelss, out_channels, kernel_size=3, stride=1,
        bias=False, norm_type=False, act_type='relu'):
        m = [default_conv(in_channelss, out_channels, kernel_size, stride=
            stride, bias=bias)]
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, out_channels) if norm_type else None
        if norm:
            m.append(norm)
        if act is not None:
            m.append(act)
        super(ConvBlock, self).__init__(*m)


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


class ResBlock(nn.Module):

    def __init__(self, n_feats, kernel_size=3, norm_type=False, act_type=
        'relu', bias=False, res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, n_feats) if norm_type else None
        for i in range(2):
            m.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm:
                m.append(norm)
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualDenseBlock5(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR18)
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
        norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(ResidualDenseBlock5, self).__init__()
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias,
            norm_type=norm_type, act_type=act_type)
        self.conv2 = ConvBlock(nc + gc, gc, kernel_size, stride, bias=bias,
            norm_type=norm_type, act_type=act_type)
        self.conv3 = ConvBlock(nc + 2 * gc, gc, kernel_size, stride, bias=
            bias, norm_type=norm_type, act_type=act_type)
        self.conv4 = ConvBlock(nc + 3 * gc, gc, kernel_size, stride, bias=
            bias, norm_type=norm_type, act_type=act_type)
        self.conv5 = ConvBlock(nc + 4 * gc, gc, kernel_size, stride, bias=
            bias, norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(self.res_scale) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
        norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.RDB1 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
            norm_type, act_type, res_scale)
        self.RDB2 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
            norm_type, act_type, res_scale)
        self.RDB3 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
            norm_type, act_type, res_scale)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(self.res_scale) + x


class SkipUpDownBlock(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR18)
    """

    def __init__(self, nc, kernel_size=3, stride=1, bias=True, norm_type=
        None, act_type='leakyrelu', res_scale=0.2):
        super(SkipUpDownBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, nc, kernel_size, stride, bias=bias,
            norm_type=norm_type, act_type=act_type)
        self.conv2 = ConvBlock(2 * nc, 2 * nc, kernel_size, stride, bias=
            bias, norm_type=norm_type, act_type=act_type)
        self.up = nn.PixelShuffle(2)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(nc, nc, kernel_size, stride, bias=bias,
            norm_type=norm_type, act_type=act_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.up(torch.cat((x, x1, x2), 1))
        x3 = self.conv3(self.pool(x3))
        return x3.mul(self.res_scale) + x


class DUDB(nn.Module):
    """
    Dense Up Down Block
    """

    def __init__(self, nc, kernel_size=3, stride=1, bias=True, norm_type=
        None, act_type='leakyrelu', res_scale=0.2):
        super(DUDB, self).__init__()
        self.res_scale = res_scale
        self.UDB1 = SkipUpDownBlock(nc, kernel_size, stride, bias,
            norm_type, act_type, res_scale)
        self.UDB2 = SkipUpDownBlock(nc, kernel_size, stride, bias,
            norm_type, act_type, res_scale)
        self.UDB3 = SkipUpDownBlock(nc, kernel_size, stride, bias,
            norm_type, act_type, res_scale)

    def forward(self, x):
        return self.UDB3(self.UDB2(self.UDB1(x))).mul(self.res_scale) + x


class Upsampler(nn.Sequential):

    def __init__(self, scale, n_feats, norm_type=False, act_type='relu',
        bias=False):
        m = []
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, n_feats) if norm_type else None
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if norm:
                    m.append(norm)
                if act is not None:
                    m.append(act)
        elif scale == 3:
            m.append(default_conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if norm:
                m.append(norm)
            if act is not None:
                m.append(act)
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class DownsamplingShuffle(nn.Module):

    def __init__(self, scale):
        super(DownsamplingShuffle, self).__init__()
        self.scale = scale

    def forward(self, input):
        """
        input should be 4D tensor N, C, H, W
        :return: N, C*scale**2,H//scale,W//scale
        """
        N, C, H, W = input.size()
        assert H % self.scale == 0, 'Please Check input and scale'
        assert W % self.scale == 0, 'Please Check input and scale'
        map_channels = self.scale ** 2
        channels = C * map_channels
        out_height = H // self.scale
        out_width = W // self.scale
        input_view = input.contiguous().view(N, C, out_height, self.scale,
            out_width, self.scale)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
        return shuffle_out.view(N, channels, out_height, out_width)


class NET(nn.Module):

    def __init__(self, sr_n_resblocks=6, dm_n_resblock=6, sr_n_feats=64,
        dm_n_feats=64, scale=2, denoise=True, bias=True, norm_type=False,
        act_type='relu', block_type='rrdb'):
        super(NET, self).__init__()
        if denoise:
            m_sr_head = [common.ConvBlock(5, sr_n_feats, 5, act_type=
                act_type, bias=True)]
        else:
            m_sr_head = [common.ConvBlock(4, sr_n_feats, 5, act_type=
                act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            m_sr_resblock = [common.RRDB(sr_n_feats, sr_n_feats, 3, 1, bias,
                norm_type, act_type, 0.2) for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_sr_resblock = [common.DUDB(sr_n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'res':
            m_sr_resblock = [common.ResBlock(sr_n_feats, 3, norm_type,
                act_type, res_scale=1, bias=bias) for _ in range(
                sr_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        m_sr_resblock += [common.ConvBlock(sr_n_feats, sr_n_feats, 3, bias=
            bias)]
        m_sr_up = [common.Upsampler(scale, sr_n_feats, norm_type, act_type,
            bias=bias), common.ConvBlock(sr_n_feats, 4, 3, bias=True)]
        m_sr_tail = [nn.PixelShuffle(2)]
        m_dm_head = [common.ConvBlock(4, dm_n_feats, 5, act_type=act_type,
            bias=True)]
        if block_type.lower() == 'rrdb':
            m_dm_resblock = [common.RRDB(dm_n_feats, dm_n_feats, 3, 1, bias,
                norm_type, act_type, 0.2) for _ in range(dm_n_resblock)]
        elif block_type.lower() == 'dudb':
            m_dm_resblock = [common.DUDB(dm_n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(dm_n_resblock)]
        elif block_type.lower() == 'res':
            m_dm_resblock = [common.ResBlock(dm_n_feats, 3, norm_type,
                act_type, res_scale=1, bias=bias) for _ in range(dm_n_resblock)
                ]
        else:
            raise RuntimeError('block_type is not supported')
        m_dm_resblock += [common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=
            bias)]
        m_dm_up = [common.Upsampler(2, dm_n_feats, norm_type, act_type,
            bias=bias), common.ConvBlock(dm_n_feats, 3, 3, bias=True)]
        self.model_sr = nn.Sequential(*m_sr_head, common.ShortcutBlock(nn.
            Sequential(*m_sr_resblock)), *m_sr_up)
        self.sr_output = nn.Sequential(*m_sr_tail)
        self.model_dm = nn.Sequential(*m_dm_head, common.ShortcutBlock(nn.
            Sequential(*m_dm_resblock)), *m_dm_up)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model_sr(x)
        sr_raw = self.sr_output(x)
        x = self.model_dm(x)
        return sr_raw, x


class NET(nn.Module):

    def __init__(self, opt):
        super(NET, self).__init__()
        denoise = opt.denoise
        block_type = opt.block_type
        n_feats = opt.channels
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type
        n_resblocks = opt.n_resblocks
        if denoise:
            dm_head = [common.ConvBlock(5, n_feats, 5, act_type=act_type,
                bias=True)]
        else:
            dm_head = [common.ConvBlock(4, n_feats, 5, act_type=act_type,
                bias=True)]
        if block_type.lower() == 'rrdb':
            dm_resblock = [common.RRDB(n_feats, n_feats, 3, 1, bias,
                norm_type, act_type, 0.2) for _ in range(n_resblocks)]
        elif block_type.lower() == 'res':
            dm_resblock = [common.ResBlock(n_feats, 3, norm_type, act_type,
                res_scale=1, bias=bias) for _ in range(n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        dm_resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True)]
        m_dm_up = [common.Upsampler(2, n_feats, norm_type, act_type, bias=
            bias), common.ConvBlock(n_feats, 3, 3, bias=True)]
        self.model_dm = nn.Sequential(*dm_head, common.ShortcutBlock(nn.
            Sequential(*dm_resblock)), *m_dm_up)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model_dm(x)
        return x


class NET(nn.Module):

    def __init__(self, opt):
        super(NET, self).__init__()
        n_resblocks = opt.n_resblocks
        n_feats = opt.channels
        bias = opt.bias
        norm_type = opt.norm_type
        act_type = opt.act_type
        block_type = opt.block_type
        head = [common.ConvBlock(5, n_feats, 5, act_type=act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            resblock = [common.RRDB(n_feats, n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(n_resblocks)]
        elif block_type.lower() == 'res':
            resblock = [common.ResBlock(n_feats, 3, norm_type, act_type,
                res_scale=1, bias=bias) for _ in range(n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True)]
        tail = [common.Upsampler(2, n_feats, norm_type, act_type, bias=bias
            ), common.ConvBlock(n_feats, 1, 3, bias=True)]
        self.model = nn.Sequential(*head, common.ShortcutBlock(nn.
            Sequential(*resblock)), *tail)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


class NET(nn.Module):

    def __init__(self, opt):
        super(NET, self).__init__()
        n_resblocks = opt.n_resblocks
        n_feats = opt.channels
        bias = opt.bias
        norm_type = opt.norm_type
        act_type = opt.act_type
        block_type = opt.block_type
        head = [common.ConvBlock(4, n_feats, 5, act_type=act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            resblock = [common.RRDB(n_feats, n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(n_resblocks)]
        elif block_type.lower() == 'res':
            resblock = [common.ResBlock(n_feats, 3, norm_type, act_type,
                res_scale=1, bias=bias) for _ in range(n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True)]
        tail = [common.ConvBlock(n_feats, 3, 3, bias=True)]
        self.model = nn.Sequential(*head, common.ShortcutBlock(nn.
            Sequential(*resblock)), *tail)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


class NET(nn.Module):

    def __init__(self, n_resblock=24, n_feats=256, scale=2, bias=True,
        norm_type=False, act_type='prelu'):
        super(NET, self).__init__()
        self.scale = scale
        m = [common.default_conv(1, n_feats, 3, stride=2)]
        m += [nn.PixelShuffle(2), common.ConvBlock(n_feats // 4, n_feats,
            bias=True, act_type=act_type)]
        m += [common.ResBlock(n_feats, 3, norm_type, act_type, res_scale=1,
            bias=bias) for _ in range(n_resblock)]
        for _ in range(int(math.log(scale, 2))):
            m += [nn.PixelShuffle(2), common.ConvBlock(n_feats // 4,
                n_feats, bias=True, act_type=act_type)]
        m += [common.default_conv(n_feats, 3, 3)]
        self.model = nn.Sequential(*m)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        return self.model(x)


class NET(nn.Module):

    def __init__(self, opt):
        super(NET, self).__init__()
        n_resblocks = opt.n_resblocks
        n_feats = opt.channels
        bias = opt.bias
        norm_type = opt.norm_type
        act_type = opt.act_type
        block_type = opt.block_type
        denoise = opt.denoise
        scale = opt.scale
        if denoise:
            head = [common.ConvBlock(5, n_feats, 5, act_type=act_type, bias
                =True)]
        else:
            head = [common.ConvBlock(4, n_feats, 5, act_type=act_type, bias
                =True)]
        if block_type.lower() == 'rrdb':
            resblock = [common.RRDB(n_feats, n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(n_resblocks)]
        elif block_type.lower() == 'res':
            resblock = [common.ResBlock(n_feats, 3, norm_type, act_type,
                res_scale=1, bias=bias) for _ in range(n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True)]
        up = [common.Upsampler(scale * 2, n_feats, norm_type, act_type,
            bias=bias), common.ConvBlock(n_feats, 1, 3, bias=True)]
        self.model = nn.Sequential(*head, common.ShortcutBlock(nn.
            Sequential(*resblock)), *up)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


class NET(nn.Module):

    def __init__(self, opt):
        super(NET, self).__init__()
        n_resblocks = opt.n_resblocks
        n_feats = opt.channels
        bias = opt.bias
        norm_type = opt.norm_type
        act_type = opt.act_type
        block_type = opt.block_type
        denoise = opt.denoise
        scale = opt.scale
        if denoise:
            head = [common.ConvBlock(4, n_feats, 5, act_type=act_type, bias
                =True)]
        else:
            head = [common.ConvBlock(3, n_feats, 5, act_type=act_type, bias
                =True)]
        if block_type.lower() == 'rrdb':
            resblock = [common.RRDB(n_feats, n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(n_resblocks)]
        elif block_type.lower() == 'res':
            resblock = [common.ResBlock(n_feats, 3, norm_type, act_type,
                res_scale=1, bias=bias) for _ in range(n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True)]
        up = [common.Upsampler(scale, n_feats, norm_type, act_type, bias=
            bias), common.ConvBlock(n_feats, 3, 3, bias=True)]
        self.model = nn.Sequential(*head, common.ShortcutBlock(nn.
            Sequential(*resblock)), *up)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


class NET(nn.Module):

    def __init__(self, opt):
        super(NET, self).__init__()
        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale
        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type
        if denoise:
            m_sr_head = [common.ConvBlock(5, sr_n_feats, 5, act_type=
                act_type, bias=True)]
        else:
            m_sr_head = [common.ConvBlock(4, sr_n_feats, 5, act_type=
                act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            m_sr_resblock = [common.RRDB(sr_n_feats, sr_n_feats, 3, 1, bias,
                norm_type, act_type, 0.2) for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_sr_resblock = [common.DUDB(sr_n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'res':
            m_sr_resblock = [common.ResBlock(sr_n_feats, 3, norm_type,
                act_type, res_scale=1, bias=bias) for _ in range(
                sr_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        m_sr_resblock += [common.ConvBlock(sr_n_feats, sr_n_feats, 3, bias=
            bias)]
        m_sr_up = [common.Upsampler(scale, sr_n_feats, norm_type, act_type,
            bias=bias), common.ConvBlock(sr_n_feats, sr_n_feats, 3, bias=True)]
        m_dm_head = [common.ConvBlock(sr_n_feats, dm_n_feats, 5, act_type=
            act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            m_dm_resblock = [common.RRDB(dm_n_feats, dm_n_feats, 3, 1, bias,
                norm_type, act_type, 0.2) for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_dm_resblock = [common.DUDB(dm_n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'res':
            m_dm_resblock = [common.ResBlock(dm_n_feats, 3, norm_type,
                act_type, res_scale=1, bias=bias) for _ in range(
                dm_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        m_dm_resblock += [common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=
            bias)]
        m_dm_up = [common.Upsampler(2, dm_n_feats, norm_type, act_type,
            bias=bias), common.ConvBlock(dm_n_feats, 3, 3, bias=True)]
        self.model_sr = nn.Sequential(*m_sr_head, common.ShortcutBlock(nn.
            Sequential(*m_sr_resblock)), *m_sr_up)
        self.model_dm = nn.Sequential(*m_dm_head, common.ShortcutBlock(nn.
            Sequential(*m_dm_resblock)), *m_dm_up)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        return self.model_dm(self.model_sr(x))


class NET(nn.Module):

    def __init__(self, opt):
        super(NET, self).__init__()
        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale
        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type
        if denoise:
            m_sr_head = [common.ConvBlock(5, sr_n_feats, 5, act_type=
                act_type, bias=True)]
        else:
            m_sr_head = [common.ConvBlock(4, sr_n_feats, 5, act_type=
                act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            m_sr_resblock = [common.RRDB(sr_n_feats, sr_n_feats, 3, 1, bias,
                norm_type, act_type, 0.2) for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_sr_resblock = [common.DUDB(sr_n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'res':
            m_sr_resblock = [common.ResBlock(sr_n_feats, 3, norm_type,
                act_type, res_scale=1, bias=bias) for _ in range(
                sr_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        m_sr_resblock += [common.ConvBlock(sr_n_feats, sr_n_feats, 3, bias=
            bias)]
        m_sr_up = [common.Upsampler(scale, sr_n_feats, norm_type, act_type,
            bias=bias), common.ConvBlock(sr_n_feats, 4, 3, bias=True)]
        m_sr_tail = [nn.PixelShuffle(2)]
        m_dm_head = [common.ConvBlock(4, dm_n_feats, 5, act_type=act_type,
            bias=True)]
        if block_type.lower() == 'rrdb':
            m_dm_resblock = [common.RRDB(dm_n_feats, dm_n_feats, 3, 1, bias,
                norm_type, act_type, 0.2) for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_dm_resblock = [common.DUDB(dm_n_feats, 3, 1, bias, norm_type,
                act_type, 0.2) for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'res':
            m_dm_resblock = [common.ResBlock(dm_n_feats, 3, norm_type,
                act_type, res_scale=1, bias=bias) for _ in range(
                dm_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')
        m_dm_resblock += [common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=
            bias)]
        m_dm_up = [common.Upsampler(2, dm_n_feats, norm_type, act_type,
            bias=bias), common.ConvBlock(dm_n_feats, 3, 3, bias=True)]
        self.model_sr = nn.Sequential(*m_sr_head, common.ShortcutBlock(nn.
            Sequential(*m_sr_resblock)), *m_sr_up)
        self.sr_output = nn.Sequential(*m_sr_tail)
        self.model_dm = nn.Sequential(*m_dm_head, common.ShortcutBlock(nn.
            Sequential(*m_dm_resblock)), *m_dm_up)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model_sr(x)
        sr_raw = self.sr_output(x)
        x = self.model_dm(x)
        return sr_raw, x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_guochengqian_TENet(_paritybench_base):
    pass
    def test_000(self):
        self._check(Charbonnier_loss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(L1_TVLoss(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(L1_TVLoss_Charbonnier(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(TV_L1LOSS(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(MSEloss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(ConvBlock(*[], **{'in_channelss': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(ResBlock(*[], **{'n_feats': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(SkipUpDownBlock(*[], **{'nc': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_008(self):
        self._check(DUDB(*[], **{'nc': 4}), [torch.rand([4, 4, 4, 4])], {})

