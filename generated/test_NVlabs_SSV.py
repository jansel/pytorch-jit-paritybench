import sys
_module = sys.modules[__name__]
del sys
prepare_lmdb = _module
preprocess_data = _module
extern = _module
network_blocks = _module
transformImage = _module
ssv = _module
test_vpnet = _module
train = _module
dataset = _module
network_blocks = _module

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


from torch import nn


from torch.nn import init


from torch.nn import functional as F


import torch.nn.utils.spectral_norm as spectralnorm


from torch.autograd import Variable


from collections import OrderedDict as odict


import numpy as np


from math import sqrt


import random


from torch.utils.data import DataLoader


from torchvision import datasets


from torchvision import transforms


import torchvision


from sklearn.linear_model import LinearRegression


from torch import optim


from torch.autograd import grad


from torchvision import utils


from torch.utils.data import Dataset


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-08)


class EqualLR:

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)
        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)
        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)
    return module


class EqualLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):

    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


class EqualConv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class StyledConvBlock2(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, style_dim=512):
        super().__init__()
        self.conv1 = EqualConv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style):
        out = self.conv1(input)
        out = self.adain1(out, style)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.adain2(out, style)
        out = self.lrelu2(out)
        return out


class AdaptiveInstanceNorm3(nn.Module):

    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(input)
        out = gamma * out + beta
        return out


class ConstantInput3(nn.Module):

    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size, size))

    def forward(self, batch_size):
        out = self.input.repeat(batch_size, 1, 1, 1, 1)
        return out


class EqualConv3d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv3d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class StyledConvBlock3(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, style_dim=512, initial=False):
        super().__init__()
        if initial:
            self.conv1 = ConstantInput3(in_channel)
        else:
            self.conv1 = EqualConv3d(in_channel, out_channel, kernel_size, padding=padding)
        self.adain1 = AdaptiveInstanceNorm3(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = EqualConv3d(out_channel, out_channel, kernel_size, padding=padding)
        self.adain2 = AdaptiveInstanceNorm3(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, batch_size, style):
        out = self.conv1(batch_size)
        out = self.adain1(out, style)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.adain2(out, style)
        out = self.lrelu2(out)
        return out


class StyledConvBlock3_noAdaIN(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, style_dim=512, initial=False):
        super().__init__()
        self.conv1 = EqualConv3d(in_channel, out_channel, kernel_size, padding=padding)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = EqualConv3d(out_channel, out_channel, kernel_size, padding=padding)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input):
        out = self.conv1(input)
        out = self.lrelu1(out)
        out = self.conv2(out)
        out = self.lrelu2(out)
        return out


class projection_unit(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=1):
        super().__init__()
        self.conv = EqualConv2d(in_channel, out_channel, kernel_size, padding=0)
        self.lrelu = nn.PReLU(1024)

    def forward(self, input):
        batch = input.shape[0]
        out = input.view(batch, input.shape[1] * input.shape[2], input.shape[3], input.shape[4])
        out = self.conv(out)
        out = self.lrelu(out)
        return out


class VPASGenerator(nn.Module):

    def __init__(self, code_dim):
        super().__init__()
        self.progression1 = nn.ModuleList([StyledConvBlock3(512, 512, 3, 1, style_dim=code_dim, initial=True), StyledConvBlock3(512, 512, 3, 1, style_dim=code_dim), StyledConvBlock3(512, 256, 3, 1, style_dim=code_dim)])
        self.progression2 = nn.ModuleList([StyledConvBlock3_noAdaIN(256, 128, 3, 1), StyledConvBlock3_noAdaIN(128, 64, 3, 1)])
        self.projection_unit = projection_unit(64 * 16, 64 * 16)
        self.scb1 = StyledConvBlock2(1024, 512, 3, 1, style_dim=code_dim)
        self.scb2 = StyledConvBlock2(512, 512, 3, 1, style_dim=code_dim)
        self.scb3 = StyledConvBlock2(512, 256, 3, 1, style_dim=code_dim)
        self.scb4 = StyledConvBlock2(256, 128, 3, 1, style_dim=code_dim)
        self.to_rgb = EqualConv2d(128, 3, 1)

    def forward(self, style, rots, batch_size):
        for i, conv in enumerate(self.progression1):
            if i == 0:
                out = conv(batch_size, style[0])
            else:
                upsample = F.interpolate(out, scale_factor=2, mode='trilinear', align_corners=False)
                out = conv(upsample, style[0])
        flow = F.affine_grid(rots, torch.Size([batch_size, 256, 16, 16, 16]))
        out = F.grid_sample(out, flow)
        for i, conv in enumerate(self.progression2):
            out = conv(out)
        out = self.projection_unit(out)
        out = self.scb1(out, style[1])
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb2(out, style[1])
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb3(out, style[1])
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.scb4(out, style[1])
        out = self.to_rgb(out)
        return out


class VPAwareSynthesizer(nn.Module):

    def __init__(self, code_dim=128, n_mlp=8):
        super().__init__()
        self.generator = VPASGenerator(code_dim)
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.style = nn.Sequential(*layers)

    def forward(self, input, rots=None):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]
        for i in input:
            styles.append(self.style(i))
        return self.generator(styles, rots, batch_size=input[0].shape[0])


class CELoss:

    def __init__(self):
        self.CELoss = nn.CrossEntropyLoss()

    def compute_loss(self, tgts, Pred, GT):
        Loss = odict()
        for tgt in tgts:
            Loss[tgt] = self.CELoss(Pred[tgt].view(Pred[tgt].size()[0], 4), GT[tgt].view(Pred[tgt].size()[0]))
        return Loss


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None, pixel_norm=True, spectral_norm=False, instance_norm=False, last=False):
        super().__init__()
        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2
        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2
        if instance_norm and last == True:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1), nn.InstanceNorm2d(out_channel), nn.LeakyReLU(0.2), EqualConv2d(out_channel, out_channel, kernel2, padding=pad2), nn.LeakyReLU(0.2))
        elif instance_norm and last == False:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1), nn.InstanceNorm2d(out_channel), nn.LeakyReLU(0.2), EqualConv2d(out_channel, out_channel, kernel2, padding=pad2), nn.InstanceNorm2d(out_channel), nn.LeakyReLU(0.2))
        else:
            self.conv = nn.Sequential(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1), nn.LeakyReLU(0.2), EqualConv2d(out_channel, out_channel, kernel2, padding=pad2), nn.LeakyReLU(0.2))

    def forward(self, input):
        out = self.conv(input)
        return out


class negDotLoss:

    def __init_(self):
        pass

    def compute_loss(self, tgts, Pred, GT):
        Loss = odict()
        for tgt in tgts:
            Loss[tgt] = torch.mean(-torch.bmm(GT[tgt].view(GT[tgt].shape[0], 1, 2).float(), Pred[tgt].view(Pred[tgt].shape[0], 2, 1).float()))
        return Loss


class VPNet(nn.Module):

    @staticmethod
    def head_seq(in_size, num_fc=1024, init_weights=True):
        """
        Creates a head with fc layer and outputs for magnitute of [sine, cosine] and direction {--, -+, +-, ++}
        """
        seq_fc8 = nn.Sequential(EqualLinear(in_size, num_fc), nn.ReLU(inplace=True), nn.Dropout())
        seq_ccss = EqualLinear(num_fc, 2)
        seq_sgnc = EqualLinear(num_fc, 4)
        return seq_fc8, seq_ccss, seq_sgnc

    def __init__(self, code_dim=128, instance_norm=False):
        super().__init__()
        self.progression = nn.ModuleList([ConvBlock(16, 32, 3, 1, instance_norm=instance_norm), ConvBlock(32, 64, 3, 1, instance_norm=instance_norm), ConvBlock(64, 128, 3, 1, instance_norm=instance_norm), ConvBlock(128, 256, 3, 1, instance_norm=instance_norm), ConvBlock(256, 512, 3, 1, instance_norm=instance_norm), ConvBlock(512, 512, 3, 1, instance_norm=instance_norm), ConvBlock(512, 512, 3, 1, instance_norm=instance_norm), ConvBlock(512, 512, 3, 1, instance_norm=instance_norm), ConvBlock(513, 512, 3, 1, 4, 0, last=True, instance_norm=instance_norm)])
        self.from_rgb = nn.ModuleList([EqualConv2d(3, 16, 1), EqualConv2d(3, 32, 1), EqualConv2d(3, 64, 1), EqualConv2d(3, 128, 1), EqualConv2d(3, 256, 1), EqualConv2d(3, 512, 1), EqualConv2d(3, 512, 1), EqualConv2d(3, 512, 1), EqualConv2d(3, 512, 1)])
        self.n_layer = len(self.progression)
        self.class_linear = EqualLinear(512, 1)
        self.z_linear = EqualLinear(512, code_dim)
        self.head_fc_a, self.head_x2_y2_mag_a, self.head_sin_cos_direc_a = self.head_seq(512, num_fc=256)
        self.head_fc_e, self.head_x2_y2_mag_e, self.head_sin_cos_direc_e = self.head_seq(512, num_fc=256)
        self.head_fc_t, self.head_x2_y2_mag_t, self.head_sin_cos_direc_t = self.head_seq(512, num_fc=256)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.loss_mag = negDotLoss()
        self.loss_direc = CELoss()
        self.balance_weight = 1.0

    def forward(self, input):
        step = 5
        alpha = 0
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1
            if i == step:
                out = self.from_rgb[index](input)
            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-08)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)
            out = self.progression[index](out)
            if i > 0:
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)
                if i == step and 0 <= alpha < 1:
                    skip_rgb = self.from_rgb[index + 1](input)
                    out = F.interpolate(skip_rgb, scale_factor=0.5, mode='bilinear', align_corners=False)
        trunk_out = out.squeeze(2).squeeze(2)
        out = out.squeeze(2).squeeze(2)
        batchsize = out.size(0)
        class_out = self.class_linear(out)
        z_out = self.z_linear(out)
        x_a = self.head_fc_a(out)
        x_e = self.head_fc_e(out)
        x_t = self.head_fc_t(out)
        mag_x2_y2_a = self.head_x2_y2_mag_a(x_a).view(batchsize, 1, 2)
        mag_x2_y2_e = self.head_x2_y2_mag_e(x_e).view(batchsize, 1, 2)
        mag_x2_y2_t = self.head_x2_y2_mag_t(x_t).view(batchsize, 1, 2)
        logsoftmax_x2_y2_a = self.logsoftmax(mag_x2_y2_a)
        logsoftmax_x2_y2_e = self.logsoftmax(mag_x2_y2_e)
        logsoftmax_x2_y2_t = self.logsoftmax(mag_x2_y2_t)
        sign_x_y_a = self.head_sin_cos_direc_a(x_a).view(batchsize, 1, 4)
        sign_x_y_e = self.head_sin_cos_direc_e(x_e).view(batchsize, 1, 4)
        sign_x_y_t = self.head_sin_cos_direc_t(x_t).view(batchsize, 1, 4)
        viewpoint_op = odict(logprob_xxyy=odict(a=logsoftmax_x2_y2_a, e=logsoftmax_x2_y2_e, t=logsoftmax_x2_y2_t), sign_x_y=odict(a=sign_x_y_a, e=sign_x_y_e, t=sign_x_y_t))
        return class_out, z_out, viewpoint_op, trunk_out

    def compute_vp_loss(self, pred, GT):
        """
        Compute loss for magnitude heads using negdot
        Compute loss for direction heads using crossentropy
        """
        Loss_c2s2 = self.loss_mag.compute_loss(['a', 'e', 't'], pred['logprob_xxyy'], dict(a=GT['ccss_a'], e=GT['ccss_e'], t=GT['ccss_t']))
        Loss_direc = self.loss_direc.compute_loss(['a', 'e', 't'], pred['sign_x_y'], dict(a=GT['sign_a'], e=GT['sign_e'], t=GT['sign_t']))
        Loss = odict(ccss_a=Loss_c2s2['a'] * self.balance_weight, ccss_e=Loss_c2s2['e'] * self.balance_weight, ccss_t=Loss_c2s2['t'] * self.balance_weight, sign_a=Loss_direc['a'], sign_e=Loss_direc['e'], sign_t=Loss_direc['t'])
        return Loss

    @staticmethod
    def compute_vp_pred(network_op):
        lmap = torch.FloatTensor([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        lmap = Variable(lmap)
        bsize = network_op['logprob_xxyy']['a'].size(0)
        vp_pred = odict()
        for tgt in network_op['logprob_xxyy'].keys():
            logprob_xx_yy = network_op['logprob_xxyy'][tgt]
            abs_cos_sin = torch.sqrt(torch.exp(logprob_xx_yy))
            vp_pred['ccss_' + tgt] = torch.exp(logprob_xx_yy)
            sign_ind = torch.argmax(network_op['sign_x_y'][tgt].view(network_op['sign_x_y'][tgt].shape[0], 4), dim=1)
            vp_pred['sign_' + tgt] = sign_ind
            i_inds = torch.from_numpy(np.arange(bsize))
            direc_cos_sin = lmap.expand(bsize, 4, 2)[i_inds, sign_ind]
            cos_sin = abs_cos_sin.view(abs_cos_sin.shape[0], 2) * direc_cos_sin
            vp_pred[tgt] = torch.atan2(cos_sin[:, (1)], cos_sin[:, (0)])
        return vp_pred

    @staticmethod
    def compute_gt_flip(network_op, dtach=False):
        """
        Takes a prediction for an image and computes the GT for the corresponding flipped image. 
        For a flipped image, the magnitude of azimuth, elevation and tilt have to be the same.
        The signs/ directions for azimuth and tilt are flipped.
        So, for correct image : [ a,  e,  t] (from the input)
        For flipped image :     [-a,  e, -t] (produce GT representation for this)
        MAP :
        +, +  ->  +, - | 0 -> 1
        +, -  ->  +, + | 1 -> 0
        -, +  ->  -, - | 2 -> 3
        -, -  ->  -, + | 3 -> 2  
        """
        lmap = torch.FloatTensor([[1, 1], [1, -1], [-1, 1], [-1, -1]])
        lmap = Variable(lmap)
        batchsize = network_op['logprob_xxyy']['a'].size(0)
        vp_pred = edict()
        for tgt in network_op['logprob_xxyy'].keys():
            logprob_xx_yy = network_op['logprob_xxyy'][tgt]
            abs_cos_sin = torch.sqrt(torch.exp(logprob_xx_yy))
            vp_pred['ccss_' + tgt] = torch.exp(logprob_xx_yy)
            sign_ind = torch.argmax(network_op['sign_x_y'][tgt].view(network_op['sign_x_y'][tgt].shape[0], 4), dim=1)
            if tgt == 'a' or tgt == 't':
                sign_ind_flipped = 1 - sign_ind % 2 + 2 * (sign_ind // 2)
            else:
                sign_ind_flipped = sign_ind
            vp_pred['sign_' + tgt] = sign_ind_flipped
            item_inds = torch.from_numpy(np.arange(batchsize))
            sign_cos_sin = lmap.expand(batchsize, 4, 2)[item_inds, sign_ind]
            cos_sin = abs_cos_sin.view(abs_cos_sin.shape[0], 2) * sign_cos_sin
            vp_pred[tgt] = torch.atan2(cos_sin[:, (1)], cos_sin[:, (0)])
        return vp_pred


import torch
from torch.nn import MSELoss, ReLU
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile


TESTCASES = [
    # (nn.Module, init_args, forward_args, jit_compiles)
    (AdaptiveInstanceNorm,
     lambda: ([], {'in_channel': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4]), torch.rand([4, 4])], {}),
     True),
    (AdaptiveInstanceNorm3,
     lambda: ([], {'in_channel': 4, 'style_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4, 4]), torch.rand([4, 4])], {}),
     True),
    (ConvBlock,
     lambda: ([], {'in_channel': 4, 'out_channel': 4, 'kernel_size': 4, 'padding': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualConv2d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (EqualConv3d,
     lambda: ([], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
    (EqualLinear,
     lambda: ([], {'in_dim': 4, 'out_dim': 4}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (PixelNorm,
     lambda: ([], {}),
     lambda: ([torch.rand([4, 4, 4, 4])], {}),
     True),
    (StyledConvBlock3_noAdaIN,
     lambda: ([], {'in_channel': 4, 'out_channel': 4}),
     lambda: ([torch.rand([4, 4, 64, 64, 64])], {}),
     True),
]

class Test_NVlabs_SSV(_paritybench_base):
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

