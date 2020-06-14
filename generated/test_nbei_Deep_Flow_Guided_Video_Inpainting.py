import sys
_module = sys.modules[__name__]
del sys
FlowInfer = _module
FlowInitial = _module
FlowRefine = _module
dataset = _module
data_list = _module
DeepFill = _module
DeepFill_Models = _module
ops = _module
util = _module
FlowNet2 = _module
FlowNetC = _module
FlowNetFusion = _module
FlowNetS = _module
FlowNetSD = _module
FlowNet2_Models = _module
channelnorm_package = _module
_ext = _module
channelnorm = _module
build = _module
functions = _module
modules = _module
channelnorm = _module
correlation_package = _module
correlation = _module
correlation = _module
resample2d_package = _module
resample2d = _module
resample2d = _module
submodules = _module
models = _module
resnet_models = _module
tools = _module
frame_inpaint = _module
infer_flownet2 = _module
propagation_inpaint = _module
test_scripts = _module
train_initial = _module
train_refine = _module
video_inpaint = _module
utils = _module
flow = _module
image = _module
io = _module
loss_func = _module
region_fill = _module
runner_func = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


from torch.nn import init


from torch.nn.modules.module import Module


from torch.utils.data import DataLoader


import torch.optim as optim


from torch import nn


from torch.utils import data


from torch.backends import cudnn


from random import *


def to_var(x, volatile=False, device=None):
    if torch.cuda.is_available():
        if device:
            x = x.cuda(device)
        else:
            x = x.cuda()
    return Variable(x, volatile=volatile)


class Generator(nn.Module):

    def __init__(self, first_dim=32, isCheck=False, device=None):
        super(Generator, self).__init__()
        self.isCheck = isCheck
        self.device = device
        self.stage_1 = CoarseNet(5, first_dim, device=device)
        self.stage_2 = RefinementNet(5, first_dim, device=device)

    def forward(self, masked_img, mask, small_mask):
        mask = mask.expand(masked_img.size(0), 1, masked_img.size(2),
            masked_img.size(3))
        small_mask = small_mask.expand(masked_img.size(0), 1, masked_img.
            size(2) // 8, masked_img.size(3) // 8)
        if self.device:
            ones = to_var(torch.ones(mask.size()), device=self.device)
        else:
            ones = to_var(torch.ones(mask.size()))
        stage1_input = torch.cat([masked_img, ones, ones * mask], dim=1)
        stage1_output, resized_mask = self.stage_1(stage1_input, mask)
        new_masked_img = stage1_output * mask.clone() + masked_img.clone() * (
            1.0 - mask.clone())
        stage2_input = torch.cat([new_masked_img, ones.clone(), ones.clone(
            ) * mask.clone()], dim=1)
        stage2_output, offset_flow = self.stage_2(stage2_input, small_mask)
        return stage1_output, stage2_output, offset_flow


def down_sample(x, scalor=2, mode='bilinear'):
    if mode == 'bilinear':
        x = F.avg_pool2d(x, kernel_size=scalor, stride=scalor)
    elif mode == 'nearest':
        x = F.max_pool2d(x, kernel_size=scalor, stride=scalor)
    return x


class CoarseNet(nn.Module):
    """
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    """

    def __init__(self, in_ch, out_ch, device=None):
        super(CoarseNet, self).__init__()
        self.down = Down_Module(in_ch, out_ch)
        self.atrous = Dilation_Module(out_ch * 4, out_ch * 4)
        self.up = Up_Module(out_ch * 4, 3)
        self.device = device

    def forward(self, x, mask):
        x = self.down(x)
        resized_mask = down_sample(mask, scale_factor=0.25, mode='nearest',
            device=self.device)
        x = self.atrous(x)
        x = self.up(x)
        return x, resized_mask


class RefinementNet(nn.Module):
    """
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    """

    def __init__(self, in_ch, out_ch, device=None):
        super(RefinementNet, self).__init__()
        self.down_conv_branch = Down_Module(in_ch, out_ch, isRefine=True)
        self.down_attn_branch = Down_Module(in_ch, out_ch, activation=nn.
            ReLU(), isRefine=True, isAttn=True)
        self.atrous = Dilation_Module(out_ch * 4, out_ch * 4)
        self.CAttn = Contextual_Attention_Module(out_ch * 4, out_ch * 4,
            device=device)
        self.up = Up_Module(out_ch * 8, 3, isRefine=True)

    def forward(self, x, resized_mask):
        conv_x = self.down_conv_branch(x)
        conv_x = self.atrous(conv_x)
        attn_x = self.down_attn_branch(x)
        attn_x, offset_flow = self.CAttn(attn_x, attn_x, mask=resized_mask)
        deconv_x = torch.cat([conv_x, attn_x], dim=1)
        x = self.up(deconv_x)
        return x, offset_flow


def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0
            ) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, 'Unsupported initialization: {}'.format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun


class Conv(nn.Module):

    def __init__(self, in_ch, out_ch, K=3, S=1, P=1, D=1, activation=nn.ELU
        (), isGated=False):
        super(Conv, self).__init__()
        if activation is not None:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=
                K, stride=S, padding=P, dilation=D), activation)
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=
                K, stride=S, padding=P, dilation=D))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init('kaiming'))

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_Downsample(nn.Module):

    def __init__(self, in_ch, out_ch, K=3, S=1, P=1, D=1, activation=nn.ELU()):
        super(Conv_Downsample, self).__init__()
        PaddingLayer = torch.nn.ZeroPad2d((0, (K - 1) // 2, 0, (K - 1) // 2))
        if activation is not None:
            self.conv = nn.Sequential(PaddingLayer, nn.Conv2d(in_ch, out_ch,
                kernel_size=K, stride=S, padding=0, dilation=D), activation)
        else:
            self.conv = nn.Sequential(PaddingLayer, nn.Conv2d(in_ch, out_ch,
                kernel_size=K, stride=S, padding=0, dilation=D))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.apply(weights_init('kaiming'))

    def forward(self, x):
        x = self.conv(x)
        return x


class Down_Module(nn.Module):

    def __init__(self, in_ch, out_ch, activation=nn.ELU(), isRefine=False,
        isAttn=False):
        super(Down_Module, self).__init__()
        layers = []
        layers.append(Conv(in_ch, out_ch, K=5, P=2))
        curr_dim = out_ch
        if isRefine:
            if isAttn:
                layers.append(Conv_Downsample(curr_dim, curr_dim, K=3, S=2))
                layers.append(Conv(curr_dim, 2 * curr_dim, K=3, S=1))
                layers.append(Conv_Downsample(2 * curr_dim, 4 * curr_dim, K
                    =3, S=2))
                layers.append(Conv(4 * curr_dim, 4 * curr_dim, K=3, S=1))
                curr_dim *= 4
            else:
                for i in range(2):
                    layers.append(Conv_Downsample(curr_dim, curr_dim, K=3, S=2)
                        )
                    layers.append(Conv(curr_dim, curr_dim * 2))
                    curr_dim *= 2
        else:
            for i in range(2):
                layers.append(Conv_Downsample(curr_dim, curr_dim * 2, K=3, S=2)
                    )
                layers.append(Conv(curr_dim * 2, curr_dim * 2))
                curr_dim *= 2
        layers.append(Conv(curr_dim, curr_dim, activation=activation))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


class Dilation_Module(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(Dilation_Module, self).__init__()
        layers = []
        dilation = 1
        for i in range(4):
            dilation *= 2
            layers.append(Conv(in_ch, out_ch, D=dilation, P=dilation))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


class Up_Module(nn.Module):

    def __init__(self, in_ch, out_ch, isRefine=False):
        super(Up_Module, self).__init__()
        layers = []
        curr_dim = in_ch
        if isRefine:
            layers.append(Conv(curr_dim, curr_dim // 2))
            curr_dim //= 2
        else:
            layers.append(Conv(curr_dim, curr_dim))
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim))
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(Conv(curr_dim, curr_dim // 2))
            curr_dim //= 2
        layers.append(Conv(curr_dim, curr_dim // 2))
        layers.append(Conv(curr_dim // 2, out_ch, activation=None))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        output = self.out(x)
        return torch.clamp(output, min=-1.0, max=1.0)


class Up_Module_CNet(nn.Module):

    def __init__(self, in_ch, out_ch, isRefine=False, isGated=False):
        super(Up_Module_CNet, self).__init__()
        layers = []
        curr_dim = in_ch
        if isRefine:
            layers.append(Conv(curr_dim, curr_dim // 2, isGated=isGated))
            curr_dim //= 2
        else:
            layers.append(Conv(curr_dim, curr_dim, isGated=isGated))
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim, isGated=isGated))
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(Conv(curr_dim, curr_dim // 2, isGated=isGated))
            curr_dim //= 2
        layers.append(Conv(curr_dim, curr_dim // 2, isGated=isGated))
        layers.append(Conv(curr_dim // 2, out_ch, activation=None, isGated=
            isGated))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        output = self.out(x)
        return output


class Flatten_Module(nn.Module):

    def __init__(self, in_ch, out_ch, isLocal=True):
        super(Flatten_Module, self).__init__()
        layers = []
        layers.append(Conv(in_ch, out_ch, K=5, S=2, P=2, activation=nn.
            LeakyReLU()))
        curr_dim = out_ch
        for i in range(2):
            layers.append(Conv(curr_dim, curr_dim * 2, K=5, S=2, P=2,
                activation=nn.LeakyReLU()))
            curr_dim *= 2
        if isLocal:
            layers.append(Conv(curr_dim, curr_dim * 2, K=5, S=2, P=2,
                activation=nn.LeakyReLU()))
        else:
            layers.append(Conv(curr_dim, curr_dim, K=5, S=2, P=2,
                activation=nn.LeakyReLU()))
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        x = self.out(x)
        return x.view(x.size(0), -1)


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


def reduce_mean(x):
    for i in range(4):
        if i == 1:
            continue
        x = torch.mean(x, dim=i, keepdim=True)
    return x


class Contextual_Attention_Module(nn.Module):

    def __init__(self, in_ch, out_ch, rate=2, stride=1, isCheck=False,
        device=None):
        super(Contextual_Attention_Module, self).__init__()
        self.rate = rate
        self.padding = nn.ZeroPad2d(1)
        self.up_sample = nn.Upsample(scale_factor=self.rate, mode='nearest')
        layers = []
        for i in range(2):
            layers.append(Conv(in_ch, out_ch))
        self.out = nn.Sequential(*layers)
        self.isCheck = isCheck
        self.device = device

    def forward(self, f, b, mask=None, ksize=3, stride=1, fuse_k=3,
        softmax_scale=10.0, training=True, fuse=True):
        """ Contextual attention layer implementation.

        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.

        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
            training: Indicating if current graph is training or inference.

        Returns:
            tf.Tensor: output

        """
        raw_fs = f.size()
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())
        kernel = 2 * self.rate
        raw_w = self.extract_patches(b, kernel=kernel, stride=self.rate)
        raw_w = raw_w.permute(0, 2, 3, 4, 5, 1)
        raw_w = raw_w.contiguous().view(raw_int_bs[0], raw_int_bs[2] / self
            .rate, raw_int_bs[3] / self.rate, -1)
        raw_w = raw_w.contiguous().view(raw_int_bs[0], -1, kernel, kernel,
            raw_int_bs[1])
        raw_w = raw_w.permute(0, 1, 4, 2, 3)
        f = down_sample(f, scale_factor=1 / self.rate, mode='nearest',
            device=self.device)
        b = down_sample(b, scale_factor=1 / self.rate, mode='nearest',
            device=self.device)
        fs = f.size()
        int_fs = list(f.size())
        f_groups = torch.split(f, 1, dim=0)
        bs = b.size()
        int_bs = list(b.size())
        w = self.extract_patches(b)
        w = w.permute(0, 2, 3, 4, 5, 1)
        w = w.contiguous().view(raw_int_bs[0], raw_int_bs[2] / self.rate, 
            raw_int_bs[3] / self.rate, -1)
        w = w.contiguous().view(raw_int_bs[0], -1, ksize, ksize, raw_int_bs[1])
        w = w.permute(0, 1, 4, 2, 3)
        mask = mask.clone()
        if mask is not None:
            if mask.size(2) != b.size(2):
                mask = down_sample(mask, scale_factor=1.0 / self.rate, mode
                    ='nearest', device=self.device)
        else:
            mask = torch.zeros([1, 1, bs[2], bs[3]])
        m = self.extract_patches(mask)
        m = m.permute(0, 2, 3, 4, 5, 1)
        m = m.contiguous().view(raw_int_bs[0], raw_int_bs[2] / self.rate, 
            raw_int_bs[3] / self.rate, -1)
        m = m.contiguous().view(raw_int_bs[0], -1, ksize, ksize, 1)
        m = m.permute(0, 4, 1, 2, 3)
        m = m[0]
        m = reduce_mean(m)
        mm = m.eq(0.0).float()
        w_groups = torch.split(w, 1, dim=0)
        raw_w_groups = torch.split(raw_w, 1, dim=0)
        y = []
        offsets = []
        k = fuse_k
        scale = softmax_scale
        fuse_weight = Variable(torch.eye(k).view(1, 1, k, k))
        y_test = []
        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            """
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            """
            wi = wi[0]
            escape_NaN = Variable(torch.FloatTensor([0.0001]))
            wi_normed = wi / torch.max(l2_norm(wi), escape_NaN)
            yi = F.conv2d(xi, wi_normed, stride=1, padding=1)
            y_test.append(yi)
            if fuse:
                yi = yi.permute(0, 2, 3, 1)
                yi = yi.contiguous().view(1, fs[2] * fs[3], bs[2] * bs[3], 1)
                yi = yi.permute(0, 3, 1, 2)
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)
                yi = yi.permute(0, 2, 3, 1)
                yi = yi.contiguous().view(1, fs[2], fs[3], bs[2], bs[3])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, fs[2] * fs[3], bs[2] * bs[3], 1)
                yi = yi.permute(0, 3, 1, 2)
                yi = F.conv2d(yi, fuse_weight, stride=1, padding=1)
                yi = yi.permute(0, 2, 3, 1)
                yi = yi.contiguous().view(1, fs[3], fs[2], bs[3], bs[2])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, fs[2], fs[3], bs[2] * bs[3])
                yi = yi.permute(0, 3, 1, 2)
            else:
                yi = yi.permute(0, 2, 3, 1)
                yi = yi.contiguous().view(1, fs[2], fs[3], bs[2] * bs[3])
                yi = yi.permute(0, 3, 1, 2)
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm
            _, offset = torch.max(yi, dim=1)
            division = torch.div(offset, fs[3]).long()
            offset = torch.stack([division, torch.div(offset, fs[3]) -
                division], dim=-1)
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1
                ) / 4.0
            y.append(yi)
            offsets.append(offset)
        y = torch.cat(y, dim=0)
        y.contiguous().view(raw_int_fs)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view([int_bs[0]] + [2] + int_bs[2:])
        h_add = Variable(torch.arange(0, float(bs[2]))).view([1, 1, bs[2], 1])
        h_add = h_add.expand(bs[0], 1, bs[2], bs[3])
        w_add = Variable(torch.arange(0, float(bs[3]))).view([1, 1, 1, bs[3]])
        w_add = w_add.expand(bs[0], 1, bs[2], bs[3])
        offsets = offsets - torch.cat([h_add, w_add], dim=1).long()
        y = self.out(y)
        return y, offsets

    def extract_patches(self, x, kernel=3, stride=1):
        x = self.padding(x)
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        return all_patches


class FlowNet2(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow=20.0, requires_grad=
        False):
        super(FlowNet2, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.flownets_d = FlowNetSD.FlowNetSD(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='nearest')
        if args.fp16:
            self.resample3 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample3 = Resample2d()
        if args.fp16:
            self.resample4 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample4 = Resample2d()
        self.flownetfusion = FlowNetFusion.FlowNetFusion(args, batchNorm=
            self.batchNorm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def init_deconv_bilinear(self, weight):
        f_shape = weight.size()
        heigh, width = f_shape[-2], f_shape[-1]
        f = np.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        min_dim = min(f_shape[0], f_shape[1])
        weight.data.fill_(0.0)
        for i in range(min_dim):
            weight.data[(i), (i), :, :] = torch.from_numpy(bilinear)
        return

    def forward(self, img1, img2):
        sz = img1.size()
        img1 = img1.view(sz[0], sz[1], 1, sz[2], sz[3])
        img2 = img2.view(sz[0], sz[1], 1, sz[2], sz[3])
        inputs = torch.cat((img1, img2), dim=2)
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim
            =-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample4(flownets2_flow2 * self.div_flow)
        norm_flownets2_flow = self.channelnorm(flownets2_flow)
        diff_flownets2_flow = self.resample4(x[:, 3:, :, :], flownets2_flow)
        diff_flownets2_img1 = self.channelnorm(x[:, :3, :, :] -
            diff_flownets2_flow)
        flownetsd_flow2 = self.flownets_d(x)[0]
        flownetsd_flow = self.upsample3(flownetsd_flow2 / self.div_flow)
        norm_flownetsd_flow = self.channelnorm(flownetsd_flow)
        diff_flownetsd_flow = self.resample3(x[:, 3:, :, :], flownetsd_flow)
        diff_flownetsd_img1 = self.channelnorm(x[:, :3, :, :] -
            diff_flownetsd_flow)
        concat3 = torch.cat((x[:, :3, :, :], flownetsd_flow, flownets2_flow,
            norm_flownetsd_flow, norm_flownets2_flow, diff_flownetsd_img1,
            diff_flownets2_img1), dim=1)
        flownetfusion_flow = self.flownetfusion(concat3)
        return flownetfusion_flow


class FlowNet2CS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow=20.0):
        super(FlowNet2CS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim
            =-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        return flownets1_flow


class FlowNet2CSS(nn.Module):

    def __init__(self, args, batchNorm=False, div_flow=20.0):
        super(FlowNet2CSS, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.rgb_max = args.rgb_max
        self.args = args
        self.channelnorm = ChannelNorm()
        self.flownetc = FlowNetC.FlowNetC(args, batchNorm=self.batchNorm)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample1 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample1 = Resample2d()
        self.flownets_1 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        if args.fp16:
            self.resample2 = nn.Sequential(tofp32(), Resample2d(), tofp16())
        else:
            self.resample2 = Resample2d()
        self.flownets_2 = FlowNetS.FlowNetS(args, batchNorm=self.batchNorm)
        self.upsample3 = nn.Upsample(scale_factor=4, mode='nearest')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2] + (-1,)).mean(dim
            =-1).view(inputs.size()[:2] + (1, 1, 1))
        x = (inputs - rgb_mean) / self.rgb_max
        x1 = x[:, :, (0), :, :]
        x2 = x[:, :, (1), :, :]
        x = torch.cat((x1, x2), dim=1)
        flownetc_flow2 = self.flownetc(x)[0]
        flownetc_flow = self.upsample1(flownetc_flow2 * self.div_flow)
        resampled_img1 = self.resample1(x[:, 3:, :, :], flownetc_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat1 = torch.cat((x, resampled_img1, flownetc_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets1_flow2 = self.flownets_1(concat1)[0]
        flownets1_flow = self.upsample2(flownets1_flow2 * self.div_flow)
        resampled_img1 = self.resample2(x[:, 3:, :, :], flownets1_flow)
        diff_img0 = x[:, :3, :, :] - resampled_img1
        norm_diff_img0 = self.channelnorm(diff_img0)
        concat2 = torch.cat((x, resampled_img1, flownets1_flow / self.
            div_flow, norm_diff_img0), dim=1)
        flownets2_flow2 = self.flownets_2(concat2)[0]
        flownets2_flow = self.upsample3(flownets2_flow2 * self.div_flow)
        return flownets2_flow


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias
        =True)


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
            bias=False), nn.BatchNorm2d(out_planes), nn.LeakyReLU(0.1,
            inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
            bias=True), nn.LeakyReLU(0.1, inplace=True))


def deconv(in_planes, out_planes):
    return nn.Sequential(nn.ConvTranspose2d(in_planes, out_planes,
        kernel_size=4, stride=2, padding=1, bias=True), nn.LeakyReLU(0.1,
        inplace=True))


class FlowNetC(nn.Module):

    def __init__(self, args, batchNorm=True, div_flow=20):
        super(FlowNetC, self).__init__()
        self.batchNorm = batchNorm
        self.div_flow = div_flow
        self.conv1 = conv(self.batchNorm, 3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(self.batchNorm, 256, 32, kernel_size=1, stride=1
            )
        if args.fp16:
            self.corr = nn.Sequential(tofp32(), Correlation(pad_size=20,
                kernel_size=1, max_displacement=20, stride1=1, stride2=2,
                corr_multiply=1), tofp16())
        else:
            self.corr = Correlation(pad_size=20, kernel_size=1,
                max_displacement=20, stride1=1, stride2=2, corr_multiply=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv3_1 = conv(self.batchNorm, 473, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True
            )
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True
            )
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True
            )
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=True
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x2 = x[:, 3:, :, :]
        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)
        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)
        out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)
        out_conv_redir = self.conv_redir(out_conv3a)
        in_conv3_1 = torch.cat((out_conv_redir, out_corr), 1)
        out_conv3_1 = self.conv3_1(in_conv3_1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3_1))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3_1, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2a, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return (flow2, flow3, flow4, flow5, flow6, out_conv3_1,
                out_conv4, out_conv5, out_conv6)
        else:
            return flow2,
        return flow2, flow3, flow4, flow5, flow6


def i_conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, bias=True
    ):
    if batchNorm:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
            bias=bias), nn.BatchNorm2d(out_planes))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
            bias=bias))


class FlowNetFusion(nn.Module):

    def __init__(self, args, batchNorm=True):
        super(FlowNetFusion, self).__init__()
        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, 11, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)
        self.deconv1 = deconv(128, 32)
        self.deconv0 = deconv(162, 16)
        self.inter_conv1 = i_conv(self.batchNorm, 162, 32)
        self.inter_conv0 = i_conv(self.batchNorm, 82, 16)
        self.predict_flow2 = predict_flow(128)
        self.predict_flow1 = predict_flow(32)
        self.predict_flow0 = predict_flow(16)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        flow2 = self.predict_flow2(out_conv2)
        flow2_up = self.upsampled_flow2_to_1(flow2)
        out_deconv1 = self.deconv1(out_conv2)
        concat1 = torch.cat((out_conv1, out_deconv1, flow2_up), 1)
        out_interconv1 = self.inter_conv1(concat1)
        flow1 = self.predict_flow1(out_interconv1)
        flow1_up = self.upsampled_flow1_to_0(flow1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((out_conv0, out_deconv0, flow1_up), 1)
        out_interconv0 = self.inter_conv0(concat0)
        flow0 = self.predict_flow0(out_interconv0)
        return flow0


class FlowNetS(nn.Module):

    def __init__(self, args, input_channels=12, batchNorm=True):
        super(FlowNetS, self).__init__()
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, input_channels, 64, kernel_size=7,
            stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=
            False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=
            False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=
            False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=
            False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        flow5 = self.predict_flow5(concat5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        flow4 = self.predict_flow4(concat4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        flow3 = self.predict_flow3(concat3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        flow2 = self.predict_flow2(concat2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class FlowNetSD(nn.Module):

    def __init__(self, args, batchNorm=True):
        super(FlowNetSD, self).__init__()
        self.batchNorm = batchNorm
        self.conv0 = conv(self.batchNorm, 6, 64)
        self.conv1 = conv(self.batchNorm, 64, 64, stride=2)
        self.conv1_1 = conv(self.batchNorm, 64, 128)
        self.conv2 = conv(self.batchNorm, 128, 128, stride=2)
        self.conv2_1 = conv(self.batchNorm, 128, 128)
        self.conv3 = conv(self.batchNorm, 128, 256, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256, 256)
        self.conv4 = conv(self.batchNorm, 256, 512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512, 512)
        self.conv5 = conv(self.batchNorm, 512, 512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512, 512)
        self.conv6 = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm, 1024, 1024)
        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1026, 256)
        self.deconv3 = deconv(770, 128)
        self.deconv2 = deconv(386, 64)
        self.inter_conv5 = i_conv(self.batchNorm, 1026, 512)
        self.inter_conv4 = i_conv(self.batchNorm, 770, 256)
        self.inter_conv3 = i_conv(self.batchNorm, 386, 128)
        self.inter_conv2 = i_conv(self.batchNorm, 194, 64)
        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(512)
        self.predict_flow4 = predict_flow(256)
        self.predict_flow3 = predict_flow(128)
        self.predict_flow2 = predict_flow(64)
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    init.uniform_(m.bias)
                init.xavier_uniform_(m.weight)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, x):
        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))
        flow6 = self.predict_flow6(out_conv6)
        flow6_up = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        concat5 = torch.cat((out_conv5, out_deconv5, flow6_up), 1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5 = self.predict_flow5(out_interconv5)
        flow5_up = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        concat4 = torch.cat((out_conv4, out_deconv4, flow5_up), 1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4 = self.predict_flow4(out_interconv4)
        flow4_up = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        concat3 = torch.cat((out_conv3, out_deconv3, flow4_up), 1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_up = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2, flow3_up), 1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)
        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2,


class ChannelNormFunction(Function):

    def __init__(self, norm_deg=2):
        super(ChannelNormFunction, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        assert input1.is_contiguous() == True
        with torch.cuda.device_of(input1):
            b, _, h, w = input1.size()
            output = input1.new().resize_(b, 1, h, w).zero_()
            channelnorm.ChannelNorm_cuda_forward(input1, output, self.norm_deg)
        self.save_for_backward(input1, output)
        return output

    def backward(self, gradOutput):
        input1, output = self.saved_tensors
        with torch.cuda.device_of(input1):
            b, c, h, w = input1.size()
            gradInput1 = input1.new().resize_(b, c, h, w).zero_()
            channelnorm.ChannelNorm_cuda_backward(input1, output,
                gradOutput, gradInput1, self.norm_deg)
        return gradInput1


class ChannelNorm(Module):

    def __init__(self, norm_deg=2):
        super(ChannelNorm, self).__init__()
        self.norm_deg = norm_deg

    def forward(self, input1):
        result = ChannelNormFunction(self.norm_deg)(input1)
        return result


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20,
        stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        assert input1.is_contiguous() == True
        assert input2.is_contiguous() == True
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()
            correlation.Correlation_forward_cuda(input1, input2, rbot1,
                rbot2, output, self.pad_size, self.kernel_size, self.
                max_displacement, self.stride1, self.stride2, self.
                corr_multiply)
        return output

    def backward(self, grad_output):
        input1, input2 = self.saved_tensors
        assert grad_output.is_contiguous() == True
        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            grad_input1 = input1.new()
            grad_input2 = input2.new()
            correlation.Correlation_backward_cuda(input1, input2, rbot1,
                rbot2, grad_output, grad_input1, grad_input2, self.pad_size,
                self.kernel_size, self.max_displacement, self.stride1, self
                .stride2, self.corr_multiply)
        return grad_input1, grad_input2


class Correlation(Module):

    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0,
        stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def forward(self, input1, input2):
        result = CorrelationFunction(self.pad_size, self.kernel_size, self.
            max_displacement, self.stride1, self.stride2, self.corr_multiply)(
            input1, input2)
        return result


class Resample2dFunction(Function):

    def __init__(self, kernel_size=1):
        super(Resample2dFunction, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        self.save_for_backward(input1, input2)
        assert input1.is_contiguous() == True
        assert input2.is_contiguous() == True
        with torch.cuda.device_of(input1):
            _, d, _, _ = input1.size()
            b, _, h, w = input2.size()
            output = input1.new().resize_(b, d, h, w).zero_()
            resample2d.Resample2d_cuda_forward(input1, input2, output, self
                .kernel_size)
        return output

    def backward(self, gradOutput):
        input1, input2 = self.saved_tensors
        assert gradOutput.is_contiguous() == True
        with torch.cuda.device_of(input1):
            b, c, h, w = input1.size()
            gradInput1 = input1.new().resize_(b, c, h, w).zero_()
            b, c, h, w = input2.size()
            gradInput2 = input2.new().resize_(b, c, h, w).zero_()
            resample2d.Resample2d_cuda_backward(input1, input2, gradOutput,
                gradInput1, gradInput2, self.kernel_size)
        return gradInput1, gradInput2


class Resample2d(Module):

    def __init__(self, kernel_size=1):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        result = Resample2dFunction(self.kernel_size)(input1_c, input2)
        return result


class tofp16(nn.Module):

    def __init__(self):
        super(tofp16, self).__init__()

    def forward(self, input):
        return input.half()


class tofp32(nn.Module):

    def __init__(self):
        super(tofp32, self).__init__()

    def forward(self, input):
        return input.float()


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


affine_par = True


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None
        ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None
        ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=
            stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
            padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class FlowBranch_Layer(nn.Module):

    def __init__(self, input_chanels, NoLabels):
        super(FlowBranch_Layer, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.upconv1 = nn.Conv2d(input_chanels, input_chanels // 2,
            kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(input_chanels // 2, 256, kernel_size=3,
            stride=1, padding=1)
        self.conv1_flow = nn.Conv2d(256, NoLabels, kernel_size=1, stride=1,
            padding=0)
        self.conv2_flow = nn.Conv2d(256, 128, kernel_size=3, stride=1,
            padding=1)
        self.conv3_flow = nn.Conv2d(128 + NoLabels, NoLabels, kernel_size=3,
            stride=1, padding=1)
        self.conv4_flow = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1
            )
        self.conv5_flow = nn.Conv2d(64 + NoLabels, NoLabels, kernel_size=3,
            stride=1, padding=1)

    def forward(self, x, input_size):
        x = self.upconv1(x)
        x = self.relu(x)
        x = F.upsample(x, (input_size[0] // 4, input_size[1] // 4), mode=
            'bilinear', align_corners=False)
        x = self.upconv2(x)
        x = self.relu(x)
        res_4x = self.conv1_flow(x)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2_flow(x)
        x = self.relu(x)
        res_4x_up = F.upsample(res_4x, scale_factor=2, mode='bilinear',
            align_corners=False)
        conv3_input = torch.cat([x, res_4x_up], dim=1)
        res_2x = self.conv3_flow(conv3_input)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv4_flow(x)
        x = self.relu(x)
        res_2x_up = F.upsample(res_2x, scale_factor=2, mode='bilinear',
            align_corners=False)
        conv5_input = torch.cat([x, res_2x_up], dim=1)
        res_1x = self.conv5_flow(conv5_input)
        return res_1x, res_2x, res_4x


class FlowModule_MultiScale(nn.Module):

    def __init__(self, input_chanels, NoLabels):
        super(FlowModule_MultiScale, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_chanels, 256, kernel_size=5, stride=1,
            padding=2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, NoLabels, kernel_size=1, padding=0)

    def forward(self, x, res_size):
        x = F.upsample(x, (res_size[0] // 4, res_size[1] // 4), mode=
            'bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.relu(x)
        x = F.upsample(x, (res_size[0] // 2, res_size[1] // 2), mode=
            'bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.relu(x)
        x = F.upsample(x, res_size, mode='bilinear', align_corners=False)
        x = self.conv3(x)
        return x


class FlowModule_SingleScale(nn.Module):

    def __init__(self, input_channels, NoLabels):
        super(FlowModule_SingleScale, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, NoLabels, kernel_size=1,
            padding=0)

    def forward(self, x, res_size):
        x = self.conv1(x)
        x = F.upsample(x, res_size, mode='bilinear')
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, input_chanels, NoLabels,
        Layer5_Module=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_chanels, 64, kernel_size=7, stride=2,
            padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
            dilation__=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation__=4)
        if Layer5_Module is not None:
            self.layer5 = Layer5_Module
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation__=1):
        downsample = None
        if (stride != 1 or self.inplanes != planes * block.expansion or 
            dilation__ == 2 or dilation__ == 4):
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation_=
            dilation__, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation_=dilation__))
        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()[2:4]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        res = self.layer5(x, input_size)
        return res

    def train(self, mode=True, freezeBn=True):
        super(ResNet, self).train(mode=mode)
        if freezeBn:
            None
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_nbei_Deep_Flow_Guided_Video_Inpainting(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Conv(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(Conv_Downsample(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(Dilation_Module(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_004(self):
        self._check(Down_Module(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(Flatten_Module(*[], **{'in_ch': 4, 'out_ch': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(FlowNetFusion(*[], **{'args': _mock_config()}), [torch.rand([4, 11, 64, 64])], {})

    @_fails_compile()
    def test_007(self):
        self._check(FlowNetS(*[], **{'args': _mock_config()}), [torch.rand([4, 12, 64, 64])], {})

    @_fails_compile()
    def test_008(self):
        self._check(FlowNetSD(*[], **{'args': _mock_config()}), [torch.rand([4, 6, 64, 64])], {})

    def test_009(self):
        self._check(Up_Module(*[], **{'in_ch': 64, 'out_ch': 4}), [torch.rand([4, 64, 64, 64])], {})

    def test_010(self):
        self._check(Up_Module_CNet(*[], **{'in_ch': 64, 'out_ch': 4}), [torch.rand([4, 64, 64, 64])], {})

    def test_011(self):
        self._check(tofp16(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_012(self):
        self._check(tofp32(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

