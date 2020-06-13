import sys
_module = sys.modules[__name__]
del sys
data = _module
dataset = _module
model = _module
networks = _module
test_single = _module
test_tf_model = _module
train = _module
trainer = _module
utils = _module
logger = _module
tools = _module

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


import torch.nn.functional as F


from torch.nn.utils import spectral_norm as spectral_norm_fn


from torch.nn.utils import weight_norm as weight_norm_fn


import random


import torch.backends.cudnn as cudnn


from torch import autograd


import numpy as np


class Generator(nn.Module):

    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum,
            self.use_cuda, self.device_ids)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum, self
            .use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0,
    rate=1, activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
        conv_padding=padding, dilation=rate, activation=activation)


class CoarseGenerator(nn.Module):

    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum * 2, 3, 2, 1)
        self.conv3 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        self.conv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)
        self.conv11 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.conv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.conv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)
        self.conv17 = gen_conv(cnum // 2, input_dim, 3, 1, 1, activation='none'
            )

    def forward(self, x, mask):
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones
            mask = mask
        x = self.conv1(torch.cat([x, ones, mask], dim=1))
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x_stage1 = torch.clamp(x, -1.0, 1.0)
        return x_stage1


class FineGenerator(nn.Module):

    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        self.conv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum * 2, cnum * 2, 3, 2, 1)
        self.conv5 = gen_conv(cnum * 2, cnum * 4, 3, 1, 1)
        self.conv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.conv7_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum * 4, cnum * 4, 3, 1, 16, rate=16)
        self.pmconv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        self.pmconv3 = gen_conv(cnum, cnum * 2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum * 2, cnum * 4, 3, 2, 1)
        self.pmconv5 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention(ksize=3, stride=1,
            rate=2, fuse_k=3, softmax_scale=10, fuse=True, use_cuda=self.
            use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum * 8, cnum * 4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum * 4, cnum * 4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum * 4, cnum * 2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum * 2, cnum * 2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum * 2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum // 2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum // 2, input_dim, 3, 1, 1, activation
            ='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1.0 - mask)
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones
            mask = mask
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = torch.clamp(x, -1.0, 1.0)
        return x_stage2, offset_flow


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    colorwheel[0:RY, (0)] = 255
    colorwheel[0:RY, (1)] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    colorwheel[col:col + YG, (0)] = 255 - np.transpose(np.floor(255 * np.
        arange(0, YG) / YG))
    colorwheel[col:col + YG, (1)] = 255
    col += YG
    colorwheel[col:col + GC, (1)] = 255
    colorwheel[col:col + GC, (2)] = np.transpose(np.floor(255 * np.arange(0,
        GC) / GC))
    col += GC
    colorwheel[col:col + CB, (1)] = 255 - np.transpose(np.floor(255 * np.
        arange(0, CB) / CB))
    colorwheel[col:col + CB, (2)] = 255
    col += CB
    colorwheel[col:col + BM, (2)] = 255
    colorwheel[col:col + BM, (0)] = np.transpose(np.floor(255 * np.arange(0,
        BM) / BM))
    col += +BM
    colorwheel[col:col + MR, (2)] = 255 - np.transpose(np.floor(255 * np.
        arange(0, MR) / MR))
    colorwheel[col:col + MR, (0)] = 255
    return colorwheel


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, (i)]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, (i)] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[(i), :, :, (0)]
        v = flow[(i), :, :, (1)]
        idxunknow = (abs(u) > 10000000.0) | (abs(v) > 10000000.0)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    padding_top = int(padding_rows / 2.0)
    padding_left = int(padding_cols / 2.0)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = padding_left, padding_right, padding_top, padding_bottom
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError(
            'Unsupported padding type: {}.                Only "same" or "valid" are supported.'
            .format(padding))
    unfold = torch.nn.Unfold(kernel_size=ksizes, dilation=rates, padding=0,
        stride=strides)
    patches = unfold(images)
    return patches


class ContextualAttention(nn.Module):

    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=
        10, fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
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
        Returns:
            torch.tensor: output
        """
        raw_int_fs = list(f.size())
        raw_int_bs = list(b.size())
        kernel = 2 * self.rate
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel], strides=[
            self.rate * self.stride, self.rate * self.stride], rates=[1, 1],
            padding='same')
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)
        raw_w_groups = torch.split(raw_w, 1, dim=0)
        f = F.interpolate(f, scale_factor=1.0 / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1.0 / self.rate, mode='nearest')
        int_fs = list(f.size())
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
            strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)
        w_groups = torch.split(w, 1, dim=0)
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask
        else:
            mask = F.interpolate(mask, scale_factor=1.0 / (4 * self.rate),
                mode='nearest')
        int_ms = list(mask.size())
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
            strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)
        m = m[0]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.0).to(torch
            .float32)
        mm = mm.permute(1, 0, 2, 3)
        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale
        fuse_weight = torch.eye(k).view(1, 1, k, k)
        if self.use_cuda:
            fuse_weight = fuse_weight
        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            """
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            """
            escape_NaN = torch.FloatTensor([0.0001])
            if self.use_cuda:
                escape_NaN = escape_NaN
            wi = wi[0]
            max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2), axis
                =[1, 2, 3], keepdim=True)), escape_NaN)
            wi_normed = wi / max_wi
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])
            yi = F.conv2d(xi, wi_normed, stride=1)
            if self.fuse:
                yi = yi.view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3]
                    )
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2
                    ], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2] * int_bs[3], 
                    int_fs[2] * int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3
                    ], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm
            offset = torch.argmax(yi, dim=1, keepdim=True)
            if int_bs != int_fs:
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] *
                    int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset // int_fs[3], offset % int_fs[3]], dim=1
                )
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1
                ) / 4.0
            y.append(yi)
            offsets.append(offset)
        y = torch.cat(y, dim=0)
        y.contiguous().view(raw_int_fs)
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])
        h_add = torch.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(
            int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(
            int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = torch.cat([h_add, w_add], dim=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate
        offsets = offsets - ref_coordinate
        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).
            cpu().data.numpy())) / 255.0
        flow = flow.permute(0, 3, 1, 2)
        if self.use_cuda:
            flow = flow
        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate * 4, mode=
                'nearest')
        return y, flow


class LocalDis(nn.Module):

    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum * 4 * 8 * 8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x


class GlobalDis(nn.Module):

    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum * 4 * 16 * 16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)
        return x


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0,
    rate=1, activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
        conv_padding=padding, dilation=rate, activation=activation)


class DisConvModule(nn.Module):

    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum * 2, 5, 2, 2)
        self.conv3 = dis_conv(cnum * 2, cnum * 4, 5, 2, 2)
        self.conv4 = dis_conv(cnum * 4, cnum * 4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class Conv2dBlock(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=
        0, conv_padding=0, dilation=1, weight_norm='none', norm='none',
        activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, 'Unsupported padding type: {}'.format(pad_type)
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(norm)
        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, 'Unsupported normalization: {}'.format(weight_norm)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, 'Unsupported activation: {}'.format(activation)
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                kernel_size, stride, padding=conv_padding, output_padding=
                conv_padding, dilation=dilation, bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size,
                stride, padding=conv_padding, dilation=dilation, bias=self.
                use_bias)
        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config['spatial_discounting_gamma']
    height, width = config['mask_shape']
    shape = [1, 1, height, width]
    if config['discounted_mask']:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = max(gamma ** min(i, height - i), gamma **
                    min(j, width - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 0)
    else:
        mask_values = np.ones(shape)
    spatial_discounting_mask_tensor = torch.tensor(mask_values, dtype=torch
        .float32)
    if config['cuda']:
        spatial_discounting_mask_tensor = spatial_discounting_mask_tensor.cuda(
            )
    return spatial_discounting_mask_tensor


def local_patch(x, bbox_list):
    assert len(x.size()) == 4
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[(i), :, t:t + h, l:l + w])
    return torch.stack(patches, dim=0)


def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if 
        os.path.isfile(os.path.join(dirname, f)) and key in f and '.pt' in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name


def date_uid():
    """Generate a unique id based on date.

    Returns:
        str: Return uid string, e.g. '20171122171307111552'.

    """
    return str(datetime.datetime.now()).replace('-', '').replace(' ', ''
        ).replace(':', '').replace('.', '')


def get_logger(checkpoint_path=None):
    """
    Get the root logger
    :param checkpoint_path: only specify this when the first time call it
    :return: the root logger
    """
    if checkpoint_path:
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        stream_hdlr = logging.StreamHandler(sys.stdout)
        log_filename = date_uid()
        file_hdlr = logging.FileHandler(os.path.join(checkpoint_path, 
            log_filename + '.log'))
        stream_hdlr.setFormatter(formatter)
        file_hdlr.setFormatter(formatter)
        logger.addHandler(stream_hdlr)
        logger.addHandler(file_hdlr)
        logger.setLevel(logging.INFO)
    else:
        logger = logging.getLogger()
    return logger


logger = get_logger()


class Trainer(nn.Module):

    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.netG = Generator(self.config['netG'], self.use_cuda, self.
            device_ids)
        self.localD = LocalDis(self.config['netD'], self.use_cuda, self.
            device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.
            device_ids)
        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self
            .config['lr'], betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.localD.parameters()) + list(self.globalD.
            parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
            betas=(self.config['beta1'], self.config['beta2']))
        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.localD.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

    def forward(self, x, bboxes, masks, ground_truth, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}
        x1, x2, offset_flow = self.netG(x, masks)
        local_patch_gt = local_patch(ground_truth, bboxes)
        x1_inpaint = x1 * masks + x * (1.0 - masks)
        x2_inpaint = x2 * masks + x * (1.0 - masks)
        local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(self
            .localD, local_patch_gt, local_patch_x2_inpaint.detach())
        global_real_pred, global_fake_pred = self.dis_forward(self.globalD,
            ground_truth, x2_inpaint.detach())
        losses['wgan_d'] = torch.mean(local_patch_fake_pred -
            local_patch_real_pred) + torch.mean(global_fake_pred -
            global_real_pred) * self.config['global_wgan_loss_alpha']
        local_penalty = self.calc_gradient_penalty(self.localD,
            local_patch_gt, local_patch_x2_inpaint.detach())
        global_penalty = self.calc_gradient_penalty(self.globalD,
            ground_truth, x2_inpaint.detach())
        losses['wgan_gp'] = local_penalty + global_penalty
        if compute_loss_g:
            sd_mask = spatial_discounting_mask(self.config)
            losses['l1'] = l1_loss(local_patch_x1_inpaint * sd_mask, 
                local_patch_gt * sd_mask) * self.config['coarse_l1_alpha'
                ] + l1_loss(local_patch_x2_inpaint * sd_mask, 
                local_patch_gt * sd_mask)
            losses['ae'] = l1_loss(x1 * (1.0 - masks), ground_truth * (1.0 -
                masks)) * self.config['coarse_l1_alpha'] + l1_loss(x2 * (
                1.0 - masks), ground_truth * (1.0 - masks))
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                self.localD, local_patch_gt, local_patch_x2_inpaint)
            global_real_pred, global_fake_pred = self.dis_forward(self.
                globalD, ground_truth, x2_inpaint)
            losses['wgan_g'] = -torch.mean(local_patch_fake_pred) - torch.mean(
                global_fake_pred) * self.config['global_wgan_loss_alpha']
        return losses, x2_inpaint, offset_flow

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)
        return real_pred, fake_pred

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()
        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())
        if self.use_cuda:
            grad_outputs = grad_outputs
        gradients = autograd.grad(outputs=disc_interpolates, inputs=
            interpolates, grad_outputs=grad_outputs, create_graph=True,
            retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks)
        x2_inpaint = x2 * masks + x * (1.0 - masks)
        return x2_inpaint, offset_flow

    def save_model(self, checkpoint_dir, iteration):
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save({'localD': self.localD.state_dict(), 'globalD': self.
            globalD.state_dict()}, dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(), 'dis': self.
            optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        last_model_name = get_model_list(checkpoint_dir, 'gen', iteration=
            iteration)
        self.netG.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])
        if not test:
            last_model_name = get_model_list(checkpoint_dir, 'dis',
                iteration=iteration)
            state_dict = torch.load(last_model_name)
            self.localD.load_state_dict(state_dict['localD'])
            self.globalD.load_state_dict(state_dict['globalD'])
            state_dict = torch.load(os.path.join(checkpoint_dir,
                'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            self.optimizer_g.load_state_dict(state_dict['gen'])
        None
        logger.info('Resume from {} at iteration {}'.format(checkpoint_dir,
            iteration))
        return iteration


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_daa233_generative_inpainting_pytorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(CoarseGenerator(*[], **{'input_dim': 4, 'cnum': 4}), [torch.rand([4, 1, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DisConvModule(*[], **{'input_dim': 4, 'cnum': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(Conv2dBlock(*[], **{'input_dim': 4, 'output_dim': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

